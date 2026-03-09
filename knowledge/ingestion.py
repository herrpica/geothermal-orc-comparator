"""
Document ingestion pipeline — PDF/text/JSON parsing, chunking,
TF-IDF embedding, and document registry.
"""

import json
import os
import re
import uuid
from datetime import datetime, timezone

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

from knowledge.vector_store import (
    add_chunks, delete_document as vs_delete, ensure_table, VECTOR_DIM,
    _sanitize_rows, _SCHEMA,
)
import pyarrow as pa

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DOCS_PATH = os.path.join(DATA_DIR, "documents.json")
MODEL_PATH = os.path.join(DATA_DIR, "tfidf_model.pkl")

CHUNK_SIZE = 800      # characters per chunk
CHUNK_OVERLAP = 100   # overlap between chunks
MIN_CHUNK_LENGTH = 50 # discard chunks shorter than this (noise from graphic-heavy PDFs)


# ── Document registry ─────────────────────────────────────────────────────

def _load_docs() -> list[dict]:
    if os.path.exists(DOCS_PATH):
        with open(DOCS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_docs(docs: list[dict]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(DOCS_PATH, "w", encoding="utf-8") as f:
        json.dump(docs, f, indent=2)


def get_all_documents() -> list[dict]:
    return _load_docs()


def delete_document(document_id: str) -> bool:
    """Remove document from registry and vector store."""
    docs = _load_docs()
    before = len(docs)
    docs = [d for d in docs if d["document_id"] != document_id]
    if len(docs) == before:
        return False
    _save_docs(docs)
    vs_delete(document_id)
    _refit_embeddings()
    return True


# ── TF-IDF Embedder ──────────────────────────────────────────────────────

class TfidfEmbedder:
    """Wraps scikit-learn TfidfVectorizer, persists to disk."""

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=VECTOR_DIM,
            sublinear_tf=True,
            ngram_range=(1, 2),
            stop_words="english",
        )

    def fit_transform(self, texts: list[str]) -> list[list[float]]:
        """Fit on texts and return vectors. Saves model to disk."""
        if not texts:
            return []
        matrix = self.vectorizer.fit_transform(texts)
        vectors = self._pad(matrix.toarray())
        os.makedirs(DATA_DIR, exist_ok=True)
        joblib.dump(self.vectorizer, MODEL_PATH)
        return vectors.tolist()

    def transform(self, texts: list[str]) -> list[list[float]]:
        """Transform texts using saved model. Falls back to fit_transform."""
        if not texts:
            return []
        if os.path.exists(MODEL_PATH):
            self.vectorizer = joblib.load(MODEL_PATH)
            matrix = self.vectorizer.transform(texts)
            return self._pad(matrix.toarray()).tolist()
        return self.fit_transform(texts)

    @staticmethod
    def _pad(arr: np.ndarray) -> np.ndarray:
        """Pad or truncate to VECTOR_DIM columns."""
        n, m = arr.shape
        if m >= VECTOR_DIM:
            return arr[:, :VECTOR_DIM].astype(np.float32)
        padded = np.zeros((n, VECTOR_DIM), dtype=np.float32)
        padded[:, :m] = arr
        return padded


# ── Chunking ──────────────────────────────────────────────────────────────

def _chunk_text(text: str, chunk_size: int = CHUNK_SIZE,
                overlap: int = CHUNK_OVERLAP) -> list[str]:
    """Split text into overlapping chunks, breaking at sentence boundaries."""
    text = text.strip()
    if not text:
        return []
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end < len(text):
            # Try to break at sentence boundary
            for sep in [". ", ".\n", "\n\n", "\n", " "]:
                idx = text.rfind(sep, start + chunk_size // 2, end)
                if idx != -1:
                    end = idx + len(sep)
                    break
        chunks.append(text[start:end].strip())
        start = end - overlap
    return [c for c in chunks if c]


def _detect_section(text: str) -> str:
    """Try to detect a section heading from the chunk text."""
    lines = text.split("\n", 3)
    first = lines[0].strip()
    if first and len(first) < 100 and first[0].isupper():
        if first.endswith(":") or first.isupper() or re.match(r"^\d+[\.\)]\s", first):
            return first.rstrip(":")
    return ""


# ── Ingestion entry points ───────────────────────────────────────────────

def _ingest_chunks(raw_chunks: list[dict], filename: str, source_type: str,
                   domain: str, tags: list[str] | None,
                   document_id: str | None = None) -> dict:
    """Common ingestion path: embed, store, register."""
    doc_id = document_id or str(uuid.uuid4())
    now = datetime.now(timezone.utc).isoformat()
    tags = tags or []

    # Filter out None, empty, whitespace-only, and too-short chunks
    raw_chunks = [
        c for c in raw_chunks
        if c.get("text") and c["text"].strip() and len(c["text"].strip()) >= MIN_CHUNK_LENGTH
    ]
    if not raw_chunks:
        return {
            "document_id": doc_id,
            "filename": filename,
            "source_type": source_type,
            "domain": domain,
            "tags": tags,
            "chunk_count": 0,
            "ingested_at": now,
            "warning": "No valid text chunks extracted. Document may be primarily graphical.",
        }

    texts = [c["text"] for c in raw_chunks]
    embedder = TfidfEmbedder()

    # Refit on full corpus for best TF-IDF representation
    all_texts = _get_all_chunk_texts() + texts
    embedder.fit_transform(all_texts)
    # Now transform just the new texts with the refitted model
    vectors = embedder.transform(texts)

    store_chunks = []
    for i, (rc, vec) in enumerate(zip(raw_chunks, vectors)):
        store_chunks.append({
            "chunk_id": str(uuid.uuid4()),
            "document_id": doc_id,
            "text": rc["text"],
            "vector": vec,
            "source_file": filename,
            "source_type": source_type,
            "section_title": rc.get("section_title", ""),
            "page_number": rc.get("page_number", 0),
            "chunk_index": i,
            "domain": domain,
            "relevance_tags": tags,
            "ingested_at": now,
        })

    count = add_chunks(store_chunks)

    # Update existing vectors with new TF-IDF model
    _update_existing_vectors(embedder, doc_id)

    doc_meta = {
        "document_id": doc_id,
        "filename": filename,
        "source_type": source_type,
        "domain": domain,
        "tags": tags,
        "chunk_count": count,
        "ingested_at": now,
    }
    docs = _load_docs()
    docs.append(doc_meta)
    _save_docs(docs)

    return doc_meta


def _get_all_chunk_texts() -> list[str]:
    """Retrieve all existing chunk texts from the vector store."""
    table = ensure_table()
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    if count == 0:
        return []
    df = table.to_pandas()
    return df["text"].tolist()


def _update_existing_vectors(embedder: TfidfEmbedder, exclude_doc_id: str):
    """Re-embed existing chunks with the refitted TF-IDF model."""
    table = ensure_table()
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    if count == 0:
        return

    df = table.to_pandas()
    existing = df[df["document_id"] != exclude_doc_id]
    if existing.empty:
        return

    texts = existing["text"].tolist()
    new_vectors = embedder.transform(texts)

    # Rebuild all chunks with updated vectors
    all_rows = []
    # Existing chunks (re-embedded)
    for idx, (_, row) in enumerate(existing.iterrows()):
        r = row.to_dict()
        r["vector"] = new_vectors[idx]
        all_rows.append(r)
    # New chunks (already in table, keep as-is)
    new_rows = df[df["document_id"] == exclude_doc_id]
    for _, row in new_rows.iterrows():
        all_rows.append(row.to_dict())

    if all_rows:
        all_rows = _sanitize_rows(all_rows)
        from knowledge.vector_store import get_db, TABLE_NAME
        db = get_db()
        db.drop_table(TABLE_NAME)
        # Always use explicit schema to prevent null-type column inference
        columns = {f.name: [row[f.name] for row in all_rows] for f in _SCHEMA}
        pa_table = pa.table(columns, schema=_SCHEMA)
        db.create_table(TABLE_NAME, data=pa_table)


def _refit_embeddings():
    """Refit TF-IDF on remaining corpus after a document deletion."""
    table = ensure_table()
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    if count == 0:
        return

    df = table.to_pandas()
    texts = df["text"].tolist()
    embedder = TfidfEmbedder()
    vectors = embedder.fit_transform(texts)

    rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        r = row.to_dict()
        r["vector"] = vectors[idx]
        rows.append(r)

    rows = _sanitize_rows(rows)
    from knowledge.vector_store import get_db, TABLE_NAME
    db = get_db()
    db.drop_table(TABLE_NAME)
    # Always use explicit schema to prevent null-type column inference
    columns = {f.name: [row[f.name] for row in rows] for f in _SCHEMA}
    pa_table = pa.table(columns, schema=_SCHEMA)
    db.create_table(TABLE_NAME, data=pa_table)


# ── Public ingest functions ──────────────────────────────────────────────

def ingest_pdf(file_bytes: bytes, filename: str,
               domain: str = "general", tags: list[str] | None = None) -> dict:
    """Extract text from PDF bytes, chunk, embed, store."""
    import pymupdf
    doc = pymupdf.open(stream=file_bytes, filetype="pdf")
    raw_chunks = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text()
        if not text.strip():
            continue
        chunks = _chunk_text(text)
        for c in chunks:
            raw_chunks.append({
                "text": c,
                "page_number": page_num + 1,
                "section_title": _detect_section(c),
            })
    doc.close()
    if not raw_chunks:
        raw_chunks = [{"text": f"[Empty PDF: {filename}]", "page_number": 0, "section_title": ""}]
    result = _ingest_chunks(raw_chunks, filename, "pdf", domain, tags)
    if result["chunk_count"] < 5 and "warning" not in result:
        result["warning"] = (
            "Document appears to be primarily graphical "
            f"({result['chunk_count']} text chunks extracted). "
            "Consider uploading a text summary instead."
        )
    return result


def ingest_text(text: str, filename: str,
                domain: str = "general", tags: list[str] | None = None) -> dict:
    """Chunk and embed plain text."""
    chunks = _chunk_text(text)
    raw_chunks = [{"text": c, "page_number": 0, "section_title": _detect_section(c)} for c in chunks]
    if not raw_chunks:
        raw_chunks = [{"text": text or "[empty]", "page_number": 0, "section_title": ""}]
    return _ingest_chunks(raw_chunks, filename, "text", domain, tags)


def ingest_structured(data: dict | list, filename: str,
                      domain: str = "general", tags: list[str] | None = None) -> dict:
    """Ingest structured JSON data."""
    text = json.dumps(data, indent=2, default=str)
    chunks = _chunk_text(text)
    raw_chunks = [{"text": c, "page_number": 0, "section_title": _detect_section(c)} for c in chunks]
    return _ingest_chunks(raw_chunks, filename, "json", domain, tags)


def ingest_manual_entry(title: str, text: str,
                        domain: str = "general", tags: list[str] | None = None) -> dict:
    """Ingest a manual knowledge entry."""
    full_text = f"{title}\n\n{text}" if title else text
    chunks = _chunk_text(full_text)
    raw_chunks = [{"text": c, "page_number": 0, "section_title": title} for c in chunks]
    return _ingest_chunks(raw_chunks, f"manual:{title}", "manual", domain, tags)
