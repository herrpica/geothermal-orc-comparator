"""
LanceDB vector store management for knowledge chunks.

Storage lives in knowledge/data/lancedb/ (gitignored, persists across sessions).
"""

import math
import os
import uuid
from datetime import datetime, timezone

import lancedb
import pyarrow as pa

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
DB_PATH = os.path.join(DATA_DIR, "lancedb")
TABLE_NAME = "knowledge_chunks"
VECTOR_DIM = 384

_SCHEMA = pa.schema([
    pa.field("chunk_id", pa.utf8()),
    pa.field("document_id", pa.utf8()),
    pa.field("text", pa.utf8()),
    pa.field("vector", pa.list_(pa.float32(), VECTOR_DIM)),
    pa.field("source_file", pa.utf8()),
    pa.field("source_type", pa.utf8()),
    pa.field("section_title", pa.utf8()),
    pa.field("page_number", pa.int32()),
    pa.field("chunk_index", pa.int32()),
    pa.field("domain", pa.utf8()),
    pa.field("relevance_tags", pa.list_(pa.utf8())),
    pa.field("ingested_at", pa.utf8()),
])


def _is_null(val) -> bool:
    """Check if a value is None, NaN, NaT, or pandas NA — any type."""
    if val is None:
        return True
    try:
        # Catches float('nan'), numpy.nan, numpy.float64('nan'), etc.
        if isinstance(val, (float, int)):
            return math.isnan(val) or math.isinf(val)
    except (TypeError, ValueError):
        pass
    try:
        # Catches pandas NA, NaT, numpy NaN of any dtype
        import numpy as np
        if isinstance(val, (np.floating, np.integer)):
            return bool(np.isnan(val))
        if isinstance(val, np.generic):
            return bool(val != val)  # NaN != NaN
    except (TypeError, ImportError):
        pass
    # pandas NA / NaT
    vtype = type(val).__name__
    if vtype in ("NAType", "NaTType"):
        return True
    # str representation check as last resort (catches edge cases)
    s = str(val)
    if s in ("nan", "None", "<NA>", "NaT", ""):
        return True
    return False


def _safe_str(val, default: str = "") -> str:
    """Convert any value to a Python str, returning default for nulls."""
    if _is_null(val):
        return default
    return str(val)


def _safe_int(val, default: int = 0) -> int:
    """Convert any value to a Python int, returning default for nulls."""
    if _is_null(val):
        return default
    try:
        return int(val)
    except (ValueError, TypeError):
        return default


def _sanitize_rows(rows: list[dict]) -> list[dict]:
    """Enforce correct schema types on every row before ANY LanceDB write.

    Handles: None, NaN (Python and numpy), pandas NA/NaT, numpy scalar
    types, missing fields, wrong types. Returns a list of dicts where
    every row has exactly the _SCHEMA fields with correct Python-native types.
    """
    clean = []
    for row in rows:
        r = {}

        # ── String fields (pyarrow utf8) — never None ────────────────
        r["chunk_id"]       = _safe_str(row.get("chunk_id"), "")
        r["document_id"]    = _safe_str(row.get("document_id"), "")
        r["text"]           = _safe_str(row.get("text"), "")
        r["source_file"]    = _safe_str(row.get("source_file"), "")
        r["source_type"]    = _safe_str(row.get("source_type"), "text") or "text"
        r["section_title"]  = _safe_str(row.get("section_title"), "")
        r["domain"]         = _safe_str(row.get("domain"), "general") or "general"
        r["ingested_at"]    = _safe_str(row.get("ingested_at"), "")

        # ── Integer fields (pyarrow int32) — never None ──────────────
        r["page_number"]    = _safe_int(row.get("page_number"), 0)
        r["chunk_index"]    = _safe_int(row.get("chunk_index"), 0)

        # ── relevance_tags (pyarrow list<utf8>) — never None ─────────
        tags = row.get("relevance_tags")
        if tags is None or not isinstance(tags, (list, tuple)):
            try:
                # Handle numpy arrays
                if hasattr(tags, "tolist"):
                    tags = tags.tolist()
                else:
                    tags = []
            except Exception:
                tags = []
        r["relevance_tags"] = [_safe_str(t) for t in tags
                               if t is not None and not _is_null(t)]

        # ── vector (pyarrow list<float32>) — never None ──────────────
        vec = row.get("vector")
        if vec is None:
            r["vector"] = [0.0] * VECTOR_DIM
        elif hasattr(vec, "tolist"):
            # numpy array → Python list of floats
            r["vector"] = [float(x) for x in vec.tolist()]
        elif isinstance(vec, (list, tuple)):
            r["vector"] = [float(x) if x is not None and not _is_null(x) else 0.0
                           for x in vec]
        else:
            r["vector"] = [0.0] * VECTOR_DIM

        # Pad or truncate vector to VECTOR_DIM
        vlen = len(r["vector"])
        if vlen < VECTOR_DIM:
            r["vector"].extend([0.0] * (VECTOR_DIM - vlen))
        elif vlen > VECTOR_DIM:
            r["vector"] = r["vector"][:VECTOR_DIM]

        clean.append(r)

    return clean


def get_db() -> lancedb.DBConnection:
    """Open (or create) the LanceDB database."""
    os.makedirs(DB_PATH, exist_ok=True)
    return lancedb.connect(DB_PATH)


def ensure_table() -> lancedb.table.Table:
    """Return the knowledge_chunks table, creating it if needed."""
    db = get_db()
    if TABLE_NAME in db.table_names():
        return db.open_table(TABLE_NAME)
    return db.create_table(TABLE_NAME, schema=_SCHEMA)


def add_chunks(chunks: list[dict]) -> int:
    """Insert chunk dicts into the table. Returns count added.

    Every value is sanitized to a Python-native type before write.
    Chunks with no meaningful text (< 10 chars) are skipped.
    """
    if not chunks:
        return 0

    table = ensure_table()
    now = datetime.now(timezone.utc).isoformat()

    # Get schema field names from the table
    schema_fields = [f.name for f in _SCHEMA]

    sanitized = []
    for chunk in chunks:
        row = {}
        for field_name in schema_fields:
            val = chunk.get(field_name)
            # Convert None/NaN to appropriate default
            if _is_null(val):
                val = "" if field_name not in ("page_number", "chunk_index",
                                                "vector", "relevance_tags") else val
            # Type-specific handling
            if field_name in ("page_number", "chunk_index"):
                row[field_name] = _safe_int(val, 0)
            elif field_name == "vector":
                if val is None:
                    row[field_name] = [0.0] * VECTOR_DIM
                elif hasattr(val, "tolist"):
                    row[field_name] = [float(x) for x in val.tolist()]
                elif isinstance(val, (list, tuple)):
                    row[field_name] = [float(x) if not _is_null(x) else 0.0
                                       for x in val]
                else:
                    row[field_name] = [0.0] * VECTOR_DIM
                # Ensure correct dimension
                vlen = len(row[field_name])
                if vlen < VECTOR_DIM:
                    row[field_name].extend([0.0] * (VECTOR_DIM - vlen))
                elif vlen > VECTOR_DIM:
                    row[field_name] = row[field_name][:VECTOR_DIM]
            elif field_name == "relevance_tags":
                tags = val if isinstance(val, (list, tuple)) else []
                if hasattr(val, "tolist"):
                    tags = val.tolist()
                row[field_name] = [_safe_str(t) for t in tags
                                   if t is not None and not _is_null(t)]
            else:
                # All other fields are utf8 strings
                row[field_name] = _safe_str(val, "")

        # Apply non-empty defaults
        if not row.get("chunk_id"):
            row["chunk_id"] = str(uuid.uuid4())
        if not row.get("source_type"):
            row["source_type"] = "text"
        if not row.get("domain"):
            row["domain"] = "general"
        if not row.get("ingested_at"):
            row["ingested_at"] = now

        # Skip chunks with no meaningful text content
        if len(str(row.get("text", "")).strip()) < 10:
            continue

        sanitized.append(row)

    if not sanitized:
        return 0

    # Build a pyarrow Table with the explicit schema to prevent
    # type inference issues (the root cause of "cast from string to null").
    columns = {f.name: [row[f.name] for row in sanitized] for f in _SCHEMA}
    pa_table = pa.table(columns, schema=_SCHEMA)
    table.add(pa_table)
    return len(sanitized)


def search(query_vector: list[float], top_k: int = 10,
           domain_filter: str | None = None) -> list[dict]:
    """Vector similarity search. Returns list of result dicts."""
    table = ensure_table()
    try:
        count = table.count_rows()
    except Exception:
        count = 0
    if count == 0:
        return []

    q = table.search(query_vector).limit(top_k)
    if domain_filter:
        q = q.where(f"domain = '{domain_filter}'")
    try:
        results = q.to_list()
    except Exception:
        return []
    return results


def delete_document(document_id: str) -> int:
    """Delete all chunks for a document_id. Returns count deleted."""
    table = ensure_table()
    try:
        before = table.count_rows()
    except Exception:
        before = 0
    table.delete(f"document_id = '{document_id}'")
    try:
        after = table.count_rows()
    except Exception:
        after = 0
    return before - after


def table_stats() -> dict:
    """Return summary stats about the knowledge store."""
    table = ensure_table()
    try:
        total = table.count_rows()
    except Exception:
        total = 0
    if total == 0:
        return {"total_chunks": 0, "total_documents": 0, "domains": []}

    df = table.to_pandas()
    return {
        "total_chunks": total,
        "total_documents": df["document_id"].nunique(),
        "domains": sorted(df["domain"].unique().tolist()),
    }
