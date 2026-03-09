"""
Knowledge Base for ORC-Comparator — persistent vector store, ingestion,
gap tracking, and self-assessment against $2,000/kW target.
"""

from knowledge.vector_store import (
    get_db, ensure_table, add_chunks, search, delete_document, table_stats,
)
from knowledge.ingestion import (
    ingest_pdf, ingest_text, ingest_structured, ingest_manual_entry,
    get_all_documents, delete_document as delete_doc_full, TfidfEmbedder,
)
from knowledge.gap_tracker import (
    add_gap, resolve_gap, get_active_gaps, get_hard_gaps, get_soft_gaps,
    estimate_total_uncertainty,
)
from knowledge.query_interface import (
    query_knowledge, query_for_technology, query_for_cost_domain,
    get_knowledge_summary,
)
from knowledge.self_assessment import run_self_assessment, KNOWLEDGE_DOMAINS, TARGET_COST_PER_KW
