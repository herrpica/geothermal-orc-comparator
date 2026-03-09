"""
Public retrieval API — query_knowledge(), auto-gap detection,
technology and cost-domain queries.
"""

from knowledge.vector_store import search, table_stats
from knowledge.ingestion import TfidfEmbedder, get_all_documents
from knowledge.gap_tracker import add_gap, get_active_gaps

# Minimum similarity score to consider a result relevant
RELEVANCE_THRESHOLD = 0.3
# Below this chunk count for a domain, flag as potential gap
SPARSE_DOMAIN_THRESHOLD = 3


def query_knowledge(query_text: str, top_k: int = 5,
                    domain: str | None = None,
                    auto_log_gaps: bool = True) -> dict:
    """
    Search the knowledge base. Returns results + metadata.
    If auto_log_gaps is True and results are sparse, logs a soft gap.
    """
    embedder = TfidfEmbedder()
    vectors = embedder.transform([query_text])
    if not vectors:
        results = []
    else:
        results = search(vectors[0], top_k=top_k, domain_filter=domain)

    # Assess quality
    relevant = [r for r in results if r.get("_distance", 999) < RELEVANCE_THRESHOLD]
    has_good_results = len(relevant) >= 1

    gap_logged = None
    if auto_log_gaps and not has_good_results:
        target_domain = domain or "general"
        existing = get_active_gaps(domain=target_domain)
        already_tracked = any(query_text.lower() in g["title"].lower()
                              or g["title"].lower() in query_text.lower()
                              for g in existing)
        if not already_tracked:
            gap_logged = add_gap(
                domain=target_domain,
                title=f"Insufficient knowledge: {query_text[:80]}",
                description=f"Query returned {len(relevant)} relevant results "
                            f"(threshold: {RELEVANCE_THRESHOLD}). "
                            f"Total results: {len(results)}.",
                severity="soft",
                detected_by="auto_query",
            )

    return {
        "query": query_text,
        "domain_filter": domain,
        "results": results,
        "result_count": len(results),
        "relevant_count": len(relevant),
        "gap_logged": gap_logged,
    }


def query_for_technology(technology_id: str,
                         aspect: str = "general") -> dict:
    """Query knowledge base for a specific technology."""
    query = f"{technology_id} {aspect} geothermal power"
    return query_knowledge(query, top_k=5, auto_log_gaps=True)


def query_for_cost_domain(domain: str, top_k: int = 5) -> dict:
    """Query within a specific cost domain."""
    return query_knowledge(
        f"{domain} cost performance specifications",
        top_k=top_k,
        domain=domain,
        auto_log_gaps=True,
    )


def get_knowledge_summary() -> dict:
    """Overview of knowledge base contents and coverage."""
    stats = table_stats()
    docs = get_all_documents()
    gaps = get_active_gaps()

    domain_doc_counts = {}
    for d in docs:
        dom = d.get("domain", "general")
        domain_doc_counts[dom] = domain_doc_counts.get(dom, 0) + 1

    sparse_domains = [
        dom for dom, cnt in domain_doc_counts.items()
        if cnt < SPARSE_DOMAIN_THRESHOLD
    ]

    return {
        "total_chunks": stats["total_chunks"],
        "total_documents": stats["total_documents"],
        "domains": stats["domains"],
        "domain_document_counts": domain_doc_counts,
        "active_gaps": len(gaps),
        "sparse_domains": sparse_domains,
    }
