"""
Knowledge Base tab — Streamlit UI with 4 sub-tabs:
  1. Upload & Ingest
  2. Knowledge Inventory
  3. Knowledge Gaps
  4. Self-Assessment
"""

import streamlit as st

from knowledge.ingestion import (
    ingest_pdf, ingest_text, ingest_manual_entry,
    get_all_documents, delete_document,
)
from knowledge.gap_tracker import (
    add_gap, resolve_gap, get_active_gaps,
    get_hard_gaps, get_soft_gaps, estimate_total_uncertainty,
)
from knowledge.query_interface import query_knowledge, get_knowledge_summary
from knowledge.self_assessment import (
    run_self_assessment, KNOWLEDGE_DOMAINS, TARGET_COST_PER_KW,
)
from knowledge.vector_store import table_stats

DOMAIN_OPTIONS = list(KNOWLEDGE_DOMAINS.keys()) + ["general"]


def render_knowledge_tab(design_basis: dict):
    """Main entry point for the Knowledge Base tab."""
    st.header("Knowledge Base")
    st.caption("Persistent knowledge store for geothermal ORC engineering data")

    sub1, sub2, sub3, sub4 = st.tabs([
        "Upload & Ingest", "Knowledge Inventory",
        "Knowledge Gaps", "Self-Assessment",
    ])

    with sub1:
        _render_upload_tab()
    with sub2:
        _render_inventory_tab()
    with sub3:
        _render_gaps_tab()
    with sub4:
        _render_assessment_tab()


# ── Sub-tab 1: Upload & Ingest ───────────────────────────────────────────

def _render_upload_tab():
    st.subheader("Upload Documents")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**File Upload**")
        uploaded = st.file_uploader(
            "Upload PDF or text file",
            type=["pdf", "txt"],
            key="kb_file_upload",
        )
        file_domain = st.selectbox(
            "Domain", DOMAIN_OPTIONS, key="kb_file_domain",
        )
        file_tags = st.text_input(
            "Tags (comma-separated)", key="kb_file_tags",
        )

        if st.button("Ingest File", key="kb_ingest_file", disabled=uploaded is None):
            tags = [t.strip() for t in file_tags.split(",") if t.strip()] if file_tags else []
            with st.spinner("Ingesting..."):
                if uploaded.name.lower().endswith(".pdf"):
                    result = ingest_pdf(uploaded.read(), uploaded.name,
                                        domain=file_domain, tags=tags)
                else:
                    text = uploaded.read().decode("utf-8", errors="replace")
                    result = ingest_text(text, uploaded.name,
                                         domain=file_domain, tags=tags)
            if result.get("warning"):
                st.warning(result["warning"])
            if result["chunk_count"] > 0:
                st.success(f"Ingested **{result['filename']}**: "
                           f"{result['chunk_count']} chunks in domain `{result['domain']}`")
            else:
                st.error(f"**{result['filename']}**: No valid text chunks extracted.")

    with col2:
        st.markdown("**Manual Entry**")
        with st.form("kb_manual_form"):
            entry_title = st.text_input("Title")
            entry_text = st.text_area("Content", height=150)
            entry_domain = st.selectbox("Domain", DOMAIN_OPTIONS,
                                        key="kb_manual_domain")
            entry_tags = st.text_input("Tags (comma-separated)",
                                       key="kb_manual_tags")
            submitted = st.form_submit_button("Add Entry")

        if submitted and entry_text.strip():
            tags = [t.strip() for t in entry_tags.split(",") if t.strip()] if entry_tags else []
            result = ingest_manual_entry(entry_title, entry_text,
                                         domain=entry_domain, tags=tags)
            st.success(f"Added manual entry: {result['chunk_count']} chunks")

    # Search test
    st.markdown("---")
    st.subheader("Search Test")
    search_query = st.text_input("Test query", key="kb_search_query")
    search_domain = st.selectbox("Filter domain", ["(all)"] + DOMAIN_OPTIONS,
                                 key="kb_search_domain")
    if st.button("Search", key="kb_search_btn") and search_query:
        domain_filter = None if search_domain == "(all)" else search_domain
        results = query_knowledge(search_query, top_k=5,
                                  domain=domain_filter, auto_log_gaps=False)
        st.write(f"**{results['result_count']} results** "
                 f"({results['relevant_count']} relevant)")
        for r in results["results"]:
            with st.expander(f"{r.get('source_file', '?')} — chunk {r.get('chunk_index', '?')}"):
                st.text(r.get("text", "")[:500])
                st.caption(f"Domain: {r.get('domain')} | "
                           f"Distance: {r.get('_distance', 'N/A'):.4f}"
                           if isinstance(r.get('_distance'), (int, float))
                           else f"Domain: {r.get('domain')}")


# ── Sub-tab 2: Knowledge Inventory ───────────────────────────────────────

def _render_inventory_tab():
    stats = table_stats()
    col1, col2, col3 = st.columns(3)
    col1.metric("Total Chunks", stats["total_chunks"])
    col2.metric("Total Documents", stats["total_documents"])
    col3.metric("Domains", len(stats["domains"]))

    docs = get_all_documents()
    if not docs:
        st.info("No documents ingested yet. Use the Upload tab to add knowledge.")
        return

    st.subheader("Documents")
    for doc in reversed(docs):  # newest first
        col_a, col_b = st.columns([4, 1])
        with col_a:
            st.markdown(
                f"**{doc['filename']}** — "
                f"{doc['chunk_count']} chunks | "
                f"`{doc['domain']}` | "
                f"{doc.get('source_type', '?')} | "
                f"{doc['ingested_at'][:10]}"
            )
            if doc.get("tags"):
                st.caption("Tags: " + ", ".join(doc["tags"]))
        with col_b:
            if st.button("Delete", key=f"kb_del_{doc['document_id']}"):
                delete_document(doc["document_id"])
                st.rerun()


# ── Sub-tab 3: Knowledge Gaps ────────────────────────────────────────────

def _render_gaps_tab():
    uncertainty = estimate_total_uncertainty()

    col1, col2, col3 = st.columns(3)
    col1.metric("Hard Gaps", uncertainty["hard_gap_count"])
    col2.metric("Soft Gaps", uncertainty["soft_gap_count"])
    col3.metric("Total $/kW Impact", f"${uncertainty['total_impact']:,.0f}")

    # Hard gaps
    hard = get_hard_gaps()
    if hard:
        st.subheader("Hard Gaps (blocking)")
        for g in hard:
            with st.expander(f"🔴 {g['title']} — {g['domain']}", expanded=True):
                st.write(g["description"])
                st.caption(f"Impact: ${g['impact_on_cost_per_kW']:,.0f}/kW | "
                           f"Detected: {g['detected_at'][:10]} by {g['detected_by']}")
                if g.get("related_technologies"):
                    st.caption("Technologies: " + ", ".join(g["related_technologies"]))
                if st.button("Resolve", key=f"kb_resolve_{g['gap_id']}"):
                    resolve_gap(g["gap_id"])
                    st.rerun()

    # Soft gaps
    soft = get_soft_gaps()
    if soft:
        st.subheader("Soft Gaps (advisory)")
        for g in soft:
            with st.expander(f"🟡 {g['title']} — {g['domain']}"):
                st.write(g["description"])
                st.caption(f"Impact: ${g['impact_on_cost_per_kW']:,.0f}/kW | "
                           f"Detected: {g['detected_at'][:10]} by {g['detected_by']}")
                if st.button("Resolve", key=f"kb_resolve_{g['gap_id']}"):
                    resolve_gap(g["gap_id"])
                    st.rerun()

    if not hard and not soft:
        st.success("No active knowledge gaps.")

    # Add gap form
    st.markdown("---")
    st.subheader("Add Knowledge Gap")
    with st.form("kb_add_gap_form"):
        gap_domain = st.selectbox("Domain", list(KNOWLEDGE_DOMAINS.keys()),
                                  key="kb_gap_domain")
        gap_title = st.text_input("Title")
        gap_desc = st.text_area("Description", height=100)
        gap_severity = st.selectbox("Severity", ["soft", "hard"],
                                    key="kb_gap_severity")
        gap_impact = st.number_input("Impact ($/kW)", min_value=0.0,
                                     value=0.0, step=10.0,
                                     key="kb_gap_impact")
        gap_submitted = st.form_submit_button("Add Gap")

    if gap_submitted and gap_title.strip():
        add_gap(
            domain=gap_domain,
            title=gap_title,
            description=gap_desc,
            severity=gap_severity,
            impact_on_cost_per_kW=gap_impact,
            detected_by="user",
        )
        st.success(f"Added {gap_severity} gap: {gap_title}")
        st.rerun()


# ── Sub-tab 4: Self-Assessment ───────────────────────────────────────────

def _render_assessment_tab():
    st.subheader(f"Self-Assessment: ${TARGET_COST_PER_KW:,}/kW Target")

    if st.button("Run Self-Assessment", key="kb_run_assessment"):
        with st.spinner("Assessing knowledge coverage..."):
            result = run_self_assessment()
        st.session_state["kb_assessment_result"] = result

    result = st.session_state.get("kb_assessment_result")
    if not result:
        st.info("Click **Run Self-Assessment** to evaluate knowledge coverage "
                "against the $2,000/kW installed cost target.")
        return

    # Summary metrics
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Target", f"${result['target_cost_per_kw']:,}/kW")
    col2.metric("Baseline", f"${result['baseline_cost_per_kw']:,.0f}/kW")
    col3.metric("Gap to Target",
                f"${result['gap_to_target']:,.0f}/kW",
                delta=f"-${abs(result['gap_to_target']):,.0f}" if result['gap_to_target'] > 0 else None,
                delta_color="inverse")
    readiness_pct = result["readiness_score"] * 100
    col4.metric("Readiness", f"{readiness_pct:.0f}%")

    # Domain cards
    st.subheader("Domain Coverage")
    assessments = result["domain_assessments"]
    for domain_id, a in assessments.items():
        coverage_color = {
            "none": "🔴", "weak": "🟠", "partial": "🟡",
            "moderate": "🔵", "strong": "🟢",
        }.get(a["coverage"], "⚪")

        with st.expander(
            f"{coverage_color} {a['label']} — {a['coverage']} "
            f"({a['score']:.0%}) | {a['cost_share_pct']}% cost share"
        ):
            info = KNOWLEDGE_DOMAINS[domain_id]
            st.write(info["description"])
            by_tag = a.get('chunk_count_by_tag', 0)
            by_content = a.get('chunk_count_by_content', 0)
            st.write(f"- **Chunks (effective):** {a['chunk_count']} "
                     f"(by tag: {by_tag}, by content: {by_content})")
            st.write(f"- **Documents:** {a['document_count']}")
            st.write(f"- **Target contribution:** ${a['target_contribution']:,}/kW")
            st.write(f"- **Active gaps:** {a['active_gaps']} "
                     f"({a['hard_gaps']} hard)")

    # Critical path
    if result["critical_path"]:
        st.subheader("Critical Path")
        for cp in result["critical_path"]:
            st.warning(f"**{cp['label']}** — {cp['coverage']} coverage, "
                       f"{cp['cost_share_pct']}% cost share, "
                       f"{cp['hard_gaps']} hard gap(s)")

    # Uncertainty
    unc = result["uncertainty"]
    if unc["total_impact"] > 0:
        st.subheader("Cost Uncertainty from Knowledge Gaps")
        st.write(f"- Hard gap impact: **${unc['total_hard_impact']:,.0f}/kW** "
                 f"({unc['hard_gap_count']} gaps)")
        st.write(f"- Soft gap impact: **${unc['total_soft_impact']:,.0f}/kW** "
                 f"({unc['soft_gap_count']} gaps)")
        st.write(f"- Total uncertainty: **${unc['total_impact']:,.0f}/kW**")

    # Next steps
    st.subheader("Recommended Next Steps")
    for i, step in enumerate(result["next_steps"], 1):
        st.write(f"{i}. {step}")
