"""
Technology Selection Engine — Streamlit UI for the rebuilt third tab.

AI-powered comparison of all viable geothermal power conversion technologies.
Uses screening, optimization, dimension evaluation, probability analysis,
and synthesis to recommend the optimal technology for a given resource.
"""

import json
import os
from typing import Any

import streamlit as st
import numpy as np

from technology_registry import (
    TECHNOLOGIES, get_technology, get_viable_technologies,
    get_research_monitored, technology_summary_table,
)
from technology_analysis_bridge import analyze_technology, ANALYZERS
from technology_optimizer import run_screening, optimize_technology, _fallback_screening
from dimension_bots import DIMENSION_BOTS, run_dimension_bot, _fallback_dimension_scores, run_all_dimension_bots
from technology_synthesis import (
    compute_horizon_npv, compute_weighted_scores, run_synthesis, _fallback_synthesis,
)
from probability_framework import run_probability_analysis, run_monte_carlo
from research_monitor import run_research_monitor, run_all_research_monitors
from proposal_system import RefinementProposal

# ── Colors ────────────────────────────────────────────────────────────────────

NAVY = "#1A1A2E"
GREEN = "#00B050"
AMBER = "#F39C12"
RED = "#E74C3C"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F0F2F6"

CATEGORY_COLORS = {
    "commercial": "#2E7D32",
    "commercial_limited": "#F57F17",
    "emerging": "#1565C0",
    "research": "#6A1B9A",
}

CONFIDENCE_COLORS = {
    "high": GREEN,
    "medium": AMBER,
    "low": RED,
}


# ── Session State ─────────────────────────────────────────────────────────────

def _init_tech_state():
    """Initialize session state keys for the technology tab."""
    defaults = {
        "tech_screening_results": None,
        "tech_optimization_results": None,
        "tech_dimension_scores": None,
        "tech_synthesis_result": None,
        "tech_research_monitor_results": None,
        "tech_structural_innovations": [],
        "tech_probability_results": None,
        "tech_comparison_complete": False,
        "tech_running": False,
        "tech_horizon_years": 20,
        "tech_objective_weights": {"efficiency": 0.30, "capital_cost": 0.40, "schedule": 0.30},
        "tech_current_stage": "idle",
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API Key ───────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        return os.environ.get("ANTHROPIC_API_KEY")


# ── Resource Summary Bar ──────────────────────────────────────────────────────

def _render_resource_summary(design_basis: dict) -> dict:
    """Show editable resource parameters in a compact strip.

    Returns an updated copy of design_basis reflecting inline edits.
    Inline widgets use their own ``tech_edit_*`` keys — the sidebar-owned
    ``shared_*`` keys are never written to (Streamlit forbids that after
    the widget has been instantiated).

    Sidebar → inline sync: when the sidebar value changes between reruns
    we detect the drift and update the ``tech_edit_*`` keys *before* the
    widgets render so they pick up the new value.
    """
    _FIELDS = [
        # (design_basis key,      widget key,              default)
        ("brine_inlet_temp_C",   "tech_edit_T_in",        215.6),
        ("brine_outlet_temp_C",  "tech_edit_T_out",        71.1),
        ("brine_flow_kg_s",      "tech_edit_m_dot",       498.95),
        ("ambient_temp_C",       "tech_edit_T_amb",        13.9),
        ("net_power_target_MW",  "tech_edit_target",       53.0),
    ]

    # Detect sidebar changes: compare incoming design_basis to the
    # snapshot we stored on the previous render.  If the sidebar moved
    # a value, overwrite the edit key so the inline widget stays in sync.
    prev = st.session_state.get("_tech_prev_db", {})
    for db_key, edit_key, default in _FIELDS:
        db_val = design_basis.get(db_key, default)
        if edit_key not in st.session_state:
            # First render — seed from design_basis
            st.session_state[edit_key] = db_val
        elif abs(prev.get(db_key, default) - db_val) > 0.01:
            # Sidebar value changed — push into inline widget
            st.session_state[edit_key] = db_val

    # Snapshot current design_basis for next-render comparison
    st.session_state["_tech_prev_db"] = {k: design_basis.get(k, d) for k, _, d in _FIELDS}

    # Editable inline inputs (own keys — never collide with shared_*)
    cols = st.columns(5)
    new_T_in   = cols[0].number_input("Brine Inlet (°C)",  step=5.0,  format="%.1f", key="tech_edit_T_in")
    new_m_dot  = cols[1].number_input("Brine Flow (kg/s)", step=25.0, format="%.1f", key="tech_edit_m_dot")
    new_T_out  = cols[2].number_input("Min Outlet (°C)",   step=5.0,  format="%.1f", key="tech_edit_T_out")
    new_T_amb  = cols[3].number_input("Ambient (°C)",      step=1.0,  format="%.1f", key="tech_edit_T_amb")
    new_target = cols[4].number_input("Target (MW)",       step=5.0,  format="%.1f", key="tech_edit_target")

    # Build updated design_basis with inline values
    updated = dict(design_basis)
    updated["brine_inlet_temp_C"]  = new_T_in
    updated["brine_outlet_temp_C"] = new_T_out
    updated["brine_flow_kg_s"]     = new_m_dot
    updated["ambient_temp_C"]      = new_T_amb
    updated["net_power_target_MW"] = new_target

    return updated


# ── Horizon Dashboard ─────────────────────────────────────────────────────────

def _render_horizon_dashboard(horizon, ppa_term, financing_cost, eff_w, cost_w, risk_w, design_basis):
    """Prominent strip showing horizon regime and economic framing."""
    if horizon <= 12:
        regime = "SHORT"
        regime_color = AMBER
        interp = "Capital recovery speed dominates. Favor lower CAPEX and faster schedule."
    elif horizon <= 22:
        regime = "MEDIUM"
        regime_color = GREEN
        interp = "Balanced evaluation. Efficiency gains begin to compound meaningfully."
    else:
        regime = "LONG"
        regime_color = NAVY
        interp = "Efficiency compounds significantly. Higher capital justified for sustained output."

    # Calculate economic context
    r = financing_cost / 100
    if r > 0:
        annuity = (1 - (1 + r) ** (-horizon)) / r
    else:
        annuity = horizon

    energy_price = design_basis.get("energy_value_per_MWh", 80)

    # Efficiency premium: NPV value of 1% efficiency gain over the horizon
    # delta_MW * 8760 * CF * $/MWh * annuity_factor
    eff_premium_per_pct = 1.0 * 8760 * 0.95 * energy_price * annuity / 1e6  # $M per MW of net power

    # Schedule value: NPV of one week of earlier revenue
    weekly_rev = 50 * 8760 * 0.95 * energy_price / 52 / 1e6  # $M per week for ~50MW plant
    schedule_value = weekly_rev / (1 + r) ** horizon  # discounted

    # Year 15+ NPV weight
    if r > 0 and horizon > 15:
        annuity_15 = (1 - (1 + r) ** (-15)) / r
        late_pct = (annuity - annuity_15) / annuity * 100
    else:
        late_pct = 0

    st.markdown(
        f"<div style='background:{regime_color}15; border:2px solid {regime_color}; "
        f"border-radius:8px; padding:12px; margin:8px 0;'>"
        f"<span style='color:{regime_color}; font-weight:700; font-size:18px;'>"
        f"HORIZON: {regime} ({horizon} years)</span>"
        f"<span style='color:#666; margin-left:16px;'>{interp}</span>"
        f"</div>",
        unsafe_allow_html=True,
    )

    mcols = st.columns(4)
    mcols[0].metric("Efficiency Premium", f"${eff_premium_per_pct:.1f}M/MW",
                     help="NPV value of 1 MW additional net power over the horizon")
    mcols[1].metric("Schedule Value", f"${schedule_value:.2f}M/wk",
                     help="NPV value of one week earlier first power")
    mcols[2].metric("Year 15+ NPV Weight", f"{late_pct:.0f}%",
                     help="Fraction of total NPV from years 15 onward")
    mcols[3].metric("Weights", f"E:{eff_w:.0%} / C:{cost_w:.0%} / R:{risk_w:.0%}")


# ── Comparison Table ──────────────────────────────────────────────────────────

def _render_comparison_table(optimization_results: dict, synthesis: dict):
    """Render the main technology comparison table with color coding."""
    st.markdown(f"<h3 style='color:{NAVY};'>Technology Comparison</h3>",
                unsafe_allow_html=True)

    recommended = synthesis.get("recommended_technology")
    weighted_scores = synthesis.get("computed_weighted_scores", {})

    # Sort by weighted score (recommended first)
    sorted_techs = sorted(
        optimization_results.keys(),
        key=lambda t: weighted_scores.get(t, 0),
        reverse=True,
    )

    # Header
    hcols = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 1])
    headers = ["Technology", "Net MW", "Eff %", "CAPEX $M", "$/kW", "NPV $M", "LCOE", "Weeks", "Score"]
    for i, h in enumerate(headers):
        hcols[i].markdown(f"**{h}**")

    st.markdown("<hr style='margin:4px 0;'>", unsafe_allow_html=True)

    # Get reference values for color coding (recommended technology)
    ref = optimization_results.get(recommended, {})

    for tech_id in sorted_techs:
        r = optimization_results[tech_id]
        tech = get_technology(tech_id)
        w_score = weighted_scores.get(tech_id, 0)
        is_rec = (tech_id == recommended)

        # Category badge
        cat = tech.category if tech else "unknown"
        cat_color = CATEGORY_COLORS.get(cat, "#666")

        prefix = ">> " if is_rec else "   "
        name = tech.name if tech else tech_id
        if is_rec:
            name = f"** {name}"

        cols = st.columns([3, 1, 1, 1, 1, 1, 1, 1, 1])

        # Technology name with category indicator
        cols[0].markdown(
            f"<span style='color:{cat_color}; font-size:10px;'>[{cat[:4].upper()}]</span> "
            f"<span style='font-weight:{700 if is_rec else 400};'>{name}</span>",
            unsafe_allow_html=True,
        )

        def _cell(col, val, fmt, ref_val=None, higher_better=True):
            text = fmt.format(val)
            if ref_val is not None and ref_val != 0:
                delta = val - ref_val
                if abs(delta / (ref_val if ref_val != 0 else 1)) < 0.02:
                    color = "#333"
                elif (delta > 0) == higher_better:
                    color = GREEN
                else:
                    color = RED
            else:
                color = "#333"
            col.markdown(f"<span style='color:{color};'>{text}</span>", unsafe_allow_html=True)

        _cell(cols[1], r.get("net_power_MW", 0), "{:.1f}", ref.get("net_power_MW"), True)
        _cell(cols[2], r.get("cycle_efficiency", 0) * 100, "{:.1f}", ref.get("cycle_efficiency", 0) * 100, True)
        _cell(cols[3], r.get("capex_total_USD", 0) / 1e6, "${:.0f}", ref.get("capex_total_USD", 0) / 1e6, False)
        _cell(cols[4], r.get("capex_per_kW", 0), "${:,.0f}", ref.get("capex_per_kW", 0), False)
        _cell(cols[5], r.get("npv_USD", 0) / 1e6, "${:.0f}", ref.get("npv_USD", 0) / 1e6, True)
        _cell(cols[6], r.get("lcoe_per_MWh", 0), "${:.1f}", ref.get("lcoe_per_MWh", 0), False)
        _cell(cols[7], r.get("construction_weeks", 0), "{:.0f}", ref.get("construction_weeks", 0), False)
        cols[8].markdown(f"**{w_score:.1f}**")

    # Legend
    st.caption(
        f"Color: green=better than recommended, red=worse. "
        f"Categories: COMM=commercial, LIMI=limited, EMER=emerging, RESE=research"
    )


# ── Dimension Scorecard ───────────────────────────────────────────────────────

def _render_dimension_scorecard(dimension_results: dict):
    """Render the six dimension scores as tabs."""
    st.markdown(f"<h3 style='color:{NAVY};'>Dimension Evaluation</h3>",
                unsafe_allow_html=True)

    dim_tabs = st.tabs([
        f"{DIMENSION_BOTS[d]['icon']} {DIMENSION_BOTS[d]['name']}"
        for d in dimension_results
    ])

    for tab, (dim_id, result) in zip(dim_tabs, dimension_results.items()):
        with tab:
            scores = result.get("scores", {})
            if scores:
                # Sort by score
                sorted_techs = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

                for tech_id in sorted_techs:
                    score = scores[tech_id]
                    bar_width = score / 10 * 100
                    if score >= 7:
                        bar_color = GREEN
                    elif score >= 4:
                        bar_color = AMBER
                    else:
                        bar_color = RED

                    tech = get_technology(tech_id)
                    name = tech.name if tech else tech_id

                    st.markdown(
                        f"<div style='display:flex; align-items:center; margin:4px 0;'>"
                        f"<span style='width:220px; font-size:13px;'>{name}</span>"
                        f"<div style='flex:1; background:#eee; border-radius:4px; height:20px; margin:0 8px;'>"
                        f"<div style='width:{bar_width}%; background:{bar_color}; "
                        f"height:100%; border-radius:4px;'></div></div>"
                        f"<span style='width:40px; text-align:right; font-weight:700;'>{score:.1f}</span>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

            # Narrative
            narrative = result.get("narrative", "")
            if narrative:
                with st.expander("Analysis narrative"):
                    st.markdown(narrative)

            # Key differentiators
            diffs = result.get("key_differentiators", [])
            if diffs:
                st.markdown("**Key differentiators:** " + " | ".join(diffs))

            # Occam's verdict (probability bot only)
            verdict = result.get("occams_verdict", "")
            if verdict:
                st.info(f"**Occam's Razor Verdict:** {verdict}")


# ── Probability Dashboard ─────────────────────────────────────────────────────

def _render_probability_dashboard(probability_results: dict):
    """Render Monte Carlo results — P10/P50/P90, joint probability, complexity penalties."""
    st.markdown(f"<h3 style='color:{NAVY};'>Probability Analysis</h3>",
                unsafe_allow_html=True)

    mc = probability_results.get("monte_carlo", {})
    penalties = probability_results.get("complexity_penalties", {})

    if not mc:
        st.info("No probability analysis available.")
        return

    # P10/P50/P90 table
    st.markdown("**NPV Distribution (P10 / P50 / P90)**")

    sorted_techs = sorted(mc.keys(), key=lambda t: mc[t]["npv_p50"], reverse=True)

    for tech_id in sorted_techs:
        r = mc[tech_id]
        tech = get_technology(tech_id)
        name = tech.name if tech else tech_id

        p10 = r["npv_p10"] / 1e6
        p50 = r["npv_p50"] / 1e6
        p90 = r["npv_p90"] / 1e6
        jp = r["joint_probability"]

        # Visual bar showing P10-P90 range
        range_width = p90 - p10
        bar_left = max(0, (p10 + 200) / 600 * 100)  # normalize to 0-100%
        bar_right = max(0, (p90 + 200) / 600 * 100)
        bar_center = max(0, (p50 + 200) / 600 * 100)

        st.markdown(
            f"<div style='display:flex; align-items:center; margin:4px 0;'>"
            f"<span style='width:200px; font-size:12px;'>{name}</span>"
            f"<span style='width:100px; font-size:11px; color:{RED};'>${p10:.0f}M</span>"
            f"<span style='width:100px; font-size:11px; font-weight:700;'>${p50:.0f}M</span>"
            f"<span style='width:100px; font-size:11px; color:{GREEN};'>${p90:.0f}M</span>"
            f"<span style='width:80px; font-size:11px;'>JP: {jp:.1%}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Complexity penalties
    if penalties:
        st.markdown("---")
        st.markdown("**Complexity Penalties vs Reference**")
        ref_id = probability_results.get("reference_technology", "orc_direct")
        st.caption(f"Reference: {ref_id}")

        for tech_id, p in penalties.items():
            justified = p["complexity_justified"]
            icon = "+" if justified else "-"
            color = GREEN if justified else RED

            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>[{icon}]</span> "
                f"**{tech_id}**: {p['summary']}",
                unsafe_allow_html=True,
            )


# ── Horizon Sensitivity ──────────────────────────────────────────────────────

def _render_horizon_sensitivity(synthesis: dict, user_horizon: int):
    """Render horizon sensitivity table and NPV vs horizon chart."""
    st.markdown(f"<h3 style='color:{NAVY};'>Horizon Sensitivity</h3>",
                unsafe_allow_html=True)

    sensitivity = synthesis.get("horizon_sensitivity", [])
    npv_data = synthesis.get("npv_vs_horizon_data", {})

    # Table
    if sensitivity:
        hcols = st.columns([1, 2, 2, 2, 3])
        hcols[0].markdown("**Horizon**")
        hcols[1].markdown("**Recommended**")
        hcols[2].markdown("**NPV Leader**")
        hcols[3].markdown("**NPV Gap**")
        hcols[4].markdown("**Driving Factor**")

        for row in sensitivity:
            h = row.get("horizon_years", 0)
            is_user = abs(h - user_horizon) <= 5
            style = "font-weight:700;" if is_user else ""

            cols = st.columns([1, 2, 2, 2, 3])
            marker = " <<" if is_user else ""
            cols[0].markdown(f"<span style='{style}'>{h}yr{marker}</span>", unsafe_allow_html=True)
            cols[1].markdown(f"<span style='{style}'>{row.get('recommended', '?')}</span>", unsafe_allow_html=True)
            cols[2].markdown(f"<span style='{style}'>{row.get('npv_leader', '?')}</span>", unsafe_allow_html=True)
            gap = row.get("npv_gap_USD", 0)
            cols[3].markdown(f"<span style='{style}'>${gap/1e6:.0f}M</span>", unsafe_allow_html=True)
            cols[4].markdown(f"<span style='{style}'>{row.get('driving_factor', '')}</span>", unsafe_allow_html=True)

    # NPV vs Horizon chart using st.line_chart
    if npv_data:
        st.markdown("**NPV vs Plant Horizon**")

        # Build chart data
        import pandas as pd

        chart_rows = []
        for tech_id, points in npv_data.items():
            tech = get_technology(tech_id)
            name = tech.name if tech else tech_id
            for p in points:
                chart_rows.append({
                    "Year": p["year"],
                    "Technology": name,
                    "NPV ($M)": p["npv_USD"] / 1e6,
                })

        if chart_rows:
            df = pd.DataFrame(chart_rows)
            # Pivot for line chart
            pivot = df.pivot(index="Year", columns="Technology", values="NPV ($M)")
            st.line_chart(pivot)

            # Mark user's horizon
            st.caption(f"Your horizon: {user_horizon} years (shown in table above)")


# ── Synthesis & Recommendation ────────────────────────────────────────────────

def _render_synthesis(synthesis: dict):
    """Render the final recommendation and executive summary."""
    st.markdown(f"<h3 style='color:{NAVY};'>Recommendation</h3>",
                unsafe_allow_html=True)

    recommended = synthesis.get("recommended_technology")
    confidence = synthesis.get("recommendation_confidence", "Low")
    conf_color = CONFIDENCE_COLORS.get(confidence.lower(), "#666")

    if recommended:
        tech = get_technology(recommended)
        name = tech.name if tech else recommended

        st.markdown(
            f"<div style='background:{conf_color}15; border:3px solid {conf_color}; "
            f"border-radius:12px; padding:20px; margin:12px 0; text-align:center;'>"
            f"<div style='font-size:14px; color:#666;'>RECOMMENDED TECHNOLOGY</div>"
            f"<div style='font-size:28px; font-weight:700; color:{NAVY}; margin:8px 0;'>"
            f"{name}</div>"
            f"<div style='font-size:16px; color:{conf_color}; font-weight:600;'>"
            f"Confidence: {confidence}</div>"
            f"</div>",
            unsafe_allow_html=True,
        )

    # Second best
    second = synthesis.get("second_best")
    if second:
        cond = synthesis.get("second_best_conditions", "")
        st.markdown(f"**Second best:** {second}" + (f" -- {cond}" if cond else ""))

    # Executive summary
    summary = synthesis.get("executive_summary", "")
    if summary:
        st.markdown("---")
        st.markdown(f"**Executive Summary**")
        st.markdown(summary)

    # Dimension tensions
    tensions = synthesis.get("dimension_tensions", [])
    if tensions:
        with st.expander("Dimension Tensions"):
            for t in tensions:
                st.markdown(f"- {t}")

    # Information gaps
    gaps = synthesis.get("information_gaps", [])
    if gaps:
        with st.expander("Information Gaps"):
            for g in gaps:
                st.markdown(f"- {g}")

    # Structural innovations
    innovations = synthesis.get("structural_innovations", [])
    if innovations:
        with st.expander("Structural Innovation Opportunities"):
            for inn in innovations:
                if isinstance(inn, dict):
                    st.markdown(f"**{inn.get('title', 'Innovation')}**")
                    st.markdown(inn.get("description", ""))
                    impact = inn.get("impact", "")
                    if impact:
                        st.markdown(f"*Impact: {impact}*")
                else:
                    st.markdown(f"- {inn}")


# ── Screening Results ─────────────────────────────────────────────────────────

def _render_screening_results(screening: dict):
    """Render screening results in an expandable section."""
    results = screening.get("screening_results", {})
    narrative = screening.get("screening_narrative", "")

    viable_count = sum(1 for r in results.values() if r["status"] == "VIABLE")
    marginal_count = sum(1 for r in results.values() if r["status"] == "MARGINAL")
    excluded_count = sum(1 for r in results.values() if r["status"] == "EXCLUDE")

    with st.expander(
        f"Screening Results: {viable_count} viable, {marginal_count} marginal, {excluded_count} excluded",
        expanded=False,
    ):
        if narrative:
            st.markdown(narrative)
            st.markdown("---")

        for tech_id, r in results.items():
            status = r["status"]
            if status == "VIABLE":
                icon, color = "+", GREEN
            elif status == "MARGINAL":
                icon, color = "~", AMBER
            else:
                icon, color = "x", RED

            tech = get_technology(tech_id)
            name = tech.name if tech else tech_id

            st.markdown(
                f"<span style='color:{color}; font-weight:700;'>[{icon}]</span> "
                f"**{name}**: {r['reasoning']}",
                unsafe_allow_html=True,
            )


# ── Downloads ─────────────────────────────────────────────────────────────────

def _render_downloads(synthesis: dict, optimization_results: dict, dimension_results: dict):
    """Render download buttons for results."""
    st.markdown("---")
    dcols = st.columns(3)

    # Full JSON
    full_data = {
        "synthesis": synthesis,
        "optimization_results": {
            k: {kk: vv for kk, vv in v.items() if kk != "optimization_metadata"}
            for k, v in optimization_results.items()
        },
        "dimension_scores": dimension_results,
    }
    dcols[0].download_button(
        "Download Full JSON",
        json.dumps(full_data, indent=2, default=str),
        "technology_comparison.json",
        "application/json",
    )

    # Executive summary
    exec_summary = synthesis.get("executive_summary", "No summary available.")
    dcols[1].download_button(
        "Download Executive Summary",
        exec_summary,
        "executive_summary.txt",
        "text/plain",
    )

    # Decision log
    log_lines = []
    screening = st.session_state.get("tech_screening_results", {})
    if screening:
        log_lines.append("=== SCREENING ===")
        for tid, r in screening.get("screening_results", {}).items():
            log_lines.append(f"{tid}: {r['status']} - {r['reasoning']}")
    log_lines.append(f"\n=== RECOMMENDATION ===")
    log_lines.append(f"Technology: {synthesis.get('recommended_technology', '?')}")
    log_lines.append(f"Confidence: {synthesis.get('recommendation_confidence', '?')}")

    dcols[2].download_button(
        "Download Decision Log",
        "\n".join(log_lines),
        "decision_log.txt",
        "text/plain",
    )


# ── Main Execution Flow ──────────────────────────────────────────────────────

def _run_technology_comparison(design_basis: dict, horizon: int, eff_w: float, cost_w: float, risk_w: float,
                                ppa_term: int, financing_cost: float):
    """Execute the full technology comparison pipeline."""
    api_key = _get_api_key()

    # Update design basis with technology tab inputs
    db = dict(design_basis)
    db["plant_life_years"] = horizon
    db["objective_weights"] = {
        "efficiency": eff_w,
        "capital_cost": cost_w,
        "schedule": risk_w,
    }
    db["ppa_term_years"] = ppa_term
    db["financing_cost_pct"] = financing_cost

    st.session_state["tech_running"] = True
    st.session_state["tech_comparison_complete"] = False

    progress = st.progress(0, text="Starting technology comparison...")

    # ── Stage 1: Research Monitor ─────────────────────────────────────────
    st.session_state["tech_current_stage"] = "research"
    progress.progress(5, text="Stage 1/6: Researching emerging technologies...")

    research_results = {}
    if api_key:
        try:
            monitored = get_research_monitored()
            for tech in monitored:
                research_results[tech.id] = run_research_monitor(
                    tech.id, db, api_key,
                )
        except Exception as e:
            st.warning(f"Research monitor error: {e}")
    st.session_state["tech_research_monitor_results"] = research_results

    # ── Stage 2: Screening ────────────────────────────────────────────────
    st.session_state["tech_current_stage"] = "screening"
    progress.progress(15, text="Stage 2/6: Screening technologies...")

    if api_key:
        screening = run_screening(db, api_key, research_results)
    else:
        screening = _fallback_screening(db)

    st.session_state["tech_screening_results"] = screening

    # Determine viable technologies
    viable = []
    for tech_id, result in screening.get("screening_results", {}).items():
        if result["status"] in ("VIABLE", "MARGINAL"):
            viable.append(tech_id)

    if not viable:
        st.error("No technologies passed screening. Adjust resource parameters.")
        st.session_state["tech_running"] = False
        return

    # ── Stage 3: Optimization ─────────────────────────────────────────────
    st.session_state["tech_current_stage"] = "optimizing"
    progress.progress(25, text=f"Stage 3/6: Optimizing {len(viable)} technologies...")

    optimization_results = {}
    for i, tech_id in enumerate(viable):
        pct = 25 + int(35 * (i + 1) / len(viable))
        tech = get_technology(tech_id)
        name = tech.name if tech else tech_id
        progress.progress(pct, text=f"Stage 3/6: Optimizing {name}...")

        if api_key:
            result = optimize_technology(
                tech_id, db, api_key,
                research_results=research_results,
                max_rounds=3,
            )
        else:
            # No API key — run with defaults
            result = analyze_technology(tech_id, {}, db)

        optimization_results[tech_id] = result

    st.session_state["tech_optimization_results"] = optimization_results

    # ── Stage 4: Probability Analysis ─────────────────────────────────────
    st.session_state["tech_current_stage"] = "probability"
    progress.progress(65, text="Stage 4/6: Running Monte Carlo simulation...")

    # Only analyze converged results
    converged_results = {k: v for k, v in optimization_results.items()
                        if v.get("converged", False)}

    probability_results = run_probability_analysis(
        converged_results, db,
        reference_id="orc_direct" if "orc_direct" in converged_results else (
            list(converged_results.keys())[0] if converged_results else "orc_direct"
        ),
        n_iterations=3000,
    )
    st.session_state["tech_probability_results"] = probability_results

    # ── Stage 5: Dimension Bots ───────────────────────────────────────────
    st.session_state["tech_current_stage"] = "dimensions"
    progress.progress(75, text="Stage 5/6: Running dimension evaluations...")

    if api_key:
        dimension_results = run_all_dimension_bots(
            converged_results, probability_results, db, api_key,
        )
    else:
        # Fallback dimension scores
        dimension_results = {}
        for dim_id in DIMENSION_BOTS:
            dimension_results[dim_id] = _fallback_dimension_scores(dim_id, converged_results)

    st.session_state["tech_dimension_scores"] = dimension_results

    # ── Stage 6: Synthesis ────────────────────────────────────────────────
    st.session_state["tech_current_stage"] = "synthesis"
    progress.progress(90, text="Stage 6/6: Synthesizing recommendation...")

    if api_key:
        synthesis = run_synthesis(
            converged_results, dimension_results,
            probability_results, db, api_key,
        )
    else:
        weighted = compute_weighted_scores(dimension_results, db["objective_weights"])
        synthesis = _fallback_synthesis(
            converged_results, dimension_results,
            probability_results, weighted, db,
        )
        synthesis["npv_vs_horizon_data"] = compute_horizon_npv(converged_results, db)
        synthesis["computed_weighted_scores"] = weighted
        from technology_synthesis import _compute_horizon_sensitivity
        synthesis["horizon_sensitivity"] = _compute_horizon_sensitivity(
            synthesis["npv_vs_horizon_data"], converged_results,
        )

    st.session_state["tech_synthesis_result"] = synthesis

    # ── Complete ──────────────────────────────────────────────────────────
    progress.progress(100, text="Complete!")
    st.session_state["tech_comparison_complete"] = True
    st.session_state["tech_running"] = False
    st.session_state["tech_current_stage"] = "complete"


# ── Main Tab Renderer ─────────────────────────────────────────────────────────

def render_technology_tab(design_basis: dict):
    """Render the Technology Selection Engine tab."""
    _init_tech_state()

    # ── Technology-specific sidebar inputs ─────────────────────────────────
    with st.sidebar:
        st.markdown("<h4 style='color:#90caf9;'>Technology Analysis</h4>",
                    unsafe_allow_html=True)
        brine_chem = st.text_area(
            "Brine chemistry / TDS notes",
            height=60,
            key="tech_brine_chemistry",
            help="Describe brine chemistry, TDS, scaling tendency",
        )
        ncg_pct = st.number_input(
            "NCG content estimate (%)",
            min_value=0.0, max_value=20.0, value=0.5, step=0.1,
            format="%.1f",
            key="tech_ncg_content",
            help="Non-condensable gas content in brine",
        )

    st.markdown(
        f"<h2 style='color:{NAVY};'>Technology Selection Engine</h2>"
        f"<p style='color:#666;'>AI-powered comparison of all viable geothermal power "
        f"conversion technologies for your resource</p>",
        unsafe_allow_html=True,
    )

    # Resource summary — editable inline, returns updated design_basis
    active_db = _render_resource_summary(design_basis)

    # Merge sidebar-only fields into active design_basis
    active_db["brine_chemistry_notes"] = brine_chem
    active_db["ncg_content_pct"] = ncg_pct

    st.markdown("---")

    # Horizon and economic inputs
    col1, col2, col3 = st.columns(3)
    horizon = col1.selectbox("Plant Horizon (years)", [10, 15, 20, 25, 30], index=2,
                              key="tech_horizon_select")
    ppa_term = col2.number_input("PPA Term (years)", 10, 30, 20, key="tech_ppa_term")
    financing_cost = col3.number_input("Financing Cost (%)", 4.0, 12.0, 7.0, step=0.5,
                                        key="tech_financing_cost")

    # Objective weights
    with st.expander("Objective Weights", expanded=False):
        wcol1, wcol2, wcol3 = st.columns(3)
        eff_w = wcol1.slider("Thermal Efficiency %", 0, 100, 30, key="tech_eff_w") / 100
        cost_w = wcol2.slider("Capital & Schedule %", 0, 100, 40, key="tech_cost_w") / 100
        risk_w = wcol3.slider("Operations & Risk %", 0, 100, 30, key="tech_risk_w") / 100

        # Normalize
        total_w = eff_w + cost_w + risk_w
        if total_w > 0:
            eff_w, cost_w, risk_w = eff_w / total_w, cost_w / total_w, risk_w / total_w

    # Horizon dashboard — uses active_db so inline changes reflect immediately
    _render_horizon_dashboard(horizon, ppa_term, financing_cost, eff_w, cost_w, risk_w, active_db)

    # Technology registry preview
    with st.expander(f"Technology Registry ({len(TECHNOLOGIES)} technologies)", expanded=False):
        for row in technology_summary_table():
            cat_color = CATEGORY_COLORS.get(row["category"], "#666")
            monitor = " [Research Monitor]" if row["research_monitor"] else ""
            st.markdown(
                f"<span style='color:{cat_color}; font-size:11px;'>[{row['category'][:4].upper()}]</span> "
                f"**{row['name']}** | {row['temp_range']} | {row['efficiency']} | "
                f"{row['model_type']}{monitor}",
                unsafe_allow_html=True,
            )

    # Run button
    st.markdown("---")
    api_key = _get_api_key()
    run_disabled = st.session_state.get("tech_running", False)

    if not api_key:
        st.warning("No API key found. Will use deterministic fallback models (no AI optimization).")

    if st.button("Run Technology Comparison", type="primary", use_container_width=True,
                  disabled=run_disabled, key="tech_run_btn"):
        _run_technology_comparison(active_db, horizon, eff_w, cost_w, risk_w, ppa_term, financing_cost)
        st.rerun()

    # ── Results ───────────────────────────────────────────────────────────
    if st.session_state.get("tech_comparison_complete"):
        screening = st.session_state.get("tech_screening_results")
        optimization = st.session_state.get("tech_optimization_results")
        dimensions = st.session_state.get("tech_dimension_scores")
        synthesis = st.session_state.get("tech_synthesis_result")
        probability = st.session_state.get("tech_probability_results")

        # Screening (expandable)
        if screening:
            _render_screening_results(screening)

        # Synthesis & Recommendation (top)
        if synthesis:
            _render_synthesis(synthesis)

        st.markdown("---")

        # Comparison table
        if optimization and synthesis:
            _render_comparison_table(optimization, synthesis)

        # Dimension scorecard
        if dimensions:
            st.markdown("---")
            _render_dimension_scorecard(dimensions)

        # Probability dashboard
        if probability:
            st.markdown("---")
            _render_probability_dashboard(probability)

        # Horizon sensitivity
        if synthesis:
            st.markdown("---")
            _render_horizon_sensitivity(synthesis, horizon)

        # Downloads
        if synthesis and optimization and dimensions:
            _render_downloads(synthesis, optimization, dimensions)

        # Re-run with new weights
        st.markdown("---")
        if st.button("Re-score with Current Weights", key="tech_rescore_btn"):
            # Recompute weighted scores and synthesis without re-running analysis
            weights = {"efficiency": eff_w, "capital_cost": cost_w, "schedule": risk_w}
            new_weighted = compute_weighted_scores(dimensions, weights)

            # Update synthesis
            synthesis["computed_weighted_scores"] = new_weighted

            # Re-rank
            ranking = sorted(new_weighted.keys(), key=lambda t: new_weighted[t], reverse=True)
            if ranking:
                synthesis["recommended_technology"] = ranking[0]
                if len(ranking) > 1:
                    gap = new_weighted[ranking[0]] - new_weighted[ranking[1]]
                    if gap > 1.5:
                        synthesis["recommendation_confidence"] = "High"
                    elif gap > 0.5:
                        synthesis["recommendation_confidence"] = "Medium"
                    else:
                        synthesis["recommendation_confidence"] = "Low"
                    synthesis["second_best"] = ranking[1]

            st.session_state["tech_synthesis_result"] = synthesis
            st.session_state["tech_objective_weights"] = weights
            st.rerun()
