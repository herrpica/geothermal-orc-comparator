"""
Path to $2,000/kW — Streamlit UI Tab
======================================

Strategic cost reduction analysis. Every number has a confidence level
and an explicit assumption. No false precision.
"""

import streamlit as st
import pandas as pd

from pathway_engine import (
    run_pathway_analysis,
    build_waterfall_data,
    PathwayAnalysis,
    TARGET_PER_KW,
    CONFIDENCE_HIGH,
    CONFIDENCE_MEDIUM,
    CONFIDENCE_LOW,
    CONTROL_FULL,
    CONTROL_PARTIAL,
    CONTROL_MINIMAL,
)


def _confidence_color(conf: str) -> str:
    if conf == CONFIDENCE_HIGH:
        return "#4CAF50"  # green
    elif conf == CONFIDENCE_MEDIUM:
        return "#FFC107"  # amber
    return "#F44336"      # red


def _confidence_emoji(conf: str) -> str:
    if conf == CONFIDENCE_HIGH:
        return "HIGH"
    elif conf == CONFIDENCE_MEDIUM:
        return "MEDIUM"
    return "LOW"


def _control_label(ctrl: str) -> str:
    if ctrl == CONTROL_FULL:
        return "Fervo controls"
    elif ctrl == CONTROL_PARTIAL:
        return "Partially controlled"
    return "Market dependent"


def render_pathway_tab(design_basis: dict):
    """Main entry point for the Path to $2,000/kW tab."""

    st.markdown("## Path to $2,000/kW Installed")
    st.markdown(
        "*Every number shows a range and confidence level. "
        "Every assumption is flagged. The goal is not to prove $2,000/kW is achievable — "
        "it is to define exactly what would have to be true, and rank those conditions "
        "by how much Fervo controls them.*"
    )

    # Run analysis (deterministic — no API calls)
    analysis = run_pathway_analysis()

    # ── Executive Summary ─────────────────────────────────────────────
    _render_executive_summary(analysis)

    st.markdown("---")

    # ── Section selector ──────────────────────────────────────────────
    section = st.radio(
        "Section",
        ["Waterfall Chart", "Pathway Details", "Pathway Comparison",
         "Critical Path", "What Must Be True", "Honest Limitations"],
        horizontal=True,
        key="pw_section",
    )

    if section == "Waterfall Chart":
        _render_waterfall(analysis)
    elif section == "Pathway Details":
        _render_pathway_details(analysis)
    elif section == "Pathway Comparison":
        _render_comparison_table(analysis)
    elif section == "Critical Path":
        _render_critical_path(analysis)
    elif section == "What Must Be True":
        _render_what_must_be_true(analysis)
    elif section == "Honest Limitations":
        _render_limitations(analysis)


# ── Executive Summary ─────────────────────────────────────────────────────────

def _render_executive_summary(a: PathwayAnalysis):
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Current Best", f"${a.best_optimizer_per_kw:,.0f}/kW")
    col2.metric("Target", f"${a.target_per_kw:,.0f}/kW")
    col3.metric("Gap to Close", f"${a.gap_per_kw:,.0f}/kW")

    # Combined mid saving across all pathways (no double-counting P1+P2 construction)
    # Use Pathway 1 + Pathway 3 as the "most likely combined" since P1 and P2 are alternatives
    pw1_mid = a.pathways[0].total_saving_mid if len(a.pathways) > 0 else 0
    pw3_mid = a.pathways[2].total_saving_mid if len(a.pathways) > 2 else 0
    combined_mid = pw1_mid + pw3_mid
    projected = a.best_optimizer_per_kw - combined_mid

    if projected <= TARGET_PER_KW:
        col4.metric("Projected (P1+P3)", f"${projected:,.0f}/kW", f"-${combined_mid:,.0f}")
    else:
        col4.metric("Projected (P1+P3)", f"${projected:,.0f}/kW",
                     f"${projected - TARGET_PER_KW:,.0f} above target")

    # Narrative
    pw2_mid = a.pathways[1].total_saving_mid if len(a.pathways) > 1 else 0
    pw2_projected = a.best_optimizer_per_kw - pw2_mid - pw3_mid

    if projected <= TARGET_PER_KW:
        verdict = (
            f"**Pathway 1 (self-perform) + Pathway 3 (equipment)** projects "
            f"${projected:,.0f}/kW at expected values — within target. "
            f"However, this requires all mechanisms to deliver as modeled. "
            f"The conservative case (least savings) reaches "
            f"${a.best_optimizer_per_kw - a.pathways[0].total_saving_high - a.pathways[2].total_saving_high:,.0f}/kW."
        )
    else:
        verdict = (
            f"**Pathway 1 + 3** projects ${projected:,.0f}/kW — "
            f"${projected - TARGET_PER_KW:,.0f}/kW above target at expected values. "
            f"Reaching $2,000/kW requires either optimistic execution on P1+P3, "
            f"or adding Pathway 2 (modular) which projects ${pw2_projected:,.0f}/kW "
            f"but carries higher risk."
        )

    st.info(
        f"**Current baseline:** ${a.best_optimizer_per_kw:,.0f}/kW "
        f"({a.baseline_source[:80]}...)\n\n{verdict}\n\n"
        f"**Most important first action:** Freeze component specifications "
        f"(requires GeoBlock parametric analysis) and commit to self-perform delivery model."
    )


# ── Waterfall Chart ───────────────────────────────────────────────────────────

def _render_waterfall(a: PathwayAnalysis):
    st.markdown("### Cost Reduction Waterfall")

    pw_options = {
        "Pathway 1 — Construction": 0,
        "Pathway 2 — Modular": 1,
        "Pathway 3 — Equipment": 2,
        "Combined (P1 + P3)": None,
    }
    selected = st.selectbox("Show pathway", list(pw_options.keys()), key="pw_waterfall_select")
    pw_idx = pw_options[selected]

    if pw_idx is None:
        # Combined P1 + P3 (not P2, since P1 and P2 are alternatives)
        wf_data = _build_combined_waterfall(a, [0, 2])
    else:
        wf_data = build_waterfall_data(a, pw_idx)

    _draw_waterfall(wf_data)


def _build_combined_waterfall(a: PathwayAnalysis, indices: list) -> dict:
    """Build waterfall for specific pathway combination."""
    labels = ["Baseline"]
    mid_values = [a.best_optimizer_per_kw]
    low_values = [a.best_optimizer_per_kw]
    high_values = [a.best_optimizer_per_kw]
    confidences = [""]
    running = a.best_optimizer_per_kw

    for idx in indices:
        pw = a.pathways[idx]
        for m in pw.mechanisms:
            short_name = m.name[:35] + ("..." if len(m.name) > 35 else "")
            labels.append(short_name)
            mid_values.append(-m.saving_mid)
            low_values.append(-m.saving_low)
            high_values.append(-m.saving_high)
            confidences.append(m.confidence)
            running -= m.saving_mid

    labels.append("Projected")
    total_low_extra = sum(m.saving_low - m.saving_mid for i in indices for m in a.pathways[i].mechanisms)
    total_high_short = sum(m.saving_mid - m.saving_high for i in indices for m in a.pathways[i].mechanisms)
    mid_values.append(running)
    low_values.append(running - total_low_extra)
    high_values.append(running + total_high_short)
    confidences.append("")

    return {
        "labels": labels,
        "mid_values": mid_values,
        "low_values": low_values,
        "high_values": high_values,
        "confidences": confidences,
        "target": a.target_per_kw,
    }


def _draw_waterfall(wf: dict):
    """Draw waterfall chart using matplotlib."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
    except ImportError:
        st.warning("matplotlib not installed — cannot render waterfall chart.")
        return

    n = len(wf["labels"])
    fig, ax = plt.subplots(figsize=(14, 7))

    running = 0
    bar_bottoms = []
    bar_heights = []
    colors = []

    conf_colors = {
        CONFIDENCE_HIGH: "#4CAF50",
        CONFIDENCE_MEDIUM: "#FFC107",
        CONFIDENCE_LOW: "#F44336",
        "": "#2196F3",
    }

    for i in range(n):
        val = wf["mid_values"][i]
        conf = wf["confidences"][i]

        if i == 0:
            # Baseline bar
            bar_bottoms.append(0)
            bar_heights.append(val)
            colors.append("#2196F3")
            running = val
        elif i == n - 1:
            # Projected total
            bar_bottoms.append(0)
            bar_heights.append(val)
            colors.append("#9C27B0" if val > wf["target"] else "#4CAF50")
        else:
            # Reduction step (negative value = cost reduction)
            if val < 0:
                bar_bottoms.append(running + val)
                bar_heights.append(abs(val))
                colors.append(conf_colors.get(conf, "#888"))
                # Error bars for uncertainty range
                lo = wf["low_values"][i]
                hi = wf["high_values"][i]
                ax.plot(
                    [i, i], [running + lo, running + hi],
                    color="#333", linewidth=2, zorder=5,
                )
                ax.plot(
                    [i - 0.15, i + 0.15], [running + lo, running + lo],
                    color="#333", linewidth=2, zorder=5,
                )
                ax.plot(
                    [i - 0.15, i + 0.15], [running + hi, running + hi],
                    color="#333", linewidth=2, zorder=5,
                )
            else:
                # Cost addition
                bar_bottoms.append(running)
                bar_heights.append(val)
                colors.append("#FF5722")
                lo = wf["low_values"][i]
                hi = wf["high_values"][i]
                ax.plot(
                    [i, i], [running + lo, running + hi],
                    color="#333", linewidth=2, zorder=5,
                )

            running += val

    bars = ax.bar(range(n), bar_heights, bottom=bar_bottoms, color=colors,
                  edgecolor="#333", linewidth=0.5, width=0.7)

    # Target line
    ax.axhline(y=wf["target"], color="#E91E63", linestyle="--", linewidth=2, label=f"Target ${wf['target']:,.0f}/kW")

    # Value labels on bars
    for i, (bot, ht, val) in enumerate(zip(bar_bottoms, bar_heights, wf["mid_values"])):
        if i == 0 or i == n - 1:
            ax.text(i, bot + ht + 30, f"${bot + ht:,.0f}", ha="center", va="bottom",
                    fontsize=9, fontweight="bold")
        else:
            ax.text(i, bot + ht / 2, f"${abs(val):,.0f}", ha="center", va="center",
                    fontsize=7, color="white" if abs(val) > 30 else "#333",
                    fontweight="bold")

    ax.set_xticks(range(n))
    ax.set_xticklabels(wf["labels"], rotation=45, ha="right", fontsize=8)
    ax.set_ylabel("Installed Cost ($/kW)", fontsize=11)
    ax.set_title("Path to $2,000/kW — Cost Reduction Waterfall", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=10)
    ax.set_ylim(0, wf["mid_values"][0] * 1.15)
    ax.grid(axis="y", alpha=0.3)

    # Legend for confidence
    legend_patches = [
        mpatches.Patch(color="#4CAF50", label="HIGH confidence"),
        mpatches.Patch(color="#FFC107", label="MEDIUM confidence"),
        mpatches.Patch(color="#F44336", label="LOW confidence"),
        mpatches.Patch(color="#FF5722", label="Cost addition"),
    ]
    ax.legend(handles=legend_patches + [plt.Line2D([0], [0], color="#E91E63", linestyle="--", label=f"Target ${wf['target']:,.0f}/kW")],
              loc="upper right", fontsize=8)

    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


# ── Pathway Details ───────────────────────────────────────────────────────────

def _render_pathway_details(a: PathwayAnalysis):
    st.markdown("### Pathway Details")

    for pw in a.pathways:
        with st.expander(f"**{pw.name}** (expected saving: ${pw.total_saving_mid:,.0f}/kW)", expanded=True):
            st.markdown(f"*{pw.description}*")

            for m in pw.mechanisms:
                conf_color = _confidence_color(m.confidence)
                addition_tag = " [COST ADDITION]" if m.is_cost_addition else ""

                st.markdown(f"""
<div style="border-left: 4px solid {conf_color}; padding: 12px; margin: 8px 0; background: #f8f9fa;">
<b style="color:#1A1A2E;">{m.name}{addition_tag}</b><br>
<table style="width:100%; color:#333; font-size:0.9em;">
<tr>
  <td><b>Optimistic:</b> ${m.saving_low:,.0f}/kW</td>
  <td><b>Expected:</b> ${m.saving_mid:,.0f}/kW</td>
  <td><b>Conservative:</b> ${m.saving_high:,.0f}/kW</td>
  <td><b>Confidence:</b> <span style="color:{conf_color}; font-weight:bold;">{m.confidence}</span></td>
</tr>
</table>
<p style="color:#555; font-size:0.85em; margin:6px 0 2px 0;"><b>Source:</b> {m.source}</p>
<p style="color:#555; font-size:0.85em; margin:2px 0;"><b>Assumption:</b> {m.assumption}</p>
<p style="color:#555; font-size:0.85em; margin:2px 0;"><b>Controllability:</b> {_control_label(m.fervo_controls)}
{' | <span style="color:#E91E63;">NEEDS UNIT 1 DATA</span>' if m.needs_unit1 else ''}</p>
{f'<p style="color:#888; font-size:0.8em; margin:2px 0;"><b>Notes:</b> {m.notes}</p>' if m.notes else ''}
</div>
""", unsafe_allow_html=True)

            # Pathway total
            st.markdown(f"""
**Pathway Total:** ${pw.total_saving_low:,.0f}/kW (optimistic) |
**${pw.total_saving_mid:,.0f}/kW** (expected) |
${pw.total_saving_high:,.0f}/kW (conservative)
""")

            # Honest limitation
            st.warning(f"**Limitation:** {pw.honest_limitation}")

            # Verdict
            projected = a.best_optimizer_per_kw - pw.total_saving_mid
            if projected <= TARGET_PER_KW:
                st.success(
                    f"This pathway alone could reach ${projected:,.0f}/kW at expected values "
                    f"(${projected - TARGET_PER_KW:+,.0f} vs target)."
                )
            else:
                st.error(
                    f"This pathway alone projects ${projected:,.0f}/kW — "
                    f"${projected - TARGET_PER_KW:,.0f}/kW above target. "
                    f"Requires additional pathways to close the gap."
                )


# ── Pathway Comparison Table ──────────────────────────────────────────────────

def _render_comparison_table(a: PathwayAnalysis):
    st.markdown("### Pathway Comparison")

    rows = []
    for pw in a.pathways:
        # Unit 1 = expected saving (no learning curve)
        unit1_mid = sum(m.saving_mid for m in pw.mechanisms if not m.needs_unit1)
        unit1_mid += sum(m.saving_mid for m in pw.mechanisms if m.needs_unit1) * 0.6  # discount unvalidated

        # Unit 8 = full expected
        unit8_mid = pw.total_saving_mid

        # Average confidence
        conf_counts = {CONFIDENCE_HIGH: 0, CONFIDENCE_MEDIUM: 0, CONFIDENCE_LOW: 0}
        for m in pw.mechanisms:
            conf_counts[m.confidence] = conf_counts.get(m.confidence, 0) + 1
        dominant = max(conf_counts, key=conf_counts.get)

        # Key dependency
        deps = [m.assumption[:60] for m in pw.mechanisms if m.confidence != CONFIDENCE_HIGH]
        key_dep = deps[0] + "..." if deps else "None identified"

        # Controllability
        ctrls = [m.fervo_controls for m in pw.mechanisms]
        if all(c == CONTROL_FULL for c in ctrls):
            ctrl = "Fully controlled"
        elif any(c == CONTROL_MINIMAL for c in ctrls):
            ctrl = "Partially market-dependent"
        else:
            ctrl = "Mostly controlled"

        rows.append({
            "Pathway": pw.name.split("—")[1].strip() if "—" in pw.name else pw.name,
            "Unit 1 Saving": f"${unit1_mid:,.0f}/kW",
            "Unit 8 Saving": f"${unit8_mid:,.0f}/kW",
            "Range": f"${pw.total_saving_high:,.0f}–${pw.total_saving_low:,.0f}/kW",
            "Confidence": dominant,
            "Controllability": ctrl,
            "Key Dependency": key_dep,
        })

    st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Combined scenarios
    st.markdown("### Combined Scenarios")
    st.markdown(
        "Pathways 1 and 2 are **alternatives** (both address construction cost — pick one). "
        "Pathway 3 **stacks** with either."
    )

    pw1 = a.pathways[0] if len(a.pathways) > 0 else None
    pw2 = a.pathways[1] if len(a.pathways) > 1 else None
    pw3 = a.pathways[2] if len(a.pathways) > 2 else None

    scenarios = []
    if pw1 and pw3:
        combined_mid = pw1.total_saving_mid + pw3.total_saving_mid
        combined_low = pw1.total_saving_low + pw3.total_saving_low
        combined_high = pw1.total_saving_high + pw3.total_saving_high
        proj = a.best_optimizer_per_kw - combined_mid
        scenarios.append({
            "Scenario": "P1 + P3 (Self-Perform + Equipment)",
            "Expected Saving": f"${combined_mid:,.0f}/kW",
            "Projected $/kW": f"${proj:,.0f}",
            "Range": f"${a.best_optimizer_per_kw - combined_low:,.0f}–${a.best_optimizer_per_kw - combined_high:,.0f}",
            "Verdict": "WITHIN TARGET" if proj <= TARGET_PER_KW else f"${proj - TARGET_PER_KW:,.0f} above",
        })

    if pw2 and pw3:
        combined_mid = pw2.total_saving_mid + pw3.total_saving_mid
        combined_low = pw2.total_saving_low + pw3.total_saving_low
        combined_high = pw2.total_saving_high + pw3.total_saving_high
        proj = a.best_optimizer_per_kw - combined_mid
        scenarios.append({
            "Scenario": "P2 + P3 (Modular + Equipment)",
            "Expected Saving": f"${combined_mid:,.0f}/kW",
            "Projected $/kW": f"${proj:,.0f}",
            "Range": f"${a.best_optimizer_per_kw - combined_low:,.0f}–${a.best_optimizer_per_kw - combined_high:,.0f}",
            "Verdict": "WITHIN TARGET" if proj <= TARGET_PER_KW else f"${proj - TARGET_PER_KW:,.0f} above",
        })

    if scenarios:
        st.dataframe(pd.DataFrame(scenarios), use_container_width=True)


# ── Critical Path ─────────────────────────────────────────────────────────────

def _render_critical_path(a: PathwayAnalysis):
    st.markdown("### Critical Path — Ordered Actions")

    for phase in a.critical_path:
        phase_name = phase["phase"]
        if "MUST" in phase_name:
            st.error(f"**{phase_name}**")
        elif "PARALLEL" in phase_name:
            st.warning(f"**{phase_name}**")
        else:
            st.info(f"**{phase_name}**")

        for item in phase["items"]:
            ctrl_color = _confidence_color(
                CONFIDENCE_HIGH if item["fervo_controls"] == CONTROL_FULL
                else CONFIDENCE_MEDIUM if item["fervo_controls"] == CONTROL_PARTIAL
                else CONFIDENCE_LOW
            )
            st.markdown(f"""
<div style="border-left: 3px solid {ctrl_color}; padding: 8px 12px; margin: 4px 0;">
<b style="color:#1A1A2E;">[ ] {item['action']}</b><br>
<span style="color:#555; font-size:0.9em;">{item['detail']}</span><br>
<span style="color:#888; font-size:0.85em;">Owner: {item['owner']} | {_control_label(item['fervo_controls'])}</span>
</div>
""", unsafe_allow_html=True)


# ── What Must Be True ─────────────────────────────────────────────────────────

def _render_what_must_be_true(a: PathwayAnalysis):
    st.markdown("### What Must Be True for $2,000/kW")
    st.markdown(
        "Ranked by controllability (Fervo-controlled first) and impact ($/kW). "
        "Items marked 'NEEDS UNIT 1' cannot be validated without building."
    )

    for i, cond in enumerate(a.what_must_be_true, 1):
        conf_color = _confidence_color(cond["confidence"])
        unit1_tag = " | NEEDS UNIT 1" if cond["needs_unit1"] else ""

        st.markdown(f"""
<div style="border: 1px solid #ddd; border-radius: 6px; padding: 10px; margin: 6px 0; background: #fafafa;">
<b style="color:#1A1A2E;">#{i}. {cond['mechanism']}</b>
<span style="float:right; color:{conf_color}; font-weight:bold;">{cond['confidence']}{unit1_tag}</span><br>
<span style="color:#555; font-size:0.9em;">{cond['condition']}</span><br>
<span style="color:#888; font-size:0.85em;">
  Saving: ${cond['saving_mid']:,.0f}/kW |
  {_control_label(cond['fervo_controls'])} |
  {cond['pathway']}
</span>
</div>
""", unsafe_allow_html=True)


# ── Honest Limitations ────────────────────────────────────────────────────────

def _render_limitations(a: PathwayAnalysis):
    st.markdown("### What This Analysis Cannot Tell You")
    st.markdown(
        "*The honest answer — 'we cannot know until Unit 1 is built' — "
        "is more valuable than a confident wrong number.*"
    )

    for lim in a.honest_limitations:
        st.markdown(f"""
<div style="border-left: 3px solid #F44336; padding: 8px 12px; margin: 6px 0; background: #fff5f5;">
<span style="color:#333; font-size:0.95em;">{lim}</span>
</div>
""", unsafe_allow_html=True)

    st.markdown("---")

    # DBD integration
    st.markdown("### Design Basis Integration")
    if st.button("Write pathway analysis to Design Basis Document (Section 10)", key="pw_dbd"):
        _update_dbd(a)
        st.success("Design Basis Document updated with Section 10: Path to $2,000/kW.")


def _update_dbd(a: PathwayAnalysis):
    """Write pathway summary into DBD Section 10."""
    try:
        from design_basis_document import load_dbd, save_dbd

        dbd = load_dbd()
        dbd["section_10_pathway_analysis"] = {
            "baseline_per_kw": a.baseline_per_kw,
            "best_optimizer_per_kw": a.best_optimizer_per_kw,
            "target_per_kw": a.target_per_kw,
            "gap_per_kw": a.gap_per_kw,
            "pathways": [
                {
                    "name": pw.name,
                    "saving_low": pw.total_saving_low,
                    "saving_mid": pw.total_saving_mid,
                    "saving_high": pw.total_saving_high,
                    "limitation": pw.honest_limitation,
                }
                for pw in a.pathways
            ],
            "honest_limitations": a.honest_limitations,
        }
        save_dbd(dbd)
    except Exception as e:
        st.error(f"Could not update DBD: {e}")
