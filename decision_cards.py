"""
Streamlit UI components for the decision card system.

Renders proposal cards with accept/lock/decline/defer/modify buttons
and a constraint status panel showing locked/working/declined constraints.
"""

import streamlit as st

from proposal_system import (
    RefinementProposal,
    ConstraintManager,
    PARAM_DISPLAY_NAMES,
)

# ── Colors ──────────────────────────────────────────────────────────────────

NAVY = "#1A1A2E"
GREEN = "#00B050"
DARK_GRAY = "#2D2D3D"
RED = "#E74C3C"
AMBER = "#F39C12"
BLUE = "#3498DB"


# ── Decision card ───────────────────────────────────────────────────────────

def render_decision_card(proposal: RefinementProposal, card_index: int, max_rounds: int = 10):
    """Render a single decision card for a RefinementProposal.

    User decisions are stored in st.session_state.dialectic_pending_decisions.
    """
    bot_color = NAVY if proposal.proposed_by == "PropaneFirst" else DARK_GRAY
    bot_icon = "⚡" if proposal.proposed_by == "PropaneFirst" else "🏭"
    bot_label = proposal.proposed_by

    # Confidence badge colors
    conf_colors = {"high": GREEN, "medium": AMBER, "low": RED}
    conf_color = conf_colors.get(proposal.confidence, AMBER)

    with st.container(border=True):
        # ── Header row ──────────────────────────────────────────────
        hcol1, hcol2 = st.columns([4, 1])
        with hcol1:
            st.markdown(
                f"<span style='color:{bot_color}; font-weight:700;'>"
                f"{bot_icon} {bot_label}</span> &mdash; "
                f"**{proposal.title}**",
                unsafe_allow_html=True,
            )
        with hcol2:
            st.markdown(
                f"<span style='background:{conf_color}20; color:{conf_color}; "
                f"padding:2px 8px; border-radius:10px; font-size:12px; font-weight:600;'>"
                f"{proposal.confidence.upper()}</span>",
                unsafe_allow_html=True,
            )

        # ── Engineering argument ────────────────────────────────────
        st.markdown(
            f"<div style='color:#555; font-size:13px; line-height:1.5; "
            f"margin:4px 0 12px 0;'>{proposal.engineering_argument}</div>",
            unsafe_allow_html=True,
        )

        # ── Parameter changes ───────────────────────────────────────
        if proposal.parameter_changes:
            param_items = [
                f"`{PARAM_DISPLAY_NAMES.get(k, k)}` = **{v}**"
                for k, v in proposal.parameter_changes.items()
                if k != "config"
            ]
            if param_items:
                st.markdown("**Proposed parameters:** " + " | ".join(param_items))

        # ── 4-metric comparison row ─────────────────────────────────
        metric_defs = [
            ("Net Power", "net_power_MW", "MW", True, 1),
            ("NPV", "npv_USD", "$M", True, 1e6),
            ("CAPEX", "capex_total_USD", "$M", False, 1e6),
            ("Schedule", "construction_weeks_critical_path", "wk", False, 1),
        ]
        mcols = st.columns(4)
        for i, (label, key, unit, higher_better, divisor) in enumerate(metric_defs):
            with mcols[i]:
                baseline = proposal.baseline_metrics.get(key)
                proposed = proposal.proposed_metrics.get(key)

                if proposed is not None:
                    val = proposed / divisor if divisor != 1 else proposed
                    if unit == "$M":
                        display = f"${val:.1f}M"
                    elif unit == "MW":
                        display = f"{val:.1f} MW"
                    else:
                        display = f"{val} {unit}"

                    delta_str = None
                    if baseline is not None and baseline != 0:
                        diff = (proposed - baseline) / divisor if divisor != 1 else (proposed - baseline)
                        if unit == "$M":
                            delta_str = f"${diff:+.1f}M"
                        elif unit == "MW":
                            delta_str = f"{diff:+.1f} MW"
                        else:
                            delta_str = f"{diff:+.0f} {unit}"

                    st.metric(label, display, delta=delta_str,
                              delta_color="normal" if higher_better else "inverse")
                else:
                    st.metric(label, "—")

        # ── Modify parameters expander ──────────────────────────────
        mod_key = f"mod_{proposal.proposal_id}_{card_index}"
        with st.expander("Modify parameters"):
            modified_vals = {}
            for k, v in proposal.parameter_changes.items():
                if k == "config":
                    continue
                display_name = PARAM_DISPLAY_NAMES.get(k, k)
                if isinstance(v, float):
                    modified_vals[k] = st.number_input(
                        display_name, value=v,
                        key=f"{mod_key}_{k}",
                        format="%.3f",
                    )
                elif isinstance(v, int):
                    modified_vals[k] = st.number_input(
                        display_name, value=v,
                        key=f"{mod_key}_{k}",
                        step=1,
                    )
                else:
                    modified_vals[k] = v

        # ── Decision buttons ────────────────────────────────────────
        bcols = st.columns(5)

        decisions_key = "dialectic_pending_decisions"
        if decisions_key not in st.session_state:
            st.session_state[decisions_key] = {}

        with bcols[0]:
            if st.button("🔒 Lock", key=f"lock_{proposal.proposal_id}_{card_index}",
                         help="Lock this parameter — bots cannot change it"):
                st.session_state[decisions_key][proposal.proposal_id] = {
                    "decision": "locked",
                }
                st.rerun()

        with bcols[1]:
            if st.button("✓ Accept", key=f"accept_{proposal.proposal_id}_{card_index}",
                         help="Accept as working assumption — bots can challenge with evidence"):
                st.session_state[decisions_key][proposal.proposal_id] = {
                    "decision": "soft",
                }
                st.rerun()

        with bcols[2]:
            defer_round = st.selectbox(
                "Defer to round",
                options=list(range(proposal.round_num + 1, max_rounds + 1)),
                key=f"defer_sel_{proposal.proposal_id}_{card_index}",
                label_visibility="collapsed",
            )
            if st.button("⏸ Defer", key=f"defer_{proposal.proposal_id}_{card_index}",
                         help="Revisit this proposal in a later round"):
                st.session_state[decisions_key][proposal.proposal_id] = {
                    "decision": "deferred",
                    "defer_round": defer_round,
                }
                st.rerun()

        with bcols[3]:
            if st.button("✗ Decline", key=f"decline_{proposal.proposal_id}_{card_index}",
                         help="Reject — bots cannot re-propose the same change"):
                st.session_state[decisions_key][proposal.proposal_id] = {
                    "decision": "declined",
                }
                st.rerun()

        with bcols[4]:
            if st.button("✏️ Modify", key=f"modify_{proposal.proposal_id}_{card_index}",
                         help="Accept with your modified parameter values"):
                st.session_state[decisions_key][proposal.proposal_id] = {
                    "decision": "modified",
                    "modified_params": modified_vals,
                }
                st.rerun()


# ── Constraint panel ────────────────────────────────────────────────────────

def render_constraint_panel(constraint_mgr: ConstraintManager):
    """Render a three-column constraint status panel."""
    state = constraint_mgr.state

    has_any = (state.locked or state.soft or state.declined or state.deferred)
    if not has_any:
        return

    st.markdown("---")
    st.markdown("**Active Constraints**")
    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown(f"<span style='color:{GREEN}; font-weight:700;'>🔒 Fixed</span>",
                     unsafe_allow_html=True)
        if state.locked:
            for p in state.locked:
                params = ", ".join(f"{PARAM_DISPLAY_NAMES.get(k,k)}={v}"
                                   for k, v in p.parameter_changes.items() if k != "config")
                st.markdown(f"- {p.title}: {params}")
        else:
            st.caption("None")

    with c2:
        st.markdown(f"<span style='color:{AMBER}; font-weight:700;'>✓ Working</span>",
                     unsafe_allow_html=True)
        items = state.soft + state.deferred
        if items:
            for p in items:
                params = ", ".join(f"{PARAM_DISPLAY_NAMES.get(k,k)}={v}"
                                   for k, v in p.parameter_changes.items() if k != "config")
                label = p.title
                if p.status == "deferred":
                    label += f" (deferred to R{p.defer_until_round})"
                st.markdown(f"- {label}: {params}")
        else:
            st.caption("None")

    with c3:
        st.markdown(f"<span style='color:{RED}; font-weight:700;'>✗ Declined</span>",
                     unsafe_allow_html=True)
        if state.declined:
            for p in state.declined:
                params = ", ".join(f"{PARAM_DISPLAY_NAMES.get(k,k)}={v}"
                                   for k, v in p.parameter_changes.items() if k != "config")
                st.markdown(f"- ~~{p.title}~~: {params}")
        else:
            st.caption("None")
