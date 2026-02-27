"""
ORC Design Dialectic Engine — Streamlit UI

Two Claude bots debate the optimal ORC configuration for a geothermal plant,
running actual analysis through the existing ORC comparison tool as evidence.

Launch: streamlit run dialectic.py
"""

import json
import time
import threading
from typing import Any

import streamlit as st

from debate_engine import (
    run_debate,
    DebateState,
    DebateMessage,
    AnalysisRun,
    transcript_to_text,
    analysis_runs_to_json,
    DEFAULT_MODEL,
)
from synthesis import (
    synthesize_debate,
    build_deck_summary,
    save_converged_design,
    save_deck_summary,
)
from analysis_bridge import get_structural_proposals, clear_structural_proposals

# ── Fervo brand colors ──────────────────────────────────────────────────────

NAVY = "#1A1A2E"
GREEN = "#00B050"
DARK_GRAY = "#2D2D3D"
WHITE = "#FFFFFF"
LIGHT_GRAY = "#F0F2F6"
RED = "#E74C3C"

# ── Custom CSS ──────────────────────────────────────────────────────────────

def _inject_dialectic_css():
    """Inject CSS styling for the dialectic UI."""
    st.markdown(f"""
<style>
    /* Global theme */
    .stApp {{
        background-color: {WHITE};
    }}

    /* Bot message containers */
    .bot-a-msg {{
        background: linear-gradient(135deg, {NAVY}08, {NAVY}15);
        border-left: 4px solid {NAVY};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    .bot-b-msg {{
        background: linear-gradient(135deg, {DARK_GRAY}08, {DARK_GRAY}15);
        border-right: 4px solid {DARK_GRAY};
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }}
    .bot-a-header {{
        color: {NAVY};
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
    }}
    .bot-b-header {{
        color: {DARK_GRAY};
        font-weight: 700;
        font-size: 14px;
        margin-bottom: 8px;
        text-align: right;
    }}

    /* Arbitrator strip */
    .arbitrator-strip {{
        background: linear-gradient(135deg, {GREEN}15, {GREEN}25);
        border: 2px solid {GREEN};
        border-radius: 8px;
        padding: 20px;
        margin: 16px auto;
        max-width: 800px;
    }}
    .arbitrator-header {{
        color: {GREEN};
        font-weight: 700;
        font-size: 16px;
        text-align: center;
        margin-bottom: 12px;
    }}

    /* Analysis result cards */
    .analysis-card {{
        background: {WHITE};
        border: 1px solid #E0E0E0;
        border-radius: 6px;
        padding: 12px;
        margin: 8px 0;
        display: grid;
        grid-template-columns: repeat(4, 1fr);
        gap: 8px;
    }}
    .metric-cell {{
        text-align: center;
        padding: 4px;
    }}
    .metric-label {{
        font-size: 11px;
        color: #666;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }}
    .metric-value {{
        font-size: 18px;
        font-weight: 700;
        margin: 2px 0;
    }}
    .metric-delta {{
        font-size: 11px;
        font-weight: 600;
    }}
    .delta-win {{
        color: {GREEN};
    }}
    .delta-lose {{
        color: {RED};
    }}

    /* Round divider */
    .round-divider {{
        text-align: center;
        padding: 12px;
        margin: 16px 0;
        background: linear-gradient(90deg, transparent, {NAVY}20, transparent);
        border-radius: 4px;
        font-weight: 700;
        color: {NAVY};
        font-size: 13px;
        letter-spacing: 1px;
    }}

    /* Convergence banner */
    .convergence-banner {{
        background: linear-gradient(135deg, {GREEN}20, {GREEN}35);
        border: 2px solid {GREEN};
        border-radius: 8px;
        padding: 16px;
        text-align: center;
        margin: 16px 0;
    }}
    .convergence-banner h3 {{
        color: {GREEN};
        margin: 0;
    }}

    /* Sidebar styling */
    section[data-testid="stSidebar"] {{
        background-color: {NAVY};
    }}
    section[data-testid="stSidebar"] .stMarkdown {{
        color: {WHITE};
    }}

    /* Pulse animation for running indicator */
    @keyframes pulse {{
        0%, 100% {{ opacity: 1; }}
        50% {{ opacity: 0.5; }}
    }}
    .pulse {{
        animation: pulse 1.5s ease-in-out infinite;
    }}
</style>
""", unsafe_allow_html=True)

# ── Session state initialization ────────────────────────────────────────────

def _init_dialectic_state():
    """Initialize session state for the dialectic tab."""
    if "dialectic_debate_state" not in st.session_state:
        st.session_state.dialectic_debate_state = None
    if "dialectic_debate_running" not in st.session_state:
        st.session_state.dialectic_debate_running = False
    if "dialectic_debate_paused" not in st.session_state:
        st.session_state.dialectic_debate_paused = False
    if "dialectic_synthesis_result" not in st.session_state:
        st.session_state.dialectic_synthesis_result = None
    if "dialectic_deck_summary" not in st.session_state:
        st.session_state.dialectic_deck_summary = None
    if "dialectic_messages_displayed" not in st.session_state:
        st.session_state.dialectic_messages_displayed = []
    if "dialectic_current_status" not in st.session_state:
        st.session_state.dialectic_current_status = ""

# ── API key handling ────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    """Get Anthropic API key from secrets or env."""
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY")

# ── Analysis result card component ──────────────────────────────────────────

def render_analysis_card(result: dict, prev_result: dict | None = None):
    """Render a compact analysis result card with delta indicators."""
    if not result or "net_power_MW" not in result:
        return

    config = result.get("_detail", {}).get("config", "?")

    metrics = [
        ("Net Power", f"{result['net_power_MW']:.1f} MW", "net_power_MW", True),
        ("NPV", f"${result['npv_USD']/1e6:.1f}M", "npv_USD", True),
        ("Schedule", f"{result['construction_weeks_critical_path']} wk", "construction_weeks_critical_path", False),
        ("CAPEX", f"${result['capex_total_USD']/1e6:.0f}M", "capex_total_USD", False),
    ]

    cols = st.columns(4)
    for i, (label, formatted, key, higher_better) in enumerate(metrics):
        with cols[i]:
            delta_html = ""
            if prev_result and key in prev_result:
                diff = result[key] - prev_result[key]
                if key == "npv_USD":
                    delta_str = f"${diff/1e6:+.1f}M"
                elif key == "capex_total_USD":
                    delta_str = f"${diff/1e6:+.0f}M"
                elif key == "construction_weeks_critical_path":
                    delta_str = f"{diff:+.0f}wk"
                else:
                    delta_str = f"{diff:+.1f}"

                is_win = (diff > 0) == higher_better
                css_class = "delta-win" if is_win else "delta-lose"
                delta_html = f'<div class="metric-delta {css_class}">{delta_str} vs opponent</div>'

            st.markdown(f"""
            <div class="metric-cell">
                <div class="metric-label">{label}</div>
                <div class="metric-value">{formatted}</div>
                {delta_html}
            </div>
            """, unsafe_allow_html=True)


# ── Sidebar: Design Basis Form ──────────────────────────────────────────────

def build_dialectic_sidebar(shared_inputs=None):
    """Render the design basis input form. Returns the design_basis dict.

    If shared_inputs provides overlapping keys (in °F / lb/s format),
    those widgets are skipped and values are converted to SI for the
    returned design_basis dict.
    """
    _init_dialectic_state()
    shared = shared_inputs or {}

    # Unit conversion helpers
    def _f_to_c(f):
        return (f - 32) * 5 / 9

    def _lbs_to_kgs(lbs):
        return lbs * 0.453592

    with st.sidebar:
        st.markdown(f"<h2 style='color:{GREEN}; margin-bottom:4px;'>ORC Design Dialectic</h2>",
                    unsafe_allow_html=True)
        st.markdown(f"<p style='color:{WHITE}; opacity:0.7; font-size:13px;'>"
                    "Two AI engineers debate the optimal design</p>",
                    unsafe_allow_html=True)

        st.markdown(f"<hr style='border-color:{WHITE}20;'>", unsafe_allow_html=True)
        st.markdown(f"<h4 style='color:{GREEN};'>Resource</h4>", unsafe_allow_html=True)

        if "T_geo_in" not in shared:
            brine_inlet = st.number_input("Brine inlet temp (°C)", value=215.6,
                                           step=5.0, key="db_brine_inlet")
        else:
            brine_inlet = _f_to_c(shared["T_geo_in"])
        if "T_geo_out_min" not in shared:
            brine_outlet = st.number_input("Brine outlet temp (°C)", value=71.1,
                                            step=5.0, key="db_brine_outlet")
        else:
            brine_outlet = _f_to_c(shared["T_geo_out_min"])
        if "m_dot_geo" not in shared:
            brine_flow = st.number_input("Brine flow (kg/s)", value=498.95,
                                          step=25.0, key="db_brine_flow")
        else:
            brine_flow = _lbs_to_kgs(shared["m_dot_geo"])

        st.markdown(f"<h4 style='color:{GREEN};'>Site</h4>", unsafe_allow_html=True)

        if "T_ambient" not in shared:
            ambient_temp = st.number_input("Ambient temp (°C)", value=13.9,
                                            step=1.0, key="db_ambient")
        else:
            ambient_temp = _f_to_c(shared["T_ambient"])
        site_notes = st.text_area("Site notes", value="", height=60, key="db_notes")

        st.markdown(f"<h4 style='color:{GREEN};'>Target</h4>", unsafe_allow_html=True)

        net_target = st.number_input("Net power target (MW)", value=53.0,
                                      step=5.0, key="db_target")

        st.markdown(f"<h4 style='color:{GREEN};'>Economics</h4>", unsafe_allow_html=True)

        if "electricity_price" not in shared:
            energy_value = st.number_input("Energy value ($/MWh)", value=80.0,
                                            step=5.0, key="db_energy_value")
        else:
            energy_value = float(shared["electricity_price"])
        if "project_life" not in shared:
            plant_life = st.number_input("Plant life (years)", value=30,
                                          step=5, key="db_life")
        else:
            plant_life = shared["project_life"]
        if "discount_rate" not in shared:
            discount_rate = st.number_input("Discount rate", value=0.08,
                                             step=0.01, format="%.2f",
                                             key="db_discount")
        else:
            discount_rate = shared["discount_rate"] / 100

        st.markdown(f"<h4 style='color:{GREEN};'>Debate Controls</h4>", unsafe_allow_html=True)

        max_rounds = st.slider("Max rounds", min_value=2, max_value=10, value=6,
                                key="db_rounds")

        st.markdown(f"<p style='color:{WHITE}; opacity:0.8; font-size:12px;'>"
                    "Objective weights (must sum to 1.0)</p>",
                    unsafe_allow_html=True)
        w_eff = st.slider("Efficiency", 0.0, 1.0, 0.4, 0.05, key="db_w_eff")
        w_cost = st.slider("Capital cost", 0.0, 1.0, 0.4, 0.05, key="db_w_cost")
        w_sched = st.slider("Schedule", 0.0, 1.0, 0.2, 0.05, key="db_w_sched")

        # Normalize weights
        total_w = w_eff + w_cost + w_sched
        if total_w > 0:
            w_eff /= total_w
            w_cost /= total_w
            w_sched /= total_w

        design_basis = {
            "brine_inlet_temp_C": brine_inlet,
            "brine_outlet_temp_C": brine_outlet,
            "brine_flow_kg_s": brine_flow,
            "ambient_temp_C": ambient_temp,
            "site_notes": site_notes,
            "net_power_target_MW": net_target,
            "energy_value_per_MWh": energy_value,
            "plant_life_years": plant_life,
            "discount_rate": discount_rate,
            "max_rounds": max_rounds,
            "objective_weights": {
                "efficiency": round(w_eff, 2),
                "capital_cost": round(w_cost, 2),
                "schedule": round(w_sched, 2),
            },
        }

        return design_basis


# ── Main content area ───────────────────────────────────────────────────────

def render_header(design_basis: dict):
    """Render the top control bar."""
    cols = st.columns([2, 1, 1, 1, 1])

    with cols[0]:
        st.markdown(f"<h1 style='color:{NAVY}; margin:0;'>ORC Design Dialectic</h1>",
                    unsafe_allow_html=True)

    with cols[1]:
        start_disabled = st.session_state.dialectic_debate_running
        if st.button("▶ Start Debate", disabled=start_disabled,
                     type="primary", key="btn_start", use_container_width=True):
            st.session_state.dialectic_debate_running = True
            st.session_state.dialectic_debate_paused = False
            st.session_state.dialectic_synthesis_result = None
            st.session_state.dialectic_deck_summary = None
            st.session_state.dialectic_messages_displayed = []
            st.rerun()

    with cols[2]:
        if st.button("⏸ Pause", disabled=not st.session_state.dialectic_debate_running,
                     key="btn_pause", use_container_width=True):
            st.session_state.dialectic_debate_paused = True

    with cols[3]:
        has_synthesis = st.session_state.dialectic_synthesis_result is not None
        if has_synthesis:
            st.download_button(
                "⬇ JSON",
                data=json.dumps(st.session_state.dialectic_synthesis_result, indent=2, default=str),
                file_name="converged_design.json",
                mime="application/json",
                key="btn_json",
                use_container_width=True,
            )
        else:
            st.button("⬇ JSON", disabled=True, key="btn_json_disabled",
                      use_container_width=True)

    with cols[4]:
        has_deck = st.session_state.dialectic_deck_summary is not None
        if has_deck:
            st.download_button(
                "📊 Build Deck",
                data=json.dumps(st.session_state.dialectic_deck_summary, indent=2, default=str),
                file_name="deck_summary.json",
                mime="application/json",
                key="btn_deck",
                use_container_width=True,
            )
        else:
            st.button("📊 Build Deck", disabled=True, key="btn_deck_disabled",
                      use_container_width=True)


def render_debate_message(msg: DebateMessage, prev_opponent_result: dict | None = None):
    """Render a single debate message with appropriate styling."""
    if msg.bot == "system":
        st.markdown(f'<div class="round-divider">ROUND {msg.round_num}</div>',
                    unsafe_allow_html=True)
        return

    if msg.bot == "PropaneFirst":
        st.markdown(f'<div class="bot-a-header">⚡ PropaneFirst — Config B Advocate</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="bot-a-msg">{msg.content}</div>',
                    unsafe_allow_html=True)
    else:
        st.markdown(f'<div class="bot-b-header">🏭 Conventionalist — Config A Advocate</div>',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="bot-b-msg">{msg.content}</div>',
                    unsafe_allow_html=True)

    # Render analysis result cards
    for tr in msg.tool_results:
        result = tr["result"]
        if isinstance(result, dict) and "net_power_MW" in result:
            with st.container():
                config_label = result.get("_detail", {}).get("config", "?")
                st.markdown(f"**Analysis Result — Config {config_label}**")
                render_analysis_card(result, prev_opponent_result)
        elif isinstance(result, dict) and "logged" in result:
            st.info(f"📋 Structural proposal logged: {result.get('proposal_id', '')}")


def render_synthesis(synthesis: dict):
    """Render the synthesis / arbitrator result."""
    st.markdown('<div class="arbitrator-strip">', unsafe_allow_html=True)
    st.markdown('<div class="arbitrator-header">⚖️ ARBITRATOR SYNTHESIS</div>',
                unsafe_allow_html=True)

    arch = synthesis.get("architecture", "TBD")
    rationale = synthesis.get("architecture_rationale", "")
    st.markdown(f"**Recommended Architecture: Config {arch}**")
    st.markdown(rationale)

    # Validated performance
    validated = synthesis.get("validated_performance", synthesis.get("predicted_performance", {}))
    if validated:
        st.markdown("---")
        st.markdown("**Validated Performance**")
        vcols = st.columns(5)
        metrics = [
            ("Net Power", f"{validated.get('net_power_MW', 0):.1f} MW"),
            ("NPV", f"${validated.get('npv_USD', 0)/1e6:.1f}M"),
            ("LCOE", f"${validated.get('lcoe_per_MWh', 0):.1f}/MWh"),
            ("CAPEX", f"${validated.get('capex_total_USD', 0)/1e6:.0f}M"),
            ("Schedule", f"{validated.get('construction_weeks_critical_path', 0)} wk"),
        ]
        for i, (label, value) in enumerate(metrics):
            with vcols[i]:
                st.metric(label, value)

    # Key outcomes
    outcomes = synthesis.get("key_debate_outcomes", [])
    if outcomes:
        st.markdown("---")
        st.markdown("**Key Debate Outcomes**")
        for o in outcomes:
            st.markdown(f"- {o}")

    # Concessions
    concessions = synthesis.get("concessions", {})
    if any(concessions.values()):
        st.markdown("---")
        st.markdown("**Concessions**")
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"**PropaneFirst:**")
            for c in concessions.get("PropaneFirst", []):
                st.markdown(f"- {c}")
        with c2:
            st.markdown(f"**Conventionalist:**")
            for c in concessions.get("Conventionalist", []):
                st.markdown(f"- {c}")

    # Structural recommendations
    recs = synthesis.get("structural_recommendations", [])
    if recs:
        st.markdown("---")
        st.markdown("**Structural Recommendations**")
        for rec in recs:
            contested = "⚠️ Contested" if rec.get("contested") else "✅ Agreed"
            conf = rec.get("confidence", "")
            st.markdown(
                f"- **{rec.get('title', '')}** ({contested}, {conf} confidence) — "
                f"{rec.get('description', '')}"
            )
            if rec.get("requires_model_extension"):
                st.caption("→ Requires model extension to validate")

    st.markdown('</div>', unsafe_allow_html=True)


# ── Debate execution (synchronous with progress updates) ────────────────────

def execute_debate(design_basis: dict):
    """Run the debate and update the UI progressively."""
    api_key = _get_api_key()
    if not api_key:
        st.error("No Anthropic API key found. Set ANTHROPIC_API_KEY in "
                 ".streamlit/secrets.toml or as an environment variable.")
        st.session_state.dialectic_debate_running = False
        return

    status_container = st.empty()
    debate_container = st.container()
    synthesis_container = st.container()

    # Track opponent's last result for delta comparison
    last_result_by_bot: dict[str, dict] = {}

    def on_message(msg: DebateMessage):
        """Callback for each bot message."""
        st.session_state.dialectic_messages_displayed.append(msg)

    def on_round_complete(round_num: int, state: DebateState):
        """Callback after each round."""
        pass

    # Run debate
    with status_container:
        st.markdown(
            f'<div class="pulse" style="text-align:center; padding:12px; '
            f'background:{NAVY}10; border-radius:8px;">'
            f'⚡ Debate in progress...</div>',
            unsafe_allow_html=True,
        )

    state = run_debate(
        design_basis=design_basis,
        api_key=api_key,
        model=DEFAULT_MODEL,
        on_message=on_message,
        on_round_complete=on_round_complete,
    )

    st.session_state.dialectic_debate_state = state
    status_container.empty()

    # Display all messages
    with debate_container:
        current_round = 0
        for msg in state.transcript:
            if msg.round_num != current_round:
                current_round = msg.round_num
                st.markdown(
                    f'<div class="round-divider">ROUND {current_round}</div>',
                    unsafe_allow_html=True,
                )

            if msg.bot == "system":
                continue

            # Find opponent's last result for delta
            opponent = "Conventionalist" if msg.bot == "PropaneFirst" else "PropaneFirst"
            prev_result = last_result_by_bot.get(opponent)

            render_debate_message(msg, prev_result)

            # Update last result tracker
            for tr in msg.tool_results:
                r = tr["result"]
                if isinstance(r, dict) and "net_power_MW" in r:
                    last_result_by_bot[msg.bot] = r

        # Convergence banner
        if state.converged:
            st.markdown(
                f'<div class="convergence-banner">'
                f'<h3>✅ Debate Converged</h3>'
                f'<p>{state.convergence_reason}</p>'
                f'<p>Completed {state.current_round} rounds with '
                f'{len(state.analysis_runs)} analysis runs</p>'
                f'</div>',
                unsafe_allow_html=True,
            )
        elif state.error:
            st.error(f"Debate ended with error: {state.error}")

    # Run synthesis
    if state.completed and not state.error:
        with synthesis_container:
            with st.spinner("Running arbitrator synthesis..."):
                try:
                    synthesis = synthesize_debate(state, api_key=api_key)
                    st.session_state.dialectic_synthesis_result = synthesis

                    deck = build_deck_summary(synthesis, state)
                    st.session_state.dialectic_deck_summary = deck

                    render_synthesis(synthesis)
                except Exception as e:
                    st.error(f"Synthesis error: {e}")

    st.session_state.dialectic_debate_running = False


# ── Render existing debate state ────────────────────────────────────────────

def render_existing_debate():
    """Render a previously completed debate from session state."""
    state = st.session_state.dialectic_debate_state
    if state is None:
        return False

    last_result_by_bot: dict[str, dict] = {}
    current_round = 0

    for msg in state.transcript:
        if msg.round_num != current_round:
            current_round = msg.round_num
            st.markdown(
                f'<div class="round-divider">ROUND {current_round}</div>',
                unsafe_allow_html=True,
            )

        if msg.bot == "system":
            continue

        opponent = "Conventionalist" if msg.bot == "PropaneFirst" else "PropaneFirst"
        prev_result = last_result_by_bot.get(opponent)
        render_debate_message(msg, prev_result)

        for tr in msg.tool_results:
            r = tr["result"]
            if isinstance(r, dict) and "net_power_MW" in r:
                last_result_by_bot[msg.bot] = r

    if state.converged:
        st.markdown(
            f'<div class="convergence-banner">'
            f'<h3>✅ Debate Converged</h3>'
            f'<p>{state.convergence_reason}</p>'
            f'<p>Completed {state.current_round} rounds with '
            f'{len(state.analysis_runs)} analysis runs</p>'
            f'</div>',
            unsafe_allow_html=True,
        )

    if st.session_state.dialectic_synthesis_result:
        render_synthesis(st.session_state.dialectic_synthesis_result)

    return True


# ── Welcome screen ──────────────────────────────────────────────────────────

def render_welcome():
    """Show the welcome / instructions screen."""
    st.markdown(f"""
    <div style="text-align:center; padding:60px 20px;">
        <h2 style="color:{NAVY};">Welcome to the ORC Design Dialectic</h2>
        <p style="color:#666; max-width:600px; margin:0 auto; line-height:1.6;">
            Two AI engineers will debate the optimal ORC configuration for your
            geothermal plant. <strong>PropaneFirst</strong> advocates for the propane
            intermediate loop (Config B), while <strong>Conventionalist</strong>
            defends the direct air-cooled design (Config A).
        </p>
        <p style="color:#666; max-width:600px; margin:16px auto; line-height:1.6;">
            Each engineer runs real thermodynamic and economic analyses to support
            their arguments. An arbitrator synthesizes the optimal design from the
            evidence.
        </p>
        <div style="margin-top:32px; padding:20px; background:{NAVY}08;
                    border-radius:8px; max-width:500px; margin-left:auto; margin-right:auto;">
            <p style="color:{NAVY}; font-weight:600; margin-bottom:8px;">Getting Started:</p>
            <p style="color:#666; font-size:14px; margin:4px 0;">
                1. Fill in the design basis in the sidebar<br>
                2. Adjust objective weights (efficiency vs cost vs schedule)<br>
                3. Click <strong>▶ Start Debate</strong>
            </p>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # Quick stats about the analysis engine
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    with c1:
        st.metric("Analysis Parameters", "50+",
                   help="Full thermodynamic cycle, equipment sizing, and lifecycle economics")
    with c2:
        st.metric("Cost Components", "14",
                   help="Bottom-up equipment + indirect cost model")
    with c3:
        st.metric("Convergence Checks", "5",
                   help="NPV delta, approach temps, concessions, weighted scores, max rounds")


# ── Main ────────────────────────────────────────────────────────────────────

def render_dialectic_tab(design_basis: dict):
    """Render the dialectic debate UI given a design_basis dict."""
    _inject_dialectic_css()
    render_header(design_basis)

    st.markdown("---")

    if st.session_state.dialectic_debate_running:
        execute_debate(design_basis)
    elif st.session_state.dialectic_debate_state is not None:
        render_existing_debate()
    else:
        render_welcome()


# ── Standalone mode ────────────────────────────────────────────────────────

if __name__ == "__main__":
    st.set_page_config(
        page_title="ORC Design Dialectic",
        page_icon="⚡",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    _inject_dialectic_css()
    design_basis = build_dialectic_sidebar()
    render_dialectic_tab(design_basis)
