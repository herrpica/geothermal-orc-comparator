"""
Free-form ORC Design Comparison Tab.

Users describe two ORC designs in natural language, Claude parses them
into run_orc_analysis parameters, runs both, and shows a weighted
comparison scorecard with a chat interface.

Launch via main.py (third tab) or standalone testing.
"""

import json
from typing import Any

import streamlit as st
import anthropic

from analysis_bridge import run_orc_analysis, TOOLS as BRIDGE_TOOLS

# ── Colors ──────────────────────────────────────────────────────────────────

NAVY = "#1A1A2E"
GREEN = "#00B050"
AMBER = "#F39C12"

# ── Tool schema reference (for NLP parsing context) ─────────────────────────

_RUN_ORC_SCHEMA = BRIDGE_TOOLS[0]["input_schema"]

# ── Metric definitions for scorecard ────────────────────────────────────────

SCORECARD_METRICS = [
    ("Net Power",    "net_power_MW",                      "MW",   True,  1),
    ("Gross Power",  "gross_power_MW",                    "MW",   True,  1),
    ("Parasitic",    "parasitic_MW",                      "MW",   False, 1),
    ("Efficiency",   "cycle_efficiency",                  "%",    True,  100),
    ("CAPEX",        "capex_total_USD",                   "$M",   False, 1e6),
    ("CAPEX/kW",     "capex_per_kW",                      "$/kW", False, 1),
    ("NPV",          "npv_USD",                           "$M",   True,  1e6),
    ("LCOE",         "lcoe_per_MWh",                      "$/MWh",False, 1),
    ("Schedule",     "construction_weeks_critical_path",  "wk",   False, 1),
]


# ── Session state initialization ────────────────────────────────────────────

def _init_comparison_state():
    """Initialize session state keys for the comparison tab."""
    defaults = {
        "comparison_design_a_text": "",
        "comparison_design_b_text": "",
        "comparison_parsed_a": None,
        "comparison_parsed_b": None,
        "comparison_results_a": None,
        "comparison_results_b": None,
        "comparison_chat_history": [],
        "comparison_phase": "input",  # input | review | results
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ── API key ─────────────────────────────────────────────────────────────────

def _get_api_key() -> str | None:
    try:
        return st.secrets["ANTHROPIC_API_KEY"]
    except Exception:
        import os
        return os.environ.get("ANTHROPIC_API_KEY")


# ── NLP design parsing ──────────────────────────────────────────────────────

_PARSE_SYSTEM = """You are an ORC (Organic Rankine Cycle) design parameter extractor.

Given a natural language description of an ORC design, extract the parameters
needed for the run_orc_analysis tool. The tool accepts these parameters:

{schema}

Rules:
1. Choose config "A" for direct air-cooled designs, "B" for propane intermediate loop designs.
2. Only include parameters explicitly stated or clearly implied by the description.
3. For parameters not mentioned, do NOT include them — the analysis will use defaults.
4. Flag any parameter you had to assume/infer (not explicitly stated) in "assumed_parameters".
5. Provide brief interpretation_notes explaining your understanding.

Respond with ONLY valid JSON in this format:
{{
  "config": "A" or "B",
  "parameters": {{...}},
  "assumed_parameters": ["param1", "param2"],
  "interpretation_notes": "..."
}}"""


def parse_design_description(
    description: str,
    api_key: str,
    design_basis: dict,
) -> dict | None:
    """Use Claude to parse a natural language design description into parameters."""
    if not description.strip():
        return None

    schema_str = json.dumps(_RUN_ORC_SCHEMA["properties"], indent=2)
    system = _PARSE_SYSTEM.format(schema=schema_str)

    context = f"Design basis context:\n{json.dumps(design_basis, indent=2, default=str)}"

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1000,
            system=system,
            messages=[{
                "role": "user",
                "content": f"{context}\n\nDesign description:\n{description}",
            }],
        )
        text = response.content[0].text.strip()
        # Extract JSON from response (handle markdown code blocks)
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)
    except Exception as e:
        st.error(f"Failed to parse design description: {e}")
        return None


# ── Comparison scorecard ────────────────────────────────────────────────────

def _render_comparison_scorecard(results_a: dict, results_b: dict, weights: dict):
    """Render a 9-row metric comparison table with winner indicators."""
    st.markdown(f"<h3 style='color:{NAVY};'>Comparison Scorecard</h3>",
                unsafe_allow_html=True)

    # Header row
    hcols = st.columns([2, 2, 2, 1])
    with hcols[0]:
        st.markdown("**Metric**")
    with hcols[1]:
        config_a_label = results_a.get("_detail", {}).get("config", "A")
        st.markdown(f"**Design A (Config {config_a_label})**")
    with hcols[2]:
        config_b_label = results_b.get("_detail", {}).get("config", "B")
        st.markdown(f"**Design B (Config {config_b_label})**")
    with hcols[3]:
        st.markdown("**Winner**")

    st.markdown("---")

    # Weighted score accumulators
    score_a = 0.0
    score_b = 0.0

    for label, key, unit, higher_better, divisor in SCORECARD_METRICS:
        val_a = results_a.get(key, 0)
        val_b = results_b.get(key, 0)

        display_a = val_a / divisor if divisor != 1 else val_a
        display_b = val_b / divisor if divisor != 1 else val_b

        # Format values
        if unit == "$M":
            fmt_a = f"${display_a:.1f}M"
            fmt_b = f"${display_b:.1f}M"
        elif unit == "$/kW":
            fmt_a = f"${display_a:,.0f}/kW"
            fmt_b = f"${display_b:,.0f}/kW"
        elif unit == "$/MWh":
            fmt_a = f"${display_a:.1f}/MWh"
            fmt_b = f"${display_b:.1f}/MWh"
        elif unit == "%":
            fmt_a = f"{display_a:.1f}%"
            fmt_b = f"{display_b:.1f}%"
        elif unit == "MW":
            fmt_a = f"{display_a:.1f} MW"
            fmt_b = f"{display_b:.1f} MW"
        elif unit == "wk":
            fmt_a = f"{display_a:.0f} wk"
            fmt_b = f"{display_b:.0f} wk"
        else:
            fmt_a = f"{display_a:.2f}"
            fmt_b = f"{display_b:.2f}"

        # Determine winner
        if higher_better:
            a_wins = val_a > val_b
        else:
            a_wins = val_a < val_b

        if val_a == val_b:
            winner = "Tie"
            winner_color = "#666"
        elif a_wins:
            winner = "A"
            winner_color = GREEN
        else:
            winner = "B"
            winner_color = NAVY

        cols = st.columns([2, 2, 2, 1])
        with cols[0]:
            st.markdown(f"**{label}**")
        with cols[1]:
            st.markdown(fmt_a)
        with cols[2]:
            st.markdown(fmt_b)
        with cols[3]:
            st.markdown(f"<span style='color:{winner_color}; font-weight:700;'>"
                        f"{winner}</span>", unsafe_allow_html=True)

    # Weighted score summary
    st.markdown("---")

    # Compute weighted scores using objective weights
    w_eff = weights.get("efficiency", 0.4)
    w_cost = weights.get("capital_cost", 0.4)
    w_sched = weights.get("schedule", 0.2)

    def _compute_weighted(r: dict) -> float:
        eff = r.get("cycle_efficiency", 0) / 0.25  # normalize to ~0-1
        capex = r.get("capex_total_USD", 250e6)
        cost = max(0, 1 - (capex - 150e6) / 150e6)
        weeks = r.get("construction_weeks_critical_path", 60)
        sched = max(0, 1 - (weeks - 40) / 40)
        return w_eff * eff + w_cost * cost + w_sched * sched

    score_a = _compute_weighted(results_a)
    score_b = _compute_weighted(results_b)

    scols = st.columns([2, 2, 2, 1])
    with scols[0]:
        st.markdown("**Weighted Score**")
    with scols[1]:
        st.markdown(f"**{score_a:.3f}**")
    with scols[2]:
        st.markdown(f"**{score_b:.3f}**")
    with scols[3]:
        if score_a > score_b:
            st.markdown(f"<span style='color:{GREEN}; font-weight:700;'>A</span>",
                        unsafe_allow_html=True)
        elif score_b > score_a:
            st.markdown(f"<span style='color:{NAVY}; font-weight:700;'>B</span>",
                        unsafe_allow_html=True)
        else:
            st.markdown("Tie")

    st.caption(
        f"Weights: Efficiency={w_eff:.0%}, Capital Cost={w_cost:.0%}, "
        f"Schedule={w_sched:.0%}. Adjust weights in the sidebar to update instantly."
    )


# ── Comparison chat ─────────────────────────────────────────────────────────

def _build_comparison_context(results_a: dict, results_b: dict, design_basis: dict) -> str:
    """Build system context for the comparison chat."""
    return (
        "You are an ORC design consultant analyzing two competing designs. "
        "Answer questions comparing them using the analysis results below.\n\n"
        f"Design basis:\n{json.dumps(design_basis, indent=2, default=str)}\n\n"
        f"Design A results:\n{json.dumps(results_a, indent=2, default=str)}\n\n"
        f"Design B results:\n{json.dumps(results_b, indent=2, default=str)}"
    )


def _render_comparison_chat(results_a: dict, results_b: dict, design_basis: dict):
    """Render an inline chat interface for comparing the two designs."""
    st.markdown("---")
    st.markdown(f"<h3 style='color:{NAVY};'>Ask About These Designs</h3>",
                unsafe_allow_html=True)

    # Display chat history
    for msg in st.session_state.comparison_chat_history:
        if msg["role"] == "user":
            st.chat_message("user").markdown(msg["content"])
        else:
            st.chat_message("assistant").markdown(msg["content"])

    # Chat input
    user_input = st.chat_input("Ask a question about these designs...",
                                key="comparison_chat_input")
    if user_input:
        api_key = _get_api_key()
        if not api_key:
            st.error("No API key found.")
            return

        st.session_state.comparison_chat_history.append({
            "role": "user", "content": user_input,
        })
        st.chat_message("user").markdown(user_input)

        system = _build_comparison_context(results_a, results_b, design_basis)

        # Build messages for Claude
        messages = [
            {"role": m["role"], "content": m["content"]}
            for m in st.session_state.comparison_chat_history
        ]

        try:
            client = anthropic.Anthropic(api_key=api_key)
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                system=system,
                messages=messages,
            )
            reply = response.content[0].text
            st.session_state.comparison_chat_history.append({
                "role": "assistant", "content": reply,
            })
            st.chat_message("assistant").markdown(reply)
        except Exception as e:
            st.error(f"Chat error: {e}")


# ── Parsed parameter review ────────────────────────────────────────────────

def _render_parsed_parameters(parsed: dict, label: str, key_prefix: str) -> dict:
    """Render editable parameter table for a parsed design. Returns edited params."""
    params = parsed.get("parameters", {})
    assumed = set(parsed.get("assumed_parameters", []))
    notes = parsed.get("interpretation_notes", "")

    st.markdown(f"**{label}** — Config {parsed.get('config', '?')}")
    if notes:
        st.caption(notes)

    edited = {}
    for k, v in params.items():
        is_assumed = k in assumed
        display_name = k.replace("_", " ").title()
        suffix = " (assumed)" if is_assumed else ""

        if isinstance(v, (int, float)):
            edited[k] = st.number_input(
                f"{display_name}{suffix}",
                value=float(v),
                key=f"{key_prefix}_{k}",
                format="%.3f" if isinstance(v, float) else "%d",
            )
        elif isinstance(v, str):
            edited[k] = st.text_input(
                f"{display_name}{suffix}",
                value=v,
                key=f"{key_prefix}_{k}",
            )
        else:
            edited[k] = v

    # Show assumed parameters in amber
    if assumed:
        assumed_list = ", ".join(assumed)
        st.markdown(
            f"<span style='color:{AMBER}; font-size:12px;'>"
            f"Assumed parameters: {assumed_list}</span>",
            unsafe_allow_html=True,
        )

    return {"config": parsed.get("config", "A"), **edited}


# ── Main tab renderer ───────────────────────────────────────────────────────

def render_comparison_tab(design_basis: dict):
    """Render the free-form design comparison tab."""
    _init_comparison_state()

    st.markdown(
        f"<h2 style='color:{NAVY};'>Design Comparison</h2>"
        f"<p style='color:#666;'>Describe two ORC designs in plain English. "
        f"Claude will parse them into analysis parameters, run both, and "
        f"show a weighted comparison.</p>",
        unsafe_allow_html=True,
    )

    phase = st.session_state.comparison_phase

    # ── Input phase ─────────────────────────────────────────────────────
    col_a, col_b = st.columns(2)

    with col_a:
        st.markdown(f"**Design A**")
        design_a_text = st.text_area(
            "Describe Design A",
            value=st.session_state.comparison_design_a_text,
            height=150,
            key="comp_design_a_input",
            placeholder=(
                "e.g., Direct air-cooled ORC with 32°F ACC approach, "
                "10°F evaporator pinch, 0.88 turbine efficiency..."
            ),
            label_visibility="collapsed",
        )

    with col_b:
        st.markdown(f"**Design B**")
        design_b_text = st.text_area(
            "Describe Design B",
            value=st.session_state.comparison_design_b_text,
            height=150,
            key="comp_design_b_input",
            placeholder=(
                "e.g., Propane intermediate loop with 20°F ACC approach, "
                "12°F IHX approach, optimized for schedule..."
            ),
            label_visibility="collapsed",
        )

    parse_disabled = not (design_a_text.strip() and design_b_text.strip())

    if st.button("Parse Designs", disabled=parse_disabled, type="primary",
                 key="btn_parse_designs"):
        api_key = _get_api_key()
        if not api_key:
            st.error("No Anthropic API key found.")
        else:
            st.session_state.comparison_design_a_text = design_a_text
            st.session_state.comparison_design_b_text = design_b_text

            with st.spinner("Parsing Design A..."):
                parsed_a = parse_design_description(design_a_text, api_key, design_basis)
            with st.spinner("Parsing Design B..."):
                parsed_b = parse_design_description(design_b_text, api_key, design_basis)

            if parsed_a and parsed_b:
                st.session_state.comparison_parsed_a = parsed_a
                st.session_state.comparison_parsed_b = parsed_b
                st.session_state.comparison_phase = "review"
                st.session_state.comparison_results_a = None
                st.session_state.comparison_results_b = None
                st.session_state.comparison_chat_history = []
                st.rerun()

    # ── Review phase ────────────────────────────────────────────────────
    parsed_a = st.session_state.comparison_parsed_a
    parsed_b = st.session_state.comparison_parsed_b

    if parsed_a and parsed_b and phase in ("review", "results"):
        st.markdown("---")
        st.markdown(f"<h3 style='color:{NAVY};'>Review Parsed Parameters</h3>",
                    unsafe_allow_html=True)
        st.markdown("Edit any values before running the comparison.")

        rcol_a, rcol_b = st.columns(2)
        with rcol_a:
            edited_a = _render_parsed_parameters(parsed_a, "Design A", "comp_param_a")
        with rcol_b:
            edited_b = _render_parsed_parameters(parsed_b, "Design B", "comp_param_b")

        if st.button("Run Comparison", type="primary", key="btn_run_comparison"):
            with st.spinner("Running Design A analysis..."):
                results_a = run_orc_analysis(edited_a, design_basis)
            with st.spinner("Running Design B analysis..."):
                results_b = run_orc_analysis(edited_b, design_basis)

            st.session_state.comparison_results_a = results_a
            st.session_state.comparison_results_b = results_b
            st.session_state.comparison_phase = "results"
            st.rerun()

    # ── Results phase ───────────────────────────────────────────────────
    results_a = st.session_state.comparison_results_a
    results_b = st.session_state.comparison_results_b

    if results_a and results_b:
        st.markdown("---")
        weights = design_basis.get("objective_weights", {
            "efficiency": 0.4, "capital_cost": 0.4, "schedule": 0.2,
        })
        _render_comparison_scorecard(results_a, results_b, weights)
        _render_comparison_chat(results_a, results_b, design_basis)
