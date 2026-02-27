"""
ORC Comparator -- Streamlit Application

Compares two geothermal ORC configurations:
  Config A: Traditional ORC with recuperator + direct air-cooled condensing
  Config B: Isopentane power block + propane intermediate loop
"""

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
import json
import re
import os
import sys

try:
    from anthropic import Anthropic
    _ANTHROPIC_AVAILABLE = True
except ImportError:
    _ANTHROPIC_AVAILABLE = False

from fluid_properties import FluidProperties
from thermodynamics import (
    solve_config_a, solve_config_b, verify_recuperator_pinch,
    run_validation_checks, validate_inputs, calc_dT_dP,
)
from cost_model import (
    calculate_costs_a, calculate_costs_b, lifecycle_cost,
    optimize_approach_temp, construction_schedule, construction_cost_savings,
    simple_payback, schedule_savings_npv, COST_FACTORS, U_VALUES,
    hx_area_vs_pinch, acc_area_vs_pinch, duct_diameter_vs_dp,
    acc_tubes_vs_dp, sizing_tradeoff_sweep, ACC_TUBE_DEFAULTS,
    _duct_segment_cost,
    calculate_fan_power, pump_sizing, acc_area_with_air_rise,
    compute_power_balance,
)



# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================

def _init_analysis_session_state():
    """Initialize session state for the analysis tab."""
    if "analysis_chat_messages" not in st.session_state:
        st.session_state.analysis_chat_messages = []
    if "analysis_pending_apply" not in st.session_state:
        st.session_state.analysis_pending_apply = {}
    if "analysis_auto_fan_bays_a" not in st.session_state:
        _seed = calculate_fan_power(40, 95, {})
        st.session_state.analysis_auto_fan_bays_a = _seed["n_fans_required"]
        st.session_state.analysis_auto_fan_bays_b = _seed["n_fans_required"]


@st.cache_resource
def get_fluid_props():
    return FluidProperties()


fp = get_fluid_props()


# ============================================================================
# CLAUDE CHAT -- CONTEXT, PARSER, DIALOG
# ============================================================================

# Widget key map: maps recommendation keys to the Streamlit widget keys used
# in the sidebar form. These are the `key=` params on st.number_input.
_WIDGET_KEY_MAP = {
    "uc_vaporizer_per_ft2": "uc_vaporizer",
    "uc_preheater_per_ft2": "uc_preheater",
    "uc_recuperator_per_ft2": "uc_recuperator",
    "uc_ihx_per_ft2": "uc_ihx",
    "uc_acc_per_ft2": "uc_acc",
    "uc_iso_duct_per_ft2": "uc_iso_duct",
    "uc_prop_pipe_per_ft2": "uc_prop_pipe",
    "uc_prop_piping_pct": "uc_prop_pct",
    "uc_turbine_per_kw": "uc_turbine",
    "uc_steel_per_lb": "uc_steel",
    "uc_foundation_pct": "uc_foundation",
    "uc_engineering_pct": "uc_eng",
    "uc_construction_mgmt_pct": "uc_cm",
    "uc_contingency_pct": "uc_contingency",
}

# Bounds for Claude AI recommendation validation -- wide ranges to allow flexibility
_WIDGET_BOUNDS = {
    "uc_vaporizer": (1, 999), "uc_preheater": (1, 999),
    "uc_recuperator": (1, 999), "uc_ihx": (1, 999),
    "uc_acc": (1, 999), "uc_iso_duct": (1, 9999),
    "uc_prop_pipe": (1, 9999), "uc_prop_pct": (0, 100),
    "uc_turbine": (1, 99999), "uc_steel": (0.01, 999.0),
    "uc_foundation": (0, 100), "uc_eng": (0, 100),
    "uc_cm": (0, 100), "uc_contingency": (0, 100),
}


def _build_chat_context():
    """Build JSON context string for Claude system prompt injection."""
    ac = st.session_state.get("analysis_computed")
    if not ac:
        return "{}"
    inputs = ac["inputs"]
    pwr_a, pwr_b = ac["pwr_a"], ac["pwr_b"]
    perf_a, perf_b = ac["perf_a"], ac["perf_b"]
    duct_a, duct_b = ac["duct_a"], ac["duct_b"]
    costs_a, costs_b = ac["costs_a"], ac["costs_b"]
    lc_a, lc_b = ac["lc_a"], ac["lc_b"]
    total_economic_advantage = ac["total_economic_advantage"]
    a_total_weeks = ac["a_total_weeks"]
    b_total_weeks = ac["b_total_weeks"]
    sched_savings = ac["sched_savings"]
    sched_info = ac["sched_info"]
    constr_savings = ac["constr_savings"]
    vol_ratio = ac["vol_ratio"]
    checks_a, checks_b = ac["checks_a"], ac["checks_b"]
    warns = ac["warns"]
    ctx = {
        "inputs": {
            "brine_inlet_T_F": inputs["T_geo_in"],
            "brine_flow_lbs": inputs["m_dot_geo"],
            "ambient_T_F": inputs["T_ambient"],
            "brine_outlet_min_T_F": inputs["T_geo_out_min"],
            "vaporizer_pinch_F": inputs["dt_pinch_vaporizer"],
            "preheater_pinch_F": inputs["dt_pinch_preheater"],
            "acc_pinch_A_F": inputs["dt_pinch_acc_a"],
            "acc_pinch_B_F": inputs["dt_pinch_acc_b"],
            "recuperator_pinch_F": inputs["dt_pinch_recup"],
            "ihx_approach_F": inputs["dt_approach_intermediate"],
            "turbine_eff": inputs["eta_turbine"],
            "pump_eff": inputs["eta_pump"],
            "electricity_price_per_MWh": inputs["electricity_price"],
        },
        "unit_costs": {
            "vaporizer_per_ft2": inputs.get("uc_vaporizer_per_ft2", COST_FACTORS["vaporizer_per_ft2"]),
            "preheater_per_ft2": inputs.get("uc_preheater_per_ft2", COST_FACTORS["preheater_per_ft2"]),
            "recuperator_per_ft2": inputs.get("uc_recuperator_per_ft2", COST_FACTORS["recup_per_ft2"]),
            "ihx_per_ft2": inputs.get("uc_ihx_per_ft2", COST_FACTORS["hx_per_ft2"]),
            "acc_per_ft2": inputs.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"]),
            "iso_duct_per_ft2": inputs.get("uc_iso_duct_per_ft2", COST_FACTORS["iso_duct_per_ft2"]),
            "prop_pipe_per_ft2": inputs.get("uc_prop_pipe_per_ft2", COST_FACTORS["prop_pipe_per_ft2"]),
            "turbine_per_kw": inputs.get("uc_turbine_per_kw", COST_FACTORS["turbine_per_kw"]),
            "steel_per_lb": inputs.get("uc_steel_per_lb", COST_FACTORS["steel_per_lb"]),
            "foundation_pct": inputs.get("uc_foundation_pct", COST_FACTORS["foundation_pct"]),
            "engineering_pct": inputs.get("uc_engineering_pct", COST_FACTORS["engineering_pct"]),
            "construction_mgmt_pct": inputs.get("uc_construction_mgmt_pct", COST_FACTORS["construction_mgmt_pct"]),
            "contingency_pct": inputs.get("uc_contingency_pct", COST_FACTORS["contingency_pct"]),
            "prop_piping_pct": inputs.get("uc_prop_piping_pct", COST_FACTORS["prop_piping_pct"]),
        },
        "config_a": {
            "net_power_kw": round(pwr_a["P_net"], 1),
            "gross_power_kw": round(perf_a["gross_power_kw"], 1),
            "thermal_eff_pct": round(pwr_a["eta_thermal"] * 100, 2),
            "T_cond_F": round(perf_a["T_cond"], 1),
            "brine_effectiveness_kW_per_lbs": round(pwr_a["brine_effectiveness"], 3),
            "tailpipe_diameter_in": round(duct_a["tailpipe_diameter_in"], 1),
            "acc_header_diameter_in": round(duct_a["acc_header_diameter_in"], 1),
            "total_installed_cost": round(costs_a["total_installed"]),
            "equipment_subtotal": round(costs_a["equipment_subtotal"]),
            "vaporizer_area_ft2": round(costs_a["vaporizer_area_ft2"]),
            "preheater_area_ft2": round(costs_a["preheater_area_ft2"]),
            "recuperator_area_ft2": round(costs_a["recuperator_area_ft2"]),
            "acc_area_ft2": round(costs_a["acc_area_ft2"]),
            "ductwork_cost": round(costs_a["ductwork"]),
            "structural_steel_cost": round(costs_a["structural_steel"]),
            "lcoe_per_MWh": round(lc_a["lcoe"], 2),
            "net_npv": round(lc_a["net_npv"]),
            "specific_cost_per_kw": round(lc_a["specific_cost_per_kw"]),
        },
        "config_b": {
            "net_power_kw": round(pwr_b["P_net"], 1),
            "gross_power_kw": round(perf_b["gross_power_kw"], 1),
            "thermal_eff_pct": round(pwr_b["eta_thermal"] * 100, 2),
            "T_cond_iso_F": round(perf_b["T_cond_iso"], 1),
            "T_propane_cond_F": round(perf_b["T_propane_cond"], 1),
            "brine_effectiveness_kW_per_lbs": round(pwr_b["brine_effectiveness"], 3),
            "tailpipe_diameter_in": round(duct_b["tailpipe_diameter_in"], 1),
            "total_installed_cost": round(costs_b["total_installed"]),
            "equipment_subtotal": round(costs_b["equipment_subtotal"]),
            "vaporizer_area_ft2": round(costs_b["vaporizer_area_ft2"]),
            "preheater_area_ft2": round(costs_b["preheater_area_ft2"]),
            "recuperator_area_ft2": round(costs_b["recuperator_area_ft2"]),
            "ihx_area_ft2": round(costs_b["intermediate_hx_area_ft2"]),
            "acc_area_ft2": round(costs_b["acc_area_ft2"]),
            "ductwork_cost": round(costs_b["ductwork"]),
            "structural_steel_cost": round(costs_b["structural_steel"]),
            "propane_system_cost": round(costs_b["propane_system"]),
            "lcoe_per_MWh": round(lc_b["lcoe"], 2),
            "net_npv": round(lc_b["net_npv"]),
            "specific_cost_per_kw": round(lc_b["specific_cost_per_kw"]),
        },
        "comparison": {
            "cost_delta_B_minus_A": round(costs_b["total_installed"] - costs_a["total_installed"]),
            "power_delta_B_minus_A_kw": round(pwr_b["P_net"] - pwr_a["P_net"], 1),
            "npv_advantage_B": round(total_economic_advantage),
            "config_a_schedule_weeks": a_total_weeks,
            "config_b_schedule_weeks": b_total_weeks,
            "schedule_savings_weeks": sched_savings,
            "schedule_delta_weeks": sched_info["net_delta"],
            "construction_cost_savings": round(constr_savings["total_construction_savings"]),
            "vol_flow_ratio_A_over_B": round(vol_ratio, 1),
        },
        "validation": {
            "config_a_pass": all(passed for _, passed, _ in checks_a),
            "config_b_pass": all(passed for _, passed, _ in checks_b),
            "warnings": warns,
        },
    }
    return json.dumps(ctx, indent=2)


def _parse_recommendations(text):
    """Extract specific cost recommendations from Claude's response.

    Looks for component-name + dollar-value or percentage patterns and maps
    them to the corresponding widget keys for the Apply button.
    """
    recs = []
    seen = set()

    # (regex, widget_key, display_label, value_type)
    patterns = [
        (r'vaporizer[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_vaporizer", "Vaporizer", "int"),
        (r'\$\s*(\d+(?:\.\d+)?)\s*/\s*ft[²2]?\s*(?:for\s+)?(?:the\s+)?vaporizer', "uc_vaporizer", "Vaporizer", "int"),
        (r'preheater[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_preheater", "Preheater", "int"),
        (r'\$\s*(\d+(?:\.\d+)?)\s*/\s*ft[²2]?\s*(?:for\s+)?(?:the\s+)?preheater', "uc_preheater", "Preheater", "int"),
        (r'recuperator[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_recuperator", "Recuperator", "int"),
        (r'\$\s*(\d+(?:\.\d+)?)\s*/\s*ft[²2]?\s*(?:for\s+)?(?:the\s+)?recuperator', "uc_recuperator", "Recuperator", "int"),
        (r'(?:IHX|intermediate)[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_ihx", "IHX", "int"),
        (r'\$\s*(\d+(?:\.\d+)?)\s*/\s*ft[²2]?\s*(?:for\s+)?(?:the\s+)?(?:IHX|intermediate)', "uc_ihx", "IHX", "int"),
        (r'ACC[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_acc", "ACC", "int"),
        (r'\$\s*(\d+(?:\.\d+)?)\s*/\s*ft[²2]?\s*(?:for\s+)?(?:the\s+)?ACC', "uc_acc", "ACC", "int"),
        (r'(?:iso(?:pentane)?\s*)?duct(?:work)?[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_iso_duct", "Iso Duct", "int"),
        (r'propane\s*pip(?:e|ing)[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*ft', "uc_prop_pipe", "Propane Pipe", "int"),
        (r'turbine[^$\n]{0,80}\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*/\s*kW', "uc_turbine", "Turbine", "int"),
        (r'\$\s*(\d+(?:,\d{3})*(?:\.\d+)?)\s*/\s*kW\s*(?:for\s+)?(?:the\s+)?turbine', "uc_turbine", "Turbine", "int"),
        (r'(?:structural\s*)?steel[^$\n]{0,80}\$\s*(\d+(?:\.\d+)?)\s*/\s*lb', "uc_steel", "Steel", "float"),
        (r'foundation\D{0,80}?(\d+(?:\.\d+)?)\s*%', "uc_foundation", "Foundation", "int"),
        (r'engineering\D{0,80}?(\d+(?:\.\d+)?)\s*%', "uc_eng", "Engineering", "int"),
        (r'construction\s*(?:management|mgmt)\D{0,80}?(\d+(?:\.\d+)?)\s*%', "uc_cm", "Constr. Mgmt", "int"),
        (r'contingency\D{0,80}?(\d+(?:\.\d+)?)\s*%', "uc_contingency", "Contingency", "int"),
    ]

    for pat, wkey, label, vtype in patterns:
        if wkey in seen:
            continue
        match = re.search(pat, text, re.IGNORECASE)
        if match:
            val_str = match.group(1).replace(",", "")
            val = float(val_str)
            if vtype == "int":
                val = int(round(val))
            # Validate against widget bounds
            bounds = _WIDGET_BOUNDS.get(wkey)
            if bounds and bounds[0] <= val <= bounds[1]:
                seen.add(wkey)
                recs.append((wkey, label, val))

    return recs


_CLAUDE_SYSTEM_PROMPT = """You are an expert geothermal ORC process engineer and cost estimator with deep knowledge of organic Rankine cycles, air cooled condensers, heat exchanger design, and geothermal power plant construction costs. You are embedded in an ORC comparison tool that analyzes a traditional isopentane ORC against a split propane loop ORC configuration. Both cycles include a preheater, vaporizer, recuperator, and air cooled condensers. The split configuration adds an isopentane/propane intermediate heat exchanger and propane loop to reduce vapor ductwork size.

You have access to the current calculation results and inputs shown below. Use these to give specific, quantitative answers where possible. When reassessing cost assumptions, provide a specific recommended value with a brief justification based on market conditions, service requirements, or industry standards. When the user asks about cost assumptions, always provide a specific number they can enter into the tool, not a range. Be direct and concise -- this is an engineering tool, not a general discussion.

IMPORTANT: When recommending a specific unit cost, always state it in the format "$XX/ft2" or "$XX/kW" or "$XX/lb" or "XX%" so the tool can parse and offer an Apply button. Always give ONE specific number, not a range.

Current calculation state:
{context}"""


def _get_anthropic_client():
    """Get Anthropic client, checking secrets then env var."""
    api_key = None
    try:
        api_key = st.secrets.get("ANTHROPIC_API_KEY")
    except Exception:
        pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    return Anthropic(api_key=api_key)


@st.dialog("Ask Claude -- ORC Engineering Assistant", width="large")
def _claude_chat_dialog():
    """Modal dialog with embedded Claude chat."""

    if not _ANTHROPIC_AVAILABLE:
        st.error("The `anthropic` package is not installed. Run: `pip install anthropic`")
        return

    client = _get_anthropic_client()
    if client is None:
        st.warning(
            "No Anthropic API key configured. Set `ANTHROPIC_API_KEY` in "
            "`.streamlit/secrets.toml` or as an environment variable."
        )
        st.code('# .streamlit/secrets.toml\nANTHROPIC_API_KEY = "sk-ant-..."', language="toml")
        return

    # --- Quick-access prompt buttons ---
    st.markdown("**Quick questions:**")
    qcols = st.columns(3)
    _quick_prompts = [
        "Reassess ACC unit cost",
        "Reassess intermediate HX cost",
        "Sanity check these results",
        "Why is Config A/B winning?",
        "Recommend pinch point adjustments",
        "Explain the key tradeoff",
    ]
    _clicked_prompt = None
    for i, qp in enumerate(_quick_prompts):
        with qcols[i % 3]:
            if st.button(qp, key=f"_qp_{i}", use_container_width=True):
                _clicked_prompt = qp

    st.divider()

    # --- Message history display ---
    chat_box = st.container(height=420)
    with chat_box:
        for i, msg in enumerate(st.session_state.analysis_chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show Apply buttons for assistant messages with recommendations
                if msg["role"] == "assistant" and msg.get("recs"):
                    for wkey, label, val in msg["recs"]:
                        btn_label = f"Apply: {label} = {val}"
                        if st.button(btn_label, key=f"_apply_{wkey}_{i}"):
                            st.session_state.analysis_pending_apply = {wkey: val}
                            st.rerun()

    # --- Chat input ---
    user_input = st.chat_input("Ask about the ORC design, costs, or assumptions...")

    # Resolve actual prompt (typed or quick-button)
    prompt = user_input or _clicked_prompt
    if not prompt:
        # --- Footer ---
        fcol1, fcol2 = st.columns([3, 1])
        with fcol2:
            if st.button("Clear Chat", key="_clear_chat"):
                st.session_state.analysis_chat_messages = []
                st.rerun(scope="fragment")
        st.caption(
            "AI recommendations should be validated against current vendor quotes "
            "and project-specific conditions before use in project decisions."
        )
        return

    # Add user message
    st.session_state.analysis_chat_messages.append({"role": "user", "content": prompt})

    # Build API messages (role/content only, no extra keys)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.analysis_chat_messages
    ]

    # Build system prompt with current context
    context_json = _build_chat_context()
    system_prompt = _CLAUDE_SYSTEM_PROMPT.format(context=context_json)

    # Call Anthropic API
    with st.spinner("Claude is thinking..."):
        try:
            response = client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1000,
                system=system_prompt,
                messages=api_messages,
            )
            reply = response.content[0].text
        except Exception as e:
            reply = f"API error: {e}"

    # Parse recommendations from response
    recs = _parse_recommendations(reply)

    # Store assistant message with recs
    st.session_state.analysis_chat_messages.append({
        "role": "assistant",
        "content": reply,
        "recs": recs,
    })

    # Rerun dialog to display the new messages
    st.rerun(scope="fragment")




# ============================================================================

# ============================================================================
# HELPERS
# ============================================================================

def _winner(val_a, val_b, lower_better=True):
    """Return winner string."""
    if lower_better:
        if val_a < val_b - 0.001:
            return "A"
        elif val_b < val_a - 0.001:
            return "B"
    else:
        if val_a > val_b + 0.001:
            return "A"
        elif val_b > val_a + 0.001:
            return "B"
    return "Tie"


def _fmt(val, fmt_str=".1f", prefix="", suffix=""):
    try:
        return f"{prefix}{val:{fmt_str}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


def _color_winner(winner):
    if winner == "A":
        return ":blue[**A**]"
    elif winner == "B":
        return ":red[**B**]"
    return "Tie"


def _svg_defs():
    """Shared SVG arrow marker definitions."""
    return """
    <defs>
      <marker id="arr-red" viewBox="0 0 8 6" refX="8" refY="3"
              markerWidth="8" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,3 L0,6 Z" fill="#c0392b"/>
      </marker>
      <marker id="arr-blue" viewBox="0 0 8 6" refX="8" refY="3"
              markerWidth="8" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,3 L0,6 Z" fill="#2980b9"/>
      </marker>
      <marker id="arr-green" viewBox="0 0 8 6" refX="8" refY="3"
              markerWidth="8" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,3 L0,6 Z" fill="#27ae60"/>
      </marker>
      <marker id="arr-orange" viewBox="0 0 8 6" refX="8" refY="3"
              markerWidth="8" markerHeight="6" orient="auto-start-reverse">
        <path d="M0,0 L8,3 L0,6 Z" fill="#d35400"/>
      </marker>
    </defs>"""


def _svg_style():
    """Shared SVG CSS styles."""
    return """
    <style>
      .eq-box { stroke: #333; stroke-width: 1.5; rx: 6; ry: 6; }
      .eq-label { font-size: 10.5px; font-weight: bold; fill: #222; font-family: sans-serif; }
      .eq-duty { font-size: 8.5px; fill: #555; font-family: sans-serif; }
      .state-label { font-size: 8px; fill: #444; font-family: monospace; }
      .state-box { fill: #fff; stroke: #bbb; stroke-width: 0.7; rx: 3; ry: 3; }
      .flow-hot { stroke: #c0392b; stroke-width: 2.5; fill: none; }
      .flow-cold { stroke: #2980b9; stroke-width: 2.5; fill: none; }
      .flow-green { stroke: #27ae60; stroke-width: 2.5; fill: none; }
      .flow-brine { stroke: #d35400; stroke-width: 2; fill: none; stroke-dasharray: 6,3; }
      .halo { stroke: #c0392b; stroke-opacity: 0.12; stroke-width: 10; fill: none; }
      .annot-box { fill: none; stroke: #c0392b; stroke-width: 1; stroke-dasharray: 4,2; rx: 4; }
      .annot-text { font-size: 8px; fill: #c0392b; font-weight: bold; font-family: sans-serif; }
      .annot-green { font-size: 8px; fill: #27ae60; font-weight: bold; font-family: sans-serif; }
      .title-text { font-size: 13px; font-weight: bold; fill: #333; font-family: sans-serif; }
      .brine-label { font-size: 8px; fill: #d35400; font-family: sans-serif; }
    </style>"""


def _eq_box(x, y, w, h, fill, name, duty_str):
    """Generate SVG rect + centered labels for an equipment block."""
    cx = x + w / 2
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" '
        f'class="eq-box" fill="{fill}"/>'
        f'<text x="{cx}" y="{y + h/2 - 4}" text-anchor="middle" '
        f'class="eq-label">{name}</text>'
        f'<text x="{cx}" y="{y + h/2 + 9}" text-anchor="middle" '
        f'class="eq-duty">{duty_str}</text>'
    )


def _state_tag(x, y, num, T, P, side="left"):
    """State point label with white background box, offset 15px from anchor."""
    label = f"{num}: {T:.0f}F / {P:.0f} psia"
    bw, bh = 115, 16
    if side == "right":
        bx = x + 15
        tx = bx + 4
        anc = "start"
    else:
        bx = x - 15 - bw
        tx = bx + bw - 4
        anc = "end"
    by = y - bh / 2
    ty = y + 3
    return (
        f'<rect x="{bx}" y="{by}" width="{bw}" height="{bh}" class="state-box"/>'
        f'<text x="{tx}" y="{ty}" text-anchor="{anc}" class="state-label">{label}</text>'
    )


def _generate_pfd_a(states, perf, duct, inputs):
    """Generate Config A PFD as inline SVG wrapped in an HTML div."""
    m = perf["m_dot_iso"]

    # Duties
    vap_duty = m * perf["q_vaporizer"] / 1e6
    pre_duty = m * perf["q_preheater"] / 1e6
    turb_kw = perf["gross_power_kw"]
    recup_duty = perf["Q_recup_mmbtu_hr"]
    rej_duty = perf["Q_reject_mmbtu_hr"]
    pump_kw = m * perf["w_pump"] / 3412.14

    s = states
    tp_dia = duct["tailpipe_diameter_in"]
    L_run = inputs.get("L_tailpipe_a", 30) + inputs.get("L_long_header", 120)

    svg = f"""<div style="background:#fafafa;border:1px solid #ddd;border-radius:8px;padding:4px;">
<svg viewBox="0 0 754 650" xmlns="http://www.w3.org/2000/svg">
{_svg_style()}
{_svg_defs()}

<!-- Title -->
<text x="377" y="26" text-anchor="middle" class="title-text">Config A -- Direct ACC</text>

<!-- Equipment boxes -->
{_eq_box(185, 55, 150, 44, '#fff3e0', 'VAPORIZER', f'{vap_duty:.2f} MMBtu/hr')}
{_eq_box(540, 80, 120, 48, '#fce4ec', 'TURBINE', f'{turb_kw:.0f} kW')}
{_eq_box(45, 205, 145, 44, '#fff3e0', 'PREHEATER', f'{pre_duty:.2f} MMBtu/hr')}
{_eq_box(300, 250, 145, 68, '#f3e5f5', 'RECUPERATOR', f'{recup_duty:.2f} MMBtu/hr')}
{_eq_box(470, 460, 145, 50, '#e3f2fd', 'ACC', f'{rej_duty:.2f} MMBtu/hr')}
{_eq_box(70, 465, 125, 40, '#e8f5e9', 'ISO PUMP', f'{pump_kw:.0f} kW')}

<!-- Recuperator internal divider -->
<line x1="310" y1="284" x2="435" y2="284" stroke="#999" stroke-width="0.8" stroke-dasharray="3,2"/>

<!-- ============ FLOW LINES ============ -->

<!-- State 1: Vaporizer -> Turbine (hot vapor, red, right across top) -->
<polyline points="335,77 438,77 438,104 540,104"
  class="flow-hot" marker-end="url(#arr-red)"/>
{_state_tag(438, 85, 1, s['1'].T, s['1'].P, 'right')}

<!-- State 2: Turbine -> Recuperator hot in (red, down right side) -->
<polyline points="600,128 600,220 445,220 445,250"
  class="flow-hot" marker-end="url(#arr-red)"/>
<polyline points="600,128 600,220 445,220 445,250"
  class="halo"/>
{_state_tag(600, 175, 2, s['2'].T, s['2'].P, 'right')}

<!-- State 3: Recuperator hot out -> ACC (red, down) -->
<polyline points="445,318 445,395 542,395 542,460"
  class="flow-hot" marker-end="url(#arr-red)"/>
<polyline points="445,318 445,395 542,395 542,460"
  class="halo"/>
{_state_tag(450, 360, 3, s['3'].T, s['3'].P, 'right')}

<!-- State 4: ACC -> Pump (blue, left across bottom) -->
<polyline points="470,485 280,485 280,530 250,530 250,485 195,485"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(265, 540, 4, s['4'].T, s['4'].P, 'right')}

<!-- State 5: Pump -> Recuperator cold in (blue, up) -->
<polyline points="70,465 70,310 300,310"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(75, 390, 5, s['5'].T, s['5'].P, 'right')}

<!-- State 6: Recuperator cold out -> Preheater (blue, left) -->
<polyline points="300,270 240,270 240,227 190,227"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(240, 255, 6, s['6'].T, s['6'].P, 'left')}

<!-- State 7: Preheater -> Vaporizer (warm, up left side) -->
<polyline points="117,205 117,130 117,99 185,77"
  class="flow-cold" marker-end="url(#arr-red)"/>
{_state_tag(117, 165, 7, s['7'].T, s['7'].P, 'right')}

<!-- ============ BRINE PATH (orange dashed) ============ -->
<polyline points="260,38 260,55"
  class="flow-brine" marker-end="url(#arr-orange)"/>
<text x="265" y="33" class="brine-label">Brine In {inputs['T_geo_in']:.0f}F</text>

<polyline points="260,99 117,99 117,205"
  class="flow-brine" marker-end="url(#arr-orange)"/>
<text x="170" y="112" class="brine-label">T_mid {perf['T_brine_mid']:.0f}F</text>

<polyline points="117,249 117,275"
  class="flow-brine"/>
<text x="65" y="290" class="brine-label">Brine Out {perf['T_geo_out_calc']:.0f}F</text>

<!-- Annotation box (bottom) -->
<rect x="190" y="575" width="370" height="55" class="annot-box"/>
<text x="375" y="595" text-anchor="middle" class="annot-text">LOW-PRESSURE VAPOR PATH</text>
<text x="375" y="607" text-anchor="middle" class="annot-text">{tp_dia:.0f}" dia x {L_run:.0f} ft run (per train, 2 trains)</text>
<text x="375" y="619" text-anchor="middle" style="font-size:7.5px;fill:#c0392b;font-family:sans-serif;">
  (field-routed isopentane vapor)</text>

</svg></div>"""
    return svg


def _generate_pfd_b(states, prop_states, perf, duct, duct_a, inputs):
    """Generate Config B PFD as inline SVG wrapped in an HTML div."""
    m = perf["m_dot_iso"]
    m_prop = perf["m_dot_prop"]

    vap_duty = m * perf["q_vaporizer"] / 1e6
    pre_duty = m * perf["q_preheater"] / 1e6
    turb_kw = perf["gross_power_kw"]
    recup_duty = perf["Q_recup_mmbtu_hr"]
    ihx_duty = m * perf["q_cond_iso"] / 1e6
    pump_iso_kw = m * perf["w_pump_iso"] / 3412.14
    pump_prop_kw_calc = m_prop * perf["w_pump_prop"] / 3412.14
    pump_prop_kw = 0.0 if inputs.get("prop_thermosiphon", True) else pump_prop_kw_calc
    prop_pump_label = "THERMOSIPHON" if inputs.get("prop_thermosiphon", True) else "PROP PUMP"
    prop_pump_detail = "no pump" if inputs.get("prop_thermosiphon", True) else f"{pump_prop_kw:.0f} kW"
    rej_duty = perf["Q_reject_mmbtu_hr"]

    s = states
    ps = prop_states
    prop_dia = duct.get("propane_header_diameter_in", 0)
    tp_a_dia = duct_a["tailpipe_diameter_in"]

    svg = f"""<div style="background:#fafafa;border:1px solid #ddd;border-radius:8px;padding:4px;">
<svg viewBox="0 0 754 650" xmlns="http://www.w3.org/2000/svg">
{_svg_style()}
{_svg_defs()}

<!-- Title -->
<text x="377" y="26" text-anchor="middle" class="title-text">Config B -- Propane Intermediate Loop</text>

<!-- Equipment boxes -->
{_eq_box(72, 55, 140, 44, '#fff3e0', 'VAPORIZER', f'{vap_duty:.2f} MMBtu/hr')}
{_eq_box(385, 80, 110, 48, '#fce4ec', 'TURBINE', f'{turb_kw:.0f} kW')}
{_eq_box(13, 205, 115, 44, '#fff3e0', 'PREHEATER', f'{pre_duty:.2f} MMBtu/hr')}
{_eq_box(182, 255, 135, 68, '#f3e5f5', 'RECUPERATOR', f'{recup_duty:.2f} MMBtu/hr')}
{_eq_box(182, 435, 135, 50, '#fffde7', 'IHX', f'{ihx_duty:.2f} MMBtu/hr')}
{_eq_box(13, 440, 105, 38, '#e8f5e9', 'ISO PUMP', f'{pump_iso_kw:.0f} kW')}
{_eq_box(520, 230, 140, 50, '#e3f2fd', 'PROPANE ACC', f'{rej_duty:.2f} MMBtu/hr')}
{_eq_box(533, 435, 120, 38, '#e8f5e9', prop_pump_label, prop_pump_detail)}

<!-- Recuperator internal divider -->
<line x1="192" y1="289" x2="307" y2="289" stroke="#999" stroke-width="0.8" stroke-dasharray="3,2"/>

<!-- ============ ISOPENTANE LOOP ============ -->

<!-- State 1: Vaporizer -> Turbine (hot vapor, red) -->
<polyline points="212,77 300,77 300,104 385,104"
  class="flow-hot" marker-end="url(#arr-red)"/>
{_state_tag(300, 85, 1, s['1'].T, s['1'].P, 'right')}

<!-- State 2: Turbine -> Recuperator hot in (red) -->
<polyline points="440,128 440,220 317,220 317,255"
  class="flow-hot" marker-end="url(#arr-red)"/>
{_state_tag(445, 175, 2, s['2'].T, s['2'].P, 'right')}

<!-- State 3: Recuperator hot out -> IHX (red, down) -->
<polyline points="317,323 317,385 249,385 249,435"
  class="flow-hot" marker-end="url(#arr-red)"/>
{_state_tag(322, 360, 3, s['3'].T, s['3'].P, 'right')}

<!-- State 4: IHX -> Pump (blue, left) -->
<polyline points="182,460 150,460 150,505 80,505 80,478"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(90, 510, 4, s['4'].T, s['4'].P, 'right')}

<!-- State 5: Pump -> Recuperator cold in (blue, up) -->
<polyline points="13,440 13,315 182,315"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(18, 380, 5, s['5'].T, s['5'].P, 'right')}

<!-- State 6: Recuperator cold out -> Preheater (blue, left) -->
<polyline points="182,275 155,275 155,227 128,227"
  class="flow-cold" marker-end="url(#arr-blue)"/>
{_state_tag(155, 260, 6, s['6'].T, s['6'].P, 'left')}

<!-- State 7: Preheater -> Vaporizer (warm, up) -->
<polyline points="70,205 70,130 70,99 72,77"
  class="flow-cold" marker-end="url(#arr-red)"/>
{_state_tag(70, 165, 7, s['7'].T, s['7'].P, 'right')}

<!-- ============ PROPANE LOOP (green) ============ -->

<!-- State A: IHX -> Propane ACC (green, right) -->
<polyline points="317,460 420,460 420,255 520,255"
  class="flow-green" marker-end="url(#arr-green)"/>
{_state_tag(425, 360, 'A', ps['A'].T, ps['A'].P, 'right')}

<!-- State B: Propane ACC -> Prop Pump (green, down) -->
<polyline points="590,280 590,370 593,435"
  class="flow-green" marker-end="url(#arr-green)"/>
{_state_tag(595, 340, 'B', ps['B'].T, ps['B'].P, 'right')}

<!-- State C: Prop Pump -> IHX (green, left) -->
<polyline points="533,454 430,454 430,485 317,485"
  class="flow-green" marker-end="url(#arr-green)"/>
{_state_tag(430, 498, 'C', ps['C'].T, ps['C'].P, 'right')}

<!-- ============ BRINE PATH (orange dashed) ============ -->
<polyline points="142,38 142,55"
  class="flow-brine" marker-end="url(#arr-orange)"/>
<text x="148" y="33" class="brine-label">Brine In {inputs['T_geo_in']:.0f}F</text>

<polyline points="142,99 70,99 70,205"
  class="flow-brine" marker-end="url(#arr-orange)"/>
<text x="90" y="112" class="brine-label">T_mid {perf['T_brine_mid']:.0f}F</text>

<polyline points="70,249 70,275"
  class="flow-brine"/>
<text x="10" y="290" class="brine-label">Brine Out {perf['T_geo_out_calc']:.0f}F</text>

<!-- Annotation boxes (bottom) -->
<rect x="40" y="575" width="170" height="35" rx="4"
  fill="none" stroke="#666" stroke-width="0.8" stroke-dasharray="3,2"/>
<text x="125" y="597" text-anchor="middle"
  style="font-size:8px;fill:#555;font-weight:bold;font-family:sans-serif;">
  SHORT ISO RUNS (all indoor)</text>

<rect x="280" y="565" width="380" height="55" rx="4"
  fill="none" stroke="#27ae60" stroke-width="1" stroke-dasharray="4,2"/>
<text x="470" y="583" text-anchor="middle" class="annot-green">HIGH-P PROPANE LOOP</text>
<text x="470" y="595" text-anchor="middle" class="annot-green">{prop_dia:.0f}" dia vs {tp_a_dia:.0f}" (Config A) per train</text>
<text x="470" y="607" text-anchor="middle"
  style="font-size:7.5px;fill:#27ae60;font-family:sans-serif;">
  (small-bore, high-pressure piping)</text>

</svg></div>"""
    return svg


# ============================================================================


# ============================================================================
# SIDEBAR BUILDER
# ============================================================================

def build_analysis_sidebar(shared_inputs=None):
    """Build the analysis sidebar form. Returns inputs dict.

    If shared_inputs provides overlapping keys, those widgets are
    skipped and the shared values are injected into the returned dict.
    """
    _init_analysis_session_state()

    # Apply pending unit cost overrides from Claude recommendations
    if st.session_state.analysis_pending_apply:
        for widget_key, value in st.session_state.analysis_pending_apply.items():
            st.session_state[widget_key] = value
        st.session_state.analysis_pending_apply = {}

    shared = shared_inputs or {}

    with st.sidebar:
        if st.button("💬 Ask Claude", use_container_width=True, type="secondary"):
            _claude_chat_dialog()
        st.header("Inputs")

        with st.form("input_form"):
            with st.expander("Brine Inputs", expanded=True):
                if "T_geo_in" not in shared:
                    T_geo_in = st.number_input("Brine inlet temperature (degF)",
                                               min_value=1, value=420, step=5)
                else:
                    T_geo_in = shared["T_geo_in"]
                if "m_dot_geo" not in shared:
                    m_dot_geo = st.number_input("Brine mass flow rate (lb/s)",
                                                min_value=1, value=1100, step=10)
                else:
                    m_dot_geo = shared["m_dot_geo"]
                cp_brine = st.number_input("Brine specific heat (BTU/lb-degF)",
                                           min_value=0.01, value=1.0, step=0.05, format="%.2f")
                if "T_geo_out_min" not in shared:
                    T_geo_out_min = st.number_input("Min brine outlet temperature (degF)",
                                                    min_value=1, value=160, step=5,
                                                    help="Silica/scaling constraint")
                else:
                    T_geo_out_min = shared["T_geo_out_min"]

            with st.expander("Cycle Parameters"):
                if "eta_turbine" not in shared:
                    eta_turbine = st.number_input("Turbine isentropic efficiency",
                                                  min_value=0.01, value=0.91, step=0.01, format="%.2f")
                else:
                    eta_turbine = shared["eta_turbine"]
                if "eta_pump" not in shared:
                    eta_pump = st.number_input("Pump isentropic efficiency",
                                               min_value=0.01, value=0.84, step=0.01, format="%.2f")
                else:
                    eta_pump = shared["eta_pump"]
                superheat = st.number_input("Turbine inlet superheat (degF above sat)",
                                            min_value=0, value=0, step=1)
                st.info("Isopentane circulation rate is solved from brine energy balance.")

            with st.expander("Ambient Conditions"):
                if "T_ambient" not in shared:
                    T_ambient = st.number_input("Ambient dry bulb temperature (degF)",
                                                min_value=-40, value=57, step=5)
                else:
                    T_ambient = shared["T_ambient"]

            with st.expander("Pinch Points"):
                dt_pinch_acc_a = st.number_input("ACC pinch Config A (degF)",
                                                 min_value=1, value=15, step=1)
                dt_pinch_acc_b = st.number_input("ACC pinch Config B (degF)",
                                                 min_value=1, value=15, step=1)
                dt_pinch_vaporizer = st.number_input("Vaporizer pinch (degF)",
                                                     min_value=1, value=10, step=1)
                dt_pinch_preheater = st.number_input("Preheater pinch (degF)",
                                                     min_value=1, value=10, step=1)
                dt_pinch_recup = st.number_input("Recuperator pinch (degF)",
                                                 min_value=1, value=15, step=1)
                dt_approach_intermediate = st.number_input(
                    "Intermediate HX approach (degF, Config B)",
                    min_value=1, value=10, step=1)

            with st.expander("Duct Parameters"):
                v_tailpipe = st.number_input("Tailpipe vapor velocity (ft/s)",
                                             min_value=1, value=70, step=1)
                v_acc_header = st.number_input("ACC header vapor velocity (ft/s)",
                                               min_value=1, value=15, step=1)
                L_tailpipe_a = st.number_input("Tailpipe length Config A (ft)",
                                               min_value=1, value=30, step=5)
                L_long_header = st.number_input("Long vapor header length (ft)",
                                                min_value=1, value=120, step=10)
                L_acc_header = st.number_input("ACC distribution header length (ft)",
                                               min_value=1, value=200, step=10)

            with st.expander("Hydraulic Parameters"):
                f_darcy = st.number_input("Darcy friction factor f",
                                          min_value=0.001, value=0.020, step=0.001, format="%.3f")
                st.markdown("**Config A Equipment dP (psi)**")
                dp_acc_tubes_a = st.number_input("ACC tube bundle dP",
                                                 min_value=0.0, value=0.5, step=0.1, key="dp_acc_tubes_a")
                dp_acc_headers_a = st.number_input("ACC vapor headers dP",
                                                   min_value=0.0, value=0.3, step=0.1, key="dp_acc_headers_a")
                dp_recup_a = st.number_input("Recuperator dP (A)",
                                             min_value=0.0, value=0.3, step=0.1, key="dp_recup_a")
                st.markdown("**Config B Isopentane Side dP (psi)**")
                dp_ihx_iso = st.number_input("IHX iso side dP",
                                             min_value=0.0, value=0.5, step=0.1, key="dp_ihx_iso")
                dp_recup_b = st.number_input("Recuperator dP (B)",
                                             min_value=0.0, value=0.3, step=0.1, key="dp_recup_b")
                dp_tailpipe_iso_b = st.number_input("ISO tailpipe dP (B)",
                                                    min_value=0.0, value=0.3, step=0.1, key="dp_tailpipe_iso_b")
                st.markdown("**Config B Propane Side dP (psi)**")
                dp_acc_tubes_prop = st.number_input("ACC tube bundle dP (prop)",
                                                    min_value=0.0, value=1.0, step=0.1, key="dp_acc_tubes_prop")
                dp_ihx_prop = st.number_input("IHX propane side dP",
                                              min_value=0.0, value=0.5, step=0.1, key="dp_ihx_prop")

            with st.expander("ACC Fan Parameters"):
                dT_air = st.number_input("Air temp rise (°F)",
                                         min_value=1, value=25, step=1,
                                         help="Temperature rise of air across ACC bundles")
                fan_static_inwc = st.number_input("Fan static pressure (in WC)",
                                                  min_value=0.01, value=0.75, step=0.05,
                                                  key="fan_static_inwc", format="%.2f")
                eta_fan = st.number_input("Fan efficiency",
                                          min_value=0.01, value=0.78, step=0.01,
                                          key="eta_fan", format="%.2f")
                eta_motor = st.number_input("Motor efficiency",
                                            min_value=0.01, value=0.95, step=0.01,
                                            key="eta_motor", format="%.2f")
                _auto_a = st.session_state.analysis_auto_fan_bays_a
                _auto_b = st.session_state.analysis_auto_fan_bays_b
                _auto_max = max(_auto_a, _auto_b)
                n_fan_bays_raw = st.number_input(
                    "Number of fan bays (blank = auto)",
                    min_value=1, value=None, step=1,
                    placeholder=str(_auto_max),
                    help="Leave blank to auto-size from airflow")
                n_fan_bays = n_fan_bays_raw if n_fan_bays_raw is not None else 0
                _auto_note = f"Auto: {_auto_a} bays (A) / {_auto_b} bays (B) calculated from airflow"
                if n_fan_bays_raw is not None:
                    _auto_note += f"  --  override: {n_fan_bays_raw} bays"
                st.caption(_auto_note)
                fan_diameter_ft = st.number_input("Fan diameter (ft)",
                                                  min_value=1, value=28, step=1)
                W_aux_kw = st.number_input("Auxiliary parasitic (kW)",
                                           min_value=0, value=150, step=10,
                                           help="Lube oil, controls, lighting, instruments")
                prop_thermosiphon = st.checkbox(
                    "Propane loop thermosiphon (no pump)",
                    value=True,
                    help="ACC elevated above IHX — gravity drives liquid return, no pump required")

            with st.expander("Economic Parameters"):
                if "electricity_price" not in shared:
                    electricity_price = st.number_input("Electricity price ($/MWh)",
                                                        min_value=1, value=35, step=5)
                else:
                    electricity_price = shared["electricity_price"]
                if "capacity_factor" not in shared:
                    capacity_factor = st.number_input("Capacity factor (%)",
                                                      min_value=1, value=95, step=1)
                else:
                    capacity_factor = shared["capacity_factor"]
                if "discount_rate" not in shared:
                    discount_rate = st.number_input("Discount rate (%)",
                                                    min_value=0, value=8, step=1)
                else:
                    discount_rate = shared["discount_rate"]
                if "project_life" not in shared:
                    project_life = st.number_input("Plant life (years)",
                                                   min_value=1, value=30, step=1)
                else:
                    project_life = shared["project_life"]

                st.markdown("**Construction Overhead**")
                weekly_site_overhead = st.number_input(
                    "Weekly site overhead ($/wk)",
                    min_value=0, value=20000, step=1000,
                    key="weekly_site_overhead",
                    help="PM, CM, QC, safety, scheduler")
                weekly_equip_rental = st.number_input(
                    "Weekly equipment rental ($/wk)",
                    min_value=0, value=15000, step=1000,
                    key="weekly_equip_rental",
                    help="Cranes, scaffolding, temporary facilities")
                craft_labor_pct = st.number_input(
                    "Craft labor budget (% of installed cost)",
                    min_value=0, max_value=50, value=15, step=1,
                    key="craft_labor_pct",
                    help="Used to estimate waiting time cost")
                craft_labor_waiting_pct = st.number_input(
                    "Craft labor waiting time (%)",
                    min_value=0, max_value=50, value=15, step=1,
                    key="craft_labor_waiting_pct",
                    help="Labor hours lost to schedule dependencies")
                construction_loan_pct = st.number_input(
                    "Construction loan (% of installed cost)",
                    min_value=0, max_value=100, value=60, step=5,
                    key="construction_loan_pct")
                construction_loan_rate = st.number_input(
                    "Construction loan interest (%)",
                    min_value=0.0, value=7.0, step=0.5,
                    key="construction_loan_rate",
                    format="%.1f")

            with st.expander("Unit Cost Assumptions (2024 USD Installed)"):
                st.markdown("**Heat Exchanger Costs**")
                uc_vaporizer_per_ft2 = st.number_input(
                    "Vaporizer ($/ft2)", min_value=1, value=35, step=1, key="uc_vaporizer")
                uc_preheater_per_ft2 = st.number_input(
                    "Preheater ($/ft2)", min_value=1, value=30, step=1, key="uc_preheater")
                uc_recuperator_per_ft2 = st.number_input(
                    "Recuperator ($/ft2)", min_value=1, value=25, step=1, key="uc_recuperator")
                uc_ihx_per_ft2 = st.number_input(
                    "IHX ($/ft2, pressure rated)", min_value=1, value=40, step=1, key="uc_ihx")
                uc_acc_per_ft2 = st.number_input(
                    "ACC ($/ft2 face area)", min_value=1, value=12, step=1, key="uc_acc")

                st.markdown("**Duct and Piping Costs**")
                uc_iso_duct_per_ft2 = st.number_input(
                    "Iso duct ($/ft2 surface)", min_value=1, value=180, step=5,
                    key="uc_iso_duct",
                    help="Diameter multiplier: >72\" x1.7, >60\" x1.4")
                uc_prop_pipe_per_ft2 = st.number_input(
                    "Propane pipe ($/ft2 surface)", min_value=1, value=120, step=5,
                    key="uc_prop_pipe")
                uc_prop_piping_pct = st.number_input(
                    "Propane system (% of IHX cost)", min_value=0, value=20, step=1,
                    key="uc_prop_pct")

                st.markdown("**Turbine and Electrical**")
                uc_turbine_per_kw = st.number_input(
                    "Turbine-generator ($/kW)", min_value=1, value=1200, step=50,
                    key="uc_turbine")

                st.markdown("**Civil and Structural**")
                uc_steel_per_lb = st.number_input(
                    "Structural steel ($/lb)", min_value=0.01, value=4.5, step=0.25,
                    key="uc_steel", format="%.2f")
                uc_foundation_pct = st.number_input(
                    "Foundation (% of equipment)", min_value=0, value=8, step=1,
                    key="uc_foundation")

                st.markdown("**Indirect Costs**")
                uc_engineering_pct = st.number_input(
                    "Engineering & procurement (%)", min_value=0, value=12, step=1,
                    key="uc_eng")
                uc_construction_mgmt_pct = st.number_input(
                    "Construction management (%)", min_value=0, value=8, step=1,
                    key="uc_cm")
                uc_contingency_pct = st.number_input(
                    "Contingency (%)", min_value=0, value=15, step=1,
                    key="uc_contingency")

                st.caption("Cost basis: AACE Class 4-5 factored estimate. Accuracy range -30% to +50%.")

            submitted = st.form_submit_button("Run Calculation", type="primary", width="stretch")

    inputs = {
        "T_geo_in": T_geo_in,
        "m_dot_geo": m_dot_geo,
        "cp_brine": cp_brine,
        "T_geo_out_min": T_geo_out_min,
        "T_ambient": T_ambient,
        "dt_pinch_vaporizer": dt_pinch_vaporizer,
        "dt_pinch_preheater": dt_pinch_preheater,
        "dt_pinch_acc_a": dt_pinch_acc_a,
        "dt_pinch_acc_b": dt_pinch_acc_b,
        "dt_pinch_recup": dt_pinch_recup,
        "dt_approach_intermediate": dt_approach_intermediate,
        "superheat": superheat,
        "eta_turbine": eta_turbine,
        "eta_pump": eta_pump,
        "v_tailpipe": v_tailpipe,
        "v_acc_header": v_acc_header,
        "L_tailpipe_a": L_tailpipe_a,
        "L_long_header": L_long_header,
        "L_acc_header": L_acc_header,
        "electricity_price": electricity_price,
        "discount_rate": discount_rate / 100,
        "project_life": project_life,
        "capacity_factor": capacity_factor / 100,
        "f_darcy": f_darcy,
        "dp_acc_tubes_a": dp_acc_tubes_a,
        "dp_acc_headers_a": dp_acc_headers_a,
        "dp_recup_a": dp_recup_a,
        "dp_ihx_iso": dp_ihx_iso,
        "dp_recup_b": dp_recup_b,
        "dp_tailpipe_iso_b": dp_tailpipe_iso_b,
        "dp_acc_tubes_prop": dp_acc_tubes_prop,
        "dp_ihx_prop": dp_ihx_prop,
        # ACC fan parameters
        "dT_air": dT_air,
        "fan_static_inwc": fan_static_inwc,
        "eta_fan": eta_fan,
        "eta_motor": eta_motor,
        "n_fan_bays": n_fan_bays,
        "fan_diameter_ft": fan_diameter_ft,
        "W_aux_kw": W_aux_kw,
        "prop_thermosiphon": prop_thermosiphon,
        # Unit cost overrides
        "uc_vaporizer_per_ft2": uc_vaporizer_per_ft2,
        "uc_preheater_per_ft2": uc_preheater_per_ft2,
        "uc_recuperator_per_ft2": uc_recuperator_per_ft2,
        "uc_ihx_per_ft2": uc_ihx_per_ft2,
        "uc_acc_per_ft2": uc_acc_per_ft2,
        "uc_iso_duct_per_ft2": uc_iso_duct_per_ft2,
        "uc_prop_pipe_per_ft2": uc_prop_pipe_per_ft2,
        "uc_prop_piping_pct": uc_prop_piping_pct,
        "uc_turbine_per_kw": uc_turbine_per_kw,
        "uc_steel_per_lb": uc_steel_per_lb,
        "uc_foundation_pct": uc_foundation_pct,
        "uc_engineering_pct": uc_engineering_pct,
        "uc_construction_mgmt_pct": uc_construction_mgmt_pct,
        "uc_contingency_pct": uc_contingency_pct,
        # Construction overhead parameters
        "weekly_site_overhead": weekly_site_overhead,
        "weekly_equip_rental": weekly_equip_rental,
        "craft_labor_pct": craft_labor_pct,
        "craft_labor_waiting_pct": craft_labor_waiting_pct,
        "construction_loan_pct": construction_loan_pct,
        "construction_loan_rate": construction_loan_rate,
    }

    return inputs


# ============================================================================
# ANALYSIS TAB RENDERER
# ============================================================================

def render_analysis_tab(inputs):
    """Run the analysis pipeline and render all content."""
    st.title("ORC Configuration Comparator")
    st.caption("Config A (Direct ACC) vs Config B (Propane Intermediate Loop) -- Geothermal Binary Cycle")

    # Validate
    warns = validate_inputs(inputs)
    for w in warns:
        st.sidebar.warning(w)

    # ============================================================================
    # SOLVE CYCLES
    # ============================================================================

    try:
        result_a = solve_config_a(inputs, fp)
    except Exception as e:
        st.error(f"**Config A solver error:** {e}\n\n"
                 "This is likely caused by an input value that is too large or too small. "
                 f"Check brine temps ({inputs['T_geo_in']}→{inputs['T_geo_out_min']}°F), ambient ({inputs['T_ambient']}°F), "
                 f"ACC pinch ({inputs['dt_pinch_acc_a']}°F), efficiencies, or duct velocities.")
        st.stop()

    try:
        result_b = solve_config_b(inputs, fp)
    except Exception as e:
        st.error(f"**Config B solver error:** {e}\n\n"
                 "This is likely caused by an input value that is too large or too small. "
                 f"Check brine temps ({inputs['T_geo_in']}→{inputs['T_geo_out_min']}°F), ambient ({inputs['T_ambient']}°F), "
                 f"ACC pinch ({inputs['dt_pinch_acc_b']}°F), IHX approach ({inputs['dt_approach_intermediate']}°F), "
                 f"or duct velocities.")
        st.stop()

    perf_a = result_a["performance"]
    perf_b = result_b["performance"]

    states_a = result_a["states"]
    states_b = result_b["states"]
    prop_states = result_b["propane_states"]
    duct_a = result_a["duct"]
    duct_b = result_b["duct"]
    hydraulic_a = result_a["hydraulic"]
    hydraulic_b = result_b["hydraulic"]

    # Power sensitivity: perturb ACC pinch by +1 degF, re-solve each config
    try:
        inp_pert_a = {**inputs, "dt_pinch_acc_a": inputs["dt_pinch_acc_a"] + 1}
        result_a_pert = solve_config_a(inp_pert_a, fp)
        fan_a_pert = calculate_fan_power(result_a_pert["performance"]["Q_reject_mmbtu_hr"], inputs["T_ambient"], inputs)
        pwr_a_pert = compute_power_balance(result_a_pert["performance"], fan_a_pert, inputs, config="A")
        dW_dT_a = pwr_a["P_net"] - pwr_a_pert["P_net"]  # kW per degF
    except Exception:
        dW_dT_a = 0.0

    try:
        inp_pert_b = {**inputs, "dt_pinch_acc_b": inputs["dt_pinch_acc_b"] + 1}
        result_b_pert = solve_config_b(inp_pert_b, fp)
        _ps_b_pert = result_b_pert["propane_states"]
        _Q_rej_b_pert = result_b_pert["performance"]["m_dot_prop"] * (_ps_b_pert["A"].h - _ps_b_pert["B"].h) / 1e6
        fan_b_pert = calculate_fan_power(_Q_rej_b_pert, inputs["T_ambient"], inputs)
        pwr_b_pert = compute_power_balance(result_b_pert["performance"], fan_b_pert, inputs, config="B")
        dW_dT_b = pwr_b["P_net"] - pwr_b_pert["P_net"]  # kW per degF
    except Exception:
        dW_dT_b = 0.0

    # Costs (equipment sizing — does not depend on net power)
    try:
        costs_a = calculate_costs_a(states_a, perf_a, inputs, duct_a)
        costs_b = calculate_costs_b(states_b, prop_states, perf_b, inputs, duct_b)
    except Exception as e:
        st.error(f"**Cost calculation error:** {e}\n\n"
                 "This is likely caused by an input value that is too large or too small. "
                 "Check unit cost assumptions, pinch points, or economic parameters.")
        st.stop()

    # ---- Fan power & pump sizing ----
    try:
        # Config A: isopentane condenses directly to air
        fan_a = calculate_fan_power(perf_a["Q_reject_mmbtu_hr"], inputs["T_ambient"], inputs)

        # Config B: propane rejects to air
        Q_reject_air_b = perf_b["m_dot_prop"] * (prop_states["A"].h - prop_states["B"].h) / 1e6
        fan_b = calculate_fan_power(Q_reject_air_b, inputs["T_ambient"], inputs)

        # Pump sizing (for equipment selection display only)
        pump_iso_a = pump_sizing(
            perf_a["m_dot_iso"], states_a["4"].rho,
            perf_a["P_high"], perf_a["P_low"],
            perf_a["w_pump"], inputs["eta_pump"],
        )
        pump_iso_b = pump_sizing(
            perf_b["m_dot_iso"], states_b["4"].rho,
            perf_b["P_high_iso"], perf_b["P_low_iso"],
            perf_b["w_pump_iso"], inputs["eta_pump"],
        )
        pump_prop_b = pump_sizing(
            perf_b["m_dot_prop"], prop_states["B"].rho,
            perf_b["P_prop_evap"], perf_b["P_prop_cond"],
            perf_b["w_pump_prop"], inputs["eta_pump"],
        )

        # Update session state so the sidebar placeholder shows the solved fan count
        st.session_state.analysis_auto_fan_bays_a = fan_a["n_fans_required"]
        st.session_state.analysis_auto_fan_bays_b = fan_b["n_fans_required"]
    except Exception as e:
        st.error(f"**Fan/pump sizing error:** {e}\n\n"
                 "This is likely caused by an input value that is too large or too small. "
                 f"Check fan parameters (dT_air={inputs['dT_air']}°F, fan efficiency={inputs['eta_fan']}, "
                 f"motor efficiency={inputs['eta_motor']}, fan diameter={inputs['fan_diameter_ft']} ft) "
                 f"or ambient temperature ({inputs['T_ambient']}°F).")
        st.stop()

    # ============================================================================
    # CENTRALIZED POWER BALANCE — single source of truth for net power, efficiency
    # ============================================================================
    pwr_a = compute_power_balance(perf_a, fan_a, inputs, config="A")
    pwr_b = compute_power_balance(perf_b, fan_b, inputs, config="B")

    # Verification assertions
    _bal_a = abs(pwr_a["P_net"] - (pwr_a["P_gross"] - pwr_a["W_total_parasitic"]))
    _bal_b = abs(pwr_b["P_net"] - (pwr_b["P_gross"] - pwr_b["W_total_parasitic"]))
    if _bal_a > 1.0 or _bal_b > 1.0:
        st.error(f"**Power balance error:** Net power does not balance.\n"
                 f"Config A: {pwr_a['P_net']:.1f} != {pwr_a['P_gross']:.1f} - {pwr_a['W_total_parasitic']:.1f} (err={_bal_a:.2f})\n"
                 f"Config B: {pwr_b['P_net']:.1f} != {pwr_b['P_gross']:.1f} - {pwr_b['W_total_parasitic']:.1f} (err={_bal_b:.2f})")
        st.stop()

    # Backward-compatible aliases used by display code
    parasitic_a = {
        "iso_pump_kw": pwr_a["W_iso_pump"],
        "prop_pump_kw": 0,
        "acc_fans_kw": pwr_a["W_fans"],
        "auxiliary_kw": pwr_a["W_auxiliary"],
        "total_kw": pwr_a["W_total_parasitic"],
        "thermosiphon": False,
    }
    parasitic_b = {
        "iso_pump_kw": pwr_b["W_iso_pump"],
        "prop_pump_kw": pwr_b["W_prop_pump"],
        "prop_pump_kw_calc": pwr_b["W_prop_pump_calc"],
        "acc_fans_kw": pwr_b["W_fans"],
        "auxiliary_kw": pwr_b["W_auxiliary"],
        "total_kw": pwr_b["W_total_parasitic"],
        "thermosiphon": pwr_b["thermosiphon"],
    }

    # Lifecycle economics — uses centralized P_net (includes ALL parasitics)
    try:
        lc_a = lifecycle_cost(costs_a["total_installed"], pwr_a["P_net"], inputs)
        lc_b = lifecycle_cost(costs_b["total_installed"], pwr_b["P_net"], inputs)
    except Exception as e:
        st.error(f"**Lifecycle cost error:** {e}")
        st.stop()

    # Seasonal fan power variation (Q_reject held constant, only air density changes)
    seasonal_temps = {
        "Winter": inputs["T_ambient"] - 30,
        "Design": inputs["T_ambient"],
        "Summer": inputs["T_ambient"] + 10,
    }
    seasonal_a = {}
    seasonal_b = {}
    for season, T_amb_s in seasonal_temps.items():
        sf_a = calculate_fan_power(perf_a["Q_reject_mmbtu_hr"], T_amb_s, inputs)
        sf_b = calculate_fan_power(Q_reject_air_b, T_amb_s, inputs)
        seasonal_a[season] = {
            "T_ambient": T_amb_s,
            "fan_kw": sf_a["W_fans_kw"],
            "P_net": pwr_a["P_gross"] - pwr_a["W_iso_pump"] - sf_a["W_fans_kw"] - pwr_a["W_auxiliary"],
        }
        seasonal_b[season] = {
            "T_ambient": T_amb_s,
            "fan_kw": sf_b["W_fans_kw"],
            "P_net": pwr_b["P_gross"] - pwr_b["W_iso_pump"] - pwr_b["W_prop_pump"] - sf_b["W_fans_kw"] - pwr_b["W_auxiliary"],
        }

    # Pinch checks
    pinch_a = verify_recuperator_pinch(states_a, fp)
    pinch_b = verify_recuperator_pinch(states_b, fp)

    # Validation checks
    checks_a = run_validation_checks(perf_a, states_a, duct_a, "A", inputs, fp)
    checks_b = run_validation_checks(perf_b, states_b, duct_b, "B", inputs, fp)

    # Schedule
    sched_info = construction_schedule(duct_a)
    sched_savings = sched_info["schedule_savings_weeks"]
    a_total_weeks = sched_info["config_a"]["total_weeks"]
    b_total_weeks = sched_info["config_b"]["total_weeks"]

    # Payback
    payback_yrs = simple_payback(
        costs_a["total_installed"], costs_b["total_installed"],
        pwr_a["P_net"], pwr_b["P_net"], inputs
    )

    # NPV of schedule savings
    sched_npv = schedule_savings_npv(sched_info, pwr_b["P_net"], inputs)

    # Construction cost savings from schedule compression
    constr_savings = construction_cost_savings(
        sched_info, costs_a["total_installed"], costs_b["total_installed"], inputs)

    # Volumetric flow ratio
    vol_a = duct_a["total_vol_flow_ft3s"]
    vol_b = duct_b.get("propane_vol_flow_ft3s", duct_b["total_vol_flow_ft3s"])
    vol_ratio = vol_a / vol_b if vol_b > 0 else float("inf")

    # Total economic advantage: capital delta + NPV power difference + early revenue + construction savings
    capital_delta = costs_a["total_installed"] - costs_b["total_installed"]  # positive = B cheaper
    npv_delta = lc_b["net_npv"] - lc_a["net_npv"]  # includes capital + revenue difference
    total_economic_advantage = npv_delta + sched_npv + constr_savings["total_construction_savings"]


    # ============================================================================
    # SIDEBAR -- COST IMPACT ANALYSIS
    # ============================================================================

    with st.sidebar:
        with st.expander("Cost Impact by Assumption"):
            # Build table of unit cost assumptions and their $ impact per config
            _uc_impact_rows = []

            # Helper: map unit cost key to the relevant area/quantity and label
            _uc_items = [
                ("Vaporizer", "uc_vaporizer_per_ft2", "vaporizer_per_ft2",
                 costs_a.get("vaporizer_area_ft2", 0), costs_b.get("vaporizer_area_ft2", 0), "ft2"),
                ("Preheater", "uc_preheater_per_ft2", "preheater_per_ft2",
                 costs_a.get("preheater_area_ft2", 0), costs_b.get("preheater_area_ft2", 0), "ft2"),
                ("Recuperator", "uc_recuperator_per_ft2", "recup_per_ft2",
                 costs_a.get("recuperator_area_ft2", 0), costs_b.get("recuperator_area_ft2", 0), "ft2"),
                ("IHX", "uc_ihx_per_ft2", "hx_per_ft2",
                 costs_a.get("intermediate_hx_area_ft2", 0), costs_b.get("intermediate_hx_area_ft2", 0), "ft2"),
                ("ACC", "uc_acc_per_ft2", "acc_per_ft2",
                 costs_a.get("acc_area_ft2", 0), costs_b.get("acc_area_ft2", 0), "ft2"),
                ("Turbine", "uc_turbine_per_kw", "turbine_per_kw",
                 perf_a["gross_power_kw"], perf_b["gross_power_kw"], "kW"),
            ]
            for label, uc_key, cf_key, qty_a, qty_b, unit in _uc_items:
                uc_val = inputs.get(uc_key, COST_FACTORS[cf_key])
                cost_a_val = uc_val * qty_a
                cost_b_val = uc_val * qty_b
                _uc_impact_rows.append({
                    "Component": label,
                    "Rate": f"${uc_val:,.0f}/{unit}",
                    "Cost A ($k)": f"{cost_a_val/1e3:,.0f}",
                    "Cost B ($k)": f"{cost_b_val/1e3:,.0f}",
                })
            st.dataframe(pd.DataFrame(_uc_impact_rows).set_index("Component"), width="stretch")

        with st.expander("Cost Sensitivity Tornado"):
            # Algebraic tornado: perturb each unit cost +/-20%, measure change in B-A delta
            _base_delta = costs_b["total_installed"] - costs_a["total_installed"]

            _tornado_items = [
                ("Vaporizer $/ft2", "uc_vaporizer_per_ft2", "vaporizer_per_ft2"),
                ("Preheater $/ft2", "uc_preheater_per_ft2", "preheater_per_ft2"),
                ("Recuperator $/ft2", "uc_recuperator_per_ft2", "recup_per_ft2"),
                ("IHX $/ft2", "uc_ihx_per_ft2", "hx_per_ft2"),
                ("ACC $/ft2", "uc_acc_per_ft2", "acc_per_ft2"),
                ("Iso Duct $/ft2", "uc_iso_duct_per_ft2", "iso_duct_per_ft2"),
                ("Propane Pipe $/ft2", "uc_prop_pipe_per_ft2", "prop_pipe_per_ft2"),
                ("Turbine $/kW", "uc_turbine_per_kw", "turbine_per_kw"),
                ("Steel $/lb", "uc_steel_per_lb", "steel_per_lb"),
                ("Foundation %", "uc_foundation_pct", "foundation_pct"),
                ("Engineering %", "uc_engineering_pct", "engineering_pct"),
                ("Constr. Mgmt %", "uc_construction_mgmt_pct", "construction_mgmt_pct"),
                ("Contingency %", "uc_contingency_pct", "contingency_pct"),
                ("Propane Sys %", "uc_prop_piping_pct", "prop_piping_pct"),
            ]

            _tornado_data = []
            for label, uc_key, cf_key in _tornado_items:
                base_val = inputs.get(uc_key, COST_FACTORS[cf_key])
                deltas = []
                for mult in [0.8, 1.2]:
                    perturbed_inputs = {**inputs, uc_key: base_val * mult}
                    ca = calculate_costs_a(states_a, perf_a, perturbed_inputs, duct_a)
                    cb = calculate_costs_b(states_b, prop_states, perf_b, perturbed_inputs, duct_b)
                    new_delta = cb["total_installed"] - ca["total_installed"]
                    deltas.append((new_delta - _base_delta) / 1e6)
                _tornado_data.append({"param": label, "low": deltas[0], "high": deltas[1]})

            _tornado_data.sort(key=lambda d: abs(d["high"] - d["low"]))
            _t_params = [d["param"] for d in _tornado_data]
            _t_lows = [d["low"] for d in _tornado_data]
            _t_highs = [d["high"] for d in _tornado_data]

            _fig_tornado_uc = go.Figure()
            _fig_tornado_uc.add_trace(go.Bar(
                y=_t_params, x=_t_lows, orientation="h",
                name="-20%", marker_color="steelblue"))
            _fig_tornado_uc.add_trace(go.Bar(
                y=_t_params, x=_t_highs, orientation="h",
                name="+20%", marker_color="indianred"))
            _fig_tornado_uc.update_layout(
                title=f"Unit Cost Sensitivity (B-A delta base: ${_base_delta/1e6:.2f}MM)",
                xaxis_title="Change in B-A Delta ($MM)",
                barmode="overlay",
                height=450,
                margin=dict(l=120),
            )
            st.plotly_chart(_fig_tornado_uc, width="stretch")


    # ============================================================================

    # Store computed state for chat dialog
    st.session_state.analysis_computed = {
        "inputs": inputs, "pwr_a": pwr_a, "pwr_b": pwr_b,
        "perf_a": perf_a, "perf_b": perf_b,
        "duct_a": duct_a, "duct_b": duct_b,
        "costs_a": costs_a, "costs_b": costs_b,
        "lc_a": lc_a, "lc_b": lc_b,
        "total_economic_advantage": total_economic_advantage,
        "a_total_weeks": a_total_weeks, "b_total_weeks": b_total_weeks,
        "sched_savings": sched_savings, "sched_info": sched_info,
        "constr_savings": constr_savings,
        "vol_ratio": vol_ratio,
        "checks_a": checks_a, "checks_b": checks_b,
        "warns": warns,
    }

    # ============================================================================
    # SECTION 1: EXECUTIVE SUMMARY
    # ============================================================================

    st.header("Executive Summary")

    # Build comparison rows: (label, val_a, val_b, fmt, lower_better_or_None)
    summary_rows = []

    # -- Power Performance --
    summary_rows.append(("group", "Power Performance", "", "", "", "", ""))
    summary_rows.append(("row", "Net power output",
                          _fmt(pwr_a["P_net"], ".0f"), "kW",
                          _fmt(pwr_b["P_net"], ".0f"), "kW",
                          _winner(pwr_a["P_net"], pwr_b["P_net"], lower_better=False)))
    summary_rows.append(("row", "Gross turbine power",
                          _fmt(perf_a["gross_power_kw"], ".0f"), "kW",
                          _fmt(perf_b["gross_power_kw"], ".0f"), "kW",
                          _winner(perf_a["gross_power_kw"], perf_b["gross_power_kw"], lower_better=False)))
    summary_rows.append(("row", "Net thermal efficiency",
                          _fmt(pwr_a["eta_thermal"]*100, ".1f"), "%",
                          _fmt(pwr_b["eta_thermal"]*100, ".1f"), "%",
                          _winner(pwr_a["eta_thermal"], pwr_b["eta_thermal"], lower_better=False)))
    summary_rows.append(("row", "Brine effectiveness",
                          _fmt(pwr_a["brine_effectiveness"], ".2f"), "kW/(lb/s)",
                          _fmt(pwr_b["brine_effectiveness"], ".2f"), "kW/(lb/s)",
                          _winner(pwr_a["brine_effectiveness"], pwr_b["brine_effectiveness"], lower_better=False)))
    summary_rows.append(("row", "Heat rejection duty",
                          _fmt(perf_a["Q_reject_mmbtu_hr"], ".2f"), "MMBtu/hr",
                          _fmt(perf_b["Q_reject_mmbtu_hr"], ".2f"), "MMBtu/hr",
                          _winner(perf_a["Q_reject_mmbtu_hr"], perf_b["Q_reject_mmbtu_hr"], lower_better=True)))

    # -- Parasitic Loads --
    summary_rows.append(("group", "Parasitic Loads", "", "", "", "", ""))
    summary_rows.append(("row", "ISO pump",
                          _fmt(parasitic_a["iso_pump_kw"], ".0f"), "kW",
                          _fmt(parasitic_b["iso_pump_kw"], ".0f"), "kW",
                          _winner(parasitic_a["iso_pump_kw"], parasitic_b["iso_pump_kw"], lower_better=True)))
    _prop_pump_label = "Propane pump (thermosiphon)" if parasitic_b["thermosiphon"] else "Propane pump"
    _prop_pump_val = "0 (thermosiphon)" if parasitic_b["thermosiphon"] else _fmt(parasitic_b["prop_pump_kw"], ".0f")
    summary_rows.append(("row", _prop_pump_label,
                          "N/A", "",
                          _prop_pump_val, "kW",
                          ""))
    summary_rows.append(("row", "ACC fans",
                          _fmt(parasitic_a["acc_fans_kw"], ".0f"), "kW",
                          _fmt(parasitic_b["acc_fans_kw"], ".0f"), "kW",
                          _winner(parasitic_a["acc_fans_kw"], parasitic_b["acc_fans_kw"], lower_better=True)))
    summary_rows.append(("row", "Auxiliary",
                          _fmt(parasitic_a["auxiliary_kw"], ".0f"), "kW",
                          _fmt(parasitic_b["auxiliary_kw"], ".0f"), "kW",
                          "Tie"))
    summary_rows.append(("row", "Total parasitic",
                          _fmt(parasitic_a["total_kw"], ".0f"), "kW",
                          _fmt(parasitic_b["total_kw"], ".0f"), "kW",
                          _winner(parasitic_a["total_kw"], parasitic_b["total_kw"], lower_better=True)))
    pct_para_a = pwr_a["parasitic_pct"]
    pct_para_b = pwr_b["parasitic_pct"]
    summary_rows.append(("row", "Parasitic % of gross",
                          _fmt(pct_para_a, ".1f"), "%",
                          _fmt(pct_para_b, ".1f"), "%",
                          _winner(pct_para_a, pct_para_b, lower_better=True)))
    # Verification: net must equal gross - total parasitic
    _check_a = abs(pwr_a["P_net"] - (pwr_a["P_gross"] - pwr_a["W_total_parasitic"])) > 1.0
    _check_b = abs(pwr_b["P_net"] - (pwr_b["P_gross"] - pwr_b["W_total_parasitic"])) > 1.0
    _net_a_style = "**:red[" + _fmt(pwr_a["P_net"], ".0f") + "]**" if _check_a else _fmt(pwr_a["P_net"], ".0f")
    _net_b_style = "**:red[" + _fmt(pwr_b["P_net"], ".0f") + "]**" if _check_b else _fmt(pwr_b["P_net"], ".0f")
    summary_rows.append(("row", "= Net power",
                          _net_a_style, "kW",
                          _net_b_style, "kW",
                          _winner(pwr_a["P_net"], pwr_b["P_net"], lower_better=False)))

    # -- Physical Scale --
    summary_rows.append(("group", "Physical Scale", "", "", "", "", ""))
    summary_rows.append(("row", "Tailpipe diameter (isopentane, both configs)",
                          _fmt(duct_a["tailpipe_diameter_in"], ".0f"), "in",
                          _fmt(duct_b["tailpipe_diameter_in"], ".0f"), "in",
                          ""))
    summary_rows.append(("row", "Isopentane ACC vapor header (per train)",
                          _fmt(duct_a["acc_header_diameter_in"], ".0f"), "in",
                          "", "",
                          ""))
    prop_header_dia = duct_b.get("propane_header_diameter_in", 0)
    summary_rows.append(("row", "Propane ACC vapor header (per train)",
                          "", "",
                          _fmt(prop_header_dia, ".0f"), "in",
                          ""))
    dia_reduction = duct_a["acc_header_diameter_in"] - prop_header_dia
    summary_rows.append(("row", "ACC header diameter reduction (B vs A)",
                          "", "",
                          _fmt(dia_reduction, ".0f"), "in",
                          "B"))
    summary_rows.append(("row", "Vapor vol. flow (plant total)",
                          _fmt(vol_a, ".0f"), "ft3/s",
                          _fmt(vol_b, ".0f"), "ft3/s",
                          _winner(vol_a, vol_b, lower_better=True)))
    summary_rows.append(("row", "Vol. flow ratio (A/B)",
                          "", "",
                          _fmt(vol_ratio, ".1f"), "x",
                          "B" if vol_ratio > 1 else ""))

    hyd_penalty_a = hydraulic_a["total_dT_penalty_F"]
    hyd_penalty_b = hydraulic_b["total_dT_penalty_F"]
    net_hyd_advantage = hyd_penalty_a - hyd_penalty_b
    power_from_hyd = net_hyd_advantage * dW_dT_a if net_hyd_advantage > 0 else net_hyd_advantage * dW_dT_b

    summary_rows.append(("row", "Total hydraulic penalty",
                          _fmt(hyd_penalty_a, ".2f"), "degF",
                          _fmt(hyd_penalty_b, ".2f"), "degF",
                          _winner(hyd_penalty_a, hyd_penalty_b, lower_better=True)))
    summary_rows.append(("row", "Net hydraulic advantage (B)",
                          "", "",
                          _fmt(net_hyd_advantage, ".2f"), "degF",
                          "B" if net_hyd_advantage > 0 else ("A" if net_hyd_advantage < 0 else "Tie")))
    summary_rows.append(("row", "Power from lower backpressure",
                          "", "",
                          _fmt(abs(power_from_hyd), ".0f"), "kW",
                          "B" if power_from_hyd > 0 else ("A" if power_from_hyd < 0 else "Tie")))

    # -- Capital Cost --
    summary_rows.append(("group", "Capital Cost", "", "", "", "", ""))
    summary_rows.append(("row", "Total installed cost",
                          _fmt(costs_a["total_installed"]/1e6, ".2f"), "$MM",
                          _fmt(costs_b["total_installed"]/1e6, ".2f"), "$MM",
                          _winner(costs_a["total_installed"], costs_b["total_installed"], lower_better=True)))
    summary_rows.append(("row", "Cost confidence band",
                          f"{costs_a['total_installed']*0.7/1e6:.2f} - {costs_a['total_installed']*1.5/1e6:.2f}", "$MM",
                          f"{costs_b['total_installed']*0.7/1e6:.2f} - {costs_b['total_installed']*1.5/1e6:.2f}", "$MM",
                          ""))
    summary_rows.append(("row", "Specific capital cost",
                          _fmt(lc_a["specific_cost_per_kw"], ".0f"), "$/kW",
                          _fmt(lc_b["specific_cost_per_kw"], ".0f"), "$/kW",
                          _winner(lc_a["specific_cost_per_kw"], lc_b["specific_cost_per_kw"], lower_better=True)))
    summary_rows.append(("row", "Equipment count",
                          str(costs_a["equipment_count"]), "",
                          str(costs_b["equipment_count"]), "",
                          _winner(costs_a["equipment_count"], costs_b["equipment_count"], lower_better=True)))

    # -- Economics --
    summary_rows.append(("group", "Economics", "", "", "", "", ""))
    summary_rows.append(("row", "Net NPV",
                          _fmt(lc_a["net_npv"]/1e6, ".2f"), "$MM",
                          _fmt(lc_b["net_npv"]/1e6, ".2f"), "$MM",
                          _winner(lc_a["net_npv"], lc_b["net_npv"], lower_better=False)))
    summary_rows.append(("row", "LCOE",
                          _fmt(lc_a["lcoe"], ".1f"), "$/MWh",
                          _fmt(lc_b["lcoe"], ".1f"), "$/MWh",
                          _winner(lc_a["lcoe"], lc_b["lcoe"], lower_better=True)))

    payback_str = f"{payback_yrs:.1f}" if payback_yrs is not None and payback_yrs < 1e6 else "N/A"
    summary_rows.append(("row", "Simple payback on cost delta",
                          "", "",
                          payback_str, "yr",
                          ""))

    summary_rows.append(("row", "NPV of schedule savings",
                          "", "",
                          _fmt(sched_npv/1e3, ".0f"), "$k",
                          ""))

    # -- Construction Cost Impact --
    summary_rows.append(("group", "Construction Cost Impact", "", "", "", "", ""))
    summary_rows.append(("row", "Schedule savings",
                          "", "",
                          f"{sched_savings} wk" if sched_savings > 0 else "0 wk", "",
                          "B" if sched_savings > 0 else ""))
    summary_rows.append(("row", "Site overhead savings",
                          "", "",
                          _fmt(constr_savings["overhead_savings"]/1e6, ".2f"), "$MM",
                          ""))
    summary_rows.append(("row", "Craft labor efficiency savings",
                          "", "",
                          _fmt(constr_savings["craft_labor_savings"]/1e6, ".2f"), "$MM",
                          ""))
    summary_rows.append(("row", "Construction financing savings",
                          "", "",
                          _fmt(constr_savings["financing_savings"]/1e6, ".2f"), "$MM",
                          ""))
    summary_rows.append(("row", "Total construction cost savings",
                          "", "",
                          _fmt(constr_savings["total_construction_savings"]/1e6, ".2f"), "$MM",
                          "B" if constr_savings["total_construction_savings"] > 0 else ""))

    # -- Total Economic Advantage --
    total_adv_str = _fmt(total_economic_advantage/1e6, ".2f")
    adv_winner = "B" if total_economic_advantage > 0 else ("A" if total_economic_advantage < 0 else "Tie")
    summary_rows.append(("group", "Total Economic Comparison", "", "", "", "", ""))
    summary_rows.append(("row", "**Total Config B economic advantage**",
                          "", "",
                          f"**{total_adv_str}**", "$MM",
                          adv_winner))

    # -- Schedule --
    summary_rows.append(("group", "Schedule", "", "", "", "", ""))
    summary_rows.append(("row", "Construction duration",
                          f"{a_total_weeks} wk", "",
                          f"{b_total_weeks} wk", "",
                          "B" if b_total_weeks < a_total_weeks else ("A" if a_total_weeks < b_total_weeks else "Tie")))
    if sched_savings > 0:
        sched_str = f"{sched_savings} wk ({sched_savings/4.33:.1f} mo) faster"
        sched_winner_val = "B"
    elif sched_savings < 0:
        sched_str = f"{abs(sched_savings)} wk slower"
        sched_winner_val = "A"
    else:
        sched_str = "Same"
        sched_winner_val = "Tie"
    summary_rows.append(("row", "Schedule advantage (B vs A)",
                          "", "",
                          sched_str, "",
                          sched_winner_val))

    # -- Validation --
    all_a_pass = all(passed for _, passed, _ in checks_a)
    all_b_pass = all(passed for _, passed, _ in checks_b)
    summary_rows.append(("group", "Validation", "", "", "", "", ""))
    summary_rows.append(("row", "Config A checks",
                          "PASS" if all_a_pass else "FAIL", "",
                          "", "",
                          ""))
    summary_rows.append(("row", "Config B checks",
                          "", "",
                          "PASS" if all_b_pass else "FAIL", "",
                          ""))

    # Render as markdown table
    table_md = "| Metric | Config A | Config B | Delta (B-A) | Winner |\n"
    table_md += "|--------|----------|----------|-------------|--------|\n"

    for row in summary_rows:
        rtype = row[0]
        if rtype == "group":
            table_md += f"| **{row[1]}** | | | | |\n"
        else:
            label = row[1]
            va_str = f"{row[2]} {row[3]}".strip() if row[2] else ""
            vb_str = f"{row[4]} {row[5]}".strip() if row[4] else ""

            # Compute delta for numeric rows
            delta_str = ""
            try:
                va_num = float(row[2].replace(",", ""))
                vb_num = float(row[4].replace(",", ""))
                d = vb_num - va_num
                unit = row[3] or row[5]
                if abs(d) < 0.005:
                    delta_str = "0"
                elif d > 0:
                    delta_str = f"+{d:.1f} {unit}".strip()
                else:
                    delta_str = f"{d:.1f} {unit}".strip()
            except (ValueError, IndexError):
                delta_str = ""

            winner = row[6]
            winner_display = _color_winner(winner) if winner else ""

            table_md += f"| {label} | {va_str} | {vb_str} | {delta_str} | {winner_display} |\n"

    st.markdown(table_md)

    # Physical scale callout
    if dia_reduction > 0 and vol_ratio > 1:
        st.info(
            f"Config B eliminates the large-bore isopentane ACC vapor header "
            f"(**{duct_a['acc_header_diameter_in']:.0f}\"**) and replaces it with a propane vapor header "
            f"(**{prop_header_dia:.0f}\"**) -- a **{vol_ratio:.1f}x** reduction in vapor volumetric flow. "
            f"The isopentane tailpipe is identical in both configurations as it is upstream of the split point."
        )

    # Auto-generated summary sentence
    power_winner = "A" if pwr_a["P_net"] >= pwr_b["P_net"] else "B"
    cost_winner = "A" if costs_a["total_installed"] <= costs_b["total_installed"] else "B"
    power_diff_kw = abs(pwr_a["P_net"] - pwr_b["P_net"])
    cost_diff_mm = abs(costs_a["total_installed"] - costs_b["total_installed"]) / 1e6

    summary_sentence = (
        f"Config **{power_winner}** produces **{power_diff_kw:.0f} kW** more net power. "
        f"Config **{cost_winner}** costs **${cost_diff_mm:.2f}MM** less to build. "
    )
    if sched_savings > 0:
        summary_sentence += f"Config B is **{sched_savings} weeks ({sched_savings/4.33:.1f} months)** faster to construct"
        sched_total_benefit = sched_npv + constr_savings["total_construction_savings"]
        if sched_total_benefit > 0:
            summary_sentence += (
                f" (construction savings: **${constr_savings['total_construction_savings']/1e6:.2f}MM**, "
                f"early revenue NPV: **${sched_npv/1e3:.0f}k**)"
            )
        summary_sentence += ". "
    elif sched_savings < 0:
        summary_sentence += f"Config A is **{abs(sched_savings)} weeks** faster to construct. "

    if total_economic_advantage > 0:
        summary_sentence += f"All-in economic advantage for B: **${total_economic_advantage/1e6:.2f}MM**."
    elif total_economic_advantage < 0:
        summary_sentence += f"All-in economic advantage for A: **${abs(total_economic_advantage)/1e6:.2f}MM**."

    st.markdown(summary_sentence)

    # Seasonal variation mini-table
    with st.expander("Seasonal Fan Power & Net Output"):
        seasonal_rows = []
        for season in ["Winter", "Design", "Summer"]:
            sa = seasonal_a[season]
            sb = seasonal_b[season]
            seasonal_rows.append({
                "Season": season,
                "T_amb (°F)": f"{sa['T_ambient']:.0f}",
                "Fan A (kW)": f"{sa['fan_kw']:.0f}",
                "Net A (kW)": f"{sa['P_net']:.0f}",
                "Fan B (kW)": f"{sb['fan_kw']:.0f}",
                "Net B (kW)": f"{sb['P_net']:.0f}",
            })
        st.dataframe(pd.DataFrame(seasonal_rows).set_index("Season"), use_container_width=True)
        st.caption("Q_rejection held constant at design value; only air density varies with temperature.")

    # Schedule breakdown detail -- Gantt chart
    with st.expander("Schedule Breakdown"):
        si = sched_info
        max_weeks = max(a_total_weeks, b_total_weeks) + 4

        def _gantt_fig(phases, total_wk, title, max_x):
            """Build a horizontal bar Gantt chart for one config."""
            fig = go.Figure()
            names = [p["name"] for p in phases]
            # Reverse so first phase is at top
            for i, p in enumerate(reversed(phases)):
                fig.add_trace(go.Bar(
                    y=[p["name"]],
                    x=[p["duration"]],
                    base=[p["start"]],
                    orientation="h",
                    marker_color=p["color"],
                    marker_line_width=2 if p["critical"] else 0,
                    marker_line_color="black" if p["critical"] else p["color"],
                    opacity=1.0 if p["critical"] else 0.7,
                    text=f'{p["duration"]}w',
                    textposition="inside",
                    hovertemplate=f'{p["name"]}<br>Wk {p["start"]}-{p["end"]} ({p["duration"]} wk)<extra></extra>',
                    showlegend=False,
                ))
            fig.add_vline(x=total_wk, line_dash="dash", line_color="red", line_width=2,
                          annotation_text=f"{total_wk} wk", annotation_position="top right")
            fig.update_layout(
                title=dict(text=title, font_size=14),
                xaxis=dict(title="Weeks", range=[0, max_x], dtick=8),
                yaxis=dict(autorange="reversed"),
                height=max(250, 40 * len(phases)),
                margin=dict(l=10, r=10, t=40, b=30),
                barmode="overlay",
            )
            return fig

        col_ga, col_gb = st.columns(2)
        with col_ga:
            fig_a = _gantt_fig(si["config_a"]["phases"], a_total_weeks,
                               f"Config A  --  {a_total_weeks} weeks", max_weeks)
            st.plotly_chart(fig_a, use_container_width=True)
        with col_gb:
            fig_b = _gantt_fig(si["config_b"]["phases"], b_total_weeks,
                               f"Config B  --  {b_total_weeks} weeks", max_weeks)
            st.plotly_chart(fig_b, use_container_width=True)

        # Phase detail tables
        col_ta, col_tb = st.columns(2)
        with col_ta:
            st.markdown("**Config A phases**")
            df_a = pd.DataFrame([
                {"Phase": p["name"], "Start": f'Wk {p["start"]}', "End": f'Wk {p["end"]}',
                 "Dur": f'{p["duration"]}w', "Critical": "Yes" if p["critical"] else ""}
                for p in si["config_a"]["phases"]
            ])
            st.dataframe(df_a.set_index("Phase"), use_container_width=True)
        with col_tb:
            st.markdown("**Config B phases**")
            df_b = pd.DataFrame([
                {"Phase": p["name"], "Start": f'Wk {p["start"]}', "End": f'Wk {p["end"]}',
                 "Dur": f'{p["duration"]}w', "Track": p["track"], "Critical": "Yes" if p["critical"] else ""}
                for p in si["config_b"]["phases"]
            ])
            st.dataframe(df_b.set_index("Phase"), use_container_width=True)

        # Summary callout
        if sched_savings > 0:
            st.success(
                f"Config B saves **{sched_savings} weeks ({sched_savings/4.33:.1f} months)** "
                f"via parallel power-block / propane-ACC construction "
                f"(tailpipe = {si['tailpipe_diameter_in']:.0f}\" per train, duct phase = {si['duct_phase_weeks']} wk)."
            )
        elif sched_savings < 0:
            st.warning(f"Config A is {abs(sched_savings)} weeks faster at this tailpipe diameter.")
        else:
            st.info("Both configs have the same construction duration.")

        # --- Schedule-Related Savings Stacked Bar Chart ---
        if sched_savings > 0:
            st.markdown("#### Schedule-Related Economic Savings")
            cs = constr_savings
            bar_categories = ["Early Revenue\nNPV", "Site Overhead\nSavings",
                              "Craft Labor\nSavings", "Financing\nSavings"]
            bar_values = [sched_npv, cs["overhead_savings"],
                          cs["craft_labor_savings"], cs["financing_savings"]]
            bar_colors = ["steelblue", "gray", "orange", "green"]
            total_sched_benefit = sched_npv + cs["total_construction_savings"]

            fig_sav = go.Figure()
            fig_sav.add_trace(go.Bar(
                x=bar_categories + ["Total"],
                y=bar_values + [total_sched_benefit],
                marker_color=bar_colors + ["darkblue"],
                text=[f"${v/1e6:.2f}MM" for v in bar_values] + [f"**${total_sched_benefit/1e6:.2f}MM**"],
                textposition="outside",
                hovertemplate="%{x}: $%{y:,.0f}<extra></extra>",
            ))
            fig_sav.update_layout(
                yaxis=dict(title="Savings ($)", tickformat="$,.0f"),
                height=350,
                margin=dict(l=10, r=10, t=30, b=10),
                showlegend=False,
            )
            # Separator line before total bar
            fig_sav.add_vline(x=3.5, line_dash="dot", line_color="gray", line_width=1)
            st.plotly_chart(fig_sav, use_container_width=True)

            st.caption(
                "Construction cost savings estimated from schedule compression. "
                "Craft labor waiting time assumes 15% of labor hours lost to serial "
                "construction dependencies in Config A — this is reduced proportionally "
                "by Config B parallel execution. Actual savings depend on contractor "
                "execution strategy and site conditions."
            )

    # Validation checks detail
    with st.expander("Validation Checks"):
        col_chk_a, col_chk_b = st.columns(2)
        with col_chk_a:
            st.markdown("**Config A**")
            for name, passed, detail in checks_a:
                icon = "PASS" if passed else "FAIL"
                color = "green" if passed else "red"
                st.markdown(f":{color}[{icon}] {name} -- {detail}")
        with col_chk_b:
            st.markdown("**Config B**")
            for name, passed, detail in checks_b:
                icon = "PASS" if passed else "FAIL"
                color = "green" if passed else "red"
                st.markdown(f":{color}[{icon}] {name} -- {detail}")

    # ============================================================================
    # PROCESS FLOW DIAGRAMS
    # ============================================================================

    st.header("Process Flow Diagrams")

    pfd_col1, pfd_col2 = st.columns(2)
    with pfd_col1:
        pfd_html_a = _generate_pfd_a(states_a, perf_a, duct_a, inputs)
        components.html(pfd_html_a, height=530, scrolling=False)
    with pfd_col2:
        pfd_html_b = _generate_pfd_b(states_b, prop_states, perf_b, duct_b, duct_a, inputs)
        components.html(pfd_html_b, height=530, scrolling=False)


    # ============================================================================
    # SECTION 2: COST WATERFALL CHARTS
    # ============================================================================

    st.header("Cost Breakdown")

    comp_list = [
        ("Turbine/Generator", "turbine_generator"),
        ("Isopentane Pump", "iso_pump"),
        ("Vaporizer", "vaporizer"),
        ("Preheater", "preheater"),
        ("Recuperator", "recuperator"),
        ("Air-Cooled Condensers", "acc"),
        ("Vapor Ductwork", "ductwork"),
        ("Structural Steel", "structural_steel"),
        ("Intermediate HX", "intermediate_hx"),
        ("Propane System", "propane_system"),
        ("Foundation", "foundation"),
        ("Engineering & Procurement", "engineering"),
        ("Construction Mgmt", "construction_mgmt"),
        ("Contingency", "contingency"),
    ]

    comp_names = [c[0] for c in comp_list]
    vals_a = [costs_a[c[1]] / 1e6 for c in comp_list]
    vals_b = [costs_b[c[1]] / 1e6 for c in comp_list]
    deltas = [vb - va for va, vb in zip(vals_a, vals_b)]

    wf_col1, wf_col2 = st.columns(2)

    # Config A waterfall
    with wf_col1:
        fig_wf_a = go.Figure(go.Waterfall(
            name="Config A",
            orientation="v",
            x=comp_names + ["Total"],
            y=vals_a + [None],
            measure=["relative"] * len(vals_a) + ["total"],
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            increasing=dict(marker=dict(color="steelblue")),
            totals=dict(marker=dict(color="darkblue")),
            text=[f"${v:.2f}M" for v in vals_a] + [f"${costs_a['total_installed']/1e6:.2f}M"],
            textposition="outside",
        ))
        max_y_a = costs_a['total_installed'] / 1e6 * 1.25
        fig_wf_a.update_layout(
            title="Config A Installed Cost ($MM)",
            yaxis_title="Cost ($MM)",
            yaxis_range=[0, max_y_a],
            height=450,
            showlegend=False,
        )
        fig_wf_a.update_traces(textfont_size=10, textangle=-45,
                               cliponaxis=False)
        st.plotly_chart(fig_wf_a, width="stretch")

    # Config B waterfall
    with wf_col2:
        fig_wf_b = go.Figure(go.Waterfall(
            name="Config B",
            orientation="v",
            x=comp_names + ["Total"],
            y=vals_b + [None],
            measure=["relative"] * len(vals_b) + ["total"],
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            increasing=dict(marker=dict(color="indianred")),
            totals=dict(marker=dict(color="darkred")),
            text=[f"${v:.2f}M" for v in vals_b] + [f"${costs_b['total_installed']/1e6:.2f}M"],
            textposition="outside",
        ))
        max_y_b = costs_b['total_installed'] / 1e6 * 1.25
        fig_wf_b.update_layout(
            title="Config B Installed Cost ($MM)",
            yaxis_title="Cost ($MM)",
            yaxis_range=[0, max_y_b],
            height=450,
            showlegend=False,
        )
        fig_wf_b.update_traces(textfont_size=10, textangle=-45,
                               cliponaxis=False)
        st.plotly_chart(fig_wf_b, width="stretch")

    # Delta chart (full width)
    delta_colors = ["green" if d < 0 else "goldenrod" for d in deltas]
    fig_delta = go.Figure(go.Bar(
        x=comp_names,
        y=deltas,
        marker_color=delta_colors,
        text=[f"{d:+.2f}M" for d in deltas],
        textposition="outside",
    ))
    fig_delta.update_layout(
        title="Cost Delta (Config B - Config A, $MM) -- Green = B saves money",
        yaxis_title="Delta ($MM)",
        height=400,
    )
    fig_delta.update_traces(textfont_size=10, textangle=-45,
                            cliponaxis=False)
    st.plotly_chart(fig_delta, width="stretch")

    # Duct segment cost breakdown
    with st.expander("Duct Segment Cost Detail"):
        duct_seg_data = []
        for seg in duct_a["segments"]:
            duct_seg_data.append({"Config": "A", "Segment": seg["name"],
                                  "Diameter (in)": f"{seg['diameter_in']:.1f}",
                                  "Length (ft)": f"{seg['length_ft']:.0f}",
                                  "Cost ($)": f"${_duct_segment_cost(seg, inputs):,.0f}"})
        for seg in duct_b["segments"]:
            duct_seg_data.append({"Config": "B", "Segment": seg["name"],
                                  "Diameter (in)": f"{seg['diameter_in']:.1f}",
                                  "Length (ft)": f"{seg['length_ft']:.0f}",
                                  "Cost ($)": f"${_duct_segment_cost(seg, inputs):,.0f}"})
        st.dataframe(pd.DataFrame(duct_seg_data).set_index("Config"), width="stretch")


    # ============================================================================
    # SECTION 3: POWER OUTPUT COMPARISON
    # ============================================================================

    st.header("Power Output Comparison")

    fig_power = make_subplots(specs=[[{"secondary_y": True}]])

    # Bar chart: gross and net power
    categories = ["Gross Power", "Net Power"]
    power_a_vals = [perf_a["gross_power_kw"], pwr_a["P_net"]]
    power_b_vals = [perf_b["gross_power_kw"], pwr_b["P_net"]]

    fig_power.add_trace(
        go.Bar(x=categories, y=power_a_vals, name="Config A", marker_color="steelblue"),
        secondary_y=False,
    )
    fig_power.add_trace(
        go.Bar(x=categories, y=power_b_vals, name="Config B", marker_color="indianred"),
        secondary_y=False,
    )

    # Efficiency line on secondary axis
    eta_categories = ["Config A", "Config B"]
    eta_vals = [pwr_a["eta_thermal"]*100, pwr_b["eta_thermal"]*100]
    fig_power.add_trace(
        go.Scatter(x=eta_categories, y=eta_vals, name="Thermal Efficiency",
                   mode="lines+markers", marker=dict(size=12, color="orange"),
                   line=dict(color="orange", width=2, dash="dot")),
        secondary_y=True,
    )

    fig_power.update_layout(
        title="Power Output and Thermal Efficiency",
        barmode="group",
        height=450,
    )
    fig_power.update_yaxes(title_text="Power (kW)", secondary_y=False)
    fig_power.update_yaxes(title_text="Thermal Efficiency (%)", secondary_y=True)

    st.plotly_chart(fig_power, width="stretch")

    # Key metrics below chart
    pwr_col1, pwr_col2, pwr_col3, pwr_col4 = st.columns(4)
    pwr_col1.metric("Net Power A", f"{pwr_a['P_net']:.0f} kW")
    pwr_col2.metric("Net Power B", f"{pwr_b['P_net']:.0f} kW")
    pwr_col3.metric("Delta (B-A)", f"{pwr_b['P_net'] - pwr_a['P_net']:.0f} kW")
    pwr_col4.metric("Vol. Flow Ratio (A/B)", f"{vol_ratio:.1f}x")

    # Power waterfall: Gross -> deductions -> Net Power
    st.subheader("Parasitic Load Breakdown")
    wf_pwr_col1, wf_pwr_col2 = st.columns(2)

    with wf_pwr_col1:
        wf_a_labels = ["Gross", "ISO Pump", "ACC Fans", "Auxiliary", "Net Power"]
        wf_a_values = [
            perf_a["gross_power_kw"],
            -parasitic_a["iso_pump_kw"],
            -parasitic_a["acc_fans_kw"],
            -parasitic_a["auxiliary_kw"],
            None,
        ]
        wf_a_measures = ["absolute", "relative", "relative", "relative", "total"]
        fig_wf_pwr_a = go.Figure(go.Waterfall(
            x=wf_a_labels, y=wf_a_values, measure=wf_a_measures,
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            decreasing=dict(marker=dict(color="indianred")),
            increasing=dict(marker=dict(color="steelblue")),
            totals=dict(marker=dict(color="darkblue")),
            text=[f"{abs(v):.0f} kW" if v is not None else f"{pwr_a["P_net"]:.0f} kW" for v in wf_a_values],
            textposition="outside",
        ))
        fig_wf_pwr_a.update_layout(
            title="Config A -- Gross to Net Power (kW)",
            yaxis_title="Power (kW)", height=400, showlegend=False,
        )
        fig_wf_pwr_a.update_traces(textfont_size=10, cliponaxis=False)
        st.plotly_chart(fig_wf_pwr_a, use_container_width=True)

    with wf_pwr_col2:
        if parasitic_b["thermosiphon"]:
            wf_b_labels = ["Gross", "ISO Pump", "ACC Fans", "Auxiliary", "Net Power"]
            wf_b_values = [
                perf_b["gross_power_kw"],
                -parasitic_b["iso_pump_kw"],
                -parasitic_b["acc_fans_kw"],
                -parasitic_b["auxiliary_kw"],
                None,
            ]
            wf_b_measures = ["absolute", "relative", "relative", "relative", "total"]
        else:
            wf_b_labels = ["Gross", "ISO Pump", "Prop Pump", "ACC Fans", "Auxiliary", "Net Power"]
            wf_b_values = [
                perf_b["gross_power_kw"],
                -parasitic_b["iso_pump_kw"],
                -parasitic_b["prop_pump_kw"],
                -parasitic_b["acc_fans_kw"],
                -parasitic_b["auxiliary_kw"],
                None,
            ]
            wf_b_measures = ["absolute", "relative", "relative", "relative", "relative", "total"]
        fig_wf_pwr_b = go.Figure(go.Waterfall(
            x=wf_b_labels, y=wf_b_values, measure=wf_b_measures,
            connector=dict(line=dict(color="rgb(63, 63, 63)")),
            decreasing=dict(marker=dict(color="indianred")),
            increasing=dict(marker=dict(color="steelblue")),
            totals=dict(marker=dict(color="darkred")),
            text=[f"{abs(v):.0f} kW" if v is not None else f"{pwr_b["P_net"]:.0f} kW" for v in wf_b_values],
            textposition="outside",
        ))
        fig_wf_pwr_b.update_layout(
            title="Config B -- Gross to Net Power (kW)",
            yaxis_title="Power (kW)", height=400, showlegend=False,
        )
        fig_wf_pwr_b.update_traces(textfont_size=10, cliponaxis=False)
        st.plotly_chart(fig_wf_pwr_b, use_container_width=True)


    # ============================================================================
    # SECTION 4: TECHNICAL CHARTS (TABS)
    # ============================================================================

    st.header("Technical Analysis")

    tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs([
        "T-s Diagram", "Duct Sizing", "Approach Optimization",
        "Brine Utilization", "Sensitivity Analysis", "Hydraulic Analysis",
        "Equipment Sizing Trade-offs", "Fan Power & ACC Optimization",
    ])

    # -- Tab 1: T-s Diagram ---------------------------------------------------
    with tab1:
        fig_ts = go.Figure()

        # Saturation dome
        try:
            T_dome = np.linspace(60, states_a["1"].T + 30, 60)
            s_f_dome, s_g_dome, T_f_dome = [], [], []
            for T in T_dome:
                try:
                    sat = fp.saturation_props("isopentane", T=T)
                    s_f_dome.append(sat["s_f"])
                    s_g_dome.append(sat["s_g"])
                    T_f_dome.append(sat["T_sat"])
                except Exception:
                    pass
            s_dome = s_f_dome + s_g_dome[::-1]
            T_dome_plot = T_f_dome + T_f_dome[::-1]
            fig_ts.add_trace(go.Scatter(
                x=s_dome, y=T_dome_plot, mode="lines",
                name="Sat Dome (Isopentane)", line=dict(color="lightgray", width=1),
                fill="toself", fillcolor="rgba(200,200,200,0.15)",
            ))
        except Exception:
            pass

        # Config A cycle
        pts = ["1", "2", "3", "4", "5", "6", "7", "1"]
        s_a = [states_a[p].s for p in pts]
        T_a = [states_a[p].T for p in pts]
        fig_ts.add_trace(go.Scatter(
            x=s_a, y=T_a, mode="lines+markers",
            name="Config A", line=dict(color="blue", width=2.5),
            marker=dict(size=8),
        ))

        # Config B cycle
        s_b = [states_b[p].s for p in pts]
        T_b = [states_b[p].T for p in pts]
        fig_ts.add_trace(go.Scatter(
            x=s_b, y=T_b, mode="lines+markers",
            name="Config B", line=dict(color="red", width=2.5, dash="dash"),
            marker=dict(size=8),
        ))

        # State point labels
        for label, st_pt in states_a.items():
            fig_ts.add_annotation(
                x=st_pt.s, y=st_pt.T, text=label,
                showarrow=True, arrowhead=2, ax=15, ay=-15,
                font=dict(color="blue", size=10),
            )

        # Condensing lines
        fig_ts.add_hline(y=perf_a["T_cond"], line_dash="dot", line_color="blue",
                         annotation_text=f"A cond: {perf_a['T_cond']:.0f}degF")
        fig_ts.add_hline(y=perf_b["T_cond_iso"], line_dash="dot", line_color="red",
                         annotation_text=f"B cond: {perf_b['T_cond_iso']:.0f}degF")

        # Net efficiency annotations
        fig_ts.add_annotation(
            x=0.98, y=0.02, xref="paper", yref="paper",
            text=(f"Net eta_th: A={pwr_a['eta_thermal']*100:.1f}%  "
                  f"B={pwr_b['eta_thermal']*100:.1f}%"),
            showarrow=False,
            font=dict(size=11, color="black"),
            bgcolor="rgba(255,255,255,0.8)", bordercolor="gray", borderwidth=1,
            xanchor="right", yanchor="bottom",
        )

        fig_ts.update_layout(
            title="T-s Diagram -- Isopentane ORC Cycles",
            xaxis_title="Entropy s (BTU/lb-R)",
            yaxis_title="Temperature T (degF)",
            legend=dict(x=0.01, y=0.99),
            height=550,
        )
        st.plotly_chart(fig_ts, width="stretch")

    # -- Tab 2: Duct Sizing ---------------------------------------------------
    with tab2:
        fig_duct = make_subplots(rows=2, cols=1,
                                 subplot_titles=["Duct Diameters by Segment (per train)",
                                                 "Volumetric Flow Rates by Segment (per train)"],
                                 vertical_spacing=0.15)

        seg_names_a = [s["name"] for s in duct_a["segments"]]
        seg_dia_a = [s["diameter_in"] for s in duct_a["segments"]]
        seg_vol_a = [s["vol_flow_ft3s"] for s in duct_a["segments"]]

        seg_names_b = [s["name"] for s in duct_b["segments"]]
        seg_dia_b = [s["diameter_in"] for s in duct_b["segments"]]
        seg_vol_b = [s["vol_flow_ft3s"] for s in duct_b["segments"]]

        all_names = list(dict.fromkeys(seg_names_a + seg_names_b))
        dia_a_padded = [seg_dia_a[seg_names_a.index(n)] if n in seg_names_a else 0 for n in all_names]
        dia_b_padded = [seg_dia_b[seg_names_b.index(n)] if n in seg_names_b else 0 for n in all_names]
        vol_a_padded = [seg_vol_a[seg_names_a.index(n)] if n in seg_names_a else 0 for n in all_names]
        vol_b_padded = [seg_vol_b[seg_names_b.index(n)] if n in seg_names_b else 0 for n in all_names]

        fig_duct.add_trace(go.Bar(x=all_names, y=dia_a_padded, name="Config A",
                                  marker_color="steelblue"), row=1, col=1)
        fig_duct.add_trace(go.Bar(x=all_names, y=dia_b_padded, name="Config B",
                                  marker_color="indianred"), row=1, col=1)

        fig_duct.add_hline(y=80, line_dash="dash", line_color="orange",
                           annotation_text="80\" ref", row=1, col=1)
        fig_duct.add_hline(y=66, line_dash="dash", line_color="gray",
                           annotation_text="66\" ref", row=1, col=1)

        fig_duct.add_trace(go.Bar(x=all_names, y=vol_a_padded, name="Config A",
                                  marker_color="steelblue", showlegend=False), row=2, col=1)
        fig_duct.add_trace(go.Bar(x=all_names, y=vol_b_padded, name="Config B",
                                  marker_color="indianred", showlegend=False), row=2, col=1)

        fig_duct.update_layout(
            title=f"Vapor Duct Sizing -- Per Train (2 trains) -- Vol Ratio A/B: {vol_ratio:.1f}x",
            barmode="group", height=700,
        )
        fig_duct.update_yaxes(title_text="Diameter (inches)", row=1, col=1)
        fig_duct.update_yaxes(title_text="Vol Flow (ft3/s)", row=2, col=1)

        st.plotly_chart(fig_duct, width="stretch")

        # Segment detail table
        seg_table = []
        for seg in duct_a["segments"]:
            seg_table.append({
                "Config": "A", "Segment": seg["name"],
                "Diameter (in)": f"{seg['diameter_in']:.1f}",
                "Velocity (ft/s)": f"{seg['velocity_fps']:.0f}",
                "dP (psia)": f"{seg['delta_P_psi']:.4f}",
                "dT penalty (degF)": f"{seg['delta_T_cond_F']:.2f}",
            })
        for seg in duct_b["segments"]:
            seg_table.append({
                "Config": "B", "Segment": seg["name"],
                "Diameter (in)": f"{seg['diameter_in']:.1f}",
                "Velocity (ft/s)": f"{seg['velocity_fps']:.0f}",
                "dP (psia)": f"{seg['delta_P_psi']:.4f}",
                "dT penalty (degF)": f"{seg['delta_T_cond_F']:.2f}",
            })
        st.dataframe(pd.DataFrame(seg_table).set_index("Config"), width="stretch")
        st.caption("Diameters and flows are per train (2 parallel trains). Plant total flow = 2x per-train values.")

    # -- Tab 3: Approach dT Sweep ----------------------------------------------
    with tab3:
        with st.spinner("Running approach temperature sweep..."):
            opt_result = optimize_approach_temp(inputs, fp)

        sweep = opt_result["sweep"]
        dts = [s["dt_approach"] for s in sweep]
        powers_b_sweep = [s["net_power_kw"] for s in sweep]
        inst_costs_b_sweep = [s["installed_cost"] / 1e6 for s in sweep]
        ihx_costs = [s["intermediate_hx_cost"] / 1e6 for s in sweep]
        lcoes = [s["lcoe"] for s in sweep]

        fig_sweep = make_subplots(rows=2, cols=1,
                                  subplot_titles=["Net Power & Installed Cost",
                                                  "LCOE"],
                                  vertical_spacing=0.12)

        fig_sweep.add_trace(go.Scatter(x=dts, y=powers_b_sweep, name="Config B Net Power (kW)",
                                       line=dict(color="red", width=2)), row=1, col=1)
        fig_sweep.add_hline(y=opt_result["ref_power_a"], line_dash="dot",
                            line_color="blue", annotation_text="Config A ref",
                            row=1, col=1)

        fig_sweep.add_trace(go.Scatter(x=dts, y=inst_costs_b_sweep, name="Config B Total Cost ($MM)",
                                       line=dict(color="orange", width=2, dash="dash"),
                                       yaxis="y3"), row=1, col=1)
        fig_sweep.add_trace(go.Scatter(x=dts, y=ihx_costs, name="IHX Cost ($MM)",
                                       line=dict(color="gray", width=1, dash="dot"),
                                       yaxis="y3"), row=1, col=1)

        fig_sweep.add_trace(go.Scatter(x=dts, y=lcoes, name="Config B LCOE ($/MWh)",
                                       line=dict(color="purple", width=2)), row=2, col=1)
        if opt_result["ref_lcoe_a"] > 0:
            fig_sweep.add_hline(y=opt_result["ref_lcoe_a"], line_dash="dot",
                                line_color="blue", annotation_text="Config A LCOE",
                                row=2, col=1)

        fig_sweep.add_vline(x=opt_result["optimal_dt"], line_dash="dash",
                            annotation_text=f"Optimal: {opt_result['optimal_dt']:.1f}degF")

        if len(powers_b_sweep) >= 3:
            dp_per_dt = (powers_b_sweep[0] - powers_b_sweep[-1]) / (dts[-1] - dts[0])
            fig_sweep.add_annotation(
                x=15, y=max(powers_b_sweep) * 0.95,
                text=f"Power penalty: {abs(dp_per_dt):.1f} kW/degF",
                showarrow=False, font=dict(size=11), row=1, col=1,
            )

        fig_sweep.update_layout(height=650,
                                yaxis3=dict(overlaying="y", side="right",
                                            title="Cost ($MM)"))
        fig_sweep.update_xaxes(title_text="Approach dT (degF)")
        fig_sweep.update_yaxes(title_text="Net Power (kW)", row=1, col=1)
        fig_sweep.update_yaxes(title_text="LCOE ($/MWh)", row=2, col=1)

        st.plotly_chart(fig_sweep, width="stretch")

    # -- Tab 4: Brine Utilization ----------------------------------------------
    with tab4:
        fig_brine = go.Figure()

        m_dot_a_hr = perf_a["m_dot_iso"]  # lb/hr
        Q_pre_a = m_dot_a_hr * perf_a["q_preheater"] / 1e6
        Q_vap_a = m_dot_a_hr * perf_a["q_vaporizer"] / 1e6

        brine_q_a = [0, Q_pre_a, Q_pre_a + Q_vap_a]
        brine_T_a = [perf_a["T_geo_out_calc"], perf_a["T_brine_mid"], inputs["T_geo_in"]]

        iso_q_a = [0, Q_pre_a, Q_pre_a + Q_vap_a]
        iso_T_a = [states_a["6"].T, states_a["7"].T, states_a["1"].T]

        fig_brine.add_trace(go.Scatter(x=brine_q_a, y=brine_T_a,
                                       name="A: Brine", mode="lines+markers",
                                       line=dict(color="blue", width=2)))
        fig_brine.add_trace(go.Scatter(x=iso_q_a, y=iso_T_a,
                                       name="A: Isopentane", mode="lines+markers",
                                       line=dict(color="blue", width=2, dash="dash")))

        m_dot_b_hr = perf_b["m_dot_iso"]
        Q_pre_b = m_dot_b_hr * perf_b["q_preheater"] / 1e6
        Q_vap_b = m_dot_b_hr * perf_b["q_vaporizer"] / 1e6

        brine_q_b = [0, Q_pre_b, Q_pre_b + Q_vap_b]
        brine_T_b = [perf_b["T_geo_out_calc"], perf_b["T_brine_mid"], inputs["T_geo_in"]]

        iso_q_b = [0, Q_pre_b, Q_pre_b + Q_vap_b]
        iso_T_b = [states_b["6"].T, states_b["7"].T, states_b["1"].T]

        fig_brine.add_trace(go.Scatter(x=brine_q_b, y=brine_T_b,
                                       name="B: Brine", mode="lines+markers",
                                       line=dict(color="red", width=2)))
        fig_brine.add_trace(go.Scatter(x=iso_q_b, y=iso_T_b,
                                       name="B: Isopentane", mode="lines+markers",
                                       line=dict(color="red", width=2, dash="dash")))

        fig_brine.add_annotation(x=Q_pre_a, y=perf_a["T_brine_mid"],
                                 text=f"A pinch: {perf_a['vaporizer_pinch']:.1f}degF",
                                 showarrow=True, arrowhead=2, font=dict(color="blue"))
        fig_brine.add_annotation(x=Q_pre_b, y=perf_b["T_brine_mid"],
                                 text=f"B pinch: {perf_b['vaporizer_pinch']:.1f}degF",
                                 showarrow=True, arrowhead=2, font=dict(color="red"))

        fig_brine.add_hline(y=inputs["T_geo_out_min"], line_dash="dash", line_color="orange",
                            annotation_text=f"Min brine outlet: {inputs['T_geo_out_min']}degF")

        fig_brine.update_layout(
            title="Brine Utilization -- Temperature vs Cumulative Heat Duty",
            xaxis_title="Cumulative Heat Duty (MMBtu/hr)",
            yaxis_title="Temperature (degF)",
            height=550,
        )
        st.plotly_chart(fig_brine, width="stretch")

    # -- Tab 5: Sensitivity Tornado --------------------------------------------
    with tab5:
        st.markdown("Impact on **Config B total cost premium over Config A** ($MM) from +/-20% parameter changes")

        base_premium = (costs_b["total_installed"] - costs_a["total_installed"]) / 1e6

        sensitivity_params = [
            ("T_ambient", inputs["T_ambient"], "Ambient Temp"),
            ("electricity_price", inputs["electricity_price"], "Elec Price"),
            ("dt_approach_intermediate", inputs["dt_approach_intermediate"], "Approach dT"),
            ("v_tailpipe", inputs["v_tailpipe"], "Duct Velocity"),
            ("T_geo_in", inputs["T_geo_in"], "Brine Inlet T"),
        ]

        tornado_data = []
        for key, base_val, label in sensitivity_params:
            low_val = base_val * 0.8
            high_val = base_val * 1.2
            premiums = []
            for v in [low_val, high_val]:
                try:
                    inp_mod = {**inputs, key: v}
                    ra = solve_config_a(inp_mod, fp)
                    rb = solve_config_b(inp_mod, fp)
                    ca = calculate_costs_a(ra["states"], ra["performance"],
                                           inp_mod, ra.get("duct"))
                    cb = calculate_costs_b(rb["states"], rb["propane_states"],
                                           rb["performance"], inp_mod, rb.get("duct"))
                    premiums.append((cb["total_installed"] - ca["total_installed"]) / 1e6)
                except Exception:
                    premiums.append(base_premium)

            tornado_data.append({
                "param": label,
                "low": premiums[0] - base_premium,
                "high": premiums[1] - base_premium,
            })

        tornado_data.sort(key=lambda d: abs(d["high"] - d["low"]))
        params = [d["param"] for d in tornado_data]
        lows = [d["low"] for d in tornado_data]
        highs = [d["high"] for d in tornado_data]

        fig_tornado = go.Figure()
        fig_tornado.add_trace(go.Bar(y=params, x=lows, orientation="h",
                                     name="-20%", marker_color="steelblue"))
        fig_tornado.add_trace(go.Bar(y=params, x=highs, orientation="h",
                                     name="+20%", marker_color="indianred"))
        fig_tornado.update_layout(
            title=f"Sensitivity Tornado -- Config B Cost Premium (base: ${base_premium:.2f}MM)",
            xaxis_title="Change in Premium ($MM)",
            barmode="overlay",
            height=450,
        )
        st.plotly_chart(fig_tornado, width="stretch")


    # -- Tab 6: Hydraulic Analysis -----------------------------------------------
    with tab6:
        st.subheader("Hydraulic Pressure Drop & Condensing Penalty")

        # 1. Side-by-side bar chart: dP contributions by component
        comp_a = hydraulic_a["components"]
        comp_b = hydraulic_b["components"]

        hyd_col1, hyd_col2 = st.columns(2)

        with hyd_col1:
            names_a = list(comp_a.keys())
            dps_a = [comp_a[n]["dP_psi"] for n in names_a]
            types_a = [comp_a[n]["type"] for n in names_a]
            colors_a = ["steelblue" if t == "pipe" else "cornflowerblue" for t in types_a]

            fig_dp_a = go.Figure(go.Bar(
                x=names_a, y=dps_a, marker_color=colors_a,
                text=[f"{v:.3f}" for v in dps_a], textposition="outside",
            ))
            fig_dp_a.update_layout(
                title=f"Config A dP Components (total: {hydraulic_a['total_dP_psi']:.3f} psi)",
                yaxis_title="Pressure Drop (psi)", height=400,
            )
            st.plotly_chart(fig_dp_a, width="stretch")

        with hyd_col2:
            names_b = list(comp_b.keys())
            dps_b = [comp_b[n]["dP_psi"] for n in names_b]
            circuit_b = [comp_b[n].get("circuit", "iso") for n in names_b]
            colors_b = []
            for n in names_b:
                c = comp_b[n]
                if c.get("circuit") == "propane":
                    colors_b.append("indianred" if c["type"] == "pipe" else "salmon")
                else:
                    colors_b.append("steelblue" if c["type"] == "pipe" else "cornflowerblue")

            fig_dp_b = go.Figure(go.Bar(
                x=names_b, y=dps_b, marker_color=colors_b,
                text=[f"{v:.3f}" for v in dps_b], textposition="outside",
            ))
            fig_dp_b.update_layout(
                title=f"Config B dP Components (total: {hydraulic_b['total_dP_psi']:.3f} psi)",
                yaxis_title="Pressure Drop (psi)", height=400,
            )
            st.plotly_chart(fig_dp_b, width="stretch")

        # 2. dT/dP sensitivity curves
        st.subheader("dT/dP Sensitivity: Isopentane vs Propane")

        # Compute dT/dP numerically from CoolProp at 20 evenly spaced pressure points
        P_iso_range = np.linspace(20, 80, 20)    # psia
        P_prop_range = np.linspace(100, 300, 20)  # psia

        T_iso_curve, dTdP_iso_curve = [], []
        for P in P_iso_range:
            try:
                sat = fp.saturation_props("isopentane", P=P)
                T_sat = sat["T_sat"]
                dp = 0.01
                sat_hi = fp.saturation_props("isopentane", P=P + dp)
                sat_lo = fp.saturation_props("isopentane", P=P - dp)
                dTdP = abs((sat_hi["T_sat"] - sat_lo["T_sat"]) / (2 * dp))
                T_iso_curve.append(T_sat)
                dTdP_iso_curve.append(dTdP)
            except Exception:
                T_iso_curve.append(float("nan"))
                dTdP_iso_curve.append(float("nan"))

        T_prop_curve, dTdP_prop_curve = [], []
        for P in P_prop_range:
            try:
                sat = fp.saturation_props("propane", P=P)
                T_sat = sat["T_sat"]
                dp = 0.01
                sat_hi = fp.saturation_props("propane", P=P + dp)
                sat_lo = fp.saturation_props("propane", P=P - dp)
                dTdP = abs((sat_hi["T_sat"] - sat_lo["T_sat"]) / (2 * dp))
                T_prop_curve.append(T_sat)
                dTdP_prop_curve.append(dTdP)
            except Exception:
                T_prop_curve.append(float("nan"))
                dTdP_prop_curve.append(float("nan"))

        fig_sens = go.Figure()
        fig_sens.add_trace(go.Scatter(
            x=T_iso_curve, y=dTdP_iso_curve, name="Isopentane (20-80 psia)",
            line=dict(color="blue", width=2.5),
        ))
        fig_sens.add_trace(go.Scatter(
            x=T_prop_curve, y=dTdP_prop_curve, name="Propane (100-300 psia)",
            line=dict(color="red", width=2.5),
        ))

        # Mark operating points
        fig_sens.add_trace(go.Scatter(
            x=[perf_a["T_cond"]], y=[hydraulic_a["dT_dP_FperPsi"]],
            mode="markers", name=f"Config A ({perf_a['T_cond']:.0f}degF, {perf_a['P_low']:.0f} psia)",
            marker=dict(size=14, color="blue", symbol="circle"),
        ))
        fig_sens.add_trace(go.Scatter(
            x=[perf_b["T_propane_cond"]], y=[hydraulic_b["prop_dT_dP_FperPsi"]],
            mode="markers", name=f"Config B propane ({perf_b['T_propane_cond']:.0f}degF, {perf_b['P_prop_cond']:.0f} psia)",
            marker=dict(size=14, color="red", symbol="circle"),
        ))

        ratio_val = hydraulic_a["dT_dP_FperPsi"] / hydraulic_b["prop_dT_dP_FperPsi"] if hydraulic_b["prop_dT_dP_FperPsi"] > 0 else 0
        fig_sens.update_layout(
            title=f"Saturation Curve Sensitivity -- Isopentane is {ratio_val:.1f}x more sensitive than propane",
            xaxis_title="Condensing Temperature (degF)",
            yaxis_title="dT/dP (degF per psi)",
            height=450,
        )
        st.plotly_chart(fig_sens, width="stretch")

        # 3. Component detail table
        st.subheader("Component Hydraulic Detail")
        detail_rows = []
        for name, c in comp_a.items():
            pwr_impact = c["dT_F"] * dW_dT_a
            detail_rows.append({
                "Config": "A",
                "Component": name,
                "Type": c["type"],
                "dP (psi)": f"{c['dP_psi']:.4f}",
                "dT penalty (degF)": f"{c['dT_F']:.3f}",
                "Power impact (kW)": f"{pwr_impact:.1f}",
            })
        for name, c in comp_b.items():
            circuit = c.get("circuit", "iso")
            sensitivity = dW_dT_b
            pwr_impact = c["dT_F"] * sensitivity
            detail_rows.append({
                "Config": "B",
                "Component": name,
                "Type": f"{c['type']} ({circuit})",
                "dP (psi)": f"{c['dP_psi']:.4f}",
                "dT penalty (degF)": f"{c['dT_F']:.3f}",
                "Power impact (kW)": f"{pwr_impact:.1f}",
            })
        st.dataframe(pd.DataFrame(detail_rows).set_index("Config"), width="stretch")

        # Summary metrics
        hyd_met1, hyd_met2, hyd_met3, hyd_met4 = st.columns(4)
        hyd_met1.metric("Config A total dT", f"{hydraulic_a['total_dT_penalty_F']:.2f} degF")
        hyd_met2.metric("Config B total dT", f"{hydraulic_b['total_dT_penalty_F']:.2f} degF")
        hyd_met3.metric("Net advantage (B)", f"{net_hyd_advantage:.2f} degF")
        hyd_met4.metric("Power recovered", f"{abs(power_from_hyd):.0f} kW")

        # 4. Interactive ACC dP sweep
        st.subheader("ACC Tube Bundle dP Sweep")
        acc_dp_range = st.slider("ACC tube dP range (psi)", 0.0, 3.0, (0.2, 2.5), 0.1, key="acc_dp_sweep")

        acc_dp_vals = np.linspace(acc_dp_range[0], acc_dp_range[1], 15)
        power_a_sweep = []
        power_b_sweep_hyd = []

        for dp_val in acc_dp_vals:
            try:
                inp_sw_a = {**inputs, "dp_acc_tubes_a": float(dp_val)}
                r_sw_a = solve_config_a(inp_sw_a, fp)
                fan_sw_a = calculate_fan_power(r_sw_a["performance"]["Q_reject_mmbtu_hr"], inputs["T_ambient"], inp_sw_a)
                pwr_sw_a = compute_power_balance(r_sw_a["performance"], fan_sw_a, inp_sw_a, config="A")
                power_a_sweep.append(pwr_sw_a["P_net"])
            except Exception:
                power_a_sweep.append(float("nan"))
            try:
                inp_sw_b = {**inputs, "dp_acc_tubes_prop": float(dp_val)}
                r_sw_b = solve_config_b(inp_sw_b, fp)
                _ps_sw_b = r_sw_b["propane_states"]
                _Q_rej_sw_b = r_sw_b["performance"]["m_dot_prop"] * (_ps_sw_b["A"].h - _ps_sw_b["B"].h) / 1e6
                fan_sw_b = calculate_fan_power(_Q_rej_sw_b, inputs["T_ambient"], inp_sw_b)
                pwr_sw_b = compute_power_balance(r_sw_b["performance"], fan_sw_b, inp_sw_b, config="B")
                power_b_sweep_hyd.append(pwr_sw_b["P_net"])
            except Exception:
                power_b_sweep_hyd.append(float("nan"))

        fig_acc_sweep = go.Figure()
        fig_acc_sweep.add_trace(go.Scatter(
            x=list(acc_dp_vals), y=power_a_sweep,
            name="Config A", line=dict(color="blue", width=2.5),
        ))
        fig_acc_sweep.add_trace(go.Scatter(
            x=list(acc_dp_vals), y=power_b_sweep_hyd,
            name="Config B", line=dict(color="red", width=2.5),
        ))

        # Annotate sensitivity slopes
        valid_a = [(x, y) for x, y in zip(acc_dp_vals, power_a_sweep) if not np.isnan(y)]
        valid_b = [(x, y) for x, y in zip(acc_dp_vals, power_b_sweep_hyd) if not np.isnan(y)]
        if len(valid_a) >= 2:
            slope_a = (valid_a[-1][1] - valid_a[0][1]) / (valid_a[-1][0] - valid_a[0][0])
            fig_acc_sweep.add_annotation(
                x=(valid_a[0][0] + valid_a[-1][0]) / 2,
                y=max(y for _, y in valid_a),
                text=f"A slope: {slope_a:.1f} kW/psi",
                showarrow=False, font=dict(color="blue", size=12),
            )
        if len(valid_b) >= 2:
            slope_b = (valid_b[-1][1] - valid_b[0][1]) / (valid_b[-1][0] - valid_b[0][0])
            fig_acc_sweep.add_annotation(
                x=(valid_b[0][0] + valid_b[-1][0]) / 2,
                y=min(y for _, y in valid_b),
                text=f"B slope: {slope_b:.1f} kW/psi",
                showarrow=False, font=dict(color="red", size=12),
            )

        fig_acc_sweep.update_layout(
            title="Net Power vs ACC Tube Bundle dP",
            xaxis_title="ACC Tube Bundle dP (psi)",
            yaxis_title="Net Power (kW)",
            height=450,
        )
        st.plotly_chart(fig_acc_sweep, width="stretch")


    # -- Tab 7: Equipment Sizing Trade-offs --------------------------------------
    with tab7:
        st.subheader("Heat Exchanger Area vs Pinch Temperature")
        st.markdown("Holding duty constant at the current operating point, "
                    "varying pinch to show area and cost sensitivity.")

        pinch_sweep = np.linspace(5, 25, 21)
        m_a = perf_a["m_dot_iso"]
        m_b = perf_b["m_dot_iso"]

        # -- Config A heat exchangers (MMBtu/hr) --
        Q_vap_a = m_a * perf_a["q_vaporizer"] / 1e6
        Q_pre_a = m_a * perf_a["q_preheater"] / 1e6
        Q_rec_a = m_a * perf_a["q_recup"] / 1e6
        Q_cond_a = m_a * perf_a["q_cond"] / 1e6

        hx_sweeps_a = {
            "Vaporizer": hx_area_vs_pinch(
                Q_vap_a, U_VALUES["vaporizer"],
                inputs["T_geo_in"], perf_a["T_brine_mid"],
                states_a["7"].T, states_a["1"].T,
                pinch_sweep, inputs.get("uc_vaporizer_per_ft2", COST_FACTORS["vaporizer_per_ft2"])),
            "Preheater": hx_area_vs_pinch(
                Q_pre_a, U_VALUES["preheater"],
                perf_a["T_brine_mid"], perf_a["T_geo_out_calc"],
                states_a["6"].T, states_a["7"].T,
                pinch_sweep, inputs.get("uc_preheater_per_ft2", COST_FACTORS["preheater_per_ft2"])),
            "Recuperator": hx_area_vs_pinch(
                Q_rec_a, U_VALUES["recuperator"],
                states_a["2"].T, states_a["3"].T,
                states_a["5"].T, states_a["6"].T,
                pinch_sweep, inputs.get("uc_recuperator_per_ft2", COST_FACTORS["recup_per_ft2"])),
        }
        acc_sweep_a = acc_area_vs_pinch(Q_cond_a, inputs["T_ambient"], pinch_sweep,
                                        inputs.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"]))

        # -- Config B heat exchangers (MMBtu/hr) --
        Q_vap_b = m_b * perf_b["q_vaporizer"] / 1e6
        Q_pre_b = m_b * perf_b["q_preheater"] / 1e6
        Q_rec_b = m_b * perf_b["q_recup"] / 1e6
        Q_ihx_b = m_b * perf_b["q_cond_iso"] / 1e6
        m_prop = perf_b["m_dot_prop"]
        Q_acc_b = m_prop * (prop_states["A"].h - prop_states["B"].h) / 1e6

        hx_sweeps_b = {
            "Vaporizer": hx_area_vs_pinch(
                Q_vap_b, U_VALUES["vaporizer"],
                inputs["T_geo_in"], perf_b["T_brine_mid"],
                states_b["7"].T, states_b["1"].T,
                pinch_sweep, inputs.get("uc_vaporizer_per_ft2", COST_FACTORS["vaporizer_per_ft2"])),
            "Preheater": hx_area_vs_pinch(
                Q_pre_b, U_VALUES["preheater"],
                perf_b["T_brine_mid"], perf_b["T_geo_out_calc"],
                states_b["6"].T, states_b["7"].T,
                pinch_sweep, inputs.get("uc_preheater_per_ft2", COST_FACTORS["preheater_per_ft2"])),
            "Recuperator": hx_area_vs_pinch(
                Q_rec_b, U_VALUES["recuperator"],
                states_b["2"].T, states_b["3"].T,
                states_b["5"].T, states_b["6"].T,
                pinch_sweep, inputs.get("uc_recuperator_per_ft2", COST_FACTORS["recup_per_ft2"])),
            "Intermediate HX": hx_area_vs_pinch(
                Q_ihx_b, U_VALUES["intermediate_hx"],
                states_b["3"].T, states_b["4"].T,
                prop_states["C"].T, prop_states["A"].T,
                pinch_sweep, inputs.get("uc_ihx_per_ft2", COST_FACTORS["hx_per_ft2"])),
        }
        acc_sweep_b = acc_area_vs_pinch(Q_acc_b, inputs["T_ambient"], pinch_sweep,
                                        inputs.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"]))

        # Current operating pinch points for markers
        pinch_ops_a = {
            "Vaporizer": inputs["dt_pinch_vaporizer"],
            "Preheater": inputs["dt_pinch_preheater"],
            "Recuperator": inputs["dt_pinch_recup"],
        }
        pinch_ops_b = {
            "Vaporizer": inputs["dt_pinch_vaporizer"],
            "Preheater": inputs["dt_pinch_preheater"],
            "Recuperator": inputs["dt_pinch_recup"],
            "Intermediate HX": inputs["dt_approach_intermediate"],
        }

        # Plot HX area subplots
        hx_names_a = list(hx_sweeps_a.keys())
        hx_names_b = list(hx_sweeps_b.keys())
        all_hx_names = list(dict.fromkeys(hx_names_a + hx_names_b))

        fig_hx = make_subplots(
            rows=2, cols=len(all_hx_names),
            subplot_titles=[f"{n} (Area)" for n in all_hx_names]
                           + [f"{n} (Cost)" for n in all_hx_names],
            vertical_spacing=0.12, horizontal_spacing=0.06,
        )

        colors_cfg = {"A": "steelblue", "B": "indianred"}

        for i, name in enumerate(all_hx_names):
            col = i + 1
            # Config A
            if name in hx_sweeps_a:
                sw = hx_sweeps_a[name]
                p_vals = [d["pinch"] for d in sw]
                areas = [d["area_ft2"] for d in sw]
                costs_hx = [d["cost"] / 1e6 for d in sw]
                fig_hx.add_trace(go.Scatter(
                    x=p_vals, y=areas, name="A" if col == 1 else None,
                    legendgroup="A", showlegend=(col == 1),
                    line=dict(color=colors_cfg["A"], width=2),
                ), row=1, col=col)
                fig_hx.add_trace(go.Scatter(
                    x=p_vals, y=costs_hx, name=None,
                    legendgroup="A", showlegend=False,
                    line=dict(color=colors_cfg["A"], width=2),
                ), row=2, col=col)
                # Operating point marker
                op = pinch_ops_a.get(name)
                if op is not None:
                    op_area = costs_a.get(f"{name.lower().replace(' ', '_')}_area_ft2",
                                          costs_a.get("vaporizer_area_ft2", 0))
                    # Find the closest sweep point to mark
                    idx = min(range(len(p_vals)), key=lambda j: abs(p_vals[j] - op))
                    fig_hx.add_trace(go.Scatter(
                        x=[p_vals[idx]], y=[areas[idx]], mode="markers",
                        marker=dict(size=10, color=colors_cfg["A"], symbol="diamond"),
                        showlegend=False,
                    ), row=1, col=col)
                    fig_hx.add_trace(go.Scatter(
                        x=[p_vals[idx]], y=[costs_hx[idx]], mode="markers",
                        marker=dict(size=10, color=colors_cfg["A"], symbol="diamond"),
                        showlegend=False,
                    ), row=2, col=col)

            # Config B
            if name in hx_sweeps_b:
                sw = hx_sweeps_b[name]
                p_vals = [d["pinch"] for d in sw]
                areas = [d["area_ft2"] for d in sw]
                costs_hx = [d["cost"] / 1e6 for d in sw]
                fig_hx.add_trace(go.Scatter(
                    x=p_vals, y=areas, name="B" if col == 1 else None,
                    legendgroup="B", showlegend=(col == 1 and name in hx_sweeps_a),
                    line=dict(color=colors_cfg["B"], width=2, dash="dash"),
                ), row=1, col=col)
                fig_hx.add_trace(go.Scatter(
                    x=p_vals, y=costs_hx, name=None,
                    legendgroup="B", showlegend=False,
                    line=dict(color=colors_cfg["B"], width=2, dash="dash"),
                ), row=2, col=col)
                op = pinch_ops_b.get(name)
                if op is not None:
                    idx = min(range(len(p_vals)), key=lambda j: abs(p_vals[j] - op))
                    fig_hx.add_trace(go.Scatter(
                        x=[p_vals[idx]], y=[areas[idx]], mode="markers",
                        marker=dict(size=10, color=colors_cfg["B"], symbol="diamond"),
                        showlegend=False,
                    ), row=1, col=col)
                    fig_hx.add_trace(go.Scatter(
                        x=[p_vals[idx]], y=[costs_hx[idx]], mode="markers",
                        marker=dict(size=10, color=colors_cfg["B"], symbol="diamond"),
                        showlegend=False,
                    ), row=2, col=col)

        fig_hx.update_layout(
            height=600,
            title_text="Heat Exchanger Area & Cost vs Pinch Temperature (diamonds = current operating point)",
        )
        for i in range(len(all_hx_names)):
            fig_hx.update_xaxes(title_text="Pinch (degF)", row=2, col=i + 1)
        fig_hx.update_yaxes(title_text="Area (ft2)", row=1, col=1)
        fig_hx.update_yaxes(title_text="Cost ($MM)", row=2, col=1)
        st.plotly_chart(fig_hx, width="stretch")

        # ACC area vs pinch
        st.subheader("ACC Face Area vs Pinch Temperature")
        fig_acc_pinch = go.Figure()
        p_vals_acc_a = [d["pinch"] for d in acc_sweep_a]
        cost_acc_a = [d["cost"] / 1e6 for d in acc_sweep_a]
        area_acc_a = [d["area_ft2"] for d in acc_sweep_a]
        p_vals_acc_b = [d["pinch"] for d in acc_sweep_b]
        cost_acc_b = [d["cost"] / 1e6 for d in acc_sweep_b]

        fig_acc_pinch.add_trace(go.Scatter(
            x=p_vals_acc_a, y=cost_acc_a, name="Config A ACC",
            line=dict(color="steelblue", width=2),
        ))
        fig_acc_pinch.add_trace(go.Scatter(
            x=p_vals_acc_b, y=cost_acc_b, name="Config B Propane ACC",
            line=dict(color="indianred", width=2, dash="dash"),
        ))
        # Operating point markers
        acc_pinch_a_op = inputs["dt_pinch_acc_a"]
        acc_pinch_b_op = inputs["dt_pinch_acc_b"]
        idx_a = min(range(len(p_vals_acc_a)), key=lambda j: abs(p_vals_acc_a[j] - acc_pinch_a_op))
        idx_b = min(range(len(p_vals_acc_b)), key=lambda j: abs(p_vals_acc_b[j] - acc_pinch_b_op))
        fig_acc_pinch.add_trace(go.Scatter(
            x=[p_vals_acc_a[idx_a]], y=[cost_acc_a[idx_a]], mode="markers",
            marker=dict(size=12, color="steelblue", symbol="diamond"),
            name="A operating point",
        ))
        fig_acc_pinch.add_trace(go.Scatter(
            x=[p_vals_acc_b[idx_b]], y=[cost_acc_b[idx_b]], mode="markers",
            marker=dict(size=12, color="indianred", symbol="diamond"),
            name="B operating point",
        ))
        fig_acc_pinch.update_layout(
            xaxis_title="ACC Pinch (degF)", yaxis_title="Installed Cost ($MM)",
            height=400,
        )
        st.plotly_chart(fig_acc_pinch, width="stretch")

        # -- Duct diameter vs allowable dP --
        st.subheader("Duct Diameter & Cost vs Allowable Pressure Drop")

        dp_sweep = np.linspace(0.1, 2.0, 20)
        m_dot_a_lbs = perf_a["m_dot_iso"] / 3600
        m_dot_b_lbs = perf_b["m_dot_iso"] / 3600
        m_dot_prop_lbs = perf_b["m_dot_prop"] / 3600

        uc_duct = inputs.get("uc_iso_duct_per_ft2", COST_FACTORS["iso_duct_per_ft2"])
        uc_prop = inputs.get("uc_prop_pipe_per_ft2", COST_FACTORS["prop_pipe_per_ft2"])

        duct_curves = {}
        # Config A segments
        for seg in duct_a["segments"]:
            duct_curves[f"A: {seg['name']}"] = {
                "data": duct_diameter_vs_dp(
                    m_dot_a_lbs, seg["rho_lbft3"], seg["length_ft"],
                    inputs.get("f_darcy", 0.02), dp_sweep, uc_duct),
                "color": "steelblue",
                "current_dp": seg["delta_P_psi"],
                "current_dia": seg["diameter_in"],
            }
        # Config B segments
        for seg in duct_b["segments"]:
            is_prop = "Propane" in seg["name"] or "propane" in seg["name"]
            m_seg = m_dot_prop_lbs if is_prop else m_dot_b_lbs
            seg_uc = uc_prop if is_prop else uc_duct
            duct_curves[f"B: {seg['name']}"] = {
                "data": duct_diameter_vs_dp(
                    m_seg, seg["rho_lbft3"], seg["length_ft"],
                    inputs.get("f_darcy", 0.02), dp_sweep, seg_uc),
                "color": "indianred",
                "current_dp": seg["delta_P_psi"],
                "current_dia": seg["diameter_in"],
            }

        fig_duct_dp = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Diameter vs Allowable dP",
                                                    "Duct Cost vs Allowable dP"])
        for name, info in duct_curves.items():
            sw = info["data"]
            dp_vals = [d["dp_psi"] for d in sw]
            dia_vals = [d["diameter_in"] for d in sw]
            cost_vals = [d["cost"] / 1e3 for d in sw]

            fig_duct_dp.add_trace(go.Scatter(
                x=dp_vals, y=dia_vals, name=name,
                line=dict(color=info["color"], width=1.5),
            ), row=1, col=1)
            fig_duct_dp.add_trace(go.Scatter(
                x=dp_vals, y=cost_vals, name=name, showlegend=False,
                line=dict(color=info["color"], width=1.5),
            ), row=1, col=2)

            # Operating point
            fig_duct_dp.add_trace(go.Scatter(
                x=[info["current_dp"]], y=[info["current_dia"]], mode="markers",
                marker=dict(size=8, color=info["color"], symbol="diamond"),
                showlegend=False,
            ), row=1, col=1)

        fig_duct_dp.update_layout(height=450, title_text="Duct Sizing vs Allowable dP (diamonds = current)")
        fig_duct_dp.update_xaxes(title_text="Allowable dP (psi)", row=1, col=1)
        fig_duct_dp.update_xaxes(title_text="Allowable dP (psi)", row=1, col=2)
        fig_duct_dp.update_yaxes(title_text="Diameter (in)", row=1, col=1)
        fig_duct_dp.update_yaxes(title_text="Cost ($k)", row=1, col=2)
        st.plotly_chart(fig_duct_dp, width="stretch")

        # -- ACC tube bundle model --
        st.subheader("ACC Tube Bundle: Face Area & Cost vs Allowable dP")

        acc_tube_a = acc_tubes_vs_dp(m_dot_a_lbs, states_a["3"].rho, dp_sweep)
        acc_tube_b_prop = acc_tubes_vs_dp(m_dot_prop_lbs, prop_states["A"].rho, dp_sweep)

        fig_acc_tube = make_subplots(rows=1, cols=2,
                                     subplot_titles=["Face Area vs dP", "Cost vs dP"])

        dp_a_vals = [d["dp_psi"] for d in acc_tube_a]
        fa_a = [d["face_area_ft2"] for d in acc_tube_a]
        c_a = [d["cost"] / 1e3 for d in acc_tube_a]
        dp_b_vals = [d["dp_psi"] for d in acc_tube_b_prop]
        fa_b = [d["face_area_ft2"] for d in acc_tube_b_prop]
        c_b = [d["cost"] / 1e3 for d in acc_tube_b_prop]

        fig_acc_tube.add_trace(go.Scatter(
            x=dp_a_vals, y=fa_a, name="A: ISO ACC tubes",
            line=dict(color="steelblue", width=2),
        ), row=1, col=1)
        fig_acc_tube.add_trace(go.Scatter(
            x=dp_b_vals, y=fa_b, name="B: Propane ACC tubes",
            line=dict(color="indianred", width=2),
        ), row=1, col=1)
        fig_acc_tube.add_trace(go.Scatter(
            x=dp_a_vals, y=c_a, name="A: ISO ACC tubes", showlegend=False,
            line=dict(color="steelblue", width=2),
        ), row=1, col=2)
        fig_acc_tube.add_trace(go.Scatter(
            x=dp_b_vals, y=c_b, name="B: Propane ACC tubes", showlegend=False,
            line=dict(color="indianred", width=2),
        ), row=1, col=2)

        # Mark current operating dP
        op_dp_a = inputs.get("dp_acc_tubes_a", 0.5)
        op_dp_b = inputs.get("dp_acc_tubes_prop", 1.0)
        idx_opa = min(range(len(dp_a_vals)), key=lambda j: abs(dp_a_vals[j] - op_dp_a))
        idx_opb = min(range(len(dp_b_vals)), key=lambda j: abs(dp_b_vals[j] - op_dp_b))
        fig_acc_tube.add_trace(go.Scatter(
            x=[dp_a_vals[idx_opa]], y=[fa_a[idx_opa]], mode="markers",
            marker=dict(size=10, color="steelblue", symbol="diamond"),
            showlegend=False,
        ), row=1, col=1)
        fig_acc_tube.add_trace(go.Scatter(
            x=[dp_b_vals[idx_opb]], y=[fa_b[idx_opb]], mode="markers",
            marker=dict(size=10, color="indianred", symbol="diamond"),
            showlegend=False,
        ), row=1, col=1)

        fig_acc_tube.update_layout(
            height=400,
            title_text=(f"ACC Tube Bundle Model (f={ACC_TUBE_DEFAULTS['f_tube']}, "
                        f"L={ACC_TUBE_DEFAULTS['L_tube_ft']}ft, "
                        f"D={ACC_TUBE_DEFAULTS['D_tube_in']}in, "
                        f"N_rows={ACC_TUBE_DEFAULTS['N_rows']})"),
        )
        fig_acc_tube.update_xaxes(title_text="Allowable dP (psi)")
        fig_acc_tube.update_yaxes(title_text="Face Area (ft2)", row=1, col=1)
        fig_acc_tube.update_yaxes(title_text="Cost ($k)", row=1, col=2)
        st.plotly_chart(fig_acc_tube, width="stretch")

        # -- Overall Trade-off: Cost vs Power Recovery --
        st.subheader("Overall Trade-off: Spend More on Equipment to Recover Power")
        st.markdown(
            "Scaling all pinch points and dP allowances together from 2x (loose) "
            "to 0.25x (tight) relative to current values. Shows implicit cost of "
            "additional capacity ($/kW)."
        )

        with st.spinner("Running trade-off sweeps..."):
            tradeoff_a = sizing_tradeoff_sweep(inputs, fp, config="A")
            tradeoff_b = sizing_tradeoff_sweep(inputs, fp, config="B")

        fig_tradeoff = make_subplots(
            rows=1, cols=2,
            subplot_titles=["Additional Cost vs Additional Power",
                            "Marginal Cost of Electricity"],
        )

        # Filter to valid points
        valid_a = [r for r in tradeoff_a
                   if not np.isnan(r["delta_cost"]) and not np.isnan(r["delta_power_kw"])]
        valid_b = [r for r in tradeoff_b
                   if not np.isnan(r["delta_cost"]) and not np.isnan(r["delta_power_kw"])]

        if valid_a:
            fig_tradeoff.add_trace(go.Scatter(
                x=[r["delta_power_kw"] for r in valid_a],
                y=[r["delta_cost"] / 1e6 for r in valid_a],
                name="Config A",
                mode="lines+markers",
                line=dict(color="steelblue", width=2),
                marker=dict(size=5),
                text=[f"mult={r['multiplier']:.2f}" for r in valid_a],
            ), row=1, col=1)
            # Operating point at mult ~1.0
            base_a = min(valid_a, key=lambda r: abs(r["multiplier"] - 1.0))
            fig_tradeoff.add_trace(go.Scatter(
                x=[base_a["delta_power_kw"]], y=[base_a["delta_cost"] / 1e6],
                mode="markers",
                marker=dict(size=12, color="steelblue", symbol="diamond"),
                showlegend=False,
            ), row=1, col=1)

        if valid_b:
            fig_tradeoff.add_trace(go.Scatter(
                x=[r["delta_power_kw"] for r in valid_b],
                y=[r["delta_cost"] / 1e6 for r in valid_b],
                name="Config B",
                mode="lines+markers",
                line=dict(color="indianred", width=2),
                marker=dict(size=5),
                text=[f"mult={r['multiplier']:.2f}" for r in valid_b],
            ), row=1, col=1)
            base_b = min(valid_b, key=lambda r: abs(r["multiplier"] - 1.0))
            fig_tradeoff.add_trace(go.Scatter(
                x=[base_b["delta_power_kw"]], y=[base_b["delta_cost"] / 1e6],
                mode="markers",
                marker=dict(size=12, color="indianred", symbol="diamond"),
                showlegend=False,
            ), row=1, col=1)

        # Marginal cost of electricity ($/kW capacity, annualized)
        # Convert to $/MWh: marginal_cost_per_kw / (8760 * CF * annuity_factor) * 1000
        cf = inputs["capacity_factor"] / 100
        r_disc = inputs["discount_rate"] / 100
        n_life = inputs["project_life"]
        if r_disc > 0:
            ann_factor = (1 - (1 + r_disc) ** (-n_life)) / r_disc
        else:
            ann_factor = n_life
        hrs_factor = 8760 * cf * ann_factor / 1000  # MWh-equiv per kW over lifetime

        marginal_a = [r for r in valid_a if r["delta_power_kw"] > 0.1
                      and not np.isnan(r["marginal_cost_per_kw"])]
        marginal_b = [r for r in valid_b if r["delta_power_kw"] > 0.1
                      and not np.isnan(r["marginal_cost_per_kw"])]

        if marginal_a:
            fig_tradeoff.add_trace(go.Scatter(
                x=[r["delta_power_kw"] for r in marginal_a],
                y=[r["marginal_cost_per_kw"] / hrs_factor for r in marginal_a],
                name="A marginal $/MWh", mode="lines",
                line=dict(color="steelblue", width=2, dash="dot"),
            ), row=1, col=2)

        if marginal_b:
            fig_tradeoff.add_trace(go.Scatter(
                x=[r["delta_power_kw"] for r in marginal_b],
                y=[r["marginal_cost_per_kw"] / hrs_factor for r in marginal_b],
                name="B marginal $/MWh", mode="lines",
                line=dict(color="indianred", width=2, dash="dot"),
            ), row=1, col=2)

        # Reference line: electricity price
        if marginal_a or marginal_b:
            all_delta_kw = ([r["delta_power_kw"] for r in marginal_a]
                            + [r["delta_power_kw"] for r in marginal_b])
            if all_delta_kw:
                fig_tradeoff.add_hline(
                    y=inputs["electricity_price"], line_dash="dash",
                    line_color="orange",
                    annotation_text=f"Elec price: ${inputs['electricity_price']}/MWh",
                    row=1, col=2,
                )

        fig_tradeoff.update_layout(height=500)
        fig_tradeoff.update_xaxes(title_text="Additional Power (kW)", row=1, col=1)
        fig_tradeoff.update_xaxes(title_text="Additional Power (kW)", row=1, col=2)
        fig_tradeoff.update_yaxes(title_text="Additional Cost ($MM)", row=1, col=1)
        fig_tradeoff.update_yaxes(title_text="Marginal Cost ($/MWh)", row=1, col=2)
        st.plotly_chart(fig_tradeoff, width="stretch")

        # Summary interpretation
        if valid_a and valid_b:
            tight_a = min(valid_a, key=lambda r: r["multiplier"])
            tight_b = min(valid_b, key=lambda r: r["multiplier"])
            st.markdown(f"""
**Interpretation:** Tightening all pinch points and dP allowances to 0.25x current values:
- Config A: **{tight_a['delta_power_kw']:+.0f} kW** for **${tight_a['delta_cost']/1e6:+.2f}MM** additional cost
- Config B: **{tight_b['delta_power_kw']:+.0f} kW** for **${tight_b['delta_cost']/1e6:+.2f}MM** additional cost

Invest in tighter heat exchangers and ducts where the marginal cost curve is
**below** the electricity price line -- those kW pay for themselves over
the plant lifetime.
""")

    # -- Tab 8: Fan Power & ACC Optimization -----------------------------------
    with tab8:
        st.subheader("Fan Power & ACC Area vs Air Temperature Rise")

        # Sweep dT_air from 15 to 40
        dT_air_sweep = np.linspace(15, 40, 26)
        sweep_fan_a = []
        sweep_fan_b = []
        sweep_acc_a = []
        sweep_acc_b = []

        T_cond_a = perf_a["T_cond"]
        T_cond_b = perf_b["T_propane_cond"]

        for dt_sweep in dT_air_sweep:
            inp_sweep = {**inputs, "dT_air": float(dt_sweep)}
            sf_a = calculate_fan_power(perf_a["Q_reject_mmbtu_hr"], inputs["T_ambient"], inp_sweep)
            sf_b = calculate_fan_power(Q_reject_air_b, inputs["T_ambient"], inp_sweep)
            sweep_fan_a.append(sf_a["W_fans_kw"])
            sweep_fan_b.append(sf_b["W_fans_kw"])
            sweep_acc_a.append(acc_area_with_air_rise(
                perf_a["Q_reject_mmbtu_hr"], T_cond_a, inputs["T_ambient"], float(dt_sweep)))
            sweep_acc_b.append(acc_area_with_air_rise(
                Q_reject_air_b, T_cond_b, inputs["T_ambient"], float(dt_sweep)))

        fig_fan_acc = make_subplots(rows=1, cols=2,
                                    subplot_titles=["Fan Power vs dT_air", "ACC Area vs dT_air"])

        fig_fan_acc.add_trace(go.Scatter(
            x=dT_air_sweep, y=sweep_fan_a, name="Config A fan",
            mode="lines", line=dict(color="steelblue", width=2),
        ), row=1, col=1)
        fig_fan_acc.add_trace(go.Scatter(
            x=dT_air_sweep, y=sweep_fan_b, name="Config B fan",
            mode="lines", line=dict(color="indianred", width=2),
        ), row=1, col=1)

        # Mark current operating point
        fig_fan_acc.add_trace(go.Scatter(
            x=[inputs["dT_air"]], y=[fan_a["W_fans_kw"]],
            mode="markers", marker=dict(size=12, color="steelblue", symbol="diamond"),
            name="A design", showlegend=False,
        ), row=1, col=1)
        fig_fan_acc.add_trace(go.Scatter(
            x=[inputs["dT_air"]], y=[fan_b["W_fans_kw"]],
            mode="markers", marker=dict(size=12, color="indianred", symbol="diamond"),
            name="B design", showlegend=False,
        ), row=1, col=1)

        fig_fan_acc.add_trace(go.Scatter(
            x=dT_air_sweep, y=[a / 1000 for a in sweep_acc_a], name="Config A ACC",
            mode="lines", line=dict(color="steelblue", width=2, dash="dash"),
        ), row=1, col=2)
        fig_fan_acc.add_trace(go.Scatter(
            x=dT_air_sweep, y=[a / 1000 for a in sweep_acc_b], name="Config B ACC",
            mode="lines", line=dict(color="indianred", width=2, dash="dash"),
        ), row=1, col=2)

        fig_fan_acc.update_xaxes(title_text="Air Temp Rise dT_air (°F)", row=1, col=1)
        fig_fan_acc.update_xaxes(title_text="Air Temp Rise dT_air (°F)", row=1, col=2)
        fig_fan_acc.update_yaxes(title_text="Fan Power (kW)", row=1, col=1)
        fig_fan_acc.update_yaxes(title_text="ACC Area (1000 ft²)", row=1, col=2)
        fig_fan_acc.update_layout(height=450)
        st.plotly_chart(fig_fan_acc, use_container_width=True)

        # --- Annualized total cost vs dT_air ---
        st.subheader("Annualized Total Cost vs Air Temperature Rise")

        elec_price = inputs["electricity_price"]  # $/MWh
        cf_val = inputs["capacity_factor"]
        r_val = inputs["discount_rate"]
        n_val = inputs["project_life"]
        if r_val > 0:
            ann_fac = (1 - (1 + r_val) ** (-n_val)) / r_val
        else:
            ann_fac = n_val
        uc_acc_val = inputs.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"])

        annual_fan_cost_a = []
        annual_fan_cost_b = []
        annual_acc_cost_a = []
        annual_acc_cost_b = []
        total_ann_a = []
        total_ann_b = []

        for i, dt_sweep in enumerate(dT_air_sweep):
            # Annual fan electricity cost: fan_kW * 8760 * CF * $/MWh / 1000
            fan_annual_a = sweep_fan_a[i] * 8760 * cf_val * elec_price / 1000
            fan_annual_b = sweep_fan_b[i] * 8760 * cf_val * elec_price / 1000
            # ACC capital annualized: area * uc / annuity_factor
            acc_ann_a = sweep_acc_a[i] * uc_acc_val / ann_fac
            acc_ann_b = sweep_acc_b[i] * uc_acc_val / ann_fac
            annual_fan_cost_a.append(fan_annual_a)
            annual_fan_cost_b.append(fan_annual_b)
            annual_acc_cost_a.append(acc_ann_a)
            annual_acc_cost_b.append(acc_ann_b)
            total_ann_a.append(fan_annual_a + acc_ann_a)
            total_ann_b.append(fan_annual_b + acc_ann_b)

        # Find optimal dT_air for each config
        opt_idx_a = int(np.argmin(total_ann_a))
        opt_idx_b = int(np.argmin(total_ann_b))
        opt_dt_a = dT_air_sweep[opt_idx_a]
        opt_dt_b = dT_air_sweep[opt_idx_b]

        fig_ann = go.Figure()
        fig_ann.add_trace(go.Scatter(
            x=dT_air_sweep, y=[t / 1000 for t in total_ann_a],
            name="Config A total", mode="lines",
            line=dict(color="steelblue", width=2),
        ))
        fig_ann.add_trace(go.Scatter(
            x=dT_air_sweep, y=[t / 1000 for t in total_ann_b],
            name="Config B total", mode="lines",
            line=dict(color="indianred", width=2),
        ))

        # Optimal markers
        fig_ann.add_trace(go.Scatter(
            x=[opt_dt_a], y=[total_ann_a[opt_idx_a] / 1000],
            mode="markers", marker=dict(size=14, color="steelblue", symbol="star"),
            name=f"A optimal ({opt_dt_a:.0f}°F)",
        ))
        fig_ann.add_trace(go.Scatter(
            x=[opt_dt_b], y=[total_ann_b[opt_idx_b] / 1000],
            mode="markers", marker=dict(size=14, color="indianred", symbol="star"),
            name=f"B optimal ({opt_dt_b:.0f}°F)",
        ))

        fig_ann.update_layout(
            title="Annualized Cost: Fan Electricity + ACC Capital",
            xaxis_title="Air Temp Rise dT_air (°F)",
            yaxis_title="Annualized Cost ($k/yr)",
            height=400,
        )
        st.plotly_chart(fig_ann, use_container_width=True)

        # --- Fan Sizing Summary Table ---
        st.subheader("Fan Sizing Summary")
        fan_summary = pd.DataFrame([
            {
                "Parameter": "Q rejection (MMBtu/hr)",
                "Config A": f"{perf_a['Q_reject_mmbtu_hr']:.2f}",
                "Config B": f"{Q_reject_air_b:.2f}",
            },
            {
                "Parameter": "Air mass flow (klb/hr)",
                "Config A": f"{fan_a['m_dot_air_lb_hr']/1000:.0f}",
                "Config B": f"{fan_b['m_dot_air_lb_hr']/1000:.0f}",
            },
            {
                "Parameter": "Vol. flow (ft³/s)",
                "Config A": f"{fan_a['vol_flow_ft3s']:.0f}",
                "Config B": f"{fan_b['vol_flow_ft3s']:.0f}",
            },
            {
                "Parameter": "Fan power (kW)",
                "Config A": f"{fan_a['W_fans_kw']:.0f}",
                "Config B": f"{fan_b['W_fans_kw']:.0f}",
            },
            {
                "Parameter": "Fan bays (required)",
                "Config A": f"{fan_a['n_fans_required']}",
                "Config B": f"{fan_b['n_fans_required']}",
            },
            {
                "Parameter": "Fan bays (used)",
                "Config A": f"{fan_a['n_fans_used']}",
                "Config B": f"{fan_b['n_fans_used']}",
            },
            {
                "Parameter": "Fan diameter (ft)",
                "Config A": f"{inputs['fan_diameter_ft']}",
                "Config B": f"{inputs['fan_diameter_ft']}",
            },
            {
                "Parameter": "Auxiliary (kW)",
                "Config A": f"{inputs['W_aux_kw']:.0f}",
                "Config B": f"{inputs['W_aux_kw']:.0f}",
            },
        ]).set_index("Parameter")
        st.dataframe(fan_summary, use_container_width=True)

        # --- Pump Sizing Summary Table ---
        st.subheader("Pump Sizing Summary")
        pump_summary_rows = [
            {
                "Pump": "ISO pump (A)",
                "Flow (gpm)": f"{pump_iso_a['flow_gpm']:.1f}",
                "dP (psi)": f"{pump_iso_a['dP_psi']:.0f}",
                "Power (kW)": f"{pump_iso_a['power_kw']:.1f}",
                "Power (HP)": f"{pump_iso_a['power_hp']:.0f}",
            },
            {
                "Pump": "ISO pump (B)",
                "Flow (gpm)": f"{pump_iso_b['flow_gpm']:.1f}",
                "dP (psi)": f"{pump_iso_b['dP_psi']:.0f}",
                "Power (kW)": f"{pump_iso_b['power_kw']:.1f}",
                "Power (HP)": f"{pump_iso_b['power_hp']:.0f}",
            },
            {
                "Pump": "Propane pump (B)" + (" -- thermosiphon" if parasitic_b["thermosiphon"] else ""),
                "Flow (gpm)": f"{pump_prop_b['flow_gpm']:.1f}" if not parasitic_b["thermosiphon"] else "N/A",
                "dP (psi)": f"{pump_prop_b['dP_psi']:.0f}",
                "Power (kW)": "0 (thermosiphon)" if parasitic_b["thermosiphon"] else f"{pump_prop_b['power_kw']:.1f}",
                "Power (HP)": "0" if parasitic_b["thermosiphon"] else f"{pump_prop_b['power_hp']:.0f}",
            },
        ]
        st.dataframe(pd.DataFrame(pump_summary_rows).set_index("Pump"), use_container_width=True)

        # --- Interpretation ---
        st.markdown(f"""
**Interpretation:**
- Optimal air temperature rise: **{opt_dt_a:.0f}°F** (Config A), **{opt_dt_b:.0f}°F** (Config B)
- Current design dT_air = **{inputs['dT_air']}°F**
- Config A parasitic: **{parasitic_a['total_kw']:.0f} kW** ({pct_para_a:.1f}% of gross)
- Config B parasitic: **{parasitic_b['total_kw']:.0f} kW** ({pct_para_b:.1f}% of gross)
- Lower dT_air = more ACC area (capital) but less fan power (operating cost)
- Higher dT_air = less ACC area but more fan power; star markers show the minimum total annualized cost
""")


    # ============================================================================
    # SECTION 5: ASSUMPTIONS AND SOFTWARE INFORMATION
    # ============================================================================

    st.divider()

    with st.expander("Assumptions and Software Information"):
        assum_col1, assum_col2 = st.columns(2)

        with assum_col1:
            st.markdown("**Heat Transfer Coefficients (U-values)**")
            u_table = []
            for name, val in U_VALUES.items():
                u_table.append({"Component": name.replace("_", " ").title(), "U (BTU/hr-ft2-degF)": val})
            st.dataframe(pd.DataFrame(u_table).set_index("Component"), width="stretch")

            st.markdown("**Unit Cost Factors (2024 USD, installed)**")
            uc_display = [
                ("Vaporizer", f"${inputs.get('uc_vaporizer_per_ft2', COST_FACTORS['vaporizer_per_ft2'])}/ft2"),
                ("Preheater", f"${inputs.get('uc_preheater_per_ft2', COST_FACTORS['preheater_per_ft2'])}/ft2"),
                ("Recuperator", f"${inputs.get('uc_recuperator_per_ft2', COST_FACTORS['recup_per_ft2'])}/ft2"),
                ("IHX", f"${inputs.get('uc_ihx_per_ft2', COST_FACTORS['hx_per_ft2'])}/ft2"),
                ("ACC", f"${inputs.get('uc_acc_per_ft2', COST_FACTORS['acc_per_ft2'])}/ft2"),
                ("Iso Duct", f"${inputs.get('uc_iso_duct_per_ft2', COST_FACTORS['iso_duct_per_ft2'])}/ft2"),
                ("Propane Pipe", f"${inputs.get('uc_prop_pipe_per_ft2', COST_FACTORS['prop_pipe_per_ft2'])}/ft2"),
                ("Turbine", f"${inputs.get('uc_turbine_per_kw', COST_FACTORS['turbine_per_kw'])}/kW"),
                ("Steel", f"${inputs.get('uc_steel_per_lb', COST_FACTORS['steel_per_lb']):.2f}/lb"),
            ]
            cf_table = [{"Factor": n, "Value": v} for n, v in uc_display]
            st.dataframe(pd.DataFrame(cf_table).set_index("Factor"), width="stretch")

            st.markdown("**Indirect Cost Percentages**")
            indirect_display = [
                ("Foundation", f"{inputs.get('uc_foundation_pct', COST_FACTORS['foundation_pct'])}%"),
                ("Engineering & Procurement", f"{inputs.get('uc_engineering_pct', COST_FACTORS['engineering_pct'])}%"),
                ("Construction Mgmt", f"{inputs.get('uc_construction_mgmt_pct', COST_FACTORS['construction_mgmt_pct'])}%"),
                ("Contingency", f"{inputs.get('uc_contingency_pct', COST_FACTORS['contingency_pct'])}%"),
                ("Propane System", f"{inputs.get('uc_prop_piping_pct', COST_FACTORS['prop_piping_pct'])}% of IHX"),
            ]
            ind_table = [{"Item": n, "Rate": v} for n, v in indirect_display]
            st.dataframe(pd.DataFrame(ind_table).set_index("Item"), width="stretch")

        with assum_col2:
            st.markdown("**Key Assumptions**")
            st.markdown(f"""
- Working fluid: Isopentane (Config A & B)
- Heat rejection fluid: Propane (Config B only)
- Duct friction factor: f = {inputs['f_darcy']:.3f} (Darcy-Weisbach)
- Equipment-internal dP modeled for ACC tubes, recuperator, IHX
- dT/dP from CoolProp saturation curve (central finite difference)
- Cycle convergence tolerance: 0.1 degF on condensing temperature
- Brine modeled as constant-cp fluid
- All costs are installed costs (2024 USD)
- ACC fan power from first-principles airflow model (Cp_air=0.24, face vel=400 fpm)
{"- Propane loop designed as thermosiphon -- ACC elevated above IHX, gravity drives liquid return, no pump required" if inputs.get("prop_thermosiphon", True) else "- Propane loop uses mechanical pump for liquid return"}
- Recuperator modeled with constant-dT pinch
""")

            st.markdown("**Software**")
            try:
                import CoolProp
                cp_version = CoolProp.__version__
            except Exception:
                cp_version = "unknown"

            st.markdown(f"""
- Thermodynamic properties: CoolProp v{cp_version}
- REFPROP bridge: {'Connected' if fp.use_refprop else 'Not available (using CoolProp)'}
- UI framework: Streamlit
- Charts: Plotly
""")

        st.caption("ORC Comparator v5.3 -- Parasitic load model, thermosiphon option, fan power optimization")


# ============================================================================
# STANDALONE MODE
# ============================================================================

if __name__ == "__main__":
    st.set_page_config(page_title="ORC Comparator", layout="wide")
    inputs = build_analysis_sidebar()
    render_analysis_tab(inputs)
