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
    optimize_approach_temp, construction_schedule_delta, simple_payback,
    schedule_savings_npv, COST_FACTORS, U_VALUES,
    hx_area_vs_pinch, acc_area_vs_pinch, duct_diameter_vs_dp,
    acc_tubes_vs_dp, sizing_tradeoff_sweep, ACC_TUBE_DEFAULTS,
    _duct_segment_cost,
)

st.set_page_config(page_title="ORC Comparator", layout="wide")

# ============================================================================
# SESSION STATE -- CHAT & APPLY
# ============================================================================

if "chat_messages" not in st.session_state:
    st.session_state.chat_messages = []
if "pending_apply" not in st.session_state:
    st.session_state.pending_apply = {}

# Apply pending unit cost overrides from Claude recommendations.
# This runs BEFORE widgets render so session_state values take effect.
if st.session_state.pending_apply:
    for widget_key, value in st.session_state.pending_apply.items():
        st.session_state[widget_key] = value
    st.session_state.pending_apply = {}


@st.cache_resource
def get_fluid_props():
    return FluidProperties()


fp = get_fluid_props()

# ============================================================================
# PAGE TITLE
# ============================================================================

st.title("ORC Configuration Comparator")
st.caption("Config A (Direct ACC) vs Config B (Propane Intermediate Loop) -- Geothermal Binary Cycle")

# ============================================================================
# SIDEBAR -- INPUTS
# ============================================================================

with st.sidebar:
    st.header("Inputs")

    with st.form("input_form"):
        with st.expander("Brine Inputs", expanded=True):
            T_geo_in = st.number_input("Brine inlet temperature (degF)", 200, 500, 300, 5)
            m_dot_geo = st.number_input("Brine mass flow rate (lb/s)", 10, 2000, 200, 10)
            cp_brine = st.number_input("Brine specific heat (BTU/lb-degF)", 0.5, 1.5, 1.0, 0.05)
            T_geo_out_min = st.number_input("Min brine outlet temperature (degF)", 80, 250, 160, 5,
                                            help="Silica/scaling constraint")

        with st.expander("Cycle Parameters"):
            eta_turbine = st.slider("Turbine isentropic efficiency", 0.60, 0.95, 0.82, 0.01)
            eta_pump = st.slider("Pump isentropic efficiency", 0.50, 0.95, 0.75, 0.01)
            superheat = st.number_input("Turbine inlet superheat (degF above sat)", 0, 50, 0, 1)
            st.info("Isopentane circulation rate is solved from brine energy balance.")

        with st.expander("Ambient Conditions"):
            T_ambient = st.number_input("Ambient dry bulb temperature (degF)", 50, 130, 95, 5)

        with st.expander("Pinch Points"):
            dt_pinch_acc_a = st.number_input("ACC pinch Config A (degF)", 5, 50, 15, 1)
            dt_pinch_acc_b = st.number_input("ACC pinch Config B (degF)", 5, 50, 15, 1)
            dt_pinch_vaporizer = st.number_input("Vaporizer pinch (degF)", 3, 30, 10, 1)
            dt_pinch_preheater = st.number_input("Preheater pinch (degF)", 3, 30, 10, 1)
            dt_pinch_recup = st.number_input("Recuperator pinch (degF)", 5, 40, 15, 1)
            dt_approach_intermediate = st.number_input(
                "Intermediate HX approach (degF, Config B)", 5, 25, 10, 1)

        with st.expander("Duct Parameters"):
            v_tailpipe = st.number_input("Tailpipe vapor velocity (ft/s)", 5, 25, 10, 1)
            v_acc_header = st.number_input("ACC header vapor velocity (ft/s)", 5, 25, 15, 1)
            L_tailpipe_a = st.number_input("Tailpipe length Config A (ft)", 10, 100, 30, 5)
            L_long_header = st.number_input("Long vapor header length (ft)", 50, 500, 120, 10)
            L_acc_header = st.number_input("ACC distribution header length (ft)", 50, 500, 200, 10)

        with st.expander("Hydraulic Parameters"):
            f_darcy = st.number_input("Darcy friction factor f", 0.015, 0.030, 0.020, 0.001, format="%.3f")
            st.markdown("**Config A Equipment dP (psi)**")
            dp_acc_tubes_a = st.number_input("ACC tube bundle dP", 0.0, 3.0, 0.5, 0.1, key="dp_acc_tubes_a")
            dp_acc_headers_a = st.number_input("ACC vapor headers dP", 0.0, 3.0, 0.3, 0.1, key="dp_acc_headers_a")
            dp_recup_a = st.number_input("Recuperator dP (A)", 0.0, 3.0, 0.3, 0.1, key="dp_recup_a")
            st.markdown("**Config B Isopentane Side dP (psi)**")
            dp_ihx_iso = st.number_input("IHX iso side dP", 0.0, 3.0, 0.5, 0.1, key="dp_ihx_iso")
            dp_recup_b = st.number_input("Recuperator dP (B)", 0.0, 3.0, 0.3, 0.1, key="dp_recup_b")
            dp_tailpipe_iso_b = st.number_input("ISO tailpipe dP (B)", 0.0, 3.0, 0.3, 0.1, key="dp_tailpipe_iso_b")
            st.markdown("**Config B Propane Side dP (psi)**")
            dp_acc_tubes_prop = st.number_input("ACC tube bundle dP (prop)", 0.0, 5.0, 1.0, 0.1, key="dp_acc_tubes_prop")
            dp_ihx_prop = st.number_input("IHX propane side dP", 0.0, 3.0, 0.5, 0.1, key="dp_ihx_prop")

        with st.expander("Economic Parameters"):
            electricity_price = st.number_input("Electricity price ($/MWh)", 10, 200, 35, 5)
            capacity_factor = st.slider("Capacity factor (%)", 70, 100, 95, 1)
            discount_rate = st.slider("Discount rate (%)", 3, 15, 8, 1)
            project_life = st.number_input("Plant life (years)", 10, 40, 30, 1)

        with st.expander("Unit Cost Assumptions (2024 USD Installed)"):
            st.markdown("**Heat Exchanger Costs**")
            uc_vaporizer_per_ft2 = st.number_input(
                "Vaporizer ($/ft2)", 20, 60, 35, 1, key="uc_vaporizer")
            uc_preheater_per_ft2 = st.number_input(
                "Preheater ($/ft2)", 15, 50, 30, 1, key="uc_preheater")
            uc_recuperator_per_ft2 = st.number_input(
                "Recuperator ($/ft2)", 15, 45, 25, 1, key="uc_recuperator")
            uc_ihx_per_ft2 = st.number_input(
                "IHX ($/ft2, pressure rated)", 25, 70, 40, 1, key="uc_ihx")
            uc_acc_per_ft2 = st.number_input(
                "ACC ($/ft2 face area)", 8, 20, 12, 1, key="uc_acc")

            st.markdown("**Duct and Piping Costs**")
            uc_iso_duct_per_ft2 = st.number_input(
                "Iso duct ($/ft2 surface)", 100, 300, 180, 5,
                key="uc_iso_duct",
                help="Diameter multiplier: >72\" x1.7, >60\" x1.4")
            uc_prop_pipe_per_ft2 = st.number_input(
                "Propane pipe ($/ft2 surface)", 80, 200, 120, 5,
                key="uc_prop_pipe")
            uc_prop_piping_pct = st.number_input(
                "Propane system (% of IHX cost)", 10, 35, 20, 1,
                key="uc_prop_pct")

            st.markdown("**Turbine and Electrical**")
            uc_turbine_per_kw = st.number_input(
                "Turbine-generator ($/kW)", 800, 1800, 1200, 50,
                key="uc_turbine")

            st.markdown("**Civil and Structural**")
            uc_steel_per_lb = st.number_input(
                "Structural steel ($/lb)", 3.0, 7.0, 4.5, 0.25,
                key="uc_steel", format="%.2f")
            uc_foundation_pct = st.number_input(
                "Foundation (% of equipment)", 5, 15, 8, 1,
                key="uc_foundation")

            st.markdown("**Indirect Costs**")
            uc_engineering_pct = st.number_input(
                "Engineering & procurement (%)", 8, 18, 12, 1,
                key="uc_eng")
            uc_construction_mgmt_pct = st.number_input(
                "Construction management (%)", 5, 12, 8, 1,
                key="uc_cm")
            uc_contingency_pct = st.number_input(
                "Contingency (%)", 10, 25, 15, 1,
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
}

# Validate
warns = validate_inputs(inputs)
for w in warns:
    st.sidebar.warning(w)

# ============================================================================
# SOLVE CYCLES
# ============================================================================

try:
    result_a = solve_config_a(inputs, fp)
    result_b = solve_config_b(inputs, fp)
except Exception as e:
    st.error(f"Cycle solver error: {e}")
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
    dW_dT_a = perf_a["net_power_kw"] - result_a_pert["performance"]["net_power_kw"]  # kW per degF
except Exception:
    dW_dT_a = 0.0

try:
    inp_pert_b = {**inputs, "dt_pinch_acc_b": inputs["dt_pinch_acc_b"] + 1}
    result_b_pert = solve_config_b(inp_pert_b, fp)
    dW_dT_b = perf_b["net_power_kw"] - result_b_pert["performance"]["net_power_kw"]  # kW per degF
except Exception:
    dW_dT_b = 0.0

# Costs
costs_a = calculate_costs_a(states_a, perf_a, inputs, duct_a)
costs_b = calculate_costs_b(states_b, prop_states, perf_b, inputs, duct_b)
lc_a = lifecycle_cost(costs_a["total_installed"], perf_a["net_power_kw"], inputs)
lc_b = lifecycle_cost(costs_b["total_installed"], perf_b["net_power_kw"], inputs)

# Pinch checks
pinch_a = verify_recuperator_pinch(states_a, fp)
pinch_b = verify_recuperator_pinch(states_b, fp)

# Validation checks
checks_a = run_validation_checks(perf_a, states_a, duct_a, "A", inputs, fp)
checks_b = run_validation_checks(perf_b, states_b, duct_b, "B", inputs, fp)

# Schedule
sched_info = construction_schedule_delta(duct_a)
sched_delta = sched_info["net_delta"]

# Payback
payback_yrs = simple_payback(
    costs_a["total_installed"], costs_b["total_installed"],
    perf_a["net_power_kw"], perf_b["net_power_kw"], inputs
)

# NPV of schedule savings
sched_npv = schedule_savings_npv(sched_info, perf_b["net_power_kw"], inputs)

# Volumetric flow ratio
vol_a = duct_a["total_vol_flow_ft3s"]
vol_b = duct_b.get("propane_vol_flow_ft3s", duct_b["total_vol_flow_ft3s"])
vol_ratio = vol_a / vol_b if vol_b > 0 else float("inf")

# Total economic advantage: NPV delta + schedule NPV
npv_delta = lc_b["net_npv"] - lc_a["net_npv"]
total_economic_advantage = npv_delta + sched_npv


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

# Bounds for validation (min, max) keyed by widget key
_WIDGET_BOUNDS = {
    "uc_vaporizer": (20, 60), "uc_preheater": (15, 50),
    "uc_recuperator": (15, 45), "uc_ihx": (25, 70),
    "uc_acc": (8, 20), "uc_iso_duct": (100, 300),
    "uc_prop_pipe": (80, 200), "uc_prop_pct": (10, 35),
    "uc_turbine": (800, 1800), "uc_steel": (3.0, 7.0),
    "uc_foundation": (5, 15), "uc_eng": (8, 18),
    "uc_cm": (5, 12), "uc_contingency": (10, 25),
}


def _build_chat_context():
    """Build JSON context string for Claude system prompt injection."""
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
            "net_power_kw": round(perf_a["net_power_kw"], 1),
            "gross_power_kw": round(perf_a["gross_power_kw"], 1),
            "thermal_eff_pct": round(perf_a["eta_thermal"] * 100, 2),
            "T_cond_F": round(perf_a["T_cond"], 1),
            "brine_effectiveness_kW_per_lbs": round(perf_a["brine_effectiveness"], 3),
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
            "net_power_kw": round(perf_b["net_power_kw"], 1),
            "gross_power_kw": round(perf_b["gross_power_kw"], 1),
            "thermal_eff_pct": round(perf_b["eta_thermal"] * 100, 2),
            "T_cond_iso_F": round(perf_b["T_cond_iso"], 1),
            "T_propane_cond_F": round(perf_b["T_propane_cond"], 1),
            "brine_effectiveness_kW_per_lbs": round(perf_b["brine_effectiveness"], 3),
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
            "power_delta_B_minus_A_kw": round(perf_b["net_power_kw"] - perf_a["net_power_kw"], 1),
            "npv_advantage_B": round(total_economic_advantage),
            "schedule_delta_weeks": sched_info["net_delta"],
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
        for i, msg in enumerate(st.session_state.chat_messages):
            with st.chat_message(msg["role"]):
                st.markdown(msg["content"])
                # Show Apply buttons for assistant messages with recommendations
                if msg["role"] == "assistant" and msg.get("recs"):
                    for wkey, label, val in msg["recs"]:
                        btn_label = f"Apply: {label} = {val}"
                        if st.button(btn_label, key=f"_apply_{wkey}_{i}"):
                            st.session_state.pending_apply = {wkey: val}
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
                st.session_state.chat_messages = []
                st.rerun(scope="fragment")
        st.caption(
            "AI recommendations should be validated against current vendor quotes "
            "and project-specific conditions before use in project decisions."
        )
        return

    # Add user message
    st.session_state.chat_messages.append({"role": "user", "content": prompt})

    # Build API messages (role/content only, no extra keys)
    api_messages = [
        {"role": m["role"], "content": m["content"]}
        for m in st.session_state.chat_messages
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
    st.session_state.chat_messages.append({
        "role": "assistant",
        "content": reply,
        "recs": recs,
    })

    # Rerun dialog to display the new messages
    st.rerun(scope="fragment")


# --- Sidebar button to open chat ---
with st.sidebar:
    st.divider()
    if st.button("💬 Ask Claude", use_container_width=True, type="secondary"):
        _claude_chat_dialog()


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
<text x="375" y="607" text-anchor="middle" class="annot-text">{tp_dia:.0f}" dia x {L_run:.0f} ft run</text>
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
    pump_prop_kw = m_prop * perf["w_pump_prop"] / 3412.14
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
{_eq_box(533, 435, 120, 38, '#e8f5e9', 'PROP PUMP', f'{pump_prop_kw:.0f} kW')}

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
<text x="470" y="595" text-anchor="middle" class="annot-green">{prop_dia:.0f}" dia vs {tp_a_dia:.0f}" (Config A)</text>
<text x="470" y="607" text-anchor="middle"
  style="font-size:7.5px;fill:#27ae60;font-family:sans-serif;">
  (small-bore, high-pressure piping)</text>

</svg></div>"""
    return svg


# ============================================================================
# SECTION 1: EXECUTIVE SUMMARY
# ============================================================================

st.header("Executive Summary")

# Build comparison rows: (label, val_a, val_b, fmt, lower_better_or_None)
summary_rows = []

# -- Power Performance --
summary_rows.append(("group", "Power Performance", "", "", "", "", ""))
summary_rows.append(("row", "Net power output",
                      _fmt(perf_a["net_power_kw"], ".0f"), "kW",
                      _fmt(perf_b["net_power_kw"], ".0f"), "kW",
                      _winner(perf_a["net_power_kw"], perf_b["net_power_kw"], lower_better=False)))
summary_rows.append(("row", "Gross turbine power",
                      _fmt(perf_a["gross_power_kw"], ".0f"), "kW",
                      _fmt(perf_b["gross_power_kw"], ".0f"), "kW",
                      _winner(perf_a["gross_power_kw"], perf_b["gross_power_kw"], lower_better=False)))
summary_rows.append(("row", "Cycle thermal efficiency",
                      _fmt(perf_a["eta_thermal"]*100, ".1f"), "%",
                      _fmt(perf_b["eta_thermal"]*100, ".1f"), "%",
                      _winner(perf_a["eta_thermal"], perf_b["eta_thermal"], lower_better=False)))
summary_rows.append(("row", "Brine effectiveness",
                      _fmt(perf_a["brine_effectiveness"], ".2f"), "kW/(lb/s)",
                      _fmt(perf_b["brine_effectiveness"], ".2f"), "kW/(lb/s)",
                      _winner(perf_a["brine_effectiveness"], perf_b["brine_effectiveness"], lower_better=False)))
summary_rows.append(("row", "Heat rejection duty",
                      _fmt(perf_a["Q_reject_mmbtu_hr"], ".2f"), "MMBtu/hr",
                      _fmt(perf_b["Q_reject_mmbtu_hr"], ".2f"), "MMBtu/hr",
                      _winner(perf_a["Q_reject_mmbtu_hr"], perf_b["Q_reject_mmbtu_hr"], lower_better=True)))

# -- Physical Scale --
summary_rows.append(("group", "Physical Scale", "", "", "", "", ""))
summary_rows.append(("row", "Tailpipe diameter",
                      _fmt(duct_a["tailpipe_diameter_in"], ".0f"), "in",
                      _fmt(duct_b["tailpipe_diameter_in"], ".0f"), "in",
                      _winner(duct_a["tailpipe_diameter_in"], duct_b["tailpipe_diameter_in"], lower_better=True)))
summary_rows.append(("row", "ACC header diameter",
                      _fmt(duct_a["acc_header_diameter_in"], ".0f"), "in",
                      _fmt(duct_b["acc_header_diameter_in"], ".0f"), "in",
                      _winner(duct_a["acc_header_diameter_in"], duct_b["acc_header_diameter_in"], lower_better=True)))
summary_rows.append(("row", "Vapor vol. flow",
                      _fmt(vol_a, ".0f"), "ft3/s",
                      _fmt(vol_b, ".0f"), "ft3/s",
                      _winner(vol_a, vol_b, lower_better=True)))
summary_rows.append(("row", "Vol. flow ratio (A/B)",
                      "", "",
                      _fmt(vol_ratio, ".1f"), "x",
                      ""))

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

total_adv_str = _fmt(total_economic_advantage/1e6, ".2f")
adv_winner = "B" if total_economic_advantage > 0 else ("A" if total_economic_advantage < 0 else "Tie")
summary_rows.append(("row", "Total economic advantage",
                      "", "",
                      total_adv_str, "$MM",
                      adv_winner))

# -- Schedule --
summary_rows.append(("group", "Schedule", "", "", "", "", ""))
if sched_delta < 0:
    sched_str = f"{abs(sched_delta)} weeks faster"
    sched_winner_val = "B"
elif sched_delta > 0:
    sched_str = f"+{sched_delta} weeks slower"
    sched_winner_val = "A"
else:
    sched_str = "Same"
    sched_winner_val = "Tie"
summary_rows.append(("row", "Schedule delta (B vs A)",
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

# Auto-generated summary sentence
power_winner = "A" if perf_a["net_power_kw"] >= perf_b["net_power_kw"] else "B"
cost_winner = "A" if costs_a["total_installed"] <= costs_b["total_installed"] else "B"
power_diff_kw = abs(perf_a["net_power_kw"] - perf_b["net_power_kw"])
cost_diff_mm = abs(costs_a["total_installed"] - costs_b["total_installed"]) / 1e6

summary_sentence = (
    f"Config **{power_winner}** produces **{power_diff_kw:.0f} kW** more net power. "
    f"Config **{cost_winner}** costs **${cost_diff_mm:.2f}MM** less to build. "
)
if sched_delta < 0:
    summary_sentence += f"Config B is **{abs(sched_delta)} weeks** faster to construct"
    if sched_npv > 0:
        summary_sentence += f" (NPV of early revenue: **${sched_npv/1e3:.0f}k**)"
    summary_sentence += ". "
elif sched_delta > 0:
    summary_sentence += f"Config A is **{sched_delta} weeks** faster to construct. "

if total_economic_advantage > 0:
    summary_sentence += f"Total economic advantage for B: **${total_economic_advantage/1e6:.2f}MM**."
elif total_economic_advantage < 0:
    summary_sentence += f"Total economic advantage for A: **${abs(total_economic_advantage)/1e6:.2f}MM**."

st.markdown(summary_sentence)

# Schedule breakdown detail
with st.expander("Schedule Breakdown"):
    si = sched_info
    st.markdown(f"""
**Config B savings** (tailpipe = {si['tailpipe_diameter_in']:.0f}"):
- Duct fabrication & erection: **-{si['duct_fab_savings']} weeks**
- Welding vs flanged connections: **-{si['weld_savings']} weeks**
- Structural steel: **-{si['steel_savings']} weeks**
- **Total savings: -{si['total_savings']} weeks**

**Config B adders** (fixed):
- Intermediate HX installation: +{si['ihx_install']} weeks
- Propane pressure test & leak check: +{si['propane_pressure_test']} weeks
- Propane safety review & commissioning: +{si['propane_safety_commissioning']} week
- **Total adder: +{si['total_adder']} weeks**

**Net: {si['net_delta']:+d} weeks** ({sched_str})
""")

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
    fig_wf_a.update_layout(
        title="Config A Installed Cost ($MM)",
        yaxis_title="Cost ($MM)",
        height=450,
        showlegend=False,
    )
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
    fig_wf_b.update_layout(
        title="Config B Installed Cost ($MM)",
        yaxis_title="Cost ($MM)",
        height=450,
        showlegend=False,
    )
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
power_a_vals = [perf_a["gross_power_kw"], perf_a["net_power_kw"]]
power_b_vals = [perf_b["gross_power_kw"], perf_b["net_power_kw"]]

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
eta_vals = [perf_a["eta_thermal"]*100, perf_b["eta_thermal"]*100]
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
pwr_col1.metric("Net Power A", f"{perf_a['net_power_kw']:.0f} kW")
pwr_col2.metric("Net Power B", f"{perf_b['net_power_kw']:.0f} kW")
pwr_col3.metric("Delta (B-A)", f"{perf_b['net_power_kw'] - perf_a['net_power_kw']:.0f} kW")
pwr_col4.metric("Vol. Flow Ratio (A/B)", f"{vol_ratio:.1f}x")


# ============================================================================
# SECTION 4: TECHNICAL CHARTS (TABS)
# ============================================================================

st.header("Technical Analysis")

tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs([
    "T-s Diagram", "Duct Sizing", "Approach Optimization",
    "Brine Utilization", "Sensitivity Analysis", "Hydraulic Analysis",
    "Equipment Sizing Trade-offs",
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
                             subplot_titles=["Duct Diameters by Segment",
                                             "Volumetric Flow Rates by Segment"],
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
        title=f"Vapor Duct Sizing -- Config A vs Config B (Vol ratio: {vol_ratio:.1f}x)",
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
    brine_T_a = [perf_a["T_geo_out_calc"], perf_a["T_brine_mid"], T_geo_in]

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
    brine_T_b = [perf_b["T_geo_out_calc"], perf_b["T_brine_mid"], T_geo_in]

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

    fig_brine.add_hline(y=T_geo_out_min, line_dash="dash", line_color="orange",
                        annotation_text=f"Min brine outlet: {T_geo_out_min}degF")

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
        ("T_ambient", T_ambient, "Ambient Temp"),
        ("electricity_price", electricity_price, "Elec Price"),
        ("dt_approach_intermediate", dt_approach_intermediate, "Approach dT"),
        ("v_tailpipe", v_tailpipe, "Duct Velocity"),
        ("T_geo_in", T_geo_in, "Brine Inlet T"),
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
            power_a_sweep.append(r_sw_a["performance"]["net_power_kw"])
        except Exception:
            power_a_sweep.append(float("nan"))
        try:
            inp_sw_b = {**inputs, "dp_acc_tubes_prop": float(dp_val)}
            r_sw_b = solve_config_b(inp_sw_b, fp)
            power_b_sweep_hyd.append(r_sw_b["performance"]["net_power_kw"])
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

    # -- Config A heat exchangers --
    Q_vap_a = m_a * perf_a["q_vaporizer"]
    Q_pre_a = m_a * perf_a["q_preheater"]
    Q_rec_a = m_a * perf_a["q_recup"]
    Q_cond_a = m_a * perf_a["q_cond"]

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

    # -- Config B heat exchangers --
    Q_vap_b = m_b * perf_b["q_vaporizer"]
    Q_pre_b = m_b * perf_b["q_preheater"]
    Q_rec_b = m_b * perf_b["q_recup"]
    Q_ihx_b = m_b * perf_b["q_cond_iso"]
    m_prop = perf_b["m_dot_prop"]
    Q_acc_b = m_prop * (prop_states["A"].h - prop_states["B"].h)

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
- Duct friction factor: f = {f_darcy:.3f} (Darcy-Weisbach)
- Equipment-internal dP modeled for ACC tubes, recuperator, IHX
- dT/dP from CoolProp saturation curve (central finite difference)
- Cycle convergence tolerance: 0.1 degF on condensing temperature
- Brine modeled as constant-cp fluid
- All costs are installed costs (2024 USD)
- No parasitic load for ACC fans included
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

    st.caption("ORC Comparator v5.0 -- Adjustable unit costs, indirect cost layers, cost sensitivity")
