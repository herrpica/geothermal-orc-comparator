"""
ORC Comparator — Streamlit Application

Compares two geothermal ORC configurations:
  Config A: Traditional ORC with recuperator + direct air-cooled condensing
  Config B: Isopentane power block + propane intermediate loop
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd

from fluid_properties import FluidProperties
from thermodynamics import (
    solve_config_a, solve_config_b, verify_recuperator_pinch,
    calculate_duct_sizing, validate_inputs,
)
from cost_model import (
    calculate_costs_a, calculate_costs_b, lifecycle_cost,
    optimize_approach_temp,
)

st.set_page_config(page_title="ORC Comparator", layout="wide")
st.title("Geothermal ORC Configuration Comparator")
st.caption("Config A (Direct ACC) vs Config B (Propane Intermediate Loop)")

# ── Initialize fluid properties ──────────────────────────────────────────────

@st.cache_resource
def get_fluid_props():
    return FluidProperties()

fp = get_fluid_props()

# ── Sidebar inputs ───────────────────────────────────────────────────────────

st.sidebar.header("Plant Inputs")

T_geo_in = st.sidebar.number_input("Geo-fluid inlet T (°F)", 200, 400, 300, 5)
T_geo_out = st.sidebar.number_input("Geo-fluid outlet T (°F)", 100, 300, 160, 5)
m_dot_geo = st.sidebar.number_input("Geo-fluid flow (lb/hr)", 100_000, 5_000_000, 500_000, 50_000)
T_ambient = st.sidebar.number_input("Ambient T (°F)", 50, 130, 95, 5)

st.sidebar.header("Cycle Parameters")

dt_pinch_evap = st.sidebar.number_input("Evaporator pinch ΔT (°F)", 3, 30, 10, 1)
dt_pinch_acc = st.sidebar.number_input("ACC pinch ΔT (°F)", 10, 50, 25, 1)
dt_pinch_recup = st.sidebar.number_input("Recuperator pinch ΔT (°F)", 5, 40, 15, 1)
dt_approach_intermediate = st.sidebar.number_input("Intermediate HX approach ΔT (°F)", 3, 30, 10, 1)
superheat = st.sidebar.number_input("Superheat (°F)", 0, 30, 5, 1)
eta_turbine = st.sidebar.slider("Turbine isentropic efficiency", 0.60, 0.95, 0.82, 0.01)
eta_pump = st.sidebar.slider("Pump isentropic efficiency", 0.50, 0.95, 0.75, 0.01)

st.sidebar.header("Duct & Economics")

v_duct = st.sidebar.number_input("Duct velocity (ft/s)", 20, 150, 60, 5)
electricity_price = st.sidebar.number_input("Electricity price ($/kWh)", 0.03, 0.25, 0.08, 0.01)
discount_rate = st.sidebar.slider("Discount rate", 0.03, 0.15, 0.08, 0.01)
project_life = st.sidebar.number_input("Project life (years)", 10, 40, 30, 1)
capacity_factor = st.sidebar.slider("Capacity factor", 0.70, 1.00, 0.95, 0.01)

inputs = {
    "T_geo_in": T_geo_in,
    "T_geo_out": T_geo_out,
    "m_dot_geo": m_dot_geo,
    "T_ambient": T_ambient,
    "dt_pinch_evap": dt_pinch_evap,
    "dt_pinch_acc": dt_pinch_acc,
    "dt_pinch_recup": dt_pinch_recup,
    "dt_approach_intermediate": dt_approach_intermediate,
    "superheat": superheat,
    "eta_turbine": eta_turbine,
    "eta_pump": eta_pump,
    "v_duct": v_duct,
    "electricity_price": electricity_price,
    "discount_rate": discount_rate,
    "project_life": project_life,
    "capacity_factor": capacity_factor,
}

# ── Validate inputs ──────────────────────────────────────────────────────────

warns = validate_inputs(inputs)
for w in warns:
    st.sidebar.warning(w)

# ── Solve cycles ─────────────────────────────────────────────────────────────

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

# ── Costs ────────────────────────────────────────────────────────────────────

costs_a = calculate_costs_a(states_a, perf_a, inputs)
costs_b = calculate_costs_b(states_b, prop_states, perf_b, inputs)
lc_a = lifecycle_cost(costs_a["total_installed"], perf_a["net_power_kw"], inputs)
lc_b = lifecycle_cost(costs_b["total_installed"], perf_b["net_power_kw"], inputs)

# ── Duct sizing ──────────────────────────────────────────────────────────────

duct_a = calculate_duct_sizing(states_a, perf_a["m_dot_iso"], v_duct, fp, "A")
duct_b = calculate_duct_sizing(
    {"2": prop_states["A"]}, perf_b["m_dot_prop"], v_duct, fp, "B"
)

# ── Recuperator pinch check ──────────────────────────────────────────────────

pinch_a = verify_recuperator_pinch(states_a, fp)
pinch_b = verify_recuperator_pinch(states_b, fp)

# ═══════════════════════════════════════════════════════════════════════════════
# DISPLAY
# ═══════════════════════════════════════════════════════════════════════════════

# ── Volumetric flow ratio (prominent) ────────────────────────────────────────

vol_ratio = duct_a["vol_flow_ft3s"] / duct_b["vol_flow_ft3s"] if duct_b["vol_flow_ft3s"] > 0 else float("inf")

col1, col2, col3 = st.columns(3)
col1.metric("Config A Net Power", f"{perf_a['net_power_kw']:.0f} kW")
col2.metric("Config B Net Power", f"{perf_b['net_power_kw']:.0f} kW",
            delta=f"{perf_b['net_power_kw'] - perf_a['net_power_kw']:.0f} kW")
col3.metric("ACC Duct Volumetric Flow Ratio (A/B)", f"{vol_ratio:.1f}x")

# ── Pinch violation alerts ───────────────────────────────────────────────────

if pinch_a["violation"]:
    st.error(f"⚠ Config A recuperator pinch violation! Min ΔT = {pinch_a['min_pinch_dT']:.1f}°F")
if pinch_b["violation"]:
    st.error(f"⚠ Config B recuperator pinch violation! Min ΔT = {pinch_b['min_pinch_dT']:.1f}°F")

# ── Performance comparison table ─────────────────────────────────────────────

st.subheader("Performance Comparison")

perf_data = {
    "Metric": [
        "Gross Power (kW)",
        "Net Power (kW)",
        "Thermal Efficiency (%)",
        "Iso Mass Flow (lb/hr)",
        "Turbine Work (BTU/lb)",
        "Total Pump Work (BTU/lb)",
        "Net Work (BTU/lb)",
        "Evaporator Duty (BTU/lb)",
        "Recuperator Duty (BTU/lb)",
        "P_high (psia)",
        "P_low (psia)",
        "Pressure Ratio",
        "Turbine Exit Vol Flow (ft³/s)",
        "ACC Duct Diameter (in)",
        "Recuperator Min Pinch (°F)",
    ],
    "Config A": [
        f"{perf_a['gross_power_kw']:.0f}",
        f"{perf_a['net_power_kw']:.0f}",
        f"{perf_a['eta_thermal']*100:.1f}",
        f"{perf_a['m_dot_iso']:,.0f}",
        f"{perf_a['w_turbine']:.2f}",
        f"{perf_a['w_pump']:.2f}",
        f"{perf_a['w_net']:.2f}",
        f"{perf_a['q_evap']:.2f}",
        f"{perf_a['q_recup']:.2f}",
        f"{perf_a['P_high']:.1f}",
        f"{perf_a['P_low']:.1f}",
        f"{perf_a['pressure_ratio']:.2f}",
        f"{perf_a['vol_flow_turbine_exit']:.1f}",
        f"{duct_a['diameter_in']:.1f}",
        f"{pinch_a['min_pinch_dT']:.1f}",
    ],
    "Config B": [
        f"{perf_b['gross_power_kw']:.0f}",
        f"{perf_b['net_power_kw']:.0f}",
        f"{perf_b['eta_thermal']*100:.1f}",
        f"{perf_b['m_dot_iso']:,.0f}",
        f"{perf_b['w_turbine']:.2f}",
        f"{perf_b.get('w_pump_total', perf_b['w_pump_iso']):.2f}",
        f"{perf_b['w_net']:.2f}",
        f"{perf_b['q_evap']:.2f}",
        f"{perf_b['q_recup']:.2f}",
        f"{perf_b['P_high_iso']:.1f}",
        f"{perf_b['P_low_iso']:.1f}",
        f"{perf_b['pressure_ratio_iso']:.2f}",
        f"{perf_b['vol_flow_turbine_exit']:.1f}",
        f"{duct_b['diameter_in']:.1f}",
        f"{pinch_b['min_pinch_dT']:.1f}",
    ],
}
st.dataframe(pd.DataFrame(perf_data).set_index("Metric"), use_container_width=True)

# ── Cost comparison table ────────────────────────────────────────────────────

st.subheader("Cost Comparison")

cost_components = [
    ("Turbine-Generator", "turbine_generator"),
    ("Isopentane Pump", "iso_pump"),
    ("Evaporator", "evaporator"),
    ("Recuperator", "recuperator"),
    ("ACC", "acc"),
    ("Ductwork", "ductwork"),
    ("Intermediate HX", "intermediate_hx"),
    ("Propane Pump", "propane_pump"),
    ("Propane Loop Piping", "propane_loop_piping"),
    ("Total Installed", "total_installed"),
]

cost_data = {
    "Component": [c[0] for c in cost_components],
    "Config A ($)": [f"${costs_a[c[1]]:,.0f}" for c in cost_components],
    "Config B ($)": [f"${costs_b[c[1]]:,.0f}" for c in cost_components],
}
st.dataframe(pd.DataFrame(cost_data).set_index("Component"), use_container_width=True)

# Lifecycle
lc_col1, lc_col2 = st.columns(2)
with lc_col1:
    st.metric("Config A NPV", f"${lc_a['net_npv']:,.0f}")
    st.metric("Config A LCOE", f"${lc_a['lcoe']:.4f}/kWh")
with lc_col2:
    st.metric("Config B NPV", f"${lc_b['net_npv']:,.0f}")
    st.metric("Config B LCOE", f"${lc_b['lcoe']:.4f}/kWh")

# ═══════════════════════════════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════════════════════════════

st.subheader("Charts")

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "T-s Diagram", "Approach ΔT Sweep", "Cost Waterfall",
    "Volumetric Flow", "Sensitivity Tornado", "Duct Pressure Drop",
])

# ── 1. T-s Diagram ──────────────────────────────────────────────────────────

with tab1:
    fig_ts = go.Figure()

    # Config A cycle
    pts_a = ["1", "2", "3", "4", "5", "6", "1"]
    s_vals_a = [states_a[p].s for p in pts_a]
    T_vals_a = [states_a[p].T for p in pts_a]
    fig_ts.add_trace(go.Scatter(
        x=s_vals_a, y=T_vals_a, mode="lines+markers",
        name="Config A (Isopentane)", line=dict(color="blue", width=2),
        marker=dict(size=8),
    ))

    # Config B iso cycle
    pts_b = ["1", "2", "3", "4", "5", "6", "1"]
    s_vals_b = [states_b[p].s for p in pts_b]
    T_vals_b = [states_b[p].T for p in pts_b]
    fig_ts.add_trace(go.Scatter(
        x=s_vals_b, y=T_vals_b, mode="lines+markers",
        name="Config B (Isopentane)", line=dict(color="red", width=2, dash="dash"),
        marker=dict(size=8),
    ))

    # Saturation dome for isopentane
    try:
        T_dome = np.linspace(80, states_a["1"].T + 20, 50)
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
            name="Sat Dome (Isopentane)", line=dict(color="gray", width=1),
        ))
    except Exception:
        pass

    # State point labels for Config A
    for label, st_pt in states_a.items():
        fig_ts.add_annotation(
            x=st_pt.s, y=st_pt.T, text=label,
            showarrow=True, arrowhead=2, ax=15, ay=-15,
            font=dict(color="blue", size=10),
        )

    fig_ts.update_layout(
        title="T-s Diagram — Isopentane ORC Cycles",
        xaxis_title="Entropy s (BTU/lb·R)",
        yaxis_title="Temperature T (°F)",
        legend=dict(x=0.01, y=0.99),
        height=500,
    )
    st.plotly_chart(fig_ts, use_container_width=True)

# ── 2. Approach ΔT Sweep ────────────────────────────────────────────────────

with tab2:
    with st.spinner("Running approach temperature sweep..."):
        opt_result = optimize_approach_temp(inputs, fp)

    sweep = opt_result["sweep"]
    dts = [s["dt_approach"] for s in sweep]
    npvs = [s["net_npv"] / 1e6 for s in sweep]
    powers = [s["net_power_kw"] for s in sweep]
    inst_costs = [s["installed_cost"] / 1e6 for s in sweep]

    fig_sweep = go.Figure()
    fig_sweep.add_trace(go.Scatter(
        x=dts, y=npvs, name="Net NPV ($M)", yaxis="y",
        line=dict(color="green", width=2),
    ))
    fig_sweep.add_trace(go.Scatter(
        x=dts, y=powers, name="Net Power (kW)", yaxis="y2",
        line=dict(color="orange", width=2, dash="dash"),
    ))
    fig_sweep.add_vline(
        x=opt_result["optimal_dt"], line_dash="dot",
        annotation_text=f"Optimal: {opt_result['optimal_dt']:.1f}°F",
    )
    fig_sweep.update_layout(
        title="Config B: Intermediate HX Approach ΔT Optimization",
        xaxis_title="Approach ΔT (°F)",
        yaxis=dict(title="Net NPV ($M)", titlefont=dict(color="green")),
        yaxis2=dict(title="Net Power (kW)", titlefont=dict(color="orange"),
                    overlaying="y", side="right"),
        height=450,
    )
    st.plotly_chart(fig_sweep, use_container_width=True)
    st.info(f"Optimal intermediate HX approach ΔT: **{opt_result['optimal_dt']:.1f}°F**")

# ── 3. Cost Waterfall ────────────────────────────────────────────────────────

with tab3:
    components = [c[0] for c in cost_components[:-1]]  # exclude total
    vals_a = [costs_a[c[1]] / 1e6 for c in cost_components[:-1]]
    vals_b = [costs_b[c[1]] / 1e6 for c in cost_components[:-1]]

    fig_wf = go.Figure()
    fig_wf.add_trace(go.Bar(
        x=components, y=vals_a, name="Config A", marker_color="steelblue",
    ))
    fig_wf.add_trace(go.Bar(
        x=components, y=vals_b, name="Config B", marker_color="indianred",
    ))
    fig_wf.update_layout(
        title="Installed Cost Breakdown ($M)",
        yaxis_title="Cost ($M)",
        barmode="group",
        height=450,
    )
    st.plotly_chart(fig_wf, use_container_width=True)

# ── 4. Volumetric Flow Comparison ────────────────────────────────────────────

with tab4:
    fig_vol = go.Figure()

    categories = ["Turbine Exit\n(Isopentane)", "ACC Duct\n(to condenser)"]
    vol_a = [perf_a["vol_flow_turbine_exit"], duct_a["vol_flow_ft3s"]]
    vol_b = [perf_b["vol_flow_turbine_exit"], duct_b["vol_flow_ft3s"]]

    fig_vol.add_trace(go.Bar(
        x=categories, y=vol_a, name="Config A",
        marker_color="steelblue", text=[f"{v:.0f}" for v in vol_a],
        textposition="outside",
    ))
    fig_vol.add_trace(go.Bar(
        x=categories, y=vol_b, name="Config B",
        marker_color="indianred", text=[f"{v:.0f}" for v in vol_b],
        textposition="outside",
    ))
    fig_vol.update_layout(
        title="Volumetric Flow Comparison (ft³/s)",
        yaxis_title="Volume Flow (ft³/s)",
        barmode="group",
        height=450,
    )
    st.plotly_chart(fig_vol, use_container_width=True)

    st.markdown(f"""
    | Metric | Config A | Config B | Ratio |
    |--------|----------|----------|-------|
    | ACC duct vol flow (ft³/s) | {duct_a['vol_flow_ft3s']:.0f} | {duct_b['vol_flow_ft3s']:.0f} | **{vol_ratio:.1f}x** |
    | ACC duct diameter (in) | {duct_a['diameter_in']:.0f} | {duct_b['diameter_in']:.0f} | {duct_a['diameter_in']/duct_b['diameter_in']:.1f}x |
    """)

# ── 5. Sensitivity Tornado ──────────────────────────────────────────────────

with tab5:
    st.markdown("Sensitivity of **Config B net NPV** to ±20% parameter changes")

    base_npv_b = lc_b["net_npv"]
    sensitivity_params = [
        ("T_geo_in", T_geo_in, "Geo Inlet T"),
        ("T_ambient", T_ambient, "Ambient T"),
        ("m_dot_geo", m_dot_geo, "Geo Flow Rate"),
        ("eta_turbine", eta_turbine, "Turbine Eff"),
        ("electricity_price", electricity_price, "Elec Price"),
        ("dt_pinch_acc", dt_pinch_acc, "ACC Pinch"),
        ("dt_approach_intermediate", dt_approach_intermediate, "Approach ΔT"),
    ]

    tornado_data = []
    for key, base_val, label in sensitivity_params:
        low_val = base_val * 0.8
        high_val = base_val * 1.2
        npvs_sens = []
        for v in [low_val, high_val]:
            try:
                inp_mod = {**inputs, key: v}
                rb = solve_config_b(inp_mod, fp)
                cb = calculate_costs_b(rb["states"], rb["propane_states"],
                                       rb["performance"], inp_mod)
                lcb = lifecycle_cost(cb["total_installed"],
                                     rb["performance"]["net_power_kw"], inp_mod)
                npvs_sens.append(lcb["net_npv"])
            except Exception:
                npvs_sens.append(base_npv_b)

        tornado_data.append({
            "param": label,
            "low": (npvs_sens[0] - base_npv_b) / 1e6,
            "high": (npvs_sens[1] - base_npv_b) / 1e6,
        })

    # Sort by total swing
    tornado_data.sort(key=lambda d: abs(d["high"] - d["low"]))

    params = [d["param"] for d in tornado_data]
    lows = [d["low"] for d in tornado_data]
    highs = [d["high"] for d in tornado_data]

    fig_tornado = go.Figure()
    fig_tornado.add_trace(go.Bar(
        y=params, x=lows, orientation="h", name="-20%",
        marker_color="steelblue",
    ))
    fig_tornado.add_trace(go.Bar(
        y=params, x=highs, orientation="h", name="+20%",
        marker_color="indianred",
    ))
    fig_tornado.update_layout(
        title="Sensitivity Tornado — Config B Net NPV ($M change from base)",
        xaxis_title="ΔNPV ($M)",
        barmode="overlay",
        height=450,
    )
    st.plotly_chart(fig_tornado, use_container_width=True)

# ── 6. Duct Pressure Drop Sensitivity ───────────────────────────────────────

with tab6:
    st.markdown("Effect of duct velocity on pressure drop and duct diameter")

    velocities = np.linspace(20, 150, 20)
    dp_a_list, dp_b_list = [], []
    dia_a_list, dia_b_list = [], []

    for v in velocities:
        da = calculate_duct_sizing(states_a, perf_a["m_dot_iso"], v, fp, "A")
        db = calculate_duct_sizing(
            {"2": prop_states["A"]}, perf_b["m_dot_prop"], v, fp, "B"
        )
        dia_a_list.append(da["diameter_in"])
        dia_b_list.append(db["diameter_in"])

        # Rough pressure drop: ΔP ∝ ½ρv² × (f L/D), use Darcy f≈0.015, L=100ft
        f = 0.015
        L = 100  # ft
        rho_a = da["rho_lbft3"]
        rho_b = db["rho_lbft3"]
        # ΔP = f * L/D * ½ * ρ * v² (lb/ft²) → psi (/144)
        dp_a = f * L / da["diameter_ft"] * 0.5 * rho_a * v**2 / 144 if da["diameter_ft"] > 0 else 0
        dp_b = f * L / db["diameter_ft"] * 0.5 * rho_b * v**2 / 144 if db["diameter_ft"] > 0 else 0
        dp_a_list.append(dp_a)
        dp_b_list.append(dp_b)

    fig_dp = go.Figure()
    fig_dp.add_trace(go.Scatter(
        x=velocities, y=dp_a_list, name="Config A ΔP",
        line=dict(color="steelblue", width=2),
    ))
    fig_dp.add_trace(go.Scatter(
        x=velocities, y=dp_b_list, name="Config B ΔP",
        line=dict(color="indianred", width=2),
    ))
    fig_dp.add_trace(go.Scatter(
        x=velocities, y=dia_a_list, name="Config A Diameter (in)",
        yaxis="y2", line=dict(color="steelblue", width=1, dash="dot"),
    ))
    fig_dp.add_trace(go.Scatter(
        x=velocities, y=dia_b_list, name="Config B Diameter (in)",
        yaxis="y2", line=dict(color="indianred", width=1, dash="dot"),
    ))
    fig_dp.add_vline(x=v_duct, line_dash="dash",
                     annotation_text=f"Design: {v_duct} ft/s")
    fig_dp.update_layout(
        title="Duct Pressure Drop & Diameter vs Velocity",
        xaxis_title="Duct Velocity (ft/s)",
        yaxis=dict(title="Pressure Drop (psi)", titlefont=dict(color="steelblue")),
        yaxis2=dict(title="Duct Diameter (in)", titlefont=dict(color="gray"),
                    overlaying="y", side="right"),
        height=450,
    )
    st.plotly_chart(fig_dp, use_container_width=True)

# ── Footer ───────────────────────────────────────────────────────────────────
st.divider()
st.caption("ORC Comparator v1.0 — CoolProp backend with optional REFPROP bridge")
