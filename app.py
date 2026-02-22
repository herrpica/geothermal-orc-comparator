"""
ORC Comparator -- Streamlit Application

Compares two geothermal ORC configurations:
  Config A: Traditional ORC with recuperator + direct air-cooled condensing
  Config B: Isopentane power block + propane intermediate loop
"""

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd

from fluid_properties import FluidProperties
from thermodynamics import (
    solve_config_a, solve_config_b, verify_recuperator_pinch,
    run_validation_checks, validate_inputs,
)
from cost_model import (
    calculate_costs_a, calculate_costs_b, lifecycle_cost,
    optimize_approach_temp, construction_schedule_delta, simple_payback,
)

st.set_page_config(page_title="ORC Comparator", layout="wide")


@st.cache_resource
def get_fluid_props():
    return FluidProperties()


fp = get_fluid_props()

# ============================================================================
# LAYOUT: Left (30%) + Right (70%)
# ============================================================================

left_col, right_col = st.columns([3, 7])

# ============================================================================
# LEFT COLUMN -- INPUTS
# ============================================================================

with left_col:
    st.header("ORC Comparator")
    st.caption("Config A (Direct ACC) vs Config B (Propane Loop)")

    with st.expander("Brine Inputs", expanded=True):
        T_geo_in = st.number_input("Brine inlet temperature (degF)", 200, 400, 300, 5)
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

    with st.expander("Economic Parameters"):
        electricity_price = st.number_input("Electricity price ($/MWh)", 10, 200, 35, 5)
        capacity_factor = st.slider("Capacity factor (%)", 70, 100, 95, 1)
        discount_rate = st.slider("Discount rate (%)", 3, 15, 8, 1)
        project_life = st.number_input("Plant life (years)", 10, 40, 30, 1)

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
}

# Validate
warns = validate_inputs(inputs)
for w in warns:
    with left_col:
        st.warning(w)

# ============================================================================
# SOLVE CYCLES
# ============================================================================

try:
    result_a = solve_config_a(inputs, fp)
    result_b = solve_config_b(inputs, fp)
except Exception as e:
    with left_col:
        st.error(f"Cycle solver error: {e}")
    st.stop()

perf_a = result_a["performance"]
perf_b = result_b["performance"]
states_a = result_a["states"]
states_b = result_b["states"]
prop_states = result_b["propane_states"]
duct_a = result_a["duct"]
duct_b = result_b["duct"]

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
sched_delta = construction_schedule_delta(duct_a)

# Payback
payback_yrs = simple_payback(
    costs_a["total_installed"], costs_b["total_installed"],
    perf_a["net_power_kw"], perf_b["net_power_kw"], inputs
)

# Volumetric flow ratio
vol_a = duct_a["total_vol_flow_ft3s"]
vol_b = duct_b.get("propane_vol_flow_ft3s", duct_b["total_vol_flow_ft3s"])
vol_ratio = vol_a / vol_b if vol_b > 0 else float("inf")

# ============================================================================
# LEFT COLUMN -- EXECUTIVE SUMMARY CARD
# ============================================================================


def _color_cell(val_a, val_b, lower_better=True):
    """Return (color_a, color_b) -- green for better, red for worse."""
    if lower_better:
        if val_a < val_b:
            return "green", "red"
        elif val_b < val_a:
            return "red", "green"
    else:
        if val_a > val_b:
            return "green", "red"
        elif val_b > val_a:
            return "red", "green"
    return "gray", "gray"


def _fmt(val, fmt_str=".1f", prefix="", suffix=""):
    try:
        return f"{prefix}{val:{fmt_str}}{suffix}"
    except (ValueError, TypeError):
        return str(val)


with left_col:
    st.subheader("Executive Summary")

    # Build table rows: (label, val_a, val_b, lower_better)
    rows = []

    # -- Thermodynamic Performance --
    rows.append(("**Thermodynamic Performance**", "", "", None))
    rows.append(("Net power output (kW)", perf_a["net_power_kw"], perf_b["net_power_kw"], False))
    rows.append(("Gross turbine power (kW)", perf_a["gross_power_kw"], perf_b["gross_power_kw"], False))
    rows.append(("Cycle thermal efficiency (%)", perf_a["eta_thermal"]*100, perf_b["eta_thermal"]*100, False))
    rows.append(("Brine effectiveness (kW/lb/s)", perf_a["brine_effectiveness"], perf_b["brine_effectiveness"], False))
    rows.append(("Iso circulation rate (lb/s)", perf_a["m_dot_iso"]/3600, perf_b["m_dot_iso"]/3600, None))
    rows.append(("Iso condensing pressure (psia)", perf_a["P_low"], perf_b.get("P_low_iso", perf_b["P_low"]), None))
    rows.append(("Turbine pressure ratio", perf_a["pressure_ratio"], perf_b.get("pressure_ratio_iso", perf_b["pressure_ratio"]), None))
    rows.append(("Heat rejection duty (MMBtu/hr)", perf_a["Q_reject_mmbtu_hr"], perf_b["Q_reject_mmbtu_hr"], True))
    rows.append(("Recuperator duty (MMBtu/hr)", perf_a["Q_recup_mmbtu_hr"], perf_b["Q_recup_mmbtu_hr"], None))

    # -- Physical Scale --
    rows.append(("**Physical Scale**", "", "", None))
    rows.append(("Tailpipe diameter (in)", duct_a["tailpipe_diameter_in"], duct_b["tailpipe_diameter_in"], True))
    rows.append(("ACC vapor header diameter (in)", duct_a["acc_header_diameter_in"], duct_b["acc_header_diameter_in"], True))
    rows.append(("Vapor volumetric flow (ft3/s)", vol_a, vol_b, True))

    # -- Economics --
    rows.append(("**Economics**", "", "", None))
    rows.append(("Total installed cost ($MM)", costs_a["total_installed"]/1e6, costs_b["total_installed"]/1e6, True))
    rows.append(("Specific capital cost ($/kW)", lc_a["specific_cost_per_kw"], lc_b["specific_cost_per_kw"], True))
    rows.append(("Net NPV ($MM)", lc_a["net_npv"]/1e6, lc_b["net_npv"]/1e6, False))
    rows.append(("LCOE ($/MWh)", lc_a["lcoe"], lc_b["lcoe"], True))

    # Build markdown table
    table_md = "| Metric | Config A | Config B |\n|--------|----------|----------|\n"
    for label, va, vb, lower_better in rows:
        if lower_better is None:
            # Header row or neutral
            if isinstance(va, str) and va == "":
                table_md += f"| {label} | | |\n"
            else:
                table_md += f"| {label} | {_fmt(va)} | {_fmt(vb)} |\n"
        else:
            ca, cb = _color_cell(va, vb, lower_better)
            table_md += f"| {label} | :{ca}[**{_fmt(va)}**] | :{cb}[**{_fmt(vb)}**] |\n"

    # Vol ratio highlight
    table_md += f"| **Vol flow ratio (A/B)** | | **{vol_ratio:.1f}x** |\n"

    # Payback
    payback_str = f"{payback_yrs:.1f} years" if payback_yrs is not None and payback_yrs < 1e6 else "N/A"
    table_md += f"| Simple payback on cost delta | | {payback_str} |\n"

    # Construction schedule
    if sched_delta < 0:
        sched_str = f"{abs(sched_delta)} weeks faster"
    elif sched_delta > 0:
        sched_str = f"+{sched_delta} weeks"
    else:
        sched_str = "Same"
    table_md += f"| Schedule delta (Config B vs A) | | {sched_str} |\n"
    table_md += f"| Equipment item count | {costs_a['equipment_count']} | {costs_b['equipment_count']} |\n"

    # Winner callout
    perf_winner = "A" if perf_a["net_power_kw"] >= perf_b["net_power_kw"] else "B"
    cost_winner = "A" if costs_a["total_installed"] <= costs_b["total_installed"] else "B"
    sched_winner = "A" if sched_delta >= 0 else "B"
    table_md += f"| **Winner** | Perf: **{perf_winner}** | Cost: **{cost_winner}** / Sched: **{sched_winner}** |\n"

    st.markdown(table_md)

    # Validation checks
    with st.expander("Validation Checks"):
        for name, passed, detail in checks_a:
            icon = "PASS" if passed else "FAIL"
            color = "green" if passed else "red"
            st.markdown(f":{color}[{icon}] **A**: {name} -- {detail}")
        for name, passed, detail in checks_b:
            icon = "PASS" if passed else "FAIL"
            color = "green" if passed else "red"
            st.markdown(f":{color}[{icon}] **B**: {name} -- {detail}")

    # Calibration warning
    gross_mw_a = perf_a["gross_power_kw"] / 1000
    if gross_mw_a > 0:
        scale_a = 50.0 / gross_mw_a
        scaled_tp_a = duct_a["tailpipe_diameter_in"] * (scale_a ** 0.5)
        if abs(scaled_tp_a - 80) / 80 > 0.20:
            st.warning(f"Calibration: At 50 MW scale, Config A tailpipe would be ~{scaled_tp_a:.0f}\" (benchmark 80\")")

    gross_mw_b = perf_b["gross_power_kw"] / 1000
    if gross_mw_b > 0 and "propane_header_diameter_in" in duct_b:
        scale_b = 50.0 / gross_mw_b
        scaled_prop_b = duct_b["propane_header_diameter_in"] * (scale_b ** 0.5)
        if abs(scaled_prop_b - 34) / 34 > 0.20:
            st.warning(f"Calibration: At 50 MW scale, Config B propane header would be ~{scaled_prop_b:.0f}\" (benchmark 34\")")

# ============================================================================
# RIGHT COLUMN -- TABBED CHARTS
# ============================================================================

with right_col:
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "T-s Diagram", "Duct Comparison", "Approach dT Sweep",
        "Cost Waterfall", "Brine Utilization", "Sensitivity Tornado",
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

        # Condensing pressure annotation
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
        st.plotly_chart(fig_ts, use_container_width=True)

    # -- Tab 2: Duct Comparison ------------------------------------------------
    with tab2:
        fig_duct = make_subplots(rows=2, cols=1,
                                 subplot_titles=["Duct Diameters by Segment",
                                                 "Volumetric Flow Rates by Segment"],
                                 vertical_spacing=0.15)

        # Gather segment data
        seg_names_a = [s["name"] for s in duct_a["segments"]]
        seg_dia_a = [s["diameter_in"] for s in duct_a["segments"]]
        seg_vol_a = [s["vol_flow_ft3s"] for s in duct_a["segments"]]

        seg_names_b = [s["name"] for s in duct_b["segments"]]
        seg_dia_b = [s["diameter_in"] for s in duct_b["segments"]]
        seg_vol_b = [s["vol_flow_ft3s"] for s in duct_b["segments"]]

        # Pad to same length for grouped bars
        all_names = list(dict.fromkeys(seg_names_a + seg_names_b))  # preserve order
        dia_a_padded = [seg_dia_a[seg_names_a.index(n)] if n in seg_names_a else 0 for n in all_names]
        dia_b_padded = [seg_dia_b[seg_names_b.index(n)] if n in seg_names_b else 0 for n in all_names]
        vol_a_padded = [seg_vol_a[seg_names_a.index(n)] if n in seg_names_a else 0 for n in all_names]
        vol_b_padded = [seg_vol_b[seg_names_b.index(n)] if n in seg_names_b else 0 for n in all_names]

        fig_duct.add_trace(go.Bar(x=all_names, y=dia_a_padded, name="Config A",
                                  marker_color="steelblue"), row=1, col=1)
        fig_duct.add_trace(go.Bar(x=all_names, y=dia_b_padded, name="Config B",
                                  marker_color="indianred"), row=1, col=1)

        # Reference lines
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

        st.plotly_chart(fig_duct, use_container_width=True)

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
        st.dataframe(pd.DataFrame(seg_table).set_index("Config"), use_container_width=True)

    # -- Tab 3: Approach dT Sweep ----------------------------------------------
    with tab3:
        with st.spinner("Running approach temperature sweep..."):
            opt_result = optimize_approach_temp(inputs, fp)

        sweep = opt_result["sweep"]
        dts = [s["dt_approach"] for s in sweep]
        powers_b = [s["net_power_kw"] for s in sweep]
        inst_costs_b = [s["installed_cost"] / 1e6 for s in sweep]
        ihx_costs = [s["intermediate_hx_cost"] / 1e6 for s in sweep]
        lcoes = [s["lcoe"] for s in sweep]

        fig_sweep = make_subplots(rows=2, cols=1,
                                  subplot_titles=["Net Power & Installed Cost",
                                                  "LCOE"],
                                  vertical_spacing=0.12)

        # Power
        fig_sweep.add_trace(go.Scatter(x=dts, y=powers_b, name="Config B Net Power (kW)",
                                       line=dict(color="red", width=2)), row=1, col=1)
        fig_sweep.add_hline(y=opt_result["ref_power_a"], line_dash="dot",
                            line_color="blue", annotation_text="Config A ref",
                            row=1, col=1)

        # Cost on secondary axis -- put on same subplot
        fig_sweep.add_trace(go.Scatter(x=dts, y=inst_costs_b, name="Config B Total Cost ($MM)",
                                       line=dict(color="orange", width=2, dash="dash"),
                                       yaxis="y3"), row=1, col=1)
        fig_sweep.add_trace(go.Scatter(x=dts, y=ihx_costs, name="IHX Cost ($MM)",
                                       line=dict(color="gray", width=1, dash="dot"),
                                       yaxis="y3"), row=1, col=1)

        # LCOE
        fig_sweep.add_trace(go.Scatter(x=dts, y=lcoes, name="Config B LCOE ($/MWh)",
                                       line=dict(color="purple", width=2)), row=2, col=1)
        if opt_result["ref_lcoe_a"] > 0:
            fig_sweep.add_hline(y=opt_result["ref_lcoe_a"], line_dash="dot",
                                line_color="blue", annotation_text="Config A LCOE",
                                row=2, col=1)

        # Optimal line
        fig_sweep.add_vline(x=opt_result["optimal_dt"], line_dash="dash",
                            annotation_text=f"Optimal: {opt_result['optimal_dt']:.1f}degF")

        # Power penalty annotation
        if len(powers_b) >= 3:
            dp_per_dt = (powers_b[0] - powers_b[-1]) / (dts[-1] - dts[0])
            fig_sweep.add_annotation(
                x=15, y=max(powers_b) * 0.95,
                text=f"Power penalty: {abs(dp_per_dt):.1f} kW/degF",
                showarrow=False, font=dict(size=11), row=1, col=1,
            )

        fig_sweep.update_layout(height=650,
                                yaxis3=dict(overlaying="y", side="right",
                                            title="Cost ($MM)"))
        fig_sweep.update_xaxes(title_text="Approach dT (degF)")
        fig_sweep.update_yaxes(title_text="Net Power (kW)", row=1, col=1)
        fig_sweep.update_yaxes(title_text="LCOE ($/MWh)", row=2, col=1)

        st.plotly_chart(fig_sweep, use_container_width=True)

    # -- Tab 4: Cost Waterfall -------------------------------------------------
    with tab4:
        # Component list matching both configs
        comp_list = [
            ("Turbine/Generator", "turbine_generator"),
            ("Isopentane Pump", "iso_pump"),
            ("Vaporizer", "vaporizer"),
            ("Preheater", "preheater"),
            ("Recuperator", "recuperator"),
            ("Air-Cooled Condensers", "acc"),
            ("Vapor Ductwork", "ductwork"),
            ("Intermediate HX", "intermediate_hx"),
            ("Propane Pump", "propane_pump"),
            ("Propane Loop Piping", "propane_loop_piping"),
        ]

        comp_names = [c[0] for c in comp_list]
        vals_a = [costs_a[c[1]] / 1e6 for c in comp_list]
        vals_b = [costs_b[c[1]] / 1e6 for c in comp_list]
        deltas = [vb - va for va, vb in zip(vals_a, vals_b)]

        fig_wf = go.Figure()
        fig_wf.add_trace(go.Bar(x=comp_names, y=vals_a, name="Config A",
                                marker_color="steelblue"))
        fig_wf.add_trace(go.Bar(x=comp_names, y=vals_b, name="Config B",
                                marker_color="indianred"))
        fig_wf.add_trace(go.Bar(x=comp_names, y=deltas, name="Delta (B-A)",
                                marker_color="goldenrod", opacity=0.6))

        fig_wf.update_layout(
            title="Installed Cost Breakdown ($MM)",
            yaxis_title="Cost ($MM)",
            barmode="group",
            height=500,
        )
        st.plotly_chart(fig_wf, use_container_width=True)

        # Duct segment cost breakdown
        st.markdown("**Duct segment cost detail:**")
        duct_seg_data = []
        for seg in duct_a["segments"]:
            from cost_model import _duct_segment_cost
            duct_seg_data.append({"Config": "A", "Segment": seg["name"],
                                  "Cost ($)": f"${_duct_segment_cost(seg):,.0f}"})
        for seg in duct_b["segments"]:
            duct_seg_data.append({"Config": "B", "Segment": seg["name"],
                                  "Cost ($)": f"${_duct_segment_cost(seg):,.0f}"})
        st.dataframe(pd.DataFrame(duct_seg_data), use_container_width=True)

    # -- Tab 5: Brine Utilization ----------------------------------------------
    with tab5:
        fig_brine = go.Figure()

        # Config A brine and isopentane temperature profiles
        # Cumulative duty: preheater then vaporizer
        m_dot_a = perf_a["m_dot_iso"]  # lb/hr
        Q_pre_a = m_dot_a * perf_a["q_preheater"] / 1e6  # MMBtu/hr
        Q_vap_a = m_dot_a * perf_a["q_vaporizer"] / 1e6

        # Brine: enters at T_geo_in, exits preheater at T_brine_mid, exits at T_geo_out
        brine_q_a = [0, Q_pre_a, Q_pre_a + Q_vap_a]
        brine_T_a = [perf_a["T_geo_out_calc"], perf_a["T_brine_mid"], T_geo_in]

        # Isopentane: enters preheater at T6, exits at T7, then vaporizer to T1
        iso_q_a = [0, Q_pre_a, Q_pre_a + Q_vap_a]
        iso_T_a = [states_a["6"].T, states_a["7"].T, states_a["1"].T]

        fig_brine.add_trace(go.Scatter(x=brine_q_a, y=brine_T_a,
                                       name="A: Brine", mode="lines+markers",
                                       line=dict(color="blue", width=2)))
        fig_brine.add_trace(go.Scatter(x=iso_q_a, y=iso_T_a,
                                       name="A: Isopentane", mode="lines+markers",
                                       line=dict(color="blue", width=2, dash="dash")))

        # Config B
        m_dot_b = perf_b["m_dot_iso"]
        Q_pre_b = m_dot_b * perf_b["q_preheater"] / 1e6
        Q_vap_b = m_dot_b * perf_b["q_vaporizer"] / 1e6

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

        # Pinch point markers
        fig_brine.add_annotation(x=Q_pre_a, y=perf_a["T_brine_mid"],
                                 text=f"A pinch: {perf_a['vaporizer_pinch']:.1f}degF",
                                 showarrow=True, arrowhead=2, font=dict(color="blue"))
        fig_brine.add_annotation(x=Q_pre_b, y=perf_b["T_brine_mid"],
                                 text=f"B pinch: {perf_b['vaporizer_pinch']:.1f}degF",
                                 showarrow=True, arrowhead=2, font=dict(color="red"))

        # Min brine outlet line
        fig_brine.add_hline(y=T_geo_out_min, line_dash="dash", line_color="orange",
                            annotation_text=f"Min brine outlet: {T_geo_out_min}degF")

        fig_brine.update_layout(
            title="Brine Utilization -- Temperature vs Cumulative Heat Duty",
            xaxis_title="Cumulative Heat Duty (MMBtu/hr)",
            yaxis_title="Temperature (degF)",
            height=550,
        )
        st.plotly_chart(fig_brine, use_container_width=True)

    # -- Tab 6: Sensitivity Tornado --------------------------------------------
    with tab6:
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
        st.plotly_chart(fig_tornado, use_container_width=True)

# Footer
st.divider()
st.caption("ORC Comparator v2.0 -- CoolProp backend with optional REFPROP bridge")
