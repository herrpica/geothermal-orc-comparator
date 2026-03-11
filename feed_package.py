"""
FEED Light Package — Generate preliminary Front-End Engineering Design
deliverables from a selected optimizer result.

6 tabs:
  1. Heat & Material Balance (stream table, duty summary, power summary, Excel export)
  2. Equipment List (flat table with sizing, weights, vendors, lead times)
  3. Process Flow Description (stream narratives, Mermaid PFD, control philosophy)
  4. Instrumentation Summary (I/O counts, instrument list, control narratives)
  5. Plot Plan Summary (footprints, separations, estimated plot area)
  6. Open Items Log (critical / important / informational tiers)
"""

import io
import math
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

# ── Reference constants ─────────────────────────────────────────────────────

EQUIP_REF = {
    "turbine_generator": {
        "vendors": ["Exergy", "Turboden", "Ormat", "Baker Hughes"],
        "lead_wk": 14, "lb_per_kw": 35, "ft2_per_mw": 200,
        "confidence": "HIGH",
    },
    "vaporizer": {
        "vendors": ["Koch Heat Transfer", "Kelvion", "Bronswerk"],
        "lead_wk": 12, "lb_per_ft2": 3.0, "tube_length_ft": 20,
        "max_shell_area_ft2": 5000, "confidence": "HIGH",
    },
    "preheater": {
        "vendors": ["Koch Heat Transfer", "Kelvion", "Bronswerk"],
        "lead_wk": 12, "lb_per_ft2": 2.5, "tube_length_ft": 20,
        "max_shell_area_ft2": 5000, "confidence": "HIGH",
    },
    "recuperator": {
        "vendors": ["Alfa Laval", "Koch Heat Transfer", "Kelvion"],
        "lead_wk": 10, "lb_per_ft2": 2.0, "tube_length_ft": 16,
        "max_shell_area_ft2": 5000, "confidence": "HIGH",
    },
    "ihx": {
        "vendors": ["Koch Heat Transfer", "Kelvion", "Bronswerk"],
        "lead_wk": 12, "lb_per_ft2": 2.5, "tube_length_ft": 20,
        "max_shell_area_ft2": 5000, "confidence": "HIGH",
    },
    "acc": {
        "vendors": ["Worldwide Air Coolers", "SPX Cooling", "Harsco"],
        "lead_wk": 14, "tons_per_bay": 12, "confidence": "HIGH",
    },
    "iso_pump": {
        "vendors": ["Flowserve", "Sulzer", "Byron Jackson"],
        "lead_wk": 10, "lb_per_hp": 12, "confidence": "MEDIUM",
    },
    "prop_pump": {
        "vendors": ["Flowserve", "Sulzer", "Byron Jackson"],
        "lead_wk": 10, "lb_per_hp": 12, "confidence": "MEDIUM",
    },
    "controls": {
        "vendors": ["Honeywell", "ABB", "Emerson", "Siemens"],
        "lead_wk": 8, "confidence": "MEDIUM",
    },
    "electrical": {
        "vendors": ["ABB", "Eaton", "Schneider Electric", "Siemens"],
        "lead_wk": 16, "confidence": "MEDIUM",
    },
    "structural_steel": {
        "vendors": ["Local fabricator (competitive bid)"],
        "lead_wk": 10, "confidence": "LOW",
    },
    "wf_inventory": {
        "vendors": ["Targa Resources", "Enterprise Products"],
        "lead_wk": 4, "confidence": "LOW",
    },
    "ductwork": {
        "vendors": ["Local fabricator (competitive bid)"],
        "lead_wk": 8, "confidence": "LOW",
    },
}

SEPARATION_RULES = [
    ("ACC", "Turbine Building", 50, "Thermal interference, API 2510"),
    ("WF Storage", "Electrical Equipment", 100, "NFPA 30"),
    ("Brine Manifold", "Process Area", 25, "Piping economics"),
    ("Transformer", "Buildings", 25, "NEC / NFPA 70"),
    ("ACC", "Property Line", 75, "Noise setback"),
    ("Brine Wellhead", "Process Area", 50, "H2S / NCG dispersion"),
]

N_TRAINS = 2

# I/O counts per MW (gross)
IO_PER_MW = {"AI": 30, "AO": 8, "DI": 25, "DO": 17}

OPEN_ITEMS_CRITICAL = [
    "Brine chemistry analysis (silica, chlorides, NCG content)",
    "Site wind rose data for ACC orientation study",
    "Soil bearing capacity and geotechnical survey",
    "Electrical interconnection voltage, capacity, and distance",
    "Water rights / cooling water permit (if hybrid wet-dry)",
]

OPEN_ITEMS_IMPORTANT = [
    "Turbine vendor efficiency confirmation at design flow",
    "ACC structural steel weight at final design height",
    "Local labor rates and craft availability survey",
    "Seismic zone classification (IBC / ASCE 7)",
    "Environmental permit status (air / noise / visual)",
]

OPEN_ITEMS_INFO = [
    "Working fluid purity specification (isopentane 99.5%+)",
    "Spare parts philosophy (installed spare pump Y/N)",
    "Painting and insulation specification",
    "Noise survey and mitigation requirements",
    "Fire protection system scope (detection vs suppression)",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def rerun_analysis(result, design_basis: dict) -> dict:
    """Re-run the full ORC analysis for a stored OptResult to get enriched _detail."""
    from optimizer_engine import OptConfig
    from analysis_bridge import run_orc_analysis

    cfg = OptConfig(**result.config)
    tool_input = cfg.to_tool_input()
    return run_orc_analysis(tool_input, design_basis)


def _costs_from_detail(detail: dict) -> dict:
    """Reconstruct a costs-like dict from _detail cost keys."""
    costs = {}
    for k, v in detail.items():
        if k.startswith("cost_"):
            costs[k[5:]] = v
    # Sizing keys
    for k in ("vaporizer_area_ft2", "preheater_area_ft2", "recuperator_area_ft2",
              "acc_area_ft2", "acc_n_bays", "intermediate_hx_area_ft2",
              "structural_steel_weight_lb"):
        costs[k] = detail.get(k, 0)
    return costs


def _config_label(result) -> str:
    """Human-readable config descriptor."""
    cfg = result.config
    topo_map = {"recuperated": "Recuperated", "basic": "Basic", "dual_pressure": "Dual-Pressure"}
    hr_map = {"direct_acc": "Direct ACC", "propane_intermediate": "Propane IHX",
              "hybrid_wet_dry": "Hybrid Wet-Dry"}
    from cost_model import STRATEGY_SHORT_LABELS
    fluid = cfg.get("working_fluid", "isopentane").title()
    topo = topo_map.get(cfg.get("topology", ""), cfg.get("topology", ""))
    hr = hr_map.get(cfg.get("heat_rejection", ""), cfg.get("heat_rejection", ""))
    strat = STRATEGY_SHORT_LABELS.get(result.procurement_strategy, result.procurement_strategy)
    return f"{fluid} / {topo} / {hr} / {strat}"


def _config_type(detail: dict) -> str:
    """Return config letter (A, B, D)."""
    return detail.get("config", "A")


def _hx_shells(area_ft2: float, max_shell: float = 5000) -> int:
    """Number of shell-and-tube shells needed."""
    if area_ft2 <= 0:
        return 0
    return max(1, math.ceil(area_ft2 / max_shell))


def _hx_shell_dia_in(area_ft2: float, tube_length_ft: float = 20) -> float:
    """Estimate shell diameter from area and tube length."""
    if area_ft2 <= 0:
        return 0
    # Approximate: area = pi * d_tube * L * N_tubes, shell_area ~ pi/4 * D^2 * pitch
    # Simplified: D ~ sqrt(area / (pi * L * 3))  (3 tubes per sq-in pitch)
    per_shell = min(area_ft2, 5000)
    return max(24, min(72, 12 * math.sqrt(per_shell / (math.pi * tube_length_ft * 3))))


# ── TAB 1: Heat & Material Balance ──────────────────────────────────────────

def _render_hmb_tab(detail: dict, result, config_label: str):
    """Heat & Material Balance — stream table, duty summary, power summary."""
    config = _config_type(detail)
    states = detail.get("states", {})
    power_bal = detail.get("power_balance", {})
    m_dot_iso_lb_hr = detail.get("m_dot_iso_lb_hr", 0)
    m_dot_geo_lb_hr = detail.get("m_dot_geo_lb_s", 0) * 3600

    # ── Stream Table ────────────────────────────────────────────
    st.markdown("### Stream Table")
    streams = []

    # Brine streams
    T_geo_in = detail.get("T_geo_in_F", 420)
    T_geo_out = detail.get("T_geo_out_min_F", 180)
    # T_brine_mid for vaporizer/preheater boundary: approximate from evap + pinch
    T_brine_mid = detail.get("T_evap_F", 300) + detail.get("P_high_psia", 0) * 0  # use evap
    # Better: use preheater area relationship — but simpler: geo_out is after preheater
    streams.append({
        "Stream": "B-1", "Description": "Brine from production wells",
        "Fluid": "Brine", "T (F)": f"{T_geo_in:.0f}",
        "P (psia)": "-", "Flow (lb/hr)": f"{m_dot_geo_lb_hr:,.0f}",
        "Phase": "Liquid", "h (BTU/lb)": "-",
    })
    streams.append({
        "Stream": "B-2", "Description": "Brine to reinjection",
        "Fluid": "Brine", "T (F)": f"{T_geo_out:.0f}",
        "P (psia)": "-", "Flow (lb/hr)": f"{m_dot_geo_lb_hr:,.0f}",
        "Phase": "Liquid", "h (BTU/lb)": "-",
    })

    # Isopentane streams
    fluid_name = detail.get("working_fluid", "isopentane").title()
    for key in sorted(states.keys(), key=lambda x: (not x.isdigit(), x)):
        sp = states[key]
        streams.append({
            "Stream": f"WF-{key}", "Description": sp.get("label", f"State {key}"),
            "Fluid": fluid_name, "T (F)": f"{sp['T']:.1f}",
            "P (psia)": f"{sp['P']:.1f}", "Flow (lb/hr)": f"{m_dot_iso_lb_hr:,.0f}",
            "Phase": sp.get("phase", ""), "h (BTU/lb)": f"{sp['h']:.1f}",
        })

    # Propane streams (Config B)
    if config == "B":
        prop_states = detail.get("prop_states", {})
        m_dot_prop = detail.get("m_dot_prop_lb_hr", 0)
        for key in sorted(prop_states.keys()):
            sp = prop_states[key]
            streams.append({
                "Stream": f"PR-{key}", "Description": sp.get("label", f"Propane {key}"),
                "Fluid": "Propane", "T (F)": f"{sp['T']:.1f}",
                "P (psia)": f"{sp['P']:.1f}", "Flow (lb/hr)": f"{m_dot_prop:,.0f}",
                "Phase": sp.get("phase", ""), "h (BTU/lb)": f"{sp['h']:.1f}",
            })

    stream_df = pd.DataFrame(streams)
    st.dataframe(stream_df, use_container_width=True, hide_index=True)

    # ── Duty Summary ────────────────────────────────────────────
    st.markdown("### Duty Summary")
    duties = []
    costs = _costs_from_detail(detail)

    vap_area = costs.get("vaporizer_area_ft2", 0)
    pre_area = costs.get("preheater_area_ft2", 0)
    recup_area = costs.get("recuperator_area_ft2", 0)
    acc_area = costs.get("acc_area_ft2", 0)
    ihx_area = costs.get("intermediate_hx_area_ft2", 0)

    # Compute duties from state points if available
    if "1" in states and "7" in states:
        q_vap_btu_lb = states["1"]["h"] - states["7"]["h"]
        q_vap_mmbtu = m_dot_iso_lb_hr * q_vap_btu_lb / 1e6
    else:
        q_vap_mmbtu = 0
    if "7" in states and "6" in states:
        q_pre_btu_lb = states["7"]["h"] - states["6"]["h"]
        q_pre_mmbtu = m_dot_iso_lb_hr * q_pre_btu_lb / 1e6
    else:
        q_pre_mmbtu = 0
    if "2" in states and "3" in states:
        q_recup_btu_lb = states["2"]["h"] - states["3"]["h"]
        q_recup_mmbtu = m_dot_iso_lb_hr * q_recup_btu_lb / 1e6
    else:
        q_recup_mmbtu = 0

    Q_reject = detail.get("Q_reject_mmbtu_hr", 0)

    duties.append({
        "Equipment": "Vaporizer", "Duty (MMBtu/hr)": f"{q_vap_mmbtu:.1f}",
        "Area (ft2)": f"{vap_area:,.0f}", "Shells": _hx_shells(vap_area),
    })
    duties.append({
        "Equipment": "Preheater", "Duty (MMBtu/hr)": f"{q_pre_mmbtu:.1f}",
        "Area (ft2)": f"{pre_area:,.0f}", "Shells": _hx_shells(pre_area),
    })
    if recup_area > 0:
        duties.append({
            "Equipment": "Recuperator", "Duty (MMBtu/hr)": f"{q_recup_mmbtu:.1f}",
            "Area (ft2)": f"{recup_area:,.0f}", "Shells": _hx_shells(recup_area),
        })
    if config == "B" and ihx_area > 0:
        duties.append({
            "Equipment": "Intermediate HX", "Duty (MMBtu/hr)": "-",
            "Area (ft2)": f"{ihx_area:,.0f}", "Shells": _hx_shells(ihx_area),
        })
    duties.append({
        "Equipment": "ACC", "Duty (MMBtu/hr)": f"{Q_reject:.1f}",
        "Area (ft2)": f"{acc_area:,.0f}", "Shells": int(costs.get("acc_n_bays", 0)),
    })

    duty_df = pd.DataFrame(duties)
    st.dataframe(duty_df, use_container_width=True, hide_index=True)

    # ── Power Summary ───────────────────────────────────────────
    st.markdown("### Power Summary")
    P_gross = power_bal.get("P_gross", 0)
    rows = []
    rows.append({"Item": "Gross Power (Turbine-Generator)", "Value (kW)": f"{P_gross:,.0f}",
                 "% of Gross": "100.0%"})
    for label, key in [("Isopentane Pump", "W_iso_pump"), ("Propane Pump", "W_prop_pump"),
                       ("ACC Fans", "W_acc_fans"), ("Auxiliary", "W_auxiliary")]:
        val = power_bal.get(key, 0)
        if val > 0:
            pct = val / P_gross * 100 if P_gross > 0 else 0
            rows.append({"Item": f"  (-) {label}", "Value (kW)": f"{val:,.0f}",
                         "% of Gross": f"{pct:.1f}%"})
    P_net = power_bal.get("P_net", 0)
    net_pct = P_net / P_gross * 100 if P_gross > 0 else 0
    rows.append({"Item": "Net Power", "Value (kW)": f"{P_net:,.0f}",
                 "% of Gross": f"{net_pct:.1f}%"})

    power_df = pd.DataFrame(rows)
    st.dataframe(power_df, use_container_width=True, hide_index=True)

    # ── Excel Export ────────────────────────────────────────────
    st.markdown("### Export")
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        stream_df.to_excel(writer, sheet_name="Stream Table", index=False)
        duty_df.to_excel(writer, sheet_name="Duty Summary", index=False)
        power_df.to_excel(writer, sheet_name="Power Summary", index=False)
    st.download_button(
        "Download H&MB (Excel)",
        data=buf.getvalue(),
        file_name=f"FEED_HMB_Run{result.run_id}.xlsx",
        mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )


# ── TAB 2: Equipment List ───────────────────────────────────────────────────

def _render_equipment_tab(detail: dict, result):
    """Equipment list with sizing, weights, vendors, lead times, confidence."""
    st.markdown("### Equipment List")
    config = _config_type(detail)
    costs = _costs_from_detail(detail)
    power_bal = detail.get("power_balance", {})
    P_gross = power_bal.get("P_gross", 0)
    P_net = power_bal.get("P_net", 0)

    rows = []

    # ── Turbine-Generator ───────────────────────────────────────
    ref = EQUIP_REF["turbine_generator"]
    tg_cost = costs.get("turbine_generator", 0)
    tg_per_kw = tg_cost / P_net if P_net > 0 else 0
    tg_weight_lb = ref["lb_per_kw"] * P_gross
    tg_footprint = ref["ft2_per_mw"] * (P_gross / 1000)
    tg_dim = f"{math.sqrt(tg_footprint / N_TRAINS):.0f} x {math.sqrt(tg_footprint / N_TRAINS):.0f}"
    rows.append({
        "Tag": "TG-0101A/B", "Service": "Turbine-Generator Set",
        "Qty": N_TRAINS,
        "Key Parameters": f"{P_gross / N_TRAINS / 1000:.1f} MW each, "
                          f"eta={detail.get('eta_turbine', 0.82):.0%}",
        "Dimensions (ft)": tg_dim,
        "Weight (tons)": f"{tg_weight_lb / 2000 / N_TRAINS:.0f} ea",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${tg_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    # ── Heat Exchangers ─────────────────────────────────────────
    hx_items = [
        ("HX-0101", "Vaporizer", "vaporizer", "vaporizer_area_ft2"),
        ("HX-0102", "Preheater", "preheater", "preheater_area_ft2"),
    ]
    # Only include recuperator if area > 0
    if costs.get("recuperator_area_ft2", 0) > 0:
        hx_items.append(("HX-0103", "Recuperator", "recuperator", "recuperator_area_ft2"))
    if config == "B" and costs.get("intermediate_hx_area_ft2", 0) > 0:
        hx_items.append(("HX-0104", "Intermediate HX", "ihx", "intermediate_hx_area_ft2"))

    for tag, service, ref_key, area_key in hx_items:
        ref = EQUIP_REF[ref_key]
        area = costs.get(area_key, 0)
        n_shells = _hx_shells(area, ref.get("max_shell_area_ft2", 5000))
        area_per_shell = area / n_shells if n_shells > 0 else 0
        shell_dia = _hx_shell_dia_in(area_per_shell, ref.get("tube_length_ft", 20))
        weight_lb = area * ref.get("lb_per_ft2", 2.5)
        hx_cost = costs.get(ref_key, 0)
        hx_per_kw = hx_cost / P_net if P_net > 0 else 0
        rows.append({
            "Tag": tag, "Service": service,
            "Qty": n_shells,
            "Key Parameters": f"{area:,.0f} ft2 total",
            "Dimensions (ft)": f"{shell_dia:.0f}\" dia x {ref.get('tube_length_ft', 20)}' lg",
            "Weight (tons)": f"{weight_lb / 2000:.0f}",
            "Vendor(s)": ", ".join(ref["vendors"][:2]),
            "Budget ($/kW)": f"${hx_per_kw:,.0f}",
            "Lead (wk)": ref["lead_wk"],
            "Confidence": ref["confidence"],
        })

    # ── ACC ──────────────────────────────────────────────────────
    ref = EQUIP_REF["acc"]
    n_bays = int(costs.get("acc_n_bays", 0))
    acc_cost = costs.get("acc", 0)
    acc_per_kw = acc_cost / P_net if P_net > 0 else 0
    acc_weight_tons = n_bays * ref["tons_per_bay"]
    acc_area = costs.get("acc_area_ft2", 0)
    # Typical bay: 40 ft x 12 ft
    bay_w, bay_l = 40, 12
    rows.append({
        "Tag": "AC-0101", "Service": "Air-Cooled Condenser",
        "Qty": n_bays,
        "Key Parameters": f"{acc_area:,.0f} ft2, {n_bays} bays, "
                          f"{detail.get('fan_n_fans_used', 0)} fans",
        "Dimensions (ft)": f"{bay_w} x {bay_l} per bay",
        "Weight (tons)": f"{acc_weight_tons:,.0f}",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${acc_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    # ── Pumps ────────────────────────────────────────────────────
    ref = EQUIP_REF["iso_pump"]
    iso_hp = detail.get("pump_iso_power_hp", 0)
    iso_gpm = detail.get("pump_iso_flow_gpm", 0)
    iso_pump_cost = costs.get("iso_pump", 0)
    iso_pump_per_kw = iso_pump_cost / P_net if P_net > 0 else 0
    iso_pump_wt = iso_hp * ref["lb_per_hp"]
    rows.append({
        "Tag": "PP-0101A/B", "Service": "Isopentane Feed Pump",
        "Qty": f"{N_TRAINS} + 1 spare",
        "Key Parameters": f"{iso_gpm:,.0f} GPM, {detail.get('pump_iso_dP_psi', 0):.0f} psi dP, "
                          f"{iso_hp:.0f} HP",
        "Dimensions (ft)": f"Skid 6 x 4",
        "Weight (tons)": f"{iso_pump_wt / 2000:.1f} ea",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${iso_pump_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    if config == "B":
        ref = EQUIP_REF["prop_pump"]
        prop_hp = detail.get("pump_prop_power_hp", 0)
        prop_gpm = detail.get("pump_prop_flow_gpm", 0)
        prop_pump_cost = costs.get("propane_system", 0) * 0.15  # pump fraction
        prop_per_kw = prop_pump_cost / P_net if P_net > 0 else 0
        rows.append({
            "Tag": "PP-0102A/B", "Service": "Propane Circulation Pump",
            "Qty": f"{N_TRAINS} + 1 spare",
            "Key Parameters": f"{prop_gpm:,.0f} GPM, {detail.get('pump_prop_dP_psi', 0):.0f} psi dP, "
                              f"{prop_hp:.0f} HP",
            "Dimensions (ft)": "Skid 6 x 4",
            "Weight (tons)": f"{prop_hp * ref['lb_per_hp'] / 2000:.1f} ea",
            "Vendor(s)": ", ".join(ref["vendors"][:2]),
            "Budget ($/kW)": f"${prop_per_kw:,.0f}",
            "Lead (wk)": ref["lead_wk"],
            "Confidence": ref["confidence"],
        })

    # ── Controls & Electrical ────────────────────────────────────
    ref = EQUIP_REF["controls"]
    ctrl_cost = costs.get("controls_instrumentation", 0)
    ctrl_per_kw = ctrl_cost / P_net if P_net > 0 else 0
    rows.append({
        "Tag": "DCS-0101", "Service": "Plant Control System (PLC/DCS)",
        "Qty": 1,
        "Key Parameters": "Redundant controllers, HMI, historian",
        "Dimensions (ft)": "Cabinet 8 x 3",
        "Weight (tons)": "1",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${ctrl_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    ref = EQUIP_REF["electrical"]
    elec_cost = costs.get("electrical_equipment", 0)
    elec_per_kw = elec_cost / P_net if P_net > 0 else 0
    rows.append({
        "Tag": "EL-0101", "Service": "Electrical Equipment (MCC, Switchgear, Xfmr)",
        "Qty": 1,
        "Key Parameters": f"{P_net / 1000:.1f} MW export capacity",
        "Dimensions (ft)": "Substation 40 x 30",
        "Weight (tons)": f"{P_net / 1000 * 8:.0f}",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${elec_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    # ── Structural Steel ─────────────────────────────────────────
    ref = EQUIP_REF["structural_steel"]
    steel_wt_lb = costs.get("structural_steel_weight_lb", 0)
    steel_cost = costs.get("structural_steel", 0)
    steel_per_kw = steel_cost / P_net if P_net > 0 else 0
    rows.append({
        "Tag": "SS-0101", "Service": "Structural Steel & Pipe Racks",
        "Qty": 1,
        "Key Parameters": f"{steel_wt_lb / 2000:,.0f} tons",
        "Dimensions (ft)": "Plant-wide",
        "Weight (tons)": f"{steel_wt_lb / 2000:,.0f}",
        "Vendor(s)": ref["vendors"][0],
        "Budget ($/kW)": f"${steel_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    # ── WF Inventory ─────────────────────────────────────────────
    ref = EQUIP_REF["wf_inventory"]
    wf_cost = costs.get("wf_inventory", 0)
    wf_per_kw = wf_cost / P_net if P_net > 0 else 0
    rows.append({
        "Tag": "WF-0101", "Service": "Working Fluid Inventory",
        "Qty": 1,
        "Key Parameters": f"{detail.get('working_fluid', 'isopentane').title()} charge",
        "Dimensions (ft)": "Storage tank TBD",
        "Weight (tons)": "-",
        "Vendor(s)": ", ".join(ref["vendors"][:2]),
        "Budget ($/kW)": f"${wf_per_kw:,.0f}",
        "Lead (wk)": ref["lead_wk"],
        "Confidence": ref["confidence"],
    })

    equip_df = pd.DataFrame(rows)
    st.dataframe(equip_df, use_container_width=True, hide_index=True)

    # Confidence legend
    st.caption(
        "**Confidence:** HIGH = vendor-quoted basis | "
        "MEDIUM = parametric estimate | LOW = allocation-based"
    )


# ── TAB 3: Process Flow Description ─────────────────────────────────────────

def _render_process_flow_tab(detail: dict, result):
    """Process flow description with stream narratives, Mermaid PFD, control philosophy."""
    config = _config_type(detail)
    states = detail.get("states", {})
    T_geo_in = detail.get("T_geo_in_F", 420)
    T_geo_out = detail.get("T_geo_out_min_F", 180)
    T_cond = detail.get("T_cond_F", 0)
    T_evap = detail.get("T_evap_F", 0)
    P_high = detail.get("P_high_psia", 0)
    P_low = detail.get("P_low_psia", 0)
    fluid = detail.get("working_fluid", "isopentane").title()
    n_fans = detail.get("fan_n_fans_used", 0)
    P_gross = detail.get("power_balance", {}).get("P_gross", 0)

    # ── Stream Descriptions ─────────────────────────────────────
    st.markdown("### Stream Descriptions")

    narratives = [
        f"**B-1:** Hot brine enters the plant at **{T_geo_in:.0f} F** from production wells "
        f"at **{detail.get('m_dot_geo_lb_s', 0) * 3600:,.0f} lb/hr**.",
        f"**B-2:** Cooled brine exits the preheater at approximately **{T_geo_out:.0f} F** "
        f"and flows to reinjection wells.",
    ]

    if config in ("A", "B"):
        if "4" in states and "5" in states and "6" in states and "7" in states:
            narratives.extend([
                f"**WF-4:** Subcooled {fluid} exits the feed pump at **{states['4']['T']:.0f} F**, "
                f"**{states['4']['P']:.0f} psia**.",
            ])
            if "5" in states and states["5"]["h"] != states["4"]["h"]:
                narratives.append(
                    f"**WF-5/6:** {fluid} is preheated through the recuperator cold side "
                    f"to **{states['6']['T']:.0f} F**."
                )
            narratives.extend([
                f"**WF-7:** {fluid} enters the vaporizer at **{states['7']['T']:.0f} F** "
                f"after passing through the preheater.",
                f"**WF-1:** Saturated {fluid} vapor exits the vaporizer at **{T_evap:.0f} F**, "
                f"**{P_high:.0f} psia** and enters the turbine.",
                f"**WF-2:** Expanded vapor exits the turbine at **{states.get('2', {}).get('T', T_cond):.0f} F**, "
                f"**{P_low:.0f} psia**.",
                f"**WF-3:** {fluid} exits the recuperator hot side (or turbine exhaust for basic cycle) "
                f"and enters the condenser.",
            ])

    if config == "B":
        prop_states = detail.get("prop_states", {})
        if prop_states:
            narratives.extend([
                f"**PR-A:** Propane vapor exits the IHX shell side at "
                f"**{prop_states.get('A', {}).get('T', 0):.0f} F** and flows to the propane ACC.",
                f"**PR-B:** Subcooled propane exits the ACC at "
                f"**{prop_states.get('B', {}).get('T', 0):.0f} F**.",
                f"**PR-C:** Propane enters the IHX tube side after the circulation pump.",
            ])

    for n in narratives:
        st.markdown(n)

    # ── Equipment Connections (Mermaid) ─────────────────────────
    st.markdown("### Process Flow Diagram")

    if config == "A":
        mermaid = f"""```mermaid
graph LR
    WELLS[Production Wells<br/>{T_geo_in:.0f} F] --> VAP[Vaporizer<br/>HX-0101]
    VAP --> PRE[Preheater<br/>HX-0102]
    PRE --> REINJ[Reinjection<br/>{T_geo_out:.0f} F]
    PP[Feed Pump<br/>PP-0101] --> RECUP_C[Recuperator<br/>Cold Side]
    RECUP_C --> PRE2[Preheater<br/>WF Side]
    PRE2 --> VAP2[Vaporizer<br/>WF Side]
    VAP2 --> TG[Turbine-Generator<br/>TG-0101<br/>{P_gross/1000:.1f} MW]
    TG --> RECUP_H[Recuperator<br/>Hot Side]
    RECUP_H --> ACC[ACC<br/>AC-0101<br/>{n_fans} fans]
    ACC --> PP
```"""
    elif config == "B":
        T_prop_cond = detail.get("T_propane_cond_F", 0)
        mermaid = f"""```mermaid
graph LR
    WELLS[Production Wells<br/>{T_geo_in:.0f} F] --> VAP[Vaporizer<br/>HX-0101]
    VAP --> PRE[Preheater<br/>HX-0102]
    PRE --> REINJ[Reinjection<br/>{T_geo_out:.0f} F]
    PP[Feed Pump<br/>PP-0101] --> RECUP_C[Recuperator<br/>Cold Side]
    RECUP_C --> PRE2[Preheater<br/>WF Side]
    PRE2 --> VAP2[Vaporizer<br/>WF Side]
    VAP2 --> TG[Turbine-Generator<br/>TG-0101<br/>{P_gross/1000:.1f} MW]
    TG --> RECUP_H[Recuperator<br/>Hot Side]
    RECUP_H --> IHX[Intermediate HX<br/>HX-0104]
    IHX --> PP
    IHX --> |Propane| ACC[Propane ACC<br/>AC-0101<br/>{T_prop_cond:.0f} F]
    ACC --> PPR[Propane Pump<br/>PP-0102]
    PPR --> IHX
```"""
    else:  # Config D
        T_split = detail.get("T_split_F", 0)
        mermaid = f"""```mermaid
graph LR
    WELLS[Production Wells<br/>{T_geo_in:.0f} F] --> HP_VAP[HP Vaporizer]
    HP_VAP --> HP_PRE[HP Preheater]
    HP_PRE --> |{T_split:.0f} F| LP_VAP[LP Vaporizer]
    LP_VAP --> LP_PRE[LP Preheater]
    LP_PRE --> REINJ[Reinjection<br/>{T_geo_out:.0f} F]
    HP_PP[HP Pump] --> HP_RECUP[HP Recuperator]
    HP_RECUP --> HP_PRE2[HP Pre WF]
    HP_PRE2 --> HP_VAP2[HP Vap WF]
    HP_VAP2 --> HP_TG[HP Turbine<br/>{detail.get('hp_gross_power_kw',0)/1000:.1f} MW]
    HP_TG --> HP_RECUP_H[HP Recup Hot]
    HP_RECUP_H --> ACC[Shared ACC<br/>{n_fans} fans]
    LP_PP[LP Pump] --> LP_PRE2[LP Pre WF]
    LP_PRE2 --> LP_VAP2[LP Vap WF]
    LP_VAP2 --> LP_TG[LP Turbine<br/>{detail.get('lp_gross_power_kw',0)/1000:.1f} MW]
    LP_TG --> ACC
    ACC --> HP_PP
    ACC --> LP_PP
```"""

    st.markdown(mermaid)
    st.caption("*Diagram is schematic — not to scale. For use by engineer to produce formal PFD.*")

    # ── Control Philosophy ──────────────────────────────────────
    st.markdown("### Control Philosophy")

    fan_resolution = 100 / n_fans if n_fans > 0 else 0
    ctrl_loops = [
        f"**1. Condensing Pressure Control:** Modulate ACC fans to maintain condensing "
        f"pressure at **{P_low:.0f} psia** (±2 psi). Staging-only control "
        f"({n_fans} fans = {fan_resolution:.1f}% resolution per step). No VFDs.",
        f"**2. Brine Flow Control:** Brine flow rate monitored at plant inlet. "
        f"Alarm on deviation >5% from design rate of "
        f"**{detail.get('m_dot_geo_lb_s', 0) * 3600:,.0f} lb/hr**. "
        f"Flow controlled at wellfield, not at plant.",
        f"**3. Turbine Speed/Load Control:** Electronic governor maintains synchronous speed. "
        f"Load controlled by working fluid mass flow rate. "
        f"Gross output target: **{P_gross / 1000:.1f} MW**.",
        f"**4. ACC Fan Staging:** Fans staged on/off based on condensing pressure setpoint. "
        f"{n_fans} fans provide step resolution of {fan_resolution:.1f}% per step. "
        f"Lead/lag rotation for even wear distribution.",
    ]
    for loop in ctrl_loops:
        st.markdown(loop)

    st.info(
        "This output is intended for use by an engineer to produce a formal PFD drawing. "
        "It is not a substitute for a stamped drawing."
    )


# ── TAB 4: Instrumentation Summary ──────────────────────────────────────────

def _render_instrumentation_tab(detail: dict, result):
    """I/O count estimation, instrument list by service area, control narratives."""
    config = _config_type(detail)
    power_bal = detail.get("power_balance", {})
    P_gross_mw = power_bal.get("P_gross", 0) / 1000
    P_low = detail.get("P_low_psia", 0)
    n_fans = detail.get("fan_n_fans_used", 0)

    # ── I/O Count ───────────────────────────────────────────────
    st.markdown("### I/O Count Estimation")
    io_rows = []
    total_io = 0
    for io_type, per_mw in IO_PER_MW.items():
        count = max(1, round(per_mw * P_gross_mw))
        total_io += count
        io_rows.append({
            "Type": io_type,
            "Count": count,
            "Per MW": per_mw,
            "Description": {
                "AI": "Temperature, pressure, flow, level transmitters",
                "AO": "Valve positioners, VFD commands",
                "DI": "Motor status, valve position, alarms",
                "DO": "Motor start/stop, valve open/close, relay trips",
            }[io_type],
        })
    io_rows.append({"Type": "TOTAL", "Count": total_io, "Per MW": sum(IO_PER_MW.values()),
                     "Description": ""})
    st.dataframe(pd.DataFrame(io_rows), use_container_width=True, hide_index=True)

    # ── Instrument List by Service Area ─────────────────────────
    st.markdown("### Instrument List by Service Area")

    instruments = []
    # Brine system
    instruments.extend([
        {"Area": "Brine System", "Tag": "FT-1001", "Service": "Brine inlet flow",
         "Type": "Vortex / Mag", "Range": "0-2000 GPM"},
        {"Area": "Brine System", "Tag": "TT-1001", "Service": "Brine inlet temperature",
         "Type": "RTD", "Range": "0-500 F"},
        {"Area": "Brine System", "Tag": "PT-1001", "Service": "Brine inlet pressure",
         "Type": "Pressure xmtr", "Range": "0-300 psig"},
        {"Area": "Brine System", "Tag": "TT-1002", "Service": "Brine outlet temperature",
         "Type": "RTD", "Range": "0-500 F"},
    ])
    # Isopentane circuit
    for i, (state_key, svc) in enumerate([
        ("4", "Pump discharge"), ("6", "Recuperator cold out"),
        ("7", "Preheater outlet"), ("1", "Vaporizer outlet / turbine inlet"),
        ("2", "Turbine exhaust"), ("3", "Condenser inlet"),
    ], start=1):
        instruments.append({
            "Area": "Isopentane Circuit", "Tag": f"TT-200{i}",
            "Service": f"WF {svc} temp", "Type": "RTD", "Range": "0-500 F",
        })
        instruments.append({
            "Area": "Isopentane Circuit", "Tag": f"PT-200{i}",
            "Service": f"WF {svc} pressure", "Type": "Pressure xmtr", "Range": "0-600 psig",
        })
    instruments.append({
        "Area": "Isopentane Circuit", "Tag": "FT-2001",
        "Service": "WF pump discharge flow", "Type": "Coriolis", "Range": "0-5000 GPM",
    })
    instruments.append({
        "Area": "Isopentane Circuit", "Tag": "LT-2001",
        "Service": "WF receiver level", "Type": "DP Level", "Range": "0-100%",
    })

    # ACC system
    instruments.extend([
        {"Area": "ACC System", "Tag": "TT-3001", "Service": "Air inlet temperature",
         "Type": "RTD", "Range": "-20 to 130 F"},
        {"Area": "ACC System", "Tag": "TT-3002", "Service": "Air outlet temperature",
         "Type": "RTD", "Range": "0-200 F"},
    ])
    for i in range(1, min(n_fans + 1, 6)):  # show first 5 fans
        instruments.append({
            "Area": "ACC System", "Tag": f"VT-300{i}",
            "Service": f"Fan {i} vibration", "Type": "Accelerometer", "Range": "0-2 in/s",
        })

    # Turbine
    instruments.extend([
        {"Area": "Turbine", "Tag": "ST-4001", "Service": "Turbine speed",
         "Type": "Proximity probe", "Range": "0-5000 RPM"},
        {"Area": "Turbine", "Tag": "TT-4001", "Service": "Bearing temp (DE)",
         "Type": "RTD", "Range": "0-300 F"},
        {"Area": "Turbine", "Tag": "TT-4002", "Service": "Bearing temp (NDE)",
         "Type": "RTD", "Range": "0-300 F"},
        {"Area": "Turbine", "Tag": "VT-4001", "Service": "Shaft vibration (DE)",
         "Type": "Proximity probe", "Range": "0-5 mil"},
        {"Area": "Turbine", "Tag": "VT-4002", "Service": "Shaft vibration (NDE)",
         "Type": "Proximity probe", "Range": "0-5 mil"},
        {"Area": "Turbine", "Tag": "PT-4001", "Service": "Lube oil pressure",
         "Type": "Pressure xmtr", "Range": "0-100 psig"},
    ])

    # Electrical
    instruments.extend([
        {"Area": "Electrical", "Tag": "JT-5001", "Service": "Generator MW output",
         "Type": "Power meter", "Range": "0-100 MW"},
        {"Area": "Electrical", "Tag": "JT-5002", "Service": "Voltage",
         "Type": "PT", "Range": "0-15 kV"},
        {"Area": "Electrical", "Tag": "JT-5003", "Service": "Frequency",
         "Type": "Freq relay", "Range": "58-62 Hz"},
    ])

    # Propane (Config B)
    if config == "B":
        instruments.extend([
            {"Area": "Propane Circuit", "Tag": "TT-6001", "Service": "Propane IHX outlet temp",
             "Type": "RTD", "Range": "0-300 F"},
            {"Area": "Propane Circuit", "Tag": "PT-6001", "Service": "Propane ACC inlet pressure",
             "Type": "Pressure xmtr", "Range": "0-400 psig"},
            {"Area": "Propane Circuit", "Tag": "FT-6001", "Service": "Propane circulation flow",
             "Type": "Coriolis", "Range": "0-5000 GPM"},
        ])

    inst_df = pd.DataFrame(instruments)
    st.dataframe(inst_df, use_container_width=True, hide_index=True)

    # ── Control Narratives ──────────────────────────────────────
    st.markdown("### Control Narratives")

    fan_res = 100 / n_fans if n_fans > 0 else 0
    narratives = [
        f"**Condensing Pressure Control:** PIC-3001 modulates ACC fan staging to maintain "
        f"condensing pressure at **{P_low:.0f} ±2 psia**. Fans staged on/off in sequence; "
        f"{n_fans} fans provide **{fan_res:.1f}%** resolution per step.",
        f"**Brine Flow Monitoring:** FT-1001 monitors brine flow. High/low alarm at ±5% "
        f"of design rate. Trip on low-low flow (<80% design).",
        f"**Turbine Governor:** ST-4001 speed input to electronic governor. "
        f"Synchronous speed maintained ±0.5%. Load rejection protection included.",
        f"**ACC Fan Staging:** Lead/lag rotation every 500 hours. "
        f"Auto-start sequence on rising condensing pressure. "
        f"Freeze protection: maintain minimum air flow at T_amb < 32 F.",
    ]
    for n in narratives:
        st.markdown(n)


# ── TAB 5: Plot Plan Summary ────────────────────────────────────────────────

def _render_plot_plan_tab(detail: dict, result):
    """Equipment footprints, separation requirements, estimated plot area."""
    config = _config_type(detail)
    costs = _costs_from_detail(detail)
    power_bal = detail.get("power_balance", {})
    P_gross = power_bal.get("P_gross", 0)
    P_net = power_bal.get("P_net", 0)
    n_bays = int(costs.get("acc_n_bays", 0))

    # ── Equipment Footprint Table ───────────────────────────────
    st.markdown("### Equipment Footprint Estimate")

    footprints = []
    total_ft2 = 0

    # Turbine building
    tg_ft2 = EQUIP_REF["turbine_generator"]["ft2_per_mw"] * (P_gross / 1000)
    tg_side = math.sqrt(tg_ft2) if tg_ft2 > 0 else 0
    footprints.append({
        "Equipment": "Turbine-Generator Building",
        "Count": N_TRAINS, "Each (ft x ft)": f"{tg_side:.0f} x {tg_side:.0f}",
        "Total (ft2)": f"{tg_ft2:,.0f}", "Source": "Calculated",
    })
    total_ft2 += tg_ft2

    # Heat exchangers
    for label, area_key, ref_key in [
        ("Vaporizer", "vaporizer_area_ft2", "vaporizer"),
        ("Preheater", "preheater_area_ft2", "preheater"),
        ("Recuperator", "recuperator_area_ft2", "recuperator"),
    ]:
        area = costs.get(area_key, 0)
        if area <= 0:
            continue
        ref = EQUIP_REF[ref_key]
        n_shells = _hx_shells(area, ref.get("max_shell_area_ft2", 5000))
        tube_len = ref.get("tube_length_ft", 20)
        shell_dia_in = _hx_shell_dia_in(area / n_shells if n_shells else area, tube_len)
        shell_dia_ft = shell_dia_in / 12
        # Footprint per shell: tube_length x (shell_dia + access)
        fp_each = tube_len * (shell_dia_ft + 6)  # 6 ft access
        fp_total = fp_each * n_shells
        footprints.append({
            "Equipment": label, "Count": n_shells,
            "Each (ft x ft)": f"{tube_len:.0f} x {shell_dia_ft + 6:.0f}",
            "Total (ft2)": f"{fp_total:,.0f}", "Source": "Calculated",
        })
        total_ft2 += fp_total

    if config == "B":
        ihx_area = costs.get("intermediate_hx_area_ft2", 0)
        if ihx_area > 0:
            ref = EQUIP_REF["ihx"]
            n_shells = _hx_shells(ihx_area, ref.get("max_shell_area_ft2", 5000))
            fp_each = 20 * 10
            fp_total = fp_each * n_shells
            footprints.append({
                "Equipment": "Intermediate HX", "Count": n_shells,
                "Each (ft x ft)": "20 x 10",
                "Total (ft2)": f"{fp_total:,.0f}", "Source": "Calculated",
            })
            total_ft2 += fp_total

    # ACC
    acc_ft2_per_bay = 40 * 12  # 40 x 12 per bay typical
    acc_total = acc_ft2_per_bay * n_bays
    footprints.append({
        "Equipment": "Air-Cooled Condenser", "Count": n_bays,
        "Each (ft x ft)": "40 x 12",
        "Total (ft2)": f"{acc_total:,.0f}", "Source": "Vendor typical",
    })
    total_ft2 += acc_total

    # Substation
    sub_ft2 = 1200  # 40 x 30
    footprints.append({
        "Equipment": "Electrical Substation", "Count": 1,
        "Each (ft x ft)": "40 x 30",
        "Total (ft2)": f"{sub_ft2:,.0f}", "Source": "Typical",
    })
    total_ft2 += sub_ft2

    # Control building
    ctrl_ft2 = 600  # 30 x 20
    footprints.append({
        "Equipment": "Control Building", "Count": 1,
        "Each (ft x ft)": "30 x 20",
        "Total (ft2)": f"{ctrl_ft2:,.0f}", "Source": "Typical",
    })
    total_ft2 += ctrl_ft2

    # WF storage
    wf_ft2 = 400  # 20 x 20 tank pad
    footprints.append({
        "Equipment": "WF Storage Tank", "Count": 1,
        "Each (ft x ft)": "20 x 20",
        "Total (ft2)": f"{wf_ft2:,.0f}", "Source": "Typical",
    })
    total_ft2 += wf_ft2

    # Total
    footprints.append({
        "Equipment": "**TOTAL EQUIPMENT FOOTPRINT**", "Count": "",
        "Each (ft x ft)": "", "Total (ft2)": f"{total_ft2:,.0f}", "Source": "",
    })

    st.dataframe(pd.DataFrame(footprints), use_container_width=True, hide_index=True)

    # ── Estimated Plot Area ─────────────────────────────────────
    spacing_factor = 3.5
    plot_ft2 = total_ft2 * spacing_factor
    plot_acres = plot_ft2 / 43560

    st.markdown(f"### Estimated Plot Area")
    c1, c2, c3 = st.columns(3)
    c1.metric("Equipment Footprint", f"{total_ft2:,.0f} ft2")
    c2.metric("Spacing Factor", f"{spacing_factor}x")
    c3.metric("Estimated Plot", f"{plot_acres:.1f} acres")

    # ── Minimum Separation Requirements ─────────────────────────
    st.markdown("### Minimum Separation Requirements")
    sep_rows = [{"From": f, "To": t, "Min Distance (ft)": d, "Basis": b}
                for f, t, d, b in SEPARATION_RULES]
    st.dataframe(pd.DataFrame(sep_rows), use_container_width=True, hide_index=True)

    # ── Adjacency Constraints ───────────────────────────────────
    st.markdown("### Adjacency Constraints")
    constraints = [
        "ACC should be located **downwind** of process area (prevailing wind study required)",
        "Turbine building requires **crane access** on at least one side for rotor pull",
        "Brine manifold should be **adjacent to production wells** to minimize piping runs",
        "Electrical substation positioned for **shortest interconnection route**",
        "WF storage tank requires **fire suppression access** on all sides",
        "Control building located **upwind** and outside blast/fire zone",
    ]
    for c in constraints:
        st.markdown(f"- {c}")

    st.warning(
        "This output requires survey, geotechnical, and site-specific review "
        "before use in layout design."
    )


# ── TAB 6: Open Items Log ───────────────────────────────────────────────────

def _render_open_items_tab(detail: dict, result):
    """Open items sorted by tier: critical, important, informational."""
    config = _config_type(detail)
    warnings = result.warnings if hasattr(result, "warnings") else []

    # ── CRITICAL ────────────────────────────────────────────────
    st.markdown("### CRITICAL — Must Resolve Before FEED")
    critical = list(OPEN_ITEMS_CRITICAL)

    # Dynamic: convergence warnings
    if not result.converged:
        critical.insert(0, "**ANALYSIS DID NOT CONVERGE** — results are unreliable")
    if warnings:
        for w in warnings:
            critical.append(f"Analysis warning: {w}")

    for i, item in enumerate(critical, 1):
        st.markdown(f"{i}. {item}")

    # ── IMPORTANT ───────────────────────────────────────────────
    st.markdown("### IMPORTANT — Resolve During FEED")
    important = list(OPEN_ITEMS_IMPORTANT)

    # Dynamic: LOW confidence equipment items
    low_conf_items = [k for k, v in EQUIP_REF.items() if v.get("confidence") == "LOW"]
    if low_conf_items:
        names = ", ".join(k.replace("_", " ").title() for k in low_conf_items)
        important.append(f"LOW confidence equipment items require vendor confirmation: {names}")

    # Config-specific
    if config == "B":
        important.append("Propane system: confirm thermosiphon feasibility vs forced circulation")
    if config == "D":
        important.append("Dual-pressure: confirm HP/LP turbine vendor availability at design size")

    for i, item in enumerate(important, 1):
        st.markdown(f"{i}. {item}")

    # ── INFORMATIONAL ───────────────────────────────────────────
    st.markdown("### INFORMATIONAL — Will Not Change Decisions")
    for i, item in enumerate(OPEN_ITEMS_INFO, 1):
        st.markdown(f"{i}. {item}")


# ── Package Header ───────────────────────────────────────────────────────────

def _render_header(result, config_label: str):
    """Render FEED package header with config descriptor and disclaimers."""
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    st.markdown(f"## FEED Light Package — Run #{result.run_id}")
    st.markdown(f"**Configuration:** {config_label}")
    st.markdown(f"**Generated:** {timestamp}")
    st.markdown(f"**Cost estimate class:** AACE CLASS 4 (±30%)")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Net Power", f"{result.net_power_MW:.1f} MW")
    c2.metric("Installed $/kW", f"${result.capex_per_kW:,.0f}")
    c3.metric("NPV", f"${result.npv_USD:,.0f}")
    c4.metric("Schedule", f"{result.construction_weeks} wk")

    st.warning(
        "NOT FOR CONSTRUCTION. NOT FOR PERMIT. For conceptual evaluation only."
    )


# ── Main entry point ────────────────────────────────────────────────────────

def render_feed_package(result, design_basis: dict):
    """Main entry point — re-run analysis and render all 6 FEED tabs."""
    config_label = _config_label(result)

    # Re-run analysis to get enriched _detail
    feed_data = st.session_state.get("opt_feed_data")
    if feed_data is None:
        with st.spinner(f"Re-running analysis for Run #{result.run_id}..."):
            feed_data = rerun_analysis(result, design_basis)
            st.session_state["opt_feed_data"] = feed_data

    if not feed_data.get("converged", False):
        st.error("Analysis did not converge — FEED package may be incomplete.")

    detail = feed_data.get("_detail", {})

    _render_header(result, config_label)

    # 6 FEED tabs
    t1, t2, t3, t4, t5, t6 = st.tabs([
        "H&MB", "Equipment List", "Process Flow",
        "Instrumentation", "Plot Plan", "Open Items",
    ])

    with t1:
        _render_hmb_tab(detail, result, config_label)
    with t2:
        _render_equipment_tab(detail, result)
    with t3:
        _render_process_flow_tab(detail, result)
    with t4:
        _render_instrumentation_tab(detail, result)
    with t5:
        _render_plot_plan_tab(detail, result)
    with t6:
        _render_open_items_tab(detail, result)
