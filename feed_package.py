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

N_TRAINS_DEFAULT = 2  # fallback if detail lacks n_trains

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
    n_trains = detail.get("n_trains", N_TRAINS_DEFAULT)
    ref = EQUIP_REF["turbine_generator"]
    tg_cost = costs.get("turbine_generator", 0)
    tg_per_kw = tg_cost / P_net if P_net > 0 else 0
    tg_weight_lb = ref["lb_per_kw"] * P_gross
    tg_footprint = ref["ft2_per_mw"] * (P_gross / 1000)
    tg_dim = f"{math.sqrt(tg_footprint / n_trains):.0f} x {math.sqrt(tg_footprint / n_trains):.0f}"
    rows.append({
        "Tag": "TG-0101A/B", "Service": "Turbine-Generator Set",
        "Qty": n_trains,
        "Key Parameters": f"{P_gross / n_trains / 1000:.1f} MW each, "
                          f"eta={detail.get('eta_turbine', 0.82):.0%}",
        "Dimensions (ft)": tg_dim,
        "Weight (tons)": f"{tg_weight_lb / 2000 / n_trains:.0f} ea",
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
        "Qty": f"{n_trains} + 1 spare",
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
            "Qty": f"{n_trains} + 1 spare",
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
    """Process flow diagram with auto-generated Mermaid PFD from state points,
    color-coded streams, equipment detail popups, and control philosophy."""
    config = _config_type(detail)
    states = detail.get("states", {})
    costs = _costs_from_detail(detail)
    power_bal = detail.get("power_balance", {})
    T_geo_in = detail.get("T_geo_in_F", 420)
    T_geo_out = detail.get("T_geo_out_min_F", 180)
    T_cond = detail.get("T_cond_F", 0)
    T_evap = detail.get("T_evap_F", 0)
    P_high = detail.get("P_high_psia", 0)
    P_low = detail.get("P_low_psia", 0)
    fluid = detail.get("working_fluid", "isopentane").title()
    n_fans = detail.get("fan_n_fans_used", 0)
    n_trains = detail.get("n_trains", N_TRAINS_DEFAULT)
    P_gross = power_bal.get("P_gross", 0)
    P_net = power_bal.get("P_net", 0)
    m_dot_geo = detail.get("m_dot_geo_lb_s", 0) * 3600

    # State helpers
    def _st(k, field="T"):
        s = states.get(str(k), {})
        return s.get(field, 0)

    def _phase(k):
        s = states.get(str(k), {})
        return s.get("phase", "?")

    # ── Mermaid PFD (color-coded) ─────────────────────────────
    st.markdown("### Process Flow Diagram")

    # Stream condition labels: T / P / Phase
    def _lbl(T, P, phase):
        return f"{T:.0f}°F / {P:.0f} psia / {phase}"

    if config == "A":
        s2_T, s2_P = _st(2, "T"), _st(2, "P")
        s3_T, s3_P = _st(3, "T"), _st(3, "P")
        s4_T, s4_P = _st(4, "T"), _st(4, "P")
        s7_T, s7_P = _st(7, "T"), _st(7, "P")
        has_recup = costs.get("recuperator_area_ft2", 0) > 0
        recup_block = """
    TG --> |"{turb_exit}"| RECUP["fa:fa-exchange-alt Recuperator<br/>HX-0103"]
    RECUP --> |"{recup_exit}"| ACC
    ACC --> |"{cond_exit}"| PP
    PP --> |"{pump_exit}"| RECUP_C[Recuperator<br/>Cold Side]
    RECUP_C --> PRE_WF""".format(
            turb_exit=_lbl(s2_T, s2_P, _phase(2)),
            recup_exit=_lbl(s3_T, s3_P, _phase(3)),
            cond_exit=_lbl(s4_T, s4_P, _phase(4)),
            pump_exit=_lbl(_st(5, "T"), _st(5, "P"), _phase(5)),
        ) if has_recup else """
    TG --> |"{turb_exit}"| ACC
    ACC --> |"{cond_exit}"| PP
    PP --> PRE_WF""".format(
            turb_exit=_lbl(s2_T, s2_P, _phase(2)),
            cond_exit=_lbl(s4_T, s4_P, _phase(4)),
        )

        mermaid = f"""```mermaid
graph LR
    classDef brine fill:#D2691E,stroke:#8B4513,color:#fff
    classDef wf fill:#1E90FF,stroke:#104E8B,color:#fff
    classDef equip fill:#2E8B57,stroke:#006400,color:#fff
    classDef elec fill:#FFD700,stroke:#B8860B,color:#000
    classDef parasitic fill:#CD853F,stroke:#8B6914,color:#fff

    WELLS["fa:fa-industry Production Wells"]:::brine
    VAP["fa:fa-fire Vaporizer<br/>HX-0101"]:::equip
    PRE["fa:fa-thermometer-half Preheater<br/>HX-0102"]:::equip
    REINJ["fa:fa-arrow-down Reinjection Wells"]:::brine
    TG["fa:fa-cog Turbine-Generator<br/>TG-0101<br/>{P_gross/1000:.1f} MW"]:::equip
    ACC["fa:fa-wind ACC<br/>AC-0101<br/>{n_fans} fans"]:::equip
    PP["fa:fa-tint Feed Pump<br/>PP-0101"]:::equip
    PRE_WF[Preheater WF Side]:::wf

    WELLS --> |"{_lbl(T_geo_in, 200, 'Liquid')}<br/>{m_dot_geo:,.0f} lb/hr"| VAP
    VAP --> PRE
    PRE --> |"{_lbl(T_geo_out, 180, 'Liquid')}"| REINJ

    PRE_WF --> |"{_lbl(s7_T, s7_P, _phase(7))}"| VAP
    VAP --> |"{_lbl(T_evap, P_high, 'Sat Vapor')}"| TG
{recup_block}

    TG -.- |"{power_bal.get('W_iso_pump', 0):.0f} kW"| PUMP_P["Pump Parasitic"]:::parasitic
    ACC -.- |"{power_bal.get('W_fans', 0):.0f} kW"| FAN_P["Fan Parasitic"]:::parasitic
```"""
    elif config == "B":
        prop_states = detail.get("prop_states", {})
        T_prop_cond = detail.get("T_propane_cond_F", 0)
        T_prop_evap = detail.get("T_propane_evap_F", T_prop_cond + 10)
        s2_T, s2_P = _st(2, "T"), _st(2, "P")
        s3_T, s3_P = _st(3, "T"), _st(3, "P")
        s4_T, s4_P = _st(4, "T"), _st(4, "P")
        s7_T, s7_P = _st(7, "T"), _st(7, "P")
        prA_T = prop_states.get("A", {}).get("T", 0)
        prA_P = prop_states.get("A", {}).get("P", 0)
        prB_T = prop_states.get("B", {}).get("T", 0)
        prB_P = prop_states.get("B", {}).get("P", 0)

        mermaid = f"""```mermaid
graph LR
    classDef brine fill:#D2691E,stroke:#8B4513,color:#fff
    classDef wf fill:#1E90FF,stroke:#104E8B,color:#fff
    classDef propane fill:#32CD32,stroke:#228B22,color:#fff
    classDef equip fill:#2E8B57,stroke:#006400,color:#fff
    classDef parasitic fill:#CD853F,stroke:#8B6914,color:#fff

    WELLS["fa:fa-industry Production Wells"]:::brine
    VAP["fa:fa-fire Vaporizer<br/>HX-0101"]:::equip
    PRE["fa:fa-thermometer-half Preheater<br/>HX-0102"]:::equip
    REINJ["fa:fa-arrow-down Reinjection Wells"]:::brine
    TG["fa:fa-cog Turbine-Generator<br/>TG-0101<br/>{P_gross/1000:.1f} MW"]:::equip
    RECUP["fa:fa-exchange-alt Recuperator<br/>HX-0103"]:::equip
    IHX["fa:fa-exchange-alt Intermediate HX<br/>HX-0104"]:::equip
    ACC["fa:fa-wind Propane ACC<br/>AC-0101<br/>{n_fans} fans"]:::equip
    PP["fa:fa-tint ISO Pump<br/>PP-0101"]:::equip
    PPR["fa:fa-tint Propane Pump<br/>PP-0102"]:::propane

    WELLS --> |"{_lbl(T_geo_in, 200, 'Liquid')}<br/>{m_dot_geo:,.0f} lb/hr"| VAP
    VAP --> PRE
    PRE --> |"{_lbl(T_geo_out, 180, 'Liquid')}"| REINJ

    PP --> |"{_lbl(s4_T, s4_P, _phase(4))}"| RECUP
    RECUP --> PRE_WF[Preheater WF]
    PRE_WF --> |"{_lbl(s7_T, s7_P, _phase(7))}"| VAP
    VAP --> |"{_lbl(T_evap, P_high, 'Sat Vapor')}"| TG
    TG --> |"{_lbl(s2_T, s2_P, _phase(2))}"| RECUP
    RECUP --> |"{_lbl(s3_T, s3_P, _phase(3))}"| IHX
    IHX --> PP

    IHX --> |"{_lbl(prA_T, prA_P, 'Sat Vapor')}"| ACC
    ACC --> |"{_lbl(prB_T, prB_P, 'Liquid')}"| PPR
    PPR --> IHX

    TG -.- |"{power_bal.get('W_iso_pump', 0):.0f} kW"| PUMP_P["Pump Parasitic"]:::parasitic
    ACC -.- |"{power_bal.get('W_fans', 0):.0f} kW"| FAN_P["Fan Parasitic"]:::parasitic
```"""
    else:  # Config D — dual-pressure with two parallel evaporation branches
        T_split = detail.get("T_split_F", 0)
        hp_kw = detail.get("hp_gross_power_kw", 0)
        lp_kw = detail.get("lp_gross_power_kw", 0)
        mermaid = f"""```mermaid
graph TD
    classDef brine fill:#D2691E,stroke:#8B4513,color:#fff
    classDef hp fill:#1E90FF,stroke:#104E8B,color:#fff
    classDef lp fill:#00BFFF,stroke:#0099CC,color:#fff
    classDef equip fill:#2E8B57,stroke:#006400,color:#fff
    classDef parasitic fill:#CD853F,stroke:#8B6914,color:#fff
    classDef acc fill:#87CEEB,stroke:#4682B4,color:#000

    %% Brine cascade (top to bottom)
    WELLS["fa:fa-industry Production Wells<br/>{T_geo_in:.0f}°F"]:::brine
    HP_VAP_B["HP Vaporizer<br/>Brine Side"]:::equip
    HP_PRE_B["HP Preheater<br/>Brine Side"]:::equip
    SPLIT{{"Brine Split<br/>{T_split:.0f}°F"}}:::brine
    LP_VAP_B["LP Vaporizer<br/>Brine Side"]:::equip
    LP_PRE_B["LP Preheater<br/>Brine Side"]:::equip
    REINJ["fa:fa-arrow-down Reinjection<br/>{T_geo_out:.0f}°F"]:::brine

    WELLS --> HP_VAP_B --> HP_PRE_B --> SPLIT --> LP_VAP_B --> LP_PRE_B --> REINJ

    %% HP WF loop (left branch)
    HP_PP["HP Pump"]:::hp
    HP_RECUP_C["HP Recuperator<br/>Cold Side"]:::hp
    HP_PRE_W["HP Preheater<br/>WF Side"]:::hp
    HP_VAP_W["HP Vaporizer<br/>WF Side"]:::hp
    HP_TG["fa:fa-cog HP Turbine<br/>{hp_kw/1000:.1f} MW"]:::equip
    HP_RECUP_H["HP Recuperator<br/>Hot Side"]:::hp

    HP_PP --> HP_RECUP_C --> HP_PRE_W --> HP_VAP_W
    HP_VAP_W --> |"{T_evap:.0f}°F / {P_high:.0f} psia"| HP_TG
    HP_TG --> HP_RECUP_H

    %% LP WF loop (right branch)
    LP_PP["LP Pump"]:::lp
    LP_PRE_W["LP Preheater<br/>WF Side"]:::lp
    LP_VAP_W["LP Vaporizer<br/>WF Side"]:::lp
    LP_TG["fa:fa-cog LP Turbine<br/>{lp_kw/1000:.1f} MW"]:::equip

    LP_PP --> LP_PRE_W --> LP_VAP_W
    LP_VAP_W --> LP_TG

    %% Shared ACC merges both exhausts
    ACC["fa:fa-wind Shared ACC<br/>{n_fans} fans"]:::acc
    HP_RECUP_H --> ACC
    LP_TG --> ACC
    ACC --> HP_PP
    ACC --> LP_PP

    %% Cross-links: brine heats WF
    HP_VAP_B -.- HP_VAP_W
    HP_PRE_B -.- HP_PRE_W
    LP_VAP_B -.- LP_VAP_W
    LP_PRE_B -.- LP_PRE_W

    ACC -.- |"{power_bal.get('W_fans', 0):.0f} kW"| FAN_P["Fan Parasitic"]:::parasitic

    ACC -.- |"{power_bal.get('W_fans', 0):.0f} kW"| FAN_P["Fan Parasitic"]:::parasitic
```"""

    st.markdown(mermaid)

    # ── Legend ─────────────────────────────────────────────────
    legend_cols = st.columns(5)
    legend_cols[0].markdown(":brown_circle: **Brine**")
    legend_cols[1].markdown(":large_blue_circle: **Working Fluid**")
    if config == "B":
        legend_cols[2].markdown(":green_circle: **Propane**")
    elif config == "D":
        legend_cols[2].markdown(":blue_heart: **HP Stage** / :droplet: **LP Stage**")
    legend_cols[3].markdown(":green_heart: **Equipment**")
    legend_cols[4].markdown(":orange_circle: **Parasitic**")

    st.caption("*Auto-generated from H&MB state points. Not to scale.*")

    # ── Equipment Detail Expanders ─────────────────────────────
    st.markdown("### Equipment Detail")
    st.caption("Click any equipment to see sizing, vendor candidates, and open items.")

    _equip_details = []
    # Turbine-Generator
    ref = EQUIP_REF["turbine_generator"]
    tg_cost = costs.get("turbine_generator", 0)
    _equip_details.append(("TG-0101", "Turbine-Generator Set", {
        "Quantity": f"{n_trains} units",
        "Rating": f"{P_gross / n_trains / 1000:.1f} MW each ({P_gross/1000:.1f} MW total)",
        "Isentropic Efficiency": f"{detail.get('eta_turbine', 0.82):.0%}",
        "Weight": f"{ref['lb_per_kw'] * P_gross / 2000:.0f} tons total",
        "Budget": f"${tg_cost:,.0f} (${tg_cost/P_net:.0f}/kW)" if P_net > 0 else "-",
        "Vendors": ", ".join(ref["vendors"]),
        "Lead Time": f"{ref['lead_wk']} weeks",
    }, ["Confirm vendor efficiency guarantee at design conditions",
        "Verify gearbox / direct-drive selection"]))

    # Vaporizer
    vap_area = costs.get("vaporizer_area_ft2", 0)
    ref = EQUIP_REF["vaporizer"]
    _equip_details.append(("HX-0101", "Vaporizer", {
        "Quantity": f"{_hx_shells(vap_area)} shell(s)",
        "Heat Transfer Area": f"{vap_area:,.0f} ft2",
        "Shell Diameter": f"{_hx_shell_dia_in(vap_area / max(_hx_shells(vap_area), 1)):.0f}\" ID",
        "Tube Length": f"{ref['tube_length_ft']}' (TEMA)",
        "Budget": f"${costs.get('vaporizer', 0):,.0f}",
        "Vendors": ", ".join(ref["vendors"]),
        "Lead Time": f"{ref['lead_wk']} weeks",
    }, ["Brine-side fouling factor and corrosion allowance",
        "Tube material selection (brine service)"]))

    # Preheater
    pre_area = costs.get("preheater_area_ft2", 0)
    ref = EQUIP_REF["preheater"]
    _equip_details.append(("HX-0102", "Preheater", {
        "Quantity": f"{_hx_shells(pre_area)} shell(s)",
        "Heat Transfer Area": f"{pre_area:,.0f} ft2",
        "Budget": f"${costs.get('preheater', 0):,.0f}",
        "Vendors": ", ".join(ref["vendors"]),
        "Lead Time": f"{ref['lead_wk']} weeks",
    }, ["Brine-side scaling potential at lower temperatures"]))

    # Recuperator (if present)
    rec_area = costs.get("recuperator_area_ft2", 0)
    if rec_area > 0:
        ref = EQUIP_REF["recuperator"]
        _equip_details.append(("HX-0103", "Recuperator", {
            "Quantity": f"{_hx_shells(rec_area)} shell(s) x {n_trains} trains",
            "Heat Transfer Area": f"{rec_area:,.0f} ft2 total",
            "Budget": f"${costs.get('recuperator', 0):,.0f}",
            "Vendors": ", ".join(ref["vendors"]),
            "Lead Time": f"{ref['lead_wk']} weeks",
        }, ["Thermal cycling fatigue analysis"]))

    # ACC
    n_bays = int(costs.get("acc_n_bays", 0))
    ref = EQUIP_REF["acc"]
    _equip_details.append(("AC-0101", "Air-Cooled Condenser", {
        "Number of Bays": f"{n_bays}",
        "Number of Fans": f"{n_fans}",
        "Fan Power": f"{power_bal.get('W_fans', 0):,.0f} kW",
        "Condensing Temperature": f"{T_cond:.0f} °F",
        "Budget": f"${costs.get('acc', 0):,.0f}",
        "Vendors": ", ".join(ref["vendors"]),
        "Lead Time": f"{ref['lead_wk']} weeks",
    }, ["Noise study at property line", "Wind effect on performance"]))

    # Feed Pump
    ref = EQUIP_REF["iso_pump"]
    _equip_details.append(("PP-0101", "Isopentane Feed Pump", {
        "Quantity": f"{n_trains} operating + 1 spare",
        "Differential Pressure": f"{detail.get('pump_iso_dP_psi', 0):.0f} psi",
        "Budget": f"${costs.get('iso_pump', 0):,.0f}",
        "Vendors": ", ".join(ref["vendors"]),
        "Lead Time": f"{ref['lead_wk']} weeks",
    }, ["Mechanical seal selection for hydrocarbon service"]))

    if config == "B":
        ihx_area = costs.get("intermediate_hx_area_ft2", 0)
        if ihx_area > 0:
            ref = EQUIP_REF["ihx"]
            _equip_details.append(("HX-0104", "Intermediate HX", {
                "Heat Transfer Area": f"{ihx_area:,.0f} ft2",
                "Budget": f"${costs.get('intermediate_hx', 0):,.0f}",
                "Vendors": ", ".join(ref["vendors"]),
            }, ["Propane-side pressure relief sizing"]))

    for tag, service, params, open_items in _equip_details:
        with st.expander(f"{tag} — {service}"):
            for k, v in params.items():
                st.markdown(f"**{k}:** {v}")
            if open_items:
                st.markdown("**Open Items:**")
                for oi in open_items:
                    st.markdown(f"- {oi}")

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

def _build_equipment_blocks(detail: dict) -> list[dict]:
    """Build list of equipment blocks with sizing for plot plan layout.

    Each block: {tag, service, w_ft, h_ft, system, color}
    system: 'brine', 'wf', 'electrical', 'civil', 'propane'
    """
    config = _config_type(detail)
    costs = _costs_from_detail(detail)
    power_bal = detail.get("power_balance", {})
    P_gross = power_bal.get("P_gross", 0)
    n_bays = int(costs.get("acc_n_bays", 0))
    n_trains = detail.get("n_trains", N_TRAINS_DEFAULT)

    blocks = []

    # Color map by system
    COLORS = {
        "brine": "#D2691E",
        "wf": "#1E90FF",
        "propane": "#32CD32",
        "electrical": "#FFD700",
        "civil": "#A9A9A9",
    }

    # Turbine building — one block per train
    tg_ft2_total = EQUIP_REF["turbine_generator"]["ft2_per_mw"] * (P_gross / 1000)
    tg_ft2_each = tg_ft2_total / max(n_trains, 1)
    tg_side = max(math.sqrt(tg_ft2_each), 15)
    for i in range(n_trains):
        suffix = chr(65 + i) if n_trains > 1 else ""
        blocks.append({
            "tag": f"TG-0101{suffix}", "service": "Turbine-Generator",
            "w_ft": tg_side, "h_ft": tg_side,
            "system": "wf", "color": COLORS["wf"], "group": "turbine",
        })

    # Heat exchangers (shared)
    for label, tag, area_key, ref_key in [
        ("Vaporizer", "HX-0101", "vaporizer_area_ft2", "vaporizer"),
        ("Preheater", "HX-0102", "preheater_area_ft2", "preheater"),
        ("Recuperator", "HX-0103", "recuperator_area_ft2", "recuperator"),
    ]:
        area = costs.get(area_key, 0)
        if area <= 0:
            continue
        ref = EQUIP_REF[ref_key]
        n_shells = _hx_shells(area, ref.get("max_shell_area_ft2", 5000))
        tube_len = ref.get("tube_length_ft", 20)
        shell_dia_in = _hx_shell_dia_in(area / max(n_shells, 1), tube_len)
        shell_w = max(shell_dia_in / 12 + 6, 8)  # add access
        for s in range(n_shells):
            suffix = chr(65 + s) if n_shells > 1 else ""
            blocks.append({
                "tag": f"{tag}{suffix}", "service": label,
                "w_ft": tube_len, "h_ft": shell_w,
                "system": "wf", "color": COLORS["wf"], "group": "hx",
            })

    # Intermediate HX (Config B)
    if config == "B":
        ihx_area = costs.get("intermediate_hx_area_ft2", 0)
        if ihx_area > 0:
            ref = EQUIP_REF["ihx"]
            n_shells = _hx_shells(ihx_area, ref.get("max_shell_area_ft2", 5000))
            for s in range(n_shells):
                suffix = chr(65 + s) if n_shells > 1 else ""
                blocks.append({
                    "tag": f"HX-0104{suffix}", "service": "Intermediate HX",
                    "w_ft": 20, "h_ft": 10,
                    "system": "propane", "color": COLORS["propane"], "group": "hx",
                })

    # ACC — modeled as single large block (bay array)
    if n_bays > 0:
        # Typical A-frame: bays in rows, each 40 x 12 ft
        # Real ACC banks: 2-4 rows, bays arrayed along the length
        bays_per_row = min(n_bays, max(4, math.ceil(math.sqrt(n_bays * 3))))
        n_rows = math.ceil(n_bays / bays_per_row)
        acc_w = bays_per_row * 12  # bays side by side (12' wide)
        acc_h = n_rows * 40  # 40 ft deep per row
        blocks.append({
            "tag": "AC-0101", "service": f"ACC ({n_bays} bays)",
            "w_ft": acc_w, "h_ft": acc_h,
            "system": "wf", "color": "#87CEEB", "group": "acc",
        })

    # Feed pumps
    for i in range(n_trains):
        suffix = chr(65 + i) if n_trains > 1 else ""
        blocks.append({
            "tag": f"PP-0101{suffix}", "service": "ISO Pump",
            "w_ft": 6, "h_ft": 4,
            "system": "wf", "color": COLORS["wf"], "group": "pump",
        })

    if config == "B":
        blocks.append({
            "tag": "PP-0102", "service": "Propane Pump",
            "w_ft": 6, "h_ft": 4,
            "system": "propane", "color": COLORS["propane"], "group": "pump",
        })

    # Electrical substation
    blocks.append({
        "tag": "SS-0101", "service": "Electrical Substation",
        "w_ft": 40, "h_ft": 30,
        "system": "electrical", "color": COLORS["electrical"], "group": "electrical",
    })

    # Control building
    blocks.append({
        "tag": "CB-0101", "service": "Control Building",
        "w_ft": 30, "h_ft": 20,
        "system": "civil", "color": COLORS["civil"], "group": "civil",
    })

    # WF storage
    blocks.append({
        "tag": "TK-0101", "service": "WF Storage",
        "w_ft": 20, "h_ft": 20,
        "system": "wf", "color": "#B0C4DE", "group": "storage",
    })

    # Brine manifold
    blocks.append({
        "tag": "BM-0101", "service": "Brine Manifold",
        "w_ft": 25, "h_ft": 15,
        "system": "brine", "color": COLORS["brine"], "group": "brine",
    })

    return blocks


# Wind direction vectors for plot plan
_WIND_VECTORS = {
    "N": (0, 1), "NE": (0.707, 0.707), "E": (1, 0), "SE": (0.707, -0.707),
    "S": (0, -1), "SW": (-0.707, -0.707), "W": (-1, 0), "NW": (-0.707, 0.707),
}


def _place_bank(blocks: list[dict], x_start: float, y_start: float,
                max_per_row: int = 6, gap_x: float = 8, gap_y: float = 10) -> tuple:
    """Place a group of blocks in rows. Returns (placed_blocks, x_extent, y_extent)."""
    if not blocks:
        return [], 0, 0
    x, y = x_start, y_start
    col = 0
    row_h = 0
    max_x = x_start
    for b in blocks:
        if col >= max_per_row:
            col = 0
            x = x_start
            y -= row_h + gap_y
            row_h = 0
        b["x"] = x + b["w_ft"] / 2
        b["y"] = y - b["h_ft"] / 2
        x += b["w_ft"] + gap_x
        max_x = max(max_x, x)
        row_h = max(row_h, b["h_ft"])
        col += 1
    y_end = y - row_h
    return blocks, max_x - x_start, y_start - y_end


def _layout_equipment(blocks: list[dict], wind_dir: str = "SW") -> list[dict]:
    """Place equipment blocks on a 2D grid respecting separation constraints.

    Layout strategy (constraint-driven):
    - Brine manifold at site boundary (top edge)
    - HX bank in center process area (grouped into rows)
    - Turbines adjacent to HX bank
    - ACC downwind of turbine building, min 50 ft separation
    - Electrical room min 25 ft from turbine, away from process
    - Control building upwind, outside process area
    - WF storage away from electrical (NFPA 30)

    Returns blocks with added 'x' and 'y' (center coordinates, ft).
    """
    wind_dx, wind_dy = _WIND_VECTORS.get(wind_dir, (-0.707, -0.707))

    # Categorize blocks
    by_group = {}
    for b in blocks:
        by_group.setdefault(b["group"], []).append(b)

    placed = []
    y_cursor = 0

    # ── Brine manifold at top (site boundary) ──
    for b in by_group.get("brine", []):
        b["x"] = 0
        b["y"] = y_cursor
        placed.append(b)
    y_cursor -= 40

    # ── HX bank (grouped into rows of max 6) ──
    hx_blocks = by_group.get("hx", [])
    hx_start_y = y_cursor
    if hx_blocks:
        _place_bank(hx_blocks, 0, y_cursor, max_per_row=6)
        placed.extend(hx_blocks)
        hx_max_x = max(b["x"] + b["w_ft"] / 2 for b in hx_blocks)
        hx_min_y = min(b["y"] - b["h_ft"] / 2 for b in hx_blocks)
        y_cursor = hx_min_y - 25
    else:
        hx_max_x = 0

    # ── Turbine buildings (side by side) ──
    turbine_blocks = by_group.get("turbine", [])
    turb_x = 0
    turb_center_y = y_cursor
    for b in turbine_blocks:
        b["x"] = turb_x + b["w_ft"] / 2
        b["y"] = y_cursor - b["h_ft"] / 2
        turb_x += b["w_ft"] + 15
        placed.append(b)
    turb_right = turb_x

    # Pumps adjacent to turbines
    pump_blocks = by_group.get("pump", [])
    for b in pump_blocks:
        b["x"] = turb_x + b["w_ft"] / 2
        b["y"] = y_cursor - b["h_ft"] / 2
        turb_x += b["w_ft"] + 8
        placed.append(b)

    turb_bottom = y_cursor - max((b["h_ft"] for b in turbine_blocks), default=30)

    # ── ACC — downwind of turbine, min 50 ft ──
    acc_blocks = by_group.get("acc", [])
    if acc_blocks:
        acc = acc_blocks[0]
        turb_cx = turbine_blocks[0]["x"] if turbine_blocks else 0
        turb_cy = turb_center_y - (turbine_blocks[0]["h_ft"] / 2 if turbine_blocks else 15)
        sep_dist = max(60, 50 + acc["h_ft"] / 4)  # ensure > 50 ft edge-to-edge
        acc["x"] = turb_cx + wind_dx * sep_dist
        acc["y"] = turb_cy + wind_dy * sep_dist
        placed.append(acc)

    # ── Electrical substation: min 25 ft from turbine, to the right ──
    process_right = max(hx_max_x, turb_right, 50)
    for b in by_group.get("electrical", []):
        b["x"] = process_right + 35 + b["w_ft"] / 2
        b["y"] = hx_start_y - b["h_ft"] / 2
        placed.append(b)
    elec_x = process_right + 35

    # ── Control building: upwind, outside process area ──
    for b in by_group.get("civil", []):
        b["x"] = -wind_dx * 80
        b["y"] = -wind_dy * 80 + hx_start_y
        placed.append(b)

    # ── WF storage: away from electrical, near process ──
    for b in by_group.get("storage", []):
        b["x"] = -50
        b["y"] = turb_bottom - 30
        placed.append(b)

    return placed


def _draw_plot_plan(blocks: list[dict], wind_dir: str = "SW",
                    annotations: list[dict] = None) -> "matplotlib.figure.Figure":
    """Render equipment blocks as a scaled matplotlib plot plan.

    Returns a matplotlib Figure with:
    - Scaled colored rectangles for each equipment
    - Tag + service labels
    - North arrow, scale bar
    - Wind direction indicator
    - Separation distance annotations
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    from matplotlib.patches import FancyArrowPatch

    fig, ax = plt.subplots(1, 1, figsize=(14, 10))
    ax.set_aspect("equal")

    # Draw equipment blocks
    all_x, all_y = [], []
    for b in blocks:
        cx, cy = b["x"], b["y"]
        w, h = b["w_ft"], b["h_ft"]
        x0 = cx - w / 2
        y0 = cy - h / 2
        rect = patches.FancyBboxPatch(
            (x0, y0), w, h,
            boxstyle="round,pad=1",
            facecolor=b["color"], edgecolor="#333333",
            linewidth=1.2, alpha=0.85,
        )
        ax.add_patch(rect)

        # Label — adapt font size to block area
        min_dim = min(w, h)
        if min_dim >= 30:
            fsize, label = 6.5, f"{b['tag']}\n{b['service']}\n{w:.0f}' x {h:.0f}'"
        elif min_dim >= 15:
            fsize, label = 5.5, f"{b['tag']}\n{b['service']}"
        else:
            fsize, label = 4.5, b['tag']
        text_color = "#000000" if b["color"] in ("#FFD700", "#A9A9A9", "#87CEEB", "#B0C4DE") else "#FFFFFF"
        ax.text(cx, cy, label, ha="center", va="center",
                fontsize=fsize, fontweight="bold", color=text_color,
                linespacing=1.3)

        all_x.extend([x0, x0 + w])
        all_y.extend([y0, y0 + h])

    if not all_x:
        return fig

    # Compute bounds with padding
    margin = 60
    x_min, x_max = min(all_x) - margin, max(all_x) + margin
    y_min, y_max = min(all_y) - margin, max(all_y) + margin
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)

    # ── Separation distance annotations ──
    if annotations:
        for ann in annotations:
            ax.annotate(
                "", xy=ann["xy1"], xytext=ann["xy0"],
                arrowprops=dict(arrowstyle="<->", color="#CC0000", lw=1.5),
            )
            mid_x = (ann["xy0"][0] + ann["xy1"][0]) / 2
            mid_y = (ann["xy0"][1] + ann["xy1"][1]) / 2
            ax.text(mid_x, mid_y + 3, ann["label"],
                    ha="center", va="bottom", fontsize=7, color="#CC0000",
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="#CC0000", alpha=0.9))

    # ── Annotate key separations automatically ──
    block_map = {b["tag"]: b for b in blocks}
    sep_pairs = [
        ("TG-0101A", "AC-0101", 50, "ACC-TG min 50'"),
        ("TG-0101", "AC-0101", 50, "ACC-TG min 50'"),
        ("TG-0101A", "SS-0101", 25, "Elec-TG min 25'"),
        ("TG-0101", "SS-0101", 25, "Elec-TG min 25'"),
        ("TK-0101", "SS-0101", 100, "WF-Elec min 100'"),
    ]
    for tag_a, tag_b, min_dist, label in sep_pairs:
        a = block_map.get(tag_a)
        b = block_map.get(tag_b)
        if a and b:
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            color = "#CC0000" if dist < min_dist else "#006400"
            status = f"{dist:.0f}'" + (" VIOLATION" if dist < min_dist else " OK")
            ax.annotate(
                "", xy=(b["x"], b["y"]), xytext=(a["x"], a["y"]),
                arrowprops=dict(arrowstyle="<->", color=color, lw=1.2, ls="--"),
            )
            mid_x = (a["x"] + b["x"]) / 2
            mid_y = (a["y"] + b["y"]) / 2
            ax.text(mid_x, mid_y + 5, f"{label}\n{status}",
                    ha="center", va="bottom", fontsize=6, color=color,
                    fontweight="bold",
                    bbox=dict(boxstyle="round,pad=0.2", fc="white", ec=color, alpha=0.85))

    # ── North arrow (upper right) ──
    arrow_x = x_max - 30
    arrow_y = y_max - 20
    ax.annotate("N", xy=(arrow_x, arrow_y + 20), fontsize=12, fontweight="bold",
                ha="center", va="bottom")
    ax.annotate("", xy=(arrow_x, arrow_y + 18), xytext=(arrow_x, arrow_y),
                arrowprops=dict(arrowstyle="-|>", color="black", lw=2))

    # ── Wind direction arrow ──
    wind_dx, wind_dy = _WIND_VECTORS.get(wind_dir, (-0.707, -0.707))
    wx = x_max - 30
    wy = y_max - 60
    ax.annotate("", xy=(wx + wind_dx * 20, wy + wind_dy * 20),
                xytext=(wx, wy),
                arrowprops=dict(arrowstyle="-|>", color="#4169E1", lw=2))
    ax.text(wx, wy - 10, f"Prevailing\nWind: {wind_dir}", ha="center",
            fontsize=7, color="#4169E1", fontweight="bold")

    # ── Scale bar (lower left) ──
    bar_x = x_min + 20
    bar_y = y_min + 15
    bar_len = 100  # 100 ft
    ax.plot([bar_x, bar_x + bar_len], [bar_y, bar_y], "k-", lw=3)
    ax.plot([bar_x, bar_x], [bar_y - 3, bar_y + 3], "k-", lw=2)
    ax.plot([bar_x + bar_len, bar_x + bar_len], [bar_y - 3, bar_y + 3], "k-", lw=2)
    ax.text(bar_x + bar_len / 2, bar_y - 8, "100 ft", ha="center", fontsize=8,
            fontweight="bold")

    # ── Plot dimensions ──
    plot_w = x_max - x_min - 2 * margin
    plot_h = y_max - y_min - 2 * margin
    area_acres = (plot_w * plot_h) / 43560
    ax.set_title(
        f"Plot Plan — {plot_w:.0f}' x {plot_h:.0f}' ({area_acres:.1f} acres)",
        fontsize=13, fontweight="bold", pad=15,
    )

    # ── Legend ──
    legend_items = [
        patches.Patch(facecolor="#D2691E", edgecolor="#333", label="Brine"),
        patches.Patch(facecolor="#1E90FF", edgecolor="#333", label="Working Fluid"),
        patches.Patch(facecolor="#87CEEB", edgecolor="#333", label="ACC"),
        patches.Patch(facecolor="#FFD700", edgecolor="#333", label="Electrical"),
        patches.Patch(facecolor="#A9A9A9", edgecolor="#333", label="Civil"),
    ]
    if any(b["system"] == "propane" for b in blocks):
        legend_items.insert(2, patches.Patch(facecolor="#32CD32", edgecolor="#333", label="Propane"))
    ax.legend(handles=legend_items, loc="lower right", fontsize=7, framealpha=0.9)

    ax.set_xlabel("East-West (ft)", fontsize=9)
    ax.set_ylabel("North-South (ft)", fontsize=9)
    ax.grid(True, alpha=0.2, linestyle="--")
    ax.tick_params(labelsize=7)

    fig.tight_layout()
    return fig


def _render_plot_plan_tab(detail: dict, result, design_basis: dict = None):
    """Visual plot plan with scaled equipment, constraint checking, and AI layout modification."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    design_basis = design_basis or {}
    config = _config_type(detail)
    costs = _costs_from_detail(detail)
    power_bal = detail.get("power_balance", {})
    P_gross = power_bal.get("P_gross", 0)
    P_net = power_bal.get("P_net", 0)
    n_bays = int(costs.get("acc_n_bays", 0))
    n_trains = detail.get("n_trains", N_TRAINS_DEFAULT)

    # ── Controls ──────────────────────────────────────────────
    # Default wind direction from shared sidebar (design basis), overridable here
    default_wind = design_basis.get("prevailing_wind", "SW")
    wind_options = ["N", "NE", "E", "SE", "S", "SW", "W", "NW"]
    default_idx = wind_options.index(default_wind) if default_wind in wind_options else 5
    ctrl_cols = st.columns(2)
    with ctrl_cols[0]:
        wind_dir = st.selectbox(
            "Prevailing wind direction",
            options=wind_options,
            index=default_idx,
            key="feed_wind_dir",
        )
    with ctrl_cols[1]:
        st.markdown(f"**Configuration:** {_config_label(result)}")
        st.markdown(f"**Trains:** {n_trains} | **ACC Bays:** {n_bays} | **Gross:** {P_gross/1000:.1f} MW")

    # ── Build and place equipment ─────────────────────────────
    blocks = _build_equipment_blocks(detail)

    # Check for AI layout modifications in session state
    layout_key = "feed_layout_overrides"
    overrides = st.session_state.get(layout_key, {})
    placed = _layout_equipment(blocks, wind_dir)

    # Apply any user overrides
    for b in placed:
        if b["tag"] in overrides:
            b["x"] = overrides[b["tag"]].get("x", b["x"])
            b["y"] = overrides[b["tag"]].get("y", b["y"])

    # ── Draw plot plan ────────────────────────────────────────
    fig = _draw_plot_plan(placed, wind_dir)
    st.pyplot(fig, use_container_width=True)
    plt.close(fig)

    # ── PNG download ──────────────────────────────────────────
    buf = io.BytesIO()
    fig2 = _draw_plot_plan(placed, wind_dir)
    fig2.savefig(buf, format="png", dpi=200, bbox_inches="tight",
                 facecolor="white", edgecolor="none")
    plt.close(fig2)
    buf.seek(0)
    st.download_button(
        "Download Plot Plan (PNG)",
        data=buf.getvalue(),
        file_name=f"plot_plan_run{result.run_id}.png",
        mime="image/png",
    )

    # ── Metrics ───────────────────────────────────────────────
    total_ft2 = sum(b["w_ft"] * b["h_ft"] for b in placed)
    all_x = [b["x"] + b["w_ft"] / 2 for b in placed] + [b["x"] - b["w_ft"] / 2 for b in placed]
    all_y = [b["y"] + b["h_ft"] / 2 for b in placed] + [b["y"] - b["h_ft"] / 2 for b in placed]
    plot_w = max(all_x) - min(all_x) if all_x else 0
    plot_h = max(all_y) - min(all_y) if all_y else 0
    plot_acres = (plot_w * plot_h) / 43560 if plot_w and plot_h else 0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Equipment Footprint", f"{total_ft2:,.0f} ft²")
    m2.metric("Plot Dimensions", f"{plot_w:.0f}' x {plot_h:.0f}'")
    m3.metric("Estimated Plot", f"{plot_acres:.1f} acres")
    m4.metric("Spacing Factor", f"{(plot_w * plot_h) / max(total_ft2, 1):.1f}x")

    # ── Constraint Check Table ────────────────────────────────
    st.markdown("### Separation Constraint Check")
    block_map = {b["tag"]: b for b in placed}
    constraint_checks = []

    checks = [
        ("ACC-Turbine", "AC-0101", ["TG-0101A", "TG-0101"], 50, "Thermal interference, API 2510"),
        ("Elec-Turbine", "SS-0101", ["TG-0101A", "TG-0101"], 25, "NEC / NFPA 70"),
        ("WF Storage-Elec", "TK-0101", ["SS-0101"], 100, "NFPA 30"),
        ("Brine-Process", "BM-0101", ["HX-0101", "HX-0101A"], 25, "Piping economics"),
    ]
    for name, tag_a, tag_b_options, min_dist, basis in checks:
        a = block_map.get(tag_a)
        b = None
        for tb in tag_b_options:
            if tb in block_map:
                b = block_map[tb]
                break
        if a and b:
            dx = b["x"] - a["x"]
            dy = b["y"] - a["y"]
            dist = math.sqrt(dx * dx + dy * dy)
            status = "PASS" if dist >= min_dist else "VIOLATION"
            constraint_checks.append({
                "Constraint": name, "Required (ft)": min_dist,
                "Actual (ft)": f"{dist:.0f}", "Status": status, "Basis": basis,
            })

    if constraint_checks:
        df_checks = pd.DataFrame(constraint_checks)
        # Highlight violations
        st.dataframe(df_checks, use_container_width=True, hide_index=True)
        violations = [c for c in constraint_checks if c["Status"] == "VIOLATION"]
        if violations:
            st.error(f"{len(violations)} separation constraint violation(s) detected — see red annotations on plan.")
        else:
            st.success("All separation constraints satisfied.")

    # ── Full separation rules reference ───────────────────────
    with st.expander("Minimum Separation Requirements (Design Basis)"):
        sep_rows = [{"From": f, "To": t, "Min Distance (ft)": d, "Basis": b}
                    for f, t, d, b in SEPARATION_RULES]
        st.dataframe(pd.DataFrame(sep_rows), use_container_width=True, hide_index=True)

    # ── Equipment Footprint Table ─────────────────────────────
    with st.expander("Equipment Footprint Details"):
        fp_rows = []
        for b in placed:
            fp_rows.append({
                "Tag": b["tag"], "Service": b["service"],
                "Width (ft)": f"{b['w_ft']:.0f}", "Depth (ft)": f"{b['h_ft']:.0f}",
                "Area (ft²)": f"{b['w_ft'] * b['h_ft']:,.0f}", "System": b["system"],
            })
        fp_rows.append({
            "Tag": "TOTAL", "Service": "", "Width (ft)": "", "Depth (ft)": "",
            "Area (ft²)": f"{total_ft2:,.0f}", "System": "",
        })
        st.dataframe(pd.DataFrame(fp_rows), use_container_width=True, hide_index=True)

    # ── Interactive Layout Modification ───────────────────────
    st.markdown("### Modify Layout")
    user_change = st.text_input(
        "Describe a layout change",
        placeholder='e.g. "Move ACC 20 ft further north" or "Swap turbine and HX positions"',
        key="feed_layout_change_input",
    )
    if user_change and st.button("Apply Change", key="feed_apply_layout"):
        _apply_layout_change(user_change, placed, wind_dir, detail, result)

    st.warning(
        "This plot plan is preliminary and requires survey, geotechnical, "
        "and site-specific review before use in layout design. "
        "Separation distances are preliminary — requires fire protection "
        "and hazardous area classification review per NFPA 30/70/497 and API 2510. "
        "Do not treat as code-compliant without licensed engineering review."
    )


def _apply_layout_change(user_request: str, placed: list[dict], wind_dir: str,
                         detail: dict, result):
    """Use Claude to interpret a natural-language layout change, check constraints, and redraw."""
    import json as _json

    try:
        import anthropic
        import streamlit as _st2
        api_key = None
        try:
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
        if not api_key:
            import os
            api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            st.error("No API key — cannot interpret layout changes. Set ANTHROPIC_API_KEY.")
            return

        # Build context for Claude
        equip_json = []
        for b in placed:
            equip_json.append({
                "tag": b["tag"], "service": b["service"],
                "x": round(b["x"], 1), "y": round(b["y"], 1),
                "w_ft": round(b["w_ft"], 1), "h_ft": round(b["h_ft"], 1),
                "system": b["system"],
            })

        constraints_text = "\n".join(
            f"- {f} to {t}: min {d} ft ({b})" for f, t, d, b in SEPARATION_RULES
        )

        prompt = f"""You are a plant layout engineer. The user wants to modify an ORC power plant plot plan.

CURRENT EQUIPMENT POSITIONS (x=East-West ft, y=North-South ft, center coordinates):
{_json.dumps(equip_json, indent=2)}

PREVAILING WIND: {wind_dir}

SEPARATION CONSTRAINTS:
{constraints_text}
- ACC minimum 50 ft from Turbine Building
- Electrical room minimum 25 ft from Turbine
- WF Storage minimum 100 ft from Electrical

USER REQUEST: "{user_request}"

Interpret the request and return a JSON object with:
1. "changes": dict of tag -> {{"x": new_x, "y": new_y}} for each moved equipment
2. "explanation": brief explanation of what you changed
3. "violations": list of any constraint violations introduced (empty if none)
4. "suggestions": if violations exist, suggest the minimum fix to satisfy BOTH the user's intent AND the constraint
5. "design_basis_updates": list of strings describing any design assumptions that changed (e.g. "ACC separation increased to 80 ft", "Plot orientation rotated 90°"). Empty list if layout-only.

Do NOT refuse a change that violates a constraint — show the violation clearly and suggest the minimum modification.

Return ONLY valid JSON, no markdown."""

        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}],
        )

        from synthesis import _extract_json
        parsed = _extract_json(response.content[0].text)

        if parsed and "changes" in parsed:
            changes = parsed["changes"]
            explanation = parsed.get("explanation", "Layout updated.")
            violations = parsed.get("violations", [])
            suggestions = parsed.get("suggestions", "")

            # Store overrides in session state
            overrides = st.session_state.get("feed_layout_overrides", {})
            overrides.update(changes)
            st.session_state["feed_layout_overrides"] = overrides

            st.success(f"Layout updated: {explanation}")

            if violations:
                st.warning(
                    f"**Constraint violations:** {'; '.join(violations)}\n\n"
                    f"**Suggested fix:** {suggestions}"
                )

            # Flag design basis updates for DBD review
            dbd_updates = parsed.get("design_basis_updates", [])
            if dbd_updates:
                st.info(
                    "**Design Basis impact detected:**\n"
                    + "\n".join(f"- {u}" for u in dbd_updates)
                    + "\n\nConsider updating the Design Basis Document "
                    "via the optimizer's DBD update workflow."
                )
                st.session_state["feed_layout_dbd_updates"] = dbd_updates

            st.rerun()
        else:
            st.error("Could not parse layout change response.")

    except Exception as e:
        st.error(f"Layout modification failed: {str(e)[:200]}")


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
        _render_plot_plan_tab(detail, result, design_basis)
    with t6:
        _render_open_items_tab(detail, result)
