"""
ORC cycle thermodynamic calculations for Config A and Config B.

Config A: Traditional ORC with recuperator + direct air-cooled condenser (ACC).
Config B: Isopentane power block + propane intermediate loop for heat rejection.

State point numbering (isopentane):
  1 - Turbine inlet (superheated or saturated vapor)
  2 - Turbine outlet
  3 - Recuperator hot-side outlet / condenser inlet
  4 - Condenser outlet (saturated liquid)
  5 - Pump outlet
  6 - Recuperator cold-side outlet / preheater inlet
  7 - Preheater exit / vaporizer inlet (subcooled liquid near bubble point)

Propane states (Config B only):
  A - Saturated vapor leaving propane evaporator (= iso condenser)
  B - Saturated liquid leaving propane condenser (ACC)
  C - Pump outlet
"""

from dataclasses import dataclass
from typing import Dict, List, Optional
import math
import numpy as np
from scipy.optimize import brentq


@dataclass
class StatePoint:
    """Thermodynamic state in imperial units."""
    T: float = 0.0          # degF
    P: float = 0.0          # psia
    h: float = 0.0          # BTU/lb
    s: float = 0.0          # BTU/lb-R
    quality: float = -1.0   # 0-1 for two-phase, -1 for single-phase
    phase: str = ""
    rho: float = 0.0        # lb/ft3
    label: str = ""


def _default_inputs():
    """Return default input dictionary."""
    return {
        "T_geo_in": 300,            # degF
        "m_dot_geo": 200,           # lb/s
        "cp_brine": 1.0,            # BTU/(lb-degF)
        "T_geo_out_min": 160,       # degF - silica constraint
        "T_ambient": 95,            # degF
        "dt_pinch_vaporizer": 10,   # degF
        "dt_pinch_preheater": 10,   # degF
        "dt_pinch_acc_a": 15,       # degF - Config A ACC pinch
        "dt_pinch_acc_b": 15,       # degF - Config B ACC pinch
        "dt_pinch_recup": 15,       # degF
        "dt_approach_intermediate": 10,  # degF (Config B only)
        "eta_turbine": 0.82,
        "eta_pump": 0.75,
        "superheat": 0,             # degF above saturation
        "v_tailpipe": 10,           # ft/s
        "v_acc_header": 15,         # ft/s
        "L_tailpipe_a": 30,         # ft
        "L_long_header": 120,       # ft
        "L_acc_header": 200,        # ft
        "L_iso_to_ihx": 40,         # ft (Config B segment 2)
        "electricity_price": 35,    # $/MWh
        "discount_rate": 0.08,
        "project_life": 30,         # years
        "capacity_factor": 0.95,
        # Hydraulic parameters
        "f_darcy": 0.02,
        # Config A equipment dP (psi)
        "dp_acc_tubes_a": 0.5,
        "dp_acc_headers_a": 0.3,
        "dp_recup_a": 0.3,
        # Config B isopentane side equipment dP (psi)
        "dp_ihx_iso": 0.5,
        "dp_recup_b": 0.3,
        "dp_tailpipe_iso_b": 0.3,
        # Config B propane side equipment dP (psi)
        "dp_acc_tubes_prop": 1.0,
        "dp_prop_headers": 0.5,
        "dp_ihx_prop": 0.5,
    }


def validate_inputs(inputs: dict) -> list:
    """Check physical constraints on user inputs. Returns list of warnings."""
    warnings = []
    inp = {**_default_inputs(), **inputs}
    T_geo = inp["T_geo_in"]
    T_amb = inp["T_ambient"]

    if T_geo <= T_amb:
        warnings.append("Geo-fluid inlet temperature must exceed ambient temperature.")
    if inp["eta_turbine"] > 1 or inp["eta_turbine"] < 0.3:
        warnings.append("Turbine isentropic efficiency should be 0.3-1.0.")
    if inp["eta_pump"] > 1 or inp["eta_pump"] < 0.3:
        warnings.append("Pump isentropic efficiency should be 0.3-1.0.")
    if inp["dt_pinch_recup"] < 0:
        warnings.append("Recuperator pinch dT must be non-negative.")
    if inp["dt_pinch_vaporizer"] < 0:
        warnings.append("Vaporizer pinch dT must be non-negative.")
    if inp["dt_pinch_preheater"] < 0:
        warnings.append("Preheater pinch dT must be non-negative.")
    return warnings


# ---------------------------------------------------------------------------
# Hydraulic helpers
# ---------------------------------------------------------------------------

def calc_dT_dP(fluid, T_sat, fp):
    """
    Central finite difference on CoolProp saturation curve.
    Returns dT/dP in degF/psi.  Fallback: 1.0 if CoolProp fails.
    """
    try:
        sat = fp.saturation_props(fluid, T=T_sat)
        P_center = sat["P_sat"]
        dp = 0.01  # psi perturbation
        sat_hi = fp.saturation_props(fluid, P=P_center + dp)
        sat_lo = fp.saturation_props(fluid, P=P_center - dp)
        dT_dP = (sat_hi["T_sat"] - sat_lo["T_sat"]) / (2 * dp)
        return abs(dT_dP)
    except Exception:
        return 1.0


# ---------------------------------------------------------------------------
# Duct sizing helpers
# ---------------------------------------------------------------------------

def _duct_segment(m_dot_lbs, rho, velocity, length, fluid, T_sat, fp, f_darcy=0.02):
    """
    Calculate a single duct segment: diameter, pressure drop, condensing
    temperature penalty.

    Args:
        m_dot_lbs: mass flow rate (lb/s)
        rho: vapor density (lb/ft3)
        velocity: design velocity (ft/s)
        length: segment length (ft)
        fluid: fluid name for dT/dP calculation
        T_sat: saturation temperature for dT/dP (degF)
        fp: FluidProperties instance

    Returns dict with diameter_in, velocity_fps, delta_P_psi, delta_T_cond_F
    """
    vol_flow = m_dot_lbs / rho  # ft3/s
    area = vol_flow / velocity  # ft2
    diameter_ft = (4 * area / math.pi) ** 0.5
    diameter_in = diameter_ft * 12

    # Darcy-Weisbach: dP = f * (L/D) * rho * v^2 / (2 * g_c)
    # g_c = 32.174 lbm·ft/(lbf·s²) converts lbm/(ft·s²) to lbf/ft²
    f = f_darcy
    g_c = 32.174
    if diameter_ft > 0:
        dp_lbft2 = f * (length / diameter_ft) * 0.5 * rho * velocity ** 2 / g_c
        dp_psi = dp_lbft2 / 144.0
    else:
        dp_psi = 0.0

    # dT/dP at saturation from Clausius-Clapeyron via CoolProp
    dt_cond = 0.0
    if dp_psi > 0 and fp is not None:
        try:
            sat1 = fp.saturation_props(fluid, T=T_sat)
            P1 = sat1["P_sat"]
            P2 = P1 - dp_psi
            if P2 > 0:
                sat2 = fp.saturation_props(fluid, P=P2)
                dt_cond = sat1["T_sat"] - sat2["T_sat"]
            else:
                dt_cond = 0.0
        except Exception:
            # Approximate: assume ~1 degF per psi for organic vapors
            dt_cond = dp_psi * 1.0

    return {
        "vol_flow_ft3s": vol_flow,
        "area_ft2": area,
        "diameter_ft": diameter_ft,
        "diameter_in": diameter_in,
        "velocity_fps": velocity,
        "length_ft": length,
        "delta_P_psi": dp_psi,
        "delta_T_cond_F": abs(dt_cond),
        "rho_lbft3": rho,
    }


N_TRAINS = 2  # parallel turbine/ACC trains sharing common vaporizer + preheater


def calculate_duct_segments_a(states, perf, inp, fp):
    """
    Config A duct segments (per-train sizing, N_TRAINS parallel trains):
      Seg 1: Tailpipe (turbine exit to recuperator) - state 2 conditions
      Seg 2: Recuperator exit header to ACC - state 3 conditions
      Seg 3: ACC distribution headers - state 3 conditions

    Diameters are per-train; total vol flow is plant-level.
    """
    m_dot_lbs = perf["m_dot_iso"] / 3600 / N_TRAINS  # per-train lb/s
    T_cond = perf["T_cond"]
    f_darcy = inp.get("f_darcy", 0.02)

    s2 = states["2"]
    s3 = states["3"]

    seg1 = _duct_segment(
        m_dot_lbs, s2.rho, inp["v_tailpipe"], inp["L_tailpipe_a"],
        "isopentane", T_cond, fp, f_darcy=f_darcy,
    )
    seg1["name"] = "Tailpipe (isopentane)"

    seg2 = _duct_segment(
        m_dot_lbs, s3.rho, inp["v_tailpipe"], inp["L_long_header"],
        "isopentane", T_cond, fp, f_darcy=f_darcy,
    )
    seg2["name"] = "ACC vapor header (isopentane)"

    seg3 = _duct_segment(
        m_dot_lbs, s3.rho, inp["v_acc_header"], inp["L_acc_header"],
        "isopentane", T_cond, fp, f_darcy=f_darcy,
    )
    seg3["name"] = "ACC distribution (isopentane)"

    segments = [seg1, seg2, seg3]
    total_dp = sum(s["delta_P_psi"] for s in segments)
    total_dt = sum(s["delta_T_cond_F"] for s in segments)

    return {
        "segments": segments,
        "n_trains": N_TRAINS,
        "total_delta_P_psi": total_dp,
        "total_delta_T_cond_F": total_dt,
        "tailpipe_diameter_in": seg1["diameter_in"],       # per-train
        "acc_header_diameter_in": seg3["diameter_in"],     # per-train
        "total_vol_flow_ft3s": seg1["vol_flow_ft3s"] * N_TRAINS,  # plant total
    }


def calculate_duct_segments_b(states, prop_states, perf, inp, fp):
    """
    Config B duct segments (per-train sizing, N_TRAINS parallel trains):
      Seg 1: ISO tailpipe (turbine exit to recuperator) - state 2
      Seg 2: ISO recup exit to intermediate HX - state 3
      Seg 3: Propane vapor header (IHX to ACC) - propane sat vapor
      Seg 4: Propane ACC distribution headers - propane sat vapor

    Diameters are per-train; total vol flow is plant-level.
    """
    m_dot_iso_lbs = perf["m_dot_iso"] / 3600 / N_TRAINS  # per-train
    m_dot_prop_lbs = perf["m_dot_prop"] / 3600 / N_TRAINS  # per-train
    T_cond_iso = perf["T_cond_iso"]
    T_propane_evap = perf["T_propane_evap"]
    T_propane_cond = perf["T_propane_cond"]
    f_darcy = inp.get("f_darcy", 0.02)

    s2 = states["2"]
    s3 = states["3"]
    sA = prop_states["A"]

    seg1 = _duct_segment(
        m_dot_iso_lbs, s2.rho, inp["v_tailpipe"], inp["L_tailpipe_a"],
        "isopentane", T_cond_iso, fp, f_darcy=f_darcy,
    )
    seg1["name"] = "Tailpipe (isopentane)"

    seg2 = _duct_segment(
        m_dot_iso_lbs, s3.rho, inp["v_tailpipe"], inp.get("L_iso_to_ihx", 40),
        "isopentane", T_cond_iso, fp, f_darcy=f_darcy,
    )
    seg2["name"] = "ISO to IHX duct"

    seg3 = _duct_segment(
        m_dot_prop_lbs, sA.rho, inp["v_tailpipe"], inp["L_long_header"],
        "propane", T_propane_cond, fp, f_darcy=f_darcy,
    )
    seg3["name"] = "Propane vapor header to ACC"

    seg4 = _duct_segment(
        m_dot_prop_lbs, sA.rho, inp["v_acc_header"], inp["L_acc_header"],
        "propane", T_propane_cond, fp, f_darcy=f_darcy,
    )
    seg4["name"] = "Propane ACC distribution"

    segments = [seg1, seg2, seg3, seg4]
    # Iso segments affect iso condensing; propane segments affect propane condensing
    iso_dt = seg1["delta_T_cond_F"] + seg2["delta_T_cond_F"]
    prop_dt = seg3["delta_T_cond_F"] + seg4["delta_T_cond_F"]
    total_dp = sum(s["delta_P_psi"] for s in segments)
    total_dt = iso_dt + prop_dt

    return {
        "segments": segments,
        "n_trains": N_TRAINS,
        "total_delta_P_psi": total_dp,
        "total_delta_T_cond_F": total_dt,
        "iso_delta_T_cond_F": iso_dt,
        "prop_delta_T_cond_F": prop_dt,
        "tailpipe_diameter_in": seg1["diameter_in"],           # per-train
        "acc_header_diameter_in": seg4["diameter_in"],         # per-train
        "propane_header_diameter_in": seg3["diameter_in"],     # per-train
        "total_vol_flow_ft3s": seg1["vol_flow_ft3s"] * N_TRAINS,    # plant total
        "propane_vol_flow_ft3s": seg3["vol_flow_ft3s"] * N_TRAINS,  # plant total
    }


# ---------------------------------------------------------------------------
# Hydraulic penalty calculations
# ---------------------------------------------------------------------------

def calculate_hydraulic_penalty_a(duct, inp, T_cond, fp):
    """
    Config A hydraulic penalty: pipe friction + equipment dP.
    Returns breakdown dict with per-component dP, dT, and totals.
    """
    segs = duct["segments"]
    # Pipe friction dP from Darcy-Weisbach (segments 0=tailpipe, 1=long header)
    pipe_dP_tailpipe = segs[0]["delta_P_psi"]
    pipe_dP_header = segs[1]["delta_P_psi"]
    pipe_dP_acc_dist = segs[2]["delta_P_psi"]
    pipe_dP = pipe_dP_tailpipe + pipe_dP_header + pipe_dP_acc_dist

    # Equipment dP from user inputs
    dp_recup = inp.get("dp_recup_a", 0.3)
    dp_acc_headers = inp.get("dp_acc_headers_a", 0.3)
    dp_acc_tubes = inp.get("dp_acc_tubes_a", 0.5)
    equip_dP = dp_recup + dp_acc_headers + dp_acc_tubes

    total_dP = pipe_dP + equip_dP

    # dT/dP sensitivity
    dT_dP_val = calc_dT_dP("isopentane", T_cond, fp)
    total_dT = total_dP * dT_dP_val

    return {
        "pipe_dP_psi": pipe_dP,
        "equip_dP_psi": equip_dP,
        "total_dP_psi": total_dP,
        "dT_dP_FperPsi": dT_dP_val,
        "total_dT_penalty_F": total_dT,
        "components": {
            "Tailpipe (D-W)": {"dP_psi": pipe_dP_tailpipe, "dT_F": pipe_dP_tailpipe * dT_dP_val, "type": "pipe"},
            "Long header (D-W)": {"dP_psi": pipe_dP_header, "dT_F": pipe_dP_header * dT_dP_val, "type": "pipe"},
            "ACC distribution (D-W)": {"dP_psi": pipe_dP_acc_dist, "dT_F": pipe_dP_acc_dist * dT_dP_val, "type": "pipe"},
            "Recuperator": {"dP_psi": dp_recup, "dT_F": dp_recup * dT_dP_val, "type": "equipment"},
            "ACC vapor headers": {"dP_psi": dp_acc_headers, "dT_F": dp_acc_headers * dT_dP_val, "type": "equipment"},
            "ACC tube bundle": {"dP_psi": dp_acc_tubes, "dT_F": dp_acc_tubes * dT_dP_val, "type": "equipment"},
        },
        "fluid": "isopentane",
    }


def calculate_hydraulic_penalty_b(duct, inp, T_cond_iso, T_propane_cond, fp):
    """
    Config B hydraulic penalty: two decoupled circuits.
    Isopentane side: all user inputs (short runs).
    Propane side: pipe D-W + equipment inputs.
    Returns breakdown dict with per-circuit and per-component details.
    """
    segs = duct["segments"]

    # --- Isopentane side (all user inputs — short runs, direct control) ---
    dp_tailpipe_iso = inp.get("dp_tailpipe_iso_b", 0.3)
    dp_recup_b = inp.get("dp_recup_b", 0.3)
    dp_ihx_iso = inp.get("dp_ihx_iso", 0.5)
    iso_dP = dp_tailpipe_iso + dp_recup_b + dp_ihx_iso

    dT_dP_iso = calc_dT_dP("isopentane", T_cond_iso, fp)
    iso_dT = iso_dP * dT_dP_iso

    # --- Propane side (pipe D-W + equipment inputs) ---
    # Segments 2,3 are propane header and propane ACC distribution
    pipe_dP_prop_header = segs[2]["delta_P_psi"]
    pipe_dP_prop_acc_dist = segs[3]["delta_P_psi"]
    pipe_dP_prop = pipe_dP_prop_header + pipe_dP_prop_acc_dist

    dp_acc_tubes_prop = inp.get("dp_acc_tubes_prop", 1.0)
    dp_ihx_prop = inp.get("dp_ihx_prop", 0.5)
    equip_dP_prop = dp_acc_tubes_prop + dp_ihx_prop

    prop_dP = pipe_dP_prop + equip_dP_prop

    dT_dP_prop = calc_dT_dP("propane", T_propane_cond, fp)
    prop_dT = prop_dP * dT_dP_prop

    # Combined: propane penalty passes through 1:1
    total_dT = iso_dT + prop_dT

    return {
        "iso_dP_psi": iso_dP,
        "iso_dT_F": iso_dT,
        "iso_dT_dP_FperPsi": dT_dP_iso,
        "prop_pipe_dP_psi": pipe_dP_prop,
        "prop_equip_dP_psi": equip_dP_prop,
        "prop_dP_psi": prop_dP,
        "prop_dT_F": prop_dT,
        "prop_dT_dP_FperPsi": dT_dP_prop,
        "total_dP_psi": iso_dP + prop_dP,
        "total_dT_penalty_F": total_dT,
        "components": {
            "ISO tailpipe": {"dP_psi": dp_tailpipe_iso, "dT_F": dp_tailpipe_iso * dT_dP_iso, "type": "equipment", "circuit": "iso"},
            "ISO recuperator": {"dP_psi": dp_recup_b, "dT_F": dp_recup_b * dT_dP_iso, "type": "equipment", "circuit": "iso"},
            "IHX iso side": {"dP_psi": dp_ihx_iso, "dT_F": dp_ihx_iso * dT_dP_iso, "type": "equipment", "circuit": "iso"},
            "Propane header (D-W)": {"dP_psi": pipe_dP_prop_header, "dT_F": pipe_dP_prop_header * dT_dP_prop, "type": "pipe", "circuit": "propane"},
            "Propane ACC dist (D-W)": {"dP_psi": pipe_dP_prop_acc_dist, "dT_F": pipe_dP_prop_acc_dist * dT_dP_prop, "type": "pipe", "circuit": "propane"},
            "ACC tube bundle (prop)": {"dP_psi": dp_acc_tubes_prop, "dT_F": dp_acc_tubes_prop * dT_dP_prop, "type": "equipment", "circuit": "propane"},
            "IHX propane side": {"dP_psi": dp_ihx_prop, "dT_F": dp_ihx_prop * dT_dP_prop, "type": "equipment", "circuit": "propane"},
        },
        "fluid_iso": "isopentane",
        "fluid_prop": "propane",
    }


# ---------------------------------------------------------------------------
# Cycle solvers
# ---------------------------------------------------------------------------

def _find_T_evap(inp, fp, fluid, T_cond):
    """
    Find T_evap that satisfies the vaporizer bubble-point pinch constraint.
    Brine energy balance determines iso mass flow, which then sets the brine
    temperature at the bubble point.
    """
    T_geo_in = inp["T_geo_in"]
    m_dot_geo = inp["m_dot_geo"]  # lb/s
    cp_geo = inp["cp_brine"]
    dt_pinch_vap = inp["dt_pinch_vaporizer"]
    dt_pinch_pre = inp["dt_pinch_preheater"]
    dt_pinch_recup = inp["dt_pinch_recup"]
    eta_turbine = inp["eta_turbine"]
    eta_pump = inp["eta_pump"]
    superheat = inp["superheat"]

    sat_cond = fp.saturation_props(fluid, T=T_cond)
    P_low = sat_cond["P_sat"]

    def residual(T_evap):
        try:
            sat_evap = fp.saturation_props(fluid, T=T_evap)
            P_high = sat_evap["P_sat"]
            h7 = sat_evap["h_f"]

            T1 = T_evap + superheat
            if superheat < 0.5:
                sp1 = fp.state_point(fluid, "T", T_evap, "Q", 1)
            else:
                sp1 = fp.state_point(fluid, "T", T1, "P", P_high)
            h1 = sp1["h"]

            sp4 = fp.state_point(fluid, "T", T_cond, "Q", 0)
            sp5s = fp.state_point(fluid, "S", sp4["s"], "P", P_high)
            h5 = sp4["h"] + (sp5s["h"] - sp4["h"]) / eta_pump
            sp5 = fp.state_point(fluid, "H", h5, "P", P_high)

            sp2s = fp.state_point(fluid, "S", sp1["s"], "P", P_low)
            h2 = h1 - eta_turbine * (h1 - sp2s["h"])

            T3 = sp5["T"] + dt_pinch_recup
            sp2_full = fp.state_point(fluid, "H", h2, "P", P_low)
            if T3 >= sp2_full["T"]:
                T3 = sp2_full["T"]
            sp3 = fp.state_point(fluid, "T", T3, "P", P_low)
            q_recup = sp2_full["h"] - sp3["h"]
            h6 = h5 + q_recup

            q_vaporizer = h1 - h7
            q_preheater = h7 - h6
            q_total = q_vaporizer + q_preheater

            if q_total <= 0:
                return -1e6

            # Energy balance: m_dot_iso * q_total = m_dot_geo * cp * (T_in - T_out)
            # But we need T_out from the preheater pinch constraint
            # Use maximum heat extraction: m_dot_iso from energy balance
            # T_brine_out = T_geo_in - m_dot_iso * q_total / (m_dot_geo * cp_geo)
            # But m_dot_iso is set by brine energy, so we need to check pinch

            # Preheater cold-end pinch: T_brine_out - T6 >= dt_pinch_pre
            sp6 = fp.state_point(fluid, "H", h6, "P", P_high)
            T6 = sp6["T"]
            T_brine_out_min_pre = T6 + dt_pinch_pre

            # Use the more restrictive outlet temperature
            T_geo_out_min = inp["T_geo_out_min"]
            T_brine_out = max(T_brine_out_min_pre, T_geo_out_min)

            Q_geo = m_dot_geo * cp_geo * (T_geo_in - T_brine_out) * 3600  # BTU/hr
            m_dot_iso = Q_geo / q_total  # lb/hr

            Q_vaporizer = m_dot_iso * q_vaporizer
            T_brine_mid = T_geo_in - Q_vaporizer / (m_dot_geo * 3600 * cp_geo)
            return T_brine_mid - T_evap - dt_pinch_vap
        except Exception:
            return -1e6

    # Clamp search range below critical temperature
    try:
        T_crit = fp.critical_point(fluid)["T_crit"]
    except Exception:
        T_crit = 369.1  # isopentane fallback
    T_lo = T_cond + 5
    T_hi = min(T_geo_in - superheat - 1, T_crit - 5)

    r_lo = residual(T_lo)
    r_hi = residual(T_hi)

    if r_lo <= 0:
        return T_lo
    if r_hi >= 0:
        return T_hi

    return brentq(residual, T_lo, T_hi, xtol=0.1)


def _solve_cycle_core(inp, fp, fluid, T_cond):
    """Solve core ORC cycle at given condensing temperature. Returns states and performance dict."""
    T_evap = _find_T_evap(inp, fp, fluid, T_cond)

    sat_cond = fp.saturation_props(fluid, T=T_cond)
    sat_evap = fp.saturation_props(fluid, T=T_evap)
    P_low = sat_cond["P_sat"]
    P_high = sat_evap["P_sat"]

    # State 1: Turbine inlet
    T1 = T_evap + inp["superheat"]
    if inp["superheat"] < 0.5:
        # At or near saturation: use quality=1 to avoid CoolProp T-P ambiguity
        sp1 = fp.state_point(fluid, "T", T_evap, "Q", 1)
    else:
        sp1 = fp.state_point(fluid, "T", T1, "P", P_high)
    s1 = StatePoint(T=sp1["T"], P=sp1["P"], h=sp1["h"], s=sp1["s"],
                    phase=sp1["phase"], rho=sp1["rho"], label="1-Turbine Inlet")

    # State 2: Turbine outlet
    sp2s = fp.state_point(fluid, "S", s1.s, "P", P_low)
    h2 = s1.h - inp["eta_turbine"] * (s1.h - sp2s["h"])
    sp2 = fp.state_point(fluid, "H", h2, "P", P_low)
    s2 = StatePoint(T=sp2["T"], P=sp2["P"], h=sp2["h"], s=sp2["s"],
                    phase=sp2["phase"], rho=sp2["rho"], label="2-Turbine Outlet",
                    quality=sp2.get("quality", -1) if isinstance(sp2, dict) else -1)

    # State 4: Condenser outlet (saturated liquid)
    sp4 = fp.state_point(fluid, "T", T_cond, "Q", 0)
    s4 = StatePoint(T=sp4["T"], P=sp4["P"], h=sp4["h"], s=sp4["s"],
                    quality=0, phase="liquid", rho=sp4["rho"],
                    label="4-Condenser Outlet")

    # State 5: Pump outlet
    sp5s = fp.state_point(fluid, "S", s4.s, "P", P_high)
    h5 = s4.h + (sp5s["h"] - s4.h) / inp["eta_pump"]
    sp5 = fp.state_point(fluid, "H", h5, "P", P_high)
    s5 = StatePoint(T=sp5["T"], P=sp5["P"], h=sp5["h"], s=sp5["s"],
                    phase=sp5["phase"], rho=sp5["rho"], label="5-Pump Outlet")

    # State 3: Recuperator hot-side outlet
    T3 = s5.T + inp["dt_pinch_recup"]
    if T3 >= s2.T:
        T3 = s2.T
    sp3 = fp.state_point(fluid, "T", T3, "P", P_low)
    s3 = StatePoint(T=sp3["T"], P=sp3["P"], h=sp3["h"], s=sp3["s"],
                    phase=sp3["phase"], rho=sp3["rho"],
                    label="3-Recuperator Hot Out")

    # State 6: Recuperator cold-side outlet
    q_recup = s2.h - s3.h
    h6 = s5.h + q_recup
    sp6 = fp.state_point(fluid, "H", h6, "P", P_high)
    s6 = StatePoint(T=sp6["T"], P=sp6["P"], h=sp6["h"], s=sp6["s"],
                    phase=sp6["phase"], rho=sp6["rho"],
                    label="6-Preheater Inlet")

    # State 7: Vaporizer inlet (saturated liquid at T_evap)
    sp7 = fp.state_point(fluid, "T", T_evap, "Q", 0)
    s7 = StatePoint(T=sp7["T"], P=sp7["P"], h=sp7["h"], s=sp7["s"],
                    quality=0, phase="liquid", rho=sp7["rho"],
                    label="7-Vaporizer Inlet")

    states = {"1": s1, "2": s2, "3": s3, "4": s4, "5": s5, "6": s6, "7": s7}

    # Heat duties (BTU/lb)
    q_vaporizer = s1.h - s7.h
    q_preheater = s7.h - s6.h
    q_evap = q_vaporizer + q_preheater

    # Mass flow from brine energy balance
    m_dot_geo = inp["m_dot_geo"]  # lb/s
    cp_geo = inp["cp_brine"]
    T_geo_in = inp["T_geo_in"]

    # Preheater pinch sets minimum brine outlet
    T_brine_out_pinch = s6.T + inp["dt_pinch_preheater"]
    T_brine_out = max(T_brine_out_pinch, inp["T_geo_out_min"])

    Q_geo = m_dot_geo * cp_geo * (T_geo_in - T_brine_out) * 3600  # BTU/hr
    m_dot_iso = Q_geo / q_evap  # lb/hr

    # Brine temperature tracking
    Q_vaporizer = m_dot_iso * q_vaporizer
    T_brine_mid = T_geo_in - Q_vaporizer / (m_dot_geo * 3600 * cp_geo)
    T_geo_out_calc = T_brine_mid - (m_dot_iso * q_preheater) / (m_dot_geo * 3600 * cp_geo)

    # Pinch verification
    vaporizer_pinch = T_brine_mid - s7.T
    vaporizer_pinch_violation = vaporizer_pinch < inp["dt_pinch_vaporizer"] - 0.1
    preheater_pinch = T_geo_out_calc - s6.T
    preheater_pinch_violation = preheater_pinch < inp["dt_pinch_preheater"] - 0.1
    brine_outlet_violation = T_geo_out_calc < inp["T_geo_out_min"] - 0.1

    # Performance
    w_turbine = s1.h - s2.h
    w_pump = s5.h - s4.h
    w_net = w_turbine - w_pump

    gross_power_kw = (m_dot_iso * w_turbine) / 3412.14
    net_power_kw = (m_dot_iso * w_net) / 3412.14
    eta_thermal = w_net / q_evap if q_evap > 0 else 0

    vol_flow_2 = m_dot_iso / s2.rho / 3600  # ft3/s

    # Heat rejection duty
    q_cond = s3.h - s4.h  # BTU/lb
    Q_reject_btu_hr = m_dot_iso * q_cond
    Q_reject_mmbtu_hr = Q_reject_btu_hr / 1e6
    Q_recup_mmbtu_hr = (m_dot_iso * q_recup) / 1e6

    performance = {
        "gross_power_kw": gross_power_kw,
        "net_power_kw": net_power_kw,
        "eta_thermal": eta_thermal,
        "m_dot_iso": m_dot_iso,
        "m_dot_iso_lbs": m_dot_iso / 3600,
        "w_turbine": w_turbine,
        "w_pump": w_pump,
        "w_net": w_net,
        "q_evap": q_evap,
        "q_vaporizer": q_vaporizer,
        "q_preheater": q_preheater,
        "q_cond": q_cond,
        "q_recup": q_recup,
        "Q_reject_mmbtu_hr": Q_reject_mmbtu_hr,
        "Q_recup_mmbtu_hr": Q_recup_mmbtu_hr,
        "P_high": P_high,
        "P_low": P_low,
        "T_cond": T_cond,
        "T_evap": T_evap,
        "T_brine_mid": T_brine_mid,
        "T_geo_out_calc": T_geo_out_calc,
        "vaporizer_pinch": vaporizer_pinch,
        "vaporizer_pinch_violation": vaporizer_pinch_violation,
        "preheater_pinch": preheater_pinch,
        "preheater_pinch_violation": preheater_pinch_violation,
        "brine_outlet_violation": brine_outlet_violation,
        "vol_flow_turbine_exit": vol_flow_2,
        "pressure_ratio": P_high / P_low if P_low > 0 else 0,
        "brine_effectiveness": net_power_kw / inp["m_dot_geo"] if inp["m_dot_geo"] > 0 else 0,
    }

    return states, performance


def solve_config_a(inputs: dict, fp) -> dict:
    """
    Solve Config A with iterative duct pressure drop convergence.
    Iterates: solve cycle -> calculate duct dP -> apply condensing penalty -> re-solve.
    """
    inp = {**_default_inputs(), **inputs}
    fluid = "isopentane"

    # Get critical temperature for guard check
    try:
        T_crit_iso = fp.critical_point(fluid)["T_crit"]
    except Exception:
        T_crit_iso = 369.1  # isopentane critical, degF

    T_cond_base = inp["T_ambient"] + inp["dt_pinch_acc_a"]

    # Pre-check: base condensing temperature must be well below critical
    # Need margin for duct penalty (~5°F) + CoolProp numerical stability near critical
    if T_cond_base >= T_crit_iso - 20:
        raise ValueError(
            f"Isopentane condensing temperature ({T_cond_base:.1f}°F) is too close to its "
            f"critical temperature ({T_crit_iso:.1f}°F). "
            f"T_ambient ({inp['T_ambient']:.1f}) + ACC pinch "
            f"({inp['dt_pinch_acc_a']:.1f}) = {T_cond_base:.1f}°F must be below "
            f"{T_crit_iso - 20:.0f}°F to allow room for duct pressure penalties."
        )

    dt_penalty = 0.0

    for iteration in range(20):
        T_cond = T_cond_base + dt_penalty

        # Guard: isopentane condensing temperature must be below critical
        if T_cond >= T_crit_iso - 5:
            raise ValueError(
                f"Isopentane condensing temperature ({T_cond:.1f}°F) exceeds its "
                f"critical temperature ({T_crit_iso:.1f}°F). "
                f"Breakdown: T_ambient ({inp['T_ambient']:.1f}) + ACC pinch "
                f"({inp['dt_pinch_acc_a']:.1f}) + duct penalty ({dt_penalty:.1f}) = "
                f"{T_cond:.1f}°F. Reduce T_ambient, dt_pinch_acc_a, or duct velocities."
            )

        states, perf = _solve_cycle_core(inp, fp, fluid, T_cond)
        perf["T_cond"] = T_cond

        duct = calculate_duct_segments_a(states, perf, inp, fp)
        hydraulic = calculate_hydraulic_penalty_a(duct, inp, T_cond, fp)
        new_penalty = hydraulic["total_dT_penalty_F"]

        if abs(new_penalty - dt_penalty) < 0.1:
            break
        dt_penalty = new_penalty

    perf["duct_penalty_F"] = dt_penalty
    perf["hydraulic_penalty_F"] = dt_penalty
    perf["converged"] = abs(new_penalty - dt_penalty) < 0.1

    return {"states": states, "performance": perf, "duct": duct, "hydraulic": hydraulic}


def solve_config_b(inputs: dict, fp) -> dict:
    """
    Solve Config B with iterative duct pressure drop convergence.
    """
    inp = {**_default_inputs(), **inputs}
    fluid = "isopentane"
    prop_fluid = "propane"

    T_amb = inp["T_ambient"]
    dt_acc = inp["dt_pinch_acc_b"]
    dt_approach = inp["dt_approach_intermediate"]

    T_propane_cond = T_amb + dt_acc
    T_cond_iso_base = T_amb + dt_acc + dt_approach

    # Get critical temperatures for guard checks
    try:
        T_crit_prop = fp.critical_point(prop_fluid)["T_crit"]
    except Exception:
        T_crit_prop = 206.1  # propane critical, degF
    try:
        T_crit_iso = fp.critical_point(fluid)["T_crit"]
    except Exception:
        T_crit_iso = 369.1  # isopentane critical, degF

    # Pre-check: propane condensing temperature must be below critical
    # Need margin for CoolProp numerical stability near critical point
    if T_propane_cond >= T_crit_prop - 10:
        raise ValueError(
            f"Propane condensing temperature ({T_propane_cond:.1f}°F) is too close to its "
            f"critical temperature ({T_crit_prop:.1f}°F). "
            f"This is set by T_ambient ({T_amb:.1f}°F) + ACC pinch ({dt_acc:.1f}°F) = "
            f"{T_propane_cond:.1f}°F. Reduce T_ambient or dt_pinch_acc_b so their sum "
            f"stays below {T_crit_prop - 10:.0f}°F."
        )

    # Pre-check: isopentane condensing temperature must be below critical
    if T_cond_iso_base >= T_crit_iso - 20:
        raise ValueError(
            f"Isopentane condensing temperature ({T_cond_iso_base:.1f}°F) is too close to its "
            f"critical temperature ({T_crit_iso:.1f}°F). "
            f"T_ambient ({T_amb:.1f}) + ACC pinch ({dt_acc:.1f}) + IHX approach "
            f"({dt_approach:.1f}) = {T_cond_iso_base:.1f}°F must be below "
            f"{T_crit_iso - 20:.0f}°F."
        )

    dt_penalty = 0.0

    for iteration in range(20):
        T_cond_iso = T_cond_iso_base + dt_penalty
        T_propane_evap = T_cond_iso - dt_approach
        if T_propane_evap <= T_propane_cond + 1:
            T_propane_evap = T_propane_cond + 2

        # Guard: propane evaporating temperature must be below critical
        if T_propane_evap >= T_crit_prop - 5:
            raise ValueError(
                f"Propane evaporating temperature ({T_propane_evap:.1f}°F) exceeds its "
                f"critical temperature ({T_crit_prop:.1f}°F). "
                f"Breakdown: T_ambient ({T_amb:.1f}) + ACC pinch ({dt_acc:.1f}) + "
                f"duct penalty ({dt_penalty:.1f}) = {T_propane_evap + dt_approach:.1f}°F "
                f"iso condensing, minus IHX approach ({dt_approach:.1f}) = "
                f"{T_propane_evap:.1f}°F propane evaporating. "
                f"Reduce T_ambient, dt_pinch_acc_b, or duct velocities."
            )

        # Guard: isopentane condensing temperature must be below critical
        if T_cond_iso >= T_crit_iso - 5:
            raise ValueError(
                f"Isopentane condensing temperature ({T_cond_iso:.1f}°F) exceeds its "
                f"critical temperature ({T_crit_iso:.1f}°F). "
                f"Breakdown: T_ambient ({T_amb:.1f}) + ACC pinch ({dt_acc:.1f}) + "
                f"IHX approach ({dt_approach:.1f}) + duct penalty ({dt_penalty:.1f}) = "
                f"{T_cond_iso:.1f}°F. Reduce inputs or duct velocities."
            )

        states, perf = _solve_cycle_core(inp, fp, fluid, T_cond_iso)

        # Override keys for Config B naming
        perf["T_cond_iso"] = T_cond_iso
        perf["P_high_iso"] = perf["P_high"]
        perf["P_low_iso"] = perf["P_low"]
        perf["pressure_ratio_iso"] = perf["pressure_ratio"]
        perf["T_propane_evap"] = T_propane_evap
        perf["T_propane_cond"] = T_propane_cond

        # Propane loop
        sat_prop_evap = fp.saturation_props(prop_fluid, T=T_propane_evap)
        sat_prop_cond = fp.saturation_props(prop_fluid, T=T_propane_cond)
        P_prop_evap = sat_prop_evap["P_sat"]
        P_prop_cond = sat_prop_cond["P_sat"]

        spA = fp.state_point(prop_fluid, "T", T_propane_evap, "Q", 1)
        sA = StatePoint(T=spA["T"], P=spA["P"], h=spA["h"], s=spA["s"],
                        quality=1, phase="vapor", rho=spA["rho"],
                        label="A-Propane Sat Vapor")

        spB = fp.state_point(prop_fluid, "T", T_propane_cond, "Q", 0)
        sB = StatePoint(T=spB["T"], P=spB["P"], h=spB["h"], s=spB["s"],
                        quality=0, phase="liquid", rho=spB["rho"],
                        label="B-Propane Sat Liquid")

        spCs = fp.state_point(prop_fluid, "S", sB.s, "P", P_prop_evap)
        h_C = sB.h + (spCs["h"] - sB.h) / inp["eta_pump"]
        spC = fp.state_point(prop_fluid, "H", h_C, "P", P_prop_evap)
        sC = StatePoint(T=spC["T"], P=spC["P"], h=spC["h"], s=spC["s"],
                        phase=spC["phase"], rho=spC["rho"],
                        label="C-Propane Pump Out")

        prop_states = {"A": sA, "B": sB, "C": sC}

        # Propane mass flow
        q_cond_iso = perf["q_cond"]
        q_evap_prop = sA.h - sC.h
        m_dot_iso = perf["m_dot_iso"]
        m_dot_prop = (m_dot_iso * q_cond_iso) / q_evap_prop

        perf["m_dot_prop"] = m_dot_prop
        perf["q_cond_iso"] = q_cond_iso
        perf["w_pump_iso"] = perf["w_pump"]
        perf["w_pump_prop"] = sC.h - sB.h
        w_pump_total = perf["w_pump_iso"] + perf["w_pump_prop"] * (m_dot_prop / m_dot_iso)
        perf["w_pump_total"] = w_pump_total
        perf["w_net"] = perf["w_turbine"] - w_pump_total
        perf["net_power_kw"] = (m_dot_iso * perf["w_net"]) / 3412.14
        perf["eta_thermal"] = perf["w_net"] / perf["q_evap"] if perf["q_evap"] > 0 else 0
        perf["brine_effectiveness"] = perf["net_power_kw"] / inp["m_dot_geo"] if inp["m_dot_geo"] > 0 else 0

        perf["P_prop_evap"] = P_prop_evap
        perf["P_prop_cond"] = P_prop_cond
        perf["pressure_ratio_prop"] = P_prop_evap / P_prop_cond if P_prop_cond > 0 else 0
        perf["vol_flow_prop_evap_exit"] = m_dot_prop / sA.rho / 3600

        # Duct segments
        duct = calculate_duct_segments_b(states, prop_states, perf, inp, fp)
        hydraulic = calculate_hydraulic_penalty_b(duct, inp, T_cond_iso, T_propane_cond, fp)
        new_penalty = hydraulic["total_dT_penalty_F"]

        if abs(new_penalty - dt_penalty) < 0.1:
            break
        dt_penalty = new_penalty

    perf["duct_penalty_F"] = dt_penalty
    perf["hydraulic_penalty_F"] = dt_penalty
    perf["iso_hydraulic_dT_F"] = hydraulic["iso_dT_F"]
    perf["prop_hydraulic_dT_F"] = hydraulic["prop_dT_F"]
    perf["converged"] = abs(new_penalty - dt_penalty) < 0.1

    return {
        "states": states,
        "propane_states": prop_states,
        "performance": perf,
        "duct": duct,
        "hydraulic": hydraulic,
    }


def verify_recuperator_pinch(states: dict, fp) -> dict:
    """Check recuperator internal pinch-point violation."""
    s2 = states["2"]
    s3 = states["3"]
    s5 = states["5"]
    s6 = states["6"]

    N = 20
    h_hot_vals = np.linspace(s2.h, s3.h, N)
    h_cold_vals = np.linspace(s6.h, s5.h, N)

    fluid = "isopentane"
    min_pinch = float("inf")
    violation = False
    pinch_profile = []

    P_low = s3.P
    P_high = s6.P

    for i in range(N):
        sp_hot = fp.state_point(fluid, "H", h_hot_vals[i], "P", P_low)
        sp_cold = fp.state_point(fluid, "H", h_cold_vals[i], "P", P_high)
        dt = sp_hot["T"] - sp_cold["T"]
        pinch_profile.append({
            "fraction": i / (N - 1),
            "T_hot": sp_hot["T"],
            "T_cold": sp_cold["T"],
            "dT": dt,
        })
        if dt < min_pinch:
            min_pinch = dt
        if dt < -0.1:
            violation = True

    return {
        "violation": violation,
        "min_pinch_dT": min_pinch,
        "profile": pinch_profile,
    }


def run_validation_checks(perf, states, duct, config, inp, fp):
    """Run convergence and validation checks. Returns list of (name, passed, detail)."""
    checks = []

    # 1. Brine outlet above minimum
    T_out = perf["T_geo_out_calc"]
    T_min = inp.get("T_geo_out_min", 160)
    checks.append(("Brine outlet >= minimum",
                    T_out >= T_min - 0.1,
                    f"{T_out:.1f} vs {T_min:.1f} degF"))

    # 2. Recuperator pinch
    pinch = verify_recuperator_pinch(states, fp)
    checks.append(("No recuperator pinch violation",
                    not pinch["violation"],
                    f"Min dT = {pinch['min_pinch_dT']:.1f} degF"))

    # 3. Turbine inlet quality >= 1.0
    s1 = states["1"]
    q1_ok = s1.quality < 0 or s1.quality >= 1.0  # -1 = superheated
    checks.append(("Turbine inlet quality >= 1.0",
                    q1_ok,
                    f"Phase: {s1.phase}"))

    # 4. Turbine exit quality > 0.90
    s2 = states["2"]
    q2 = s2.quality if s2.quality >= 0 else 1.0
    checks.append(("Turbine exit quality > 0.90",
                    q2 > 0.90,
                    f"Quality: {q2:.3f}" if s2.quality >= 0 else f"Phase: {s2.phase}"))

    # 5. Duct velocities within 5-25 ft/s
    all_vels_ok = True
    vel_detail = []
    for seg in duct["segments"]:
        v = seg["velocity_fps"]
        ok = 5 <= v <= 25
        if not ok:
            all_vels_ok = False
        vel_detail.append(f"{seg['name']}: {v:.0f} ft/s")
    checks.append(("Duct velocities 5-25 ft/s",
                    all_vels_ok,
                    "; ".join(vel_detail)))

    # 6. Cycle converged
    checks.append(("Cycle converged",
                    perf.get("converged", False),
                    f"Duct penalty: {perf.get('duct_penalty_F', 0):.2f} degF"))

    # 7. Tailpipe diameter benchmark (at 50 MW scale)
    gross_mw = perf["gross_power_kw"] / 1000
    if gross_mw > 0:
        scale = 50.0 / gross_mw
        tp_dia = duct["tailpipe_diameter_in"]
        scaled_dia = tp_dia * (scale ** 0.5)  # diameter scales with sqrt of flow
        if config == "A":
            benchmark = 80
            deviation = abs(scaled_dia - benchmark) / benchmark
            checks.append(("Tailpipe diameter benchmark",
                            deviation <= 0.20,
                            f"Scaled: {scaled_dia:.0f}\" vs {benchmark}\" benchmark ({deviation*100:.0f}% dev)"))

    return checks
