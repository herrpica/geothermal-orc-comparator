"""
ORC cycle thermodynamic calculations for Config A and Config B.

Config A: Traditional ORC with recuperator + direct air-cooled condenser (ACC).
Config B: Isopentane power block + propane intermediate loop for heat rejection.

State point numbering (isopentane):
  1 – Turbine inlet (superheated or saturated vapor)
  2 – Turbine outlet
  3 – Recuperator hot-side outlet / condenser inlet
  4 – Condenser outlet (saturated liquid)
  5 – Pump outlet
  6 – Recuperator cold-side outlet / evaporator inlet

Propane states (Config B only):
  A – Saturated vapor leaving propane evaporator (= iso condenser)
  B – Saturated liquid leaving propane condenser (ACC)
  C – Pump outlet
"""

from dataclasses import dataclass, field
from typing import Dict, Optional
import numpy as np


@dataclass
class StatePoint:
    """Thermodynamic state in imperial units."""
    T: float = 0.0          # °F
    P: float = 0.0          # psia
    h: float = 0.0          # BTU/lb
    s: float = 0.0          # BTU/lb·R
    quality: float = -1.0   # 0-1 for two-phase, -1 for single-phase
    phase: str = ""
    rho: float = 0.0        # lb/ft³
    label: str = ""


def validate_inputs(inputs: dict):
    """Check physical constraints on user inputs. Returns list of warnings."""
    warnings = []
    T_geo = inputs.get("T_geo_in", 300)
    T_amb = inputs.get("T_ambient", 95)

    if T_geo <= T_amb:
        warnings.append("Geo-fluid inlet temperature must exceed ambient temperature.")
    if inputs.get("T_geo_out", 160) >= T_geo:
        warnings.append("Geo-fluid outlet must be below inlet temperature.")
    if inputs.get("eta_turbine", 0.82) > 1 or inputs.get("eta_turbine", 0.82) < 0.3:
        warnings.append("Turbine isentropic efficiency should be 0.3–1.0.")
    if inputs.get("eta_pump", 0.75) > 1 or inputs.get("eta_pump", 0.75) < 0.3:
        warnings.append("Pump isentropic efficiency should be 0.3–1.0.")
    if inputs.get("dt_pinch_recup", 15) < 0:
        warnings.append("Recuperator pinch ΔT must be non-negative.")

    return warnings


def _default_inputs():
    """Return default input dictionary."""
    return {
        "T_geo_in": 300,          # °F
        "T_geo_out": 160,         # °F
        "m_dot_geo": 500_000,     # lb/hr
        "T_ambient": 95,          # °F
        "dt_pinch_evap": 10,      # °F
        "dt_pinch_acc": 25,       # °F  (ACC approach for Config A)
        "dt_pinch_recup": 15,     # °F  (recuperator pinch)
        "dt_approach_intermediate": 10,  # °F  (Config B intermediate HX)
        "eta_turbine": 0.82,
        "eta_pump": 0.75,
        "superheat": 5,           # °F above saturation
        "T_propane_cond": None,   # °F – auto-computed if None
        "v_duct": 60,             # ft/s duct velocity
        "electricity_price": 0.08,  # $/kWh
        "discount_rate": 0.08,
        "project_life": 30,       # years
        "capacity_factor": 0.95,
    }


def solve_config_a(inputs: dict, fp) -> dict:
    """
    Solve the traditional ORC cycle (Config A).

    Returns dict:
      states: dict of StatePoint (keys "1"–"6")
      performance: dict with net_power_kw, eta_thermal, m_dot_iso, etc.
    """
    inp = {**_default_inputs(), **inputs}
    fluid = "isopentane"

    # ── Condenser temperature ────────────────────────────────────────────
    T_cond = inp["T_ambient"] + inp["dt_pinch_acc"]  # °F

    # ── Evaporator temperature ───────────────────────────────────────────
    T_evap = inp["T_geo_in"] - inp["dt_pinch_evap"]  # °F

    # ── Saturation pressures ─────────────────────────────────────────────
    sat_cond = fp.saturation_props(fluid, T=T_cond)
    sat_evap = fp.saturation_props(fluid, T=T_evap)
    P_low = sat_cond["P_sat"]
    P_high = sat_evap["P_sat"]

    # ── State 1: Turbine inlet (superheated vapor) ───────────────────────
    T1 = T_evap + inp["superheat"]
    sp1 = fp.state_point(fluid, "T", T1, "P", P_high)
    s1 = StatePoint(T=sp1["T"], P=sp1["P"], h=sp1["h"], s=sp1["s"],
                    phase=sp1["phase"], rho=sp1["rho"], label="1-Turbine Inlet")

    # ── State 2: Turbine outlet (isentropic efficiency) ──────────────────
    sp2s = fp.state_point(fluid, "S", s1.s, "P", P_low)
    h2s = sp2s["h"]
    h2 = s1.h - inp["eta_turbine"] * (s1.h - h2s)
    sp2 = fp.state_point(fluid, "H", h2, "P", P_low)
    s2 = StatePoint(T=sp2["T"], P=sp2["P"], h=sp2["h"], s=sp2["s"],
                    phase=sp2["phase"], rho=sp2["rho"], label="2-Turbine Outlet")

    # ── State 4: Condenser outlet (saturated liquid) ─────────────────────
    sp4 = fp.state_point(fluid, "T", T_cond, "Q", 0)
    s4 = StatePoint(T=sp4["T"], P=sp4["P"], h=sp4["h"], s=sp4["s"],
                    quality=0, phase="liquid", rho=sp4["rho"],
                    label="4-Condenser Outlet")

    # ── State 5: Pump outlet (isentropic efficiency) ─────────────────────
    sp5s = fp.state_point(fluid, "S", s4.s, "P", P_high)
    h5s = sp5s["h"]
    h5 = s4.h + (h5s - s4.h) / inp["eta_pump"]
    sp5 = fp.state_point(fluid, "H", h5, "P", P_high)
    s5 = StatePoint(T=sp5["T"], P=sp5["P"], h=sp5["h"], s=sp5["s"],
                    phase=sp5["phase"], rho=sp5["rho"], label="5-Pump Outlet")

    # ── State 3: Recuperator hot-side outlet ─────────────────────────────
    # T3 = T5 + dt_pinch_recup
    T3 = s5.T + inp["dt_pinch_recup"]
    if T3 >= s2.T:
        # No useful recuperation — skip
        T3 = s2.T
    sp3 = fp.state_point(fluid, "T", T3, "P", P_low)
    s3 = StatePoint(T=sp3["T"], P=sp3["P"], h=sp3["h"], s=sp3["s"],
                    phase=sp3["phase"], rho=sp3["rho"],
                    label="3-Recuperator Hot Out")

    # ── State 6: Recuperator cold-side outlet ────────────────────────────
    q_recup = s2.h - s3.h  # BTU/lb recovered
    h6 = s5.h + q_recup
    sp6 = fp.state_point(fluid, "H", h6, "P", P_high)
    s6 = StatePoint(T=sp6["T"], P=sp6["P"], h=sp6["h"], s=sp6["s"],
                    phase=sp6["phase"], rho=sp6["rho"],
                    label="6-Recuperator Cold Out")

    states = {"1": s1, "2": s2, "3": s3, "4": s4, "5": s5, "6": s6}

    # ── Mass flow rate (energy balance on evaporator) ────────────────────
    cp_geo = 1.0  # BTU/(lb·°F) approx for brine
    q_evap = s1.h - s6.h  # BTU/lb isopentane
    Q_geo = inp["m_dot_geo"] * cp_geo * (inp["T_geo_in"] - inp["T_geo_out"])  # BTU/hr
    m_dot_iso = Q_geo / q_evap  # lb/hr

    # ── Performance ──────────────────────────────────────────────────────
    w_turbine = s1.h - s2.h       # BTU/lb
    w_pump = s5.h - s4.h          # BTU/lb
    w_net = w_turbine - w_pump    # BTU/lb

    gross_power_btu_hr = m_dot_iso * w_turbine
    net_power_btu_hr = m_dot_iso * w_net
    gross_power_kw = gross_power_btu_hr / 3412.14
    net_power_kw = net_power_btu_hr / 3412.14

    eta_thermal = w_net / q_evap if q_evap > 0 else 0

    # Volumetric flow at turbine outlet (State 2)
    vol_flow_2 = m_dot_iso / s2.rho / 3600  # ft³/s

    performance = {
        "gross_power_kw": gross_power_kw,
        "net_power_kw": net_power_kw,
        "eta_thermal": eta_thermal,
        "m_dot_iso": m_dot_iso,
        "w_turbine": w_turbine,
        "w_pump": w_pump,
        "w_net": w_net,
        "q_evap": q_evap,
        "q_cond": s3.h - s4.h,
        "q_recup": q_recup,
        "P_high": P_high,
        "P_low": P_low,
        "T_cond": T_cond,
        "T_evap": T_evap,
        "vol_flow_turbine_exit": vol_flow_2,
        "pressure_ratio": P_high / P_low if P_low > 0 else 0,
    }

    return {"states": states, "performance": performance}


def solve_config_b(inputs: dict, fp) -> dict:
    """
    Solve Config B: isopentane ORC with propane intermediate loop.

    The iso condenser rejects heat to propane (phase-change to phase-change),
    and the propane loop rejects heat through an ACC.
    """
    inp = {**_default_inputs(), **inputs}
    fluid = "isopentane"
    prop_fluid = "propane"

    # ── Temperatures ─────────────────────────────────────────────────────
    T_amb = inp["T_ambient"]
    dt_acc = inp["dt_pinch_acc"]
    dt_approach = inp["dt_approach_intermediate"]
    dt_evap = inp["dt_pinch_evap"]

    # Propane condensing in ACC at ambient + ACC pinch
    T_propane_cond = inp.get("T_propane_cond") or (T_amb + dt_acc)

    # Isopentane condensing temperature is elevated above Config A:
    #   T_cond_iso = T_ambient + ΔT_ACC_pinch + ΔT_approach_intermediate
    T_cond_iso = T_amb + dt_acc + dt_approach

    # Propane evaporates in the intermediate HX, absorbing heat from
    # condensing isopentane. The approach ΔT spans iso condensing → propane
    # evaporation, so:
    #   T_propane_evap = T_cond_iso - dt_approach
    # This equals T_amb + dt_acc (= T_propane_cond) by construction.
    # The propane loop therefore operates with minimal temperature lift;
    # its value comes from replacing low-density isopentane vapor in the
    # ACC duct with higher-density propane vapor.
    T_propane_evap = T_cond_iso - dt_approach
    # Ensure at least 2°F lift so CoolProp can distinguish the two states
    if T_propane_evap <= T_propane_cond + 1:
        T_propane_evap = T_propane_cond + 2

    T_evap = inp["T_geo_in"] - dt_evap

    # ── Saturation pressures (isopentane) ────────────────────────────────
    sat_cond_iso = fp.saturation_props(fluid, T=T_cond_iso)
    sat_evap_iso = fp.saturation_props(fluid, T=T_evap)
    P_low_iso = sat_cond_iso["P_sat"]
    P_high_iso = sat_evap_iso["P_sat"]

    # ── Isopentane states (same logic as Config A with different T_cond) ─
    T1 = T_evap + inp["superheat"]
    sp1 = fp.state_point(fluid, "T", T1, "P", P_high_iso)
    s1 = StatePoint(T=sp1["T"], P=sp1["P"], h=sp1["h"], s=sp1["s"],
                    phase=sp1["phase"], rho=sp1["rho"], label="1-Turbine Inlet")

    sp2s = fp.state_point(fluid, "S", s1.s, "P", P_low_iso)
    h2s = sp2s["h"]
    h2 = s1.h - inp["eta_turbine"] * (s1.h - h2s)
    sp2 = fp.state_point(fluid, "H", h2, "P", P_low_iso)
    s2 = StatePoint(T=sp2["T"], P=sp2["P"], h=sp2["h"], s=sp2["s"],
                    phase=sp2["phase"], rho=sp2["rho"], label="2-Turbine Outlet")

    sp4 = fp.state_point(fluid, "T", T_cond_iso, "Q", 0)
    s4 = StatePoint(T=sp4["T"], P=sp4["P"], h=sp4["h"], s=sp4["s"],
                    quality=0, phase="liquid", rho=sp4["rho"],
                    label="4-Condenser Outlet")

    sp5s = fp.state_point(fluid, "S", s4.s, "P", P_high_iso)
    h5s = sp5s["h"]
    h5 = s4.h + (h5s - s4.h) / inp["eta_pump"]
    sp5 = fp.state_point(fluid, "H", h5, "P", P_high_iso)
    s5 = StatePoint(T=sp5["T"], P=sp5["P"], h=sp5["h"], s=sp5["s"],
                    phase=sp5["phase"], rho=sp5["rho"], label="5-Pump Outlet")

    T3 = s5.T + inp["dt_pinch_recup"]
    if T3 >= s2.T:
        T3 = s2.T
    sp3 = fp.state_point(fluid, "T", T3, "P", P_low_iso)
    s3 = StatePoint(T=sp3["T"], P=sp3["P"], h=sp3["h"], s=sp3["s"],
                    phase=sp3["phase"], rho=sp3["rho"],
                    label="3-Recuperator Hot Out")

    q_recup = s2.h - s3.h
    h6 = s5.h + q_recup
    sp6 = fp.state_point(fluid, "H", h6, "P", P_high_iso)
    s6 = StatePoint(T=sp6["T"], P=sp6["P"], h=sp6["h"], s=sp6["s"],
                    phase=sp6["phase"], rho=sp6["rho"],
                    label="6-Recuperator Cold Out")

    iso_states = {"1": s1, "2": s2, "3": s3, "4": s4, "5": s5, "6": s6}

    # ── Mass flow rate (isopentane) ──────────────────────────────────────
    cp_geo = 1.0
    q_evap = s1.h - s6.h
    Q_geo = inp["m_dot_geo"] * cp_geo * (inp["T_geo_in"] - inp["T_geo_out"])
    m_dot_iso = Q_geo / q_evap

    # ── Propane loop ─────────────────────────────────────────────────────
    sat_prop_evap = fp.saturation_props(prop_fluid, T=T_propane_evap)
    sat_prop_cond = fp.saturation_props(prop_fluid, T=T_propane_cond)

    P_prop_evap = sat_prop_evap["P_sat"]
    P_prop_cond = sat_prop_cond["P_sat"]

    # State A: saturated vapor leaving propane evaporator
    spA = fp.state_point(prop_fluid, "T", T_propane_evap, "Q", 1)
    sA = StatePoint(T=spA["T"], P=spA["P"], h=spA["h"], s=spA["s"],
                    quality=1, phase="vapor", rho=spA["rho"],
                    label="A-Propane Sat Vapor")

    # State B: saturated liquid leaving propane condenser (ACC)
    spB = fp.state_point(prop_fluid, "T", T_propane_cond, "Q", 0)
    sB = StatePoint(T=spB["T"], P=spB["P"], h=spB["h"], s=spB["s"],
                    quality=0, phase="liquid", rho=spB["rho"],
                    label="B-Propane Sat Liquid")

    # State C: propane pump outlet
    spCs = fp.state_point(prop_fluid, "S", sB.s, "P", P_prop_evap)
    h_Cs = spCs["h"]
    h_C = sB.h + (h_Cs - sB.h) / inp["eta_pump"]
    spC = fp.state_point(prop_fluid, "H", h_C, "P", P_prop_evap)
    sC = StatePoint(T=spC["T"], P=spC["P"], h=spC["h"], s=spC["s"],
                    phase=spC["phase"], rho=spC["rho"],
                    label="C-Propane Pump Out")

    prop_states = {"A": sA, "B": sB, "C": sC}

    # Propane mass flow (energy balance on intermediate HX)
    q_cond_iso = s3.h - s4.h  # BTU/lb iso — heat rejected by isopentane
    q_evap_prop = sA.h - sC.h  # BTU/lb propane — heat absorbed by propane
    m_dot_prop = (m_dot_iso * q_cond_iso) / q_evap_prop  # lb/hr

    # ── Performance ──────────────────────────────────────────────────────
    w_turbine = s1.h - s2.h
    w_pump_iso = s5.h - s4.h
    w_pump_prop = sC.h - sB.h
    w_pump_total = w_pump_iso + w_pump_prop * (m_dot_prop / m_dot_iso)
    w_net = w_turbine - w_pump_total

    gross_power_kw = (m_dot_iso * w_turbine) / 3412.14
    net_power_kw = (m_dot_iso * w_net) / 3412.14

    eta_thermal = w_net / q_evap if q_evap > 0 else 0

    # Volumetric flow at turbine outlet
    vol_flow_2 = m_dot_iso / s2.rho / 3600  # ft³/s

    # Propane volumetric flow at evaporator exit (State A)
    vol_flow_prop_A = m_dot_prop / sA.rho / 3600  # ft³/s

    performance = {
        "gross_power_kw": gross_power_kw,
        "net_power_kw": net_power_kw,
        "eta_thermal": eta_thermal,
        "m_dot_iso": m_dot_iso,
        "m_dot_prop": m_dot_prop,
        "w_turbine": w_turbine,
        "w_pump_iso": w_pump_iso,
        "w_pump_prop": w_pump_prop,
        "w_pump_total": w_pump_total,
        "w_net": w_net,
        "q_evap": q_evap,
        "q_cond_iso": q_cond_iso,
        "q_recup": q_recup,
        "P_high_iso": P_high_iso,
        "P_low_iso": P_low_iso,
        "T_cond_iso": T_cond_iso,
        "T_evap": T_evap,
        "T_propane_evap": T_propane_evap,
        "T_propane_cond": T_propane_cond,
        "P_prop_evap": P_prop_evap,
        "P_prop_cond": P_prop_cond,
        "vol_flow_turbine_exit": vol_flow_2,
        "vol_flow_prop_evap_exit": vol_flow_prop_A,
        "pressure_ratio_iso": P_high_iso / P_low_iso if P_low_iso > 0 else 0,
        "pressure_ratio_prop": P_prop_evap / P_prop_cond if P_prop_cond > 0 else 0,
    }

    return {
        "states": iso_states,
        "propane_states": prop_states,
        "performance": performance,
    }


def verify_recuperator_pinch(states: dict, fp) -> dict:
    """
    Check that the recuperator has no internal pinch-point violation.

    Divides the hot side (State 2 → State 3) into N intervals and checks
    that T_hot − T_cold >= 0 at every point.
    """
    s2 = states["2"]
    s3 = states["3"]
    s5 = states["5"]
    s6 = states["6"]

    N = 20
    h_hot_vals = np.linspace(s2.h, s3.h, N)
    h_cold_vals = np.linspace(s6.h, s5.h, N)  # reversed — cold side enters at 5, exits at 6

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
        if dt < -0.1:  # small tolerance
            violation = True

    return {
        "violation": violation,
        "min_pinch_dT": min_pinch,
        "profile": pinch_profile,
    }


def calculate_duct_sizing(states: dict, m_dot: float, v_duct: float,
                          fp, config: str = "A") -> dict:
    """
    Calculate duct area and diameter for vapor piping from turbine outlet
    to condenser inlet.

    m_dot: lb/hr
    v_duct: ft/s (design velocity)
    """
    s2 = states["2"]  # turbine outlet — highest vapor volume

    rho = s2.rho  # lb/ft³
    m_dot_fps = m_dot / 3600  # lb/s

    vol_flow = m_dot_fps / rho  # ft³/s
    area = vol_flow / v_duct   # ft²
    diameter = (4 * area / np.pi) ** 0.5  # ft
    diameter_in = diameter * 12  # inches

    return {
        "config": config,
        "vol_flow_ft3s": vol_flow,
        "area_ft2": area,
        "diameter_ft": diameter,
        "diameter_in": diameter_in,
        "rho_lbft3": rho,
        "v_duct_fps": v_duct,
        "m_dot_lbhr": m_dot,
    }
