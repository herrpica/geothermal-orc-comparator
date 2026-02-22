"""
Equipment sizing and cost estimation for ORC Config A and Config B.

All costs in 2024 USD (installed). Uses parametric unit-cost factors.
"""

import numpy as np
from scipy.optimize import minimize_scalar


# ── Unit cost factors (2024 USD, installed) ──────────────────────────────────

COST_FACTORS = {
    # Air-cooled condenser: $/ft² of face area
    "acc_per_ft2": 250,
    # Ductwork: $/ft² of cross-section area (installed, insulated)
    "duct_per_ft2": 800,
    # Shell-and-tube / plate HX (intermediate): $/ft² of heat transfer area
    "hx_per_ft2": 150,
    # Propane circulation pump: $/hp
    "pump_per_hp": 1500,
    # Recuperator: $/ft² heat transfer area
    "recup_per_ft2": 120,
    # Turbine-generator: $/kW (installed)
    "turbine_per_kw": 1200,
    # Evaporator (geo-fluid → isopentane): $/ft² HT area
    "evap_per_ft2": 130,
    # Isopentane pump: $/hp
    "iso_pump_per_hp": 1500,
}

# Heat transfer coefficients for HX sizing (BTU/(hr·ft²·°F))
U_VALUES = {
    "acc": 8,             # air-cooled condenser (air side limited)
    "intermediate_hx": 150,  # propane–isopentane phase change
    "recuperator": 40,    # vapor–liquid recuperator
    "evaporator": 120,    # brine–isopentane
}


def _lmtd(dT1, dT2):
    """Log-mean temperature difference. Handles equal ΔT gracefully."""
    if abs(dT1 - dT2) < 0.01:
        return (dT1 + dT2) / 2
    if dT1 <= 0 or dT2 <= 0:
        return max(dT1, dT2, 0.1)
    return (dT1 - dT2) / np.log(dT1 / dT2)


def _acc_face_area(Q_reject_btu_hr, T_cond, T_ambient, U=None):
    """
    Estimate ACC face area (ft²) from heat duty and approach.
    Q_reject_btu_hr: total heat rejection (BTU/hr)
    """
    U = U or U_VALUES["acc"]
    dT = T_cond - T_ambient
    if dT <= 0:
        dT = 1
    # For ACC, use simple Q = U·A·ΔT (air-side limited, conservative)
    area = Q_reject_btu_hr / (U * dT)
    return area


def _hx_area(Q_btu_hr, lmtd_val, U):
    """Heat exchanger area (ft²) from duty, LMTD, and U."""
    if lmtd_val <= 0:
        lmtd_val = 0.1
    return Q_btu_hr / (U * lmtd_val)


def calculate_costs_a(states, performance, inputs) -> dict:
    """
    Component-by-component installed cost for Config A.

    Returns dict with each component cost and total.
    """
    inp = {**_default_inputs(), **inputs}
    perf = performance
    m_dot = perf["m_dot_iso"]  # lb/hr

    costs = {}

    # ── Turbine-generator ────────────────────────────────────────────────
    costs["turbine_generator"] = perf["gross_power_kw"] * COST_FACTORS["turbine_per_kw"]

    # ── Isopentane pump ──────────────────────────────────────────────────
    pump_power_btu_hr = m_dot * perf["w_pump"]
    pump_hp = pump_power_btu_hr / 2544.43  # BTU/hr → hp
    costs["iso_pump"] = pump_hp * COST_FACTORS["iso_pump_per_hp"]

    # ── Evaporator ───────────────────────────────────────────────────────
    Q_evap = m_dot * perf["q_evap"]  # BTU/hr
    dT1 = inp["T_geo_in"] - states["1"].T
    dT2 = inp["T_geo_out"] - states["6"].T
    lmtd_evap = _lmtd(dT1, dT2)
    evap_area = _hx_area(Q_evap, lmtd_evap, U_VALUES["evaporator"])
    costs["evaporator"] = evap_area * COST_FACTORS["evap_per_ft2"]
    costs["evaporator_area_ft2"] = evap_area

    # ── Recuperator ──────────────────────────────────────────────────────
    Q_recup = m_dot * perf["q_recup"]  # BTU/hr
    if Q_recup > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area = _hx_area(Q_recup, lmtd_recup, U_VALUES["recuperator"])
        costs["recuperator"] = recup_area * COST_FACTORS["recup_per_ft2"]
        costs["recuperator_area_ft2"] = recup_area
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # ── ACC (direct condensing) ──────────────────────────────────────────
    Q_cond = m_dot * perf["q_cond"]  # BTU/hr
    acc_area = _acc_face_area(Q_cond, perf["T_cond"], inp["T_ambient"])
    costs["acc"] = acc_area * COST_FACTORS["acc_per_ft2"]
    costs["acc_area_ft2"] = acc_area

    # ── Ductwork (turbine outlet to ACC) ─────────────────────────────────
    from thermodynamics import calculate_duct_sizing
    duct = calculate_duct_sizing(states, m_dot, inp.get("v_duct", 60), None, "A")
    duct_area = duct["area_ft2"]
    # Cost per linear foot × assumed 100 ft equivalent length
    duct_length = 100  # ft assumed
    costs["ductwork"] = duct_area * duct_length * COST_FACTORS["duct_per_ft2"] / 10
    costs["duct_diameter_in"] = duct["diameter_in"]
    costs["duct_area_ft2"] = duct_area

    # ── No intermediate HX or propane loop ───────────────────────────────
    costs["intermediate_hx"] = 0
    costs["propane_pump"] = 0
    costs["propane_loop_piping"] = 0

    # ── Total ────────────────────────────────────────────────────────────
    component_keys = [
        "turbine_generator", "iso_pump", "evaporator", "recuperator",
        "acc", "ductwork", "intermediate_hx", "propane_pump",
        "propane_loop_piping",
    ]
    costs["total_installed"] = sum(costs[k] for k in component_keys)

    return costs


def calculate_costs_b(states, propane_states, performance, inputs) -> dict:
    """
    Component-by-component installed cost for Config B.
    """
    inp = {**_default_inputs(), **inputs}
    perf = performance
    m_dot_iso = perf["m_dot_iso"]
    m_dot_prop = perf["m_dot_prop"]

    costs = {}

    # ── Turbine-generator ────────────────────────────────────────────────
    costs["turbine_generator"] = perf["gross_power_kw"] * COST_FACTORS["turbine_per_kw"]

    # ── Isopentane pump ──────────────────────────────────────────────────
    pump_power_iso = m_dot_iso * perf["w_pump_iso"]
    pump_hp_iso = pump_power_iso / 2544.43
    costs["iso_pump"] = pump_hp_iso * COST_FACTORS["iso_pump_per_hp"]

    # ── Evaporator ───────────────────────────────────────────────────────
    Q_evap = m_dot_iso * perf["q_evap"]
    dT1 = inp["T_geo_in"] - states["1"].T
    dT2 = inp["T_geo_out"] - states["6"].T
    lmtd_evap = _lmtd(dT1, dT2)
    evap_area = _hx_area(Q_evap, lmtd_evap, U_VALUES["evaporator"])
    costs["evaporator"] = evap_area * COST_FACTORS["evap_per_ft2"]
    costs["evaporator_area_ft2"] = evap_area

    # ── Recuperator ──────────────────────────────────────────────────────
    Q_recup = m_dot_iso * perf["q_recup"]
    if Q_recup > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area = _hx_area(Q_recup, lmtd_recup, U_VALUES["recuperator"])
        costs["recuperator"] = recup_area * COST_FACTORS["recup_per_ft2"]
        costs["recuperator_area_ft2"] = recup_area
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # ── Intermediate HX (iso condenser / propane evaporator) ─────────────
    Q_intermediate = m_dot_iso * perf["q_cond_iso"]
    # Both sides phase-changing → LMTD ≈ approach ΔT
    lmtd_int = inp.get("dt_approach_intermediate", 10)
    int_area = _hx_area(Q_intermediate, lmtd_int, U_VALUES["intermediate_hx"])
    costs["intermediate_hx"] = int_area * COST_FACTORS["hx_per_ft2"]
    costs["intermediate_hx_area_ft2"] = int_area

    # ── Propane ACC ──────────────────────────────────────────────────────
    Q_prop_cond = m_dot_prop * (propane_states["A"].h - propane_states["B"].h)
    T_propane_cond = perf["T_propane_cond"]
    acc_area = _acc_face_area(Q_prop_cond, T_propane_cond, inp["T_ambient"])
    costs["acc"] = acc_area * COST_FACTORS["acc_per_ft2"]
    costs["acc_area_ft2"] = acc_area

    # ── Propane pump ─────────────────────────────────────────────────────
    pump_power_prop = m_dot_prop * perf["w_pump_prop"]
    pump_hp_prop = pump_power_prop / 2544.43
    costs["propane_pump"] = pump_hp_prop * COST_FACTORS["pump_per_hp"]

    # ── Ductwork (propane vapor to ACC — much smaller) ───────────────────
    from thermodynamics import calculate_duct_sizing
    # For Config B, the ACC duct carries propane vapor
    prop_duct_states = {
        "2": propane_states["A"],  # use propane vapor state for duct sizing
    }
    duct = calculate_duct_sizing(prop_duct_states, m_dot_prop,
                                 inp.get("v_duct", 60), None, "B")
    duct_area = duct["area_ft2"]
    duct_length = 100
    costs["ductwork"] = duct_area * duct_length * COST_FACTORS["duct_per_ft2"] / 10
    costs["duct_diameter_in"] = duct["diameter_in"]
    costs["duct_area_ft2"] = duct_area

    # ── Propane loop piping (liquid side — small) ────────────────────────
    costs["propane_loop_piping"] = 50_000  # lump sum estimate

    # ── Total ────────────────────────────────────────────────────────────
    component_keys = [
        "turbine_generator", "iso_pump", "evaporator", "recuperator",
        "acc", "ductwork", "intermediate_hx", "propane_pump",
        "propane_loop_piping",
    ]
    costs["total_installed"] = sum(costs[k] for k in component_keys)

    return costs


def lifecycle_cost(installed_cost, net_power_kw, inputs) -> dict:
    """
    Simple NPV / lifecycle cost calculation.

    Returns dict with annual_revenue, npv_revenue, net_npv, lcoe.
    """
    inp = {**_default_inputs(), **inputs}
    price = inp["electricity_price"]
    r = inp["discount_rate"]
    n = inp["project_life"]
    cf = inp["capacity_factor"]

    annual_energy_kwh = net_power_kw * 8760 * cf
    annual_revenue = annual_energy_kwh * price

    # NPV of revenue stream (annuity)
    if r > 0:
        annuity_factor = (1 - (1 + r) ** (-n)) / r
    else:
        annuity_factor = n

    npv_revenue = annual_revenue * annuity_factor
    net_npv = npv_revenue - installed_cost

    lcoe = installed_cost / (annual_energy_kwh * annuity_factor) if annual_energy_kwh > 0 else float("inf")

    return {
        "installed_cost": installed_cost,
        "annual_energy_kwh": annual_energy_kwh,
        "annual_revenue": annual_revenue,
        "npv_revenue": npv_revenue,
        "net_npv": net_npv,
        "lcoe": lcoe,
        "annuity_factor": annuity_factor,
    }


def optimize_approach_temp(inputs, fp):
    """
    Find optimal intermediate HX approach ΔT (5–25°F) that minimises lifecycle cost
    for Config B.

    Returns dict with optimal_dt, min_lifecycle_cost, sweep data.
    """
    from thermodynamics import solve_config_b

    sweep_dts = np.linspace(5, 25, 21)
    sweep_results = []

    def objective(dt):
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result = solve_config_b(inp, fp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp,
            )
            lc = lifecycle_cost(costs["total_installed"],
                                result["performance"]["net_power_kw"], inp)
            return -lc["net_npv"]  # minimise negative NPV = maximise NPV
        except Exception:
            return 1e12

    # Sweep for plotting
    for dt in sweep_dts:
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result = solve_config_b(inp, fp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp,
            )
            lc = lifecycle_cost(costs["total_installed"],
                                result["performance"]["net_power_kw"], inp)
            sweep_results.append({
                "dt_approach": float(dt),
                "net_power_kw": result["performance"]["net_power_kw"],
                "installed_cost": costs["total_installed"],
                "net_npv": lc["net_npv"],
                "lcoe": lc["lcoe"],
            })
        except Exception:
            sweep_results.append({
                "dt_approach": float(dt),
                "net_power_kw": 0,
                "installed_cost": 0,
                "net_npv": -1e12,
                "lcoe": float("inf"),
            })

    # Optimise
    opt = minimize_scalar(objective, bounds=(5, 25), method="bounded")
    optimal_dt = opt.x

    return {
        "optimal_dt": optimal_dt,
        "sweep": sweep_results,
    }


def _default_inputs():
    """Mirror of thermodynamics defaults."""
    return {
        "T_geo_in": 300,
        "T_geo_out": 160,
        "m_dot_geo": 500_000,
        "T_ambient": 95,
        "dt_pinch_evap": 10,
        "dt_pinch_acc": 25,
        "dt_pinch_recup": 15,
        "dt_approach_intermediate": 10,
        "eta_turbine": 0.82,
        "eta_pump": 0.75,
        "superheat": 5,
        "v_duct": 60,
        "electricity_price": 0.08,
        "discount_rate": 0.08,
        "project_life": 30,
        "capacity_factor": 0.95,
    }
