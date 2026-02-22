"""
Equipment sizing and cost estimation for ORC Config A and Config B.
All costs in 2024 USD (installed). Uses parametric unit-cost factors.
"""

import numpy as np
from scipy.optimize import minimize_scalar


# -- Unit cost factors (2024 USD, installed) -----------------------------------

COST_FACTORS = {
    "acc_per_ft2": 250,
    "duct_per_ft_per_in": 15,       # $/ft of duct per inch of diameter
    "hx_per_ft2": 150,              # intermediate HX
    "pump_per_hp": 1500,
    "recup_per_ft2": 120,
    "turbine_per_kw": 1200,
    "vaporizer_per_ft2": 130,
    "preheater_per_ft2": 100,
    "iso_pump_per_hp": 1500,
    "propane_loop_piping_lump": 50_000,
}

U_VALUES = {
    "acc": 8,
    "intermediate_hx": 150,
    "recuperator": 40,
    "vaporizer": 150,
    "preheater": 80,
}


def _default_inputs():
    return {
        "T_geo_in": 300,
        "m_dot_geo": 200,
        "cp_brine": 1.0,
        "T_geo_out_min": 160,
        "T_ambient": 95,
        "dt_pinch_vaporizer": 10,
        "dt_pinch_preheater": 10,
        "dt_pinch_acc": 15,
        "dt_pinch_recup": 15,
        "dt_approach_intermediate": 10,
        "eta_turbine": 0.82,
        "eta_pump": 0.75,
        "superheat": 0,
        "v_tailpipe": 10,
        "v_acc_header": 15,
        "L_tailpipe_a": 30,
        "L_long_header": 120,
        "L_acc_header": 200,
        "L_iso_to_ihx": 40,
        "electricity_price": 35,
        "discount_rate": 0.08,
        "project_life": 30,
        "capacity_factor": 0.95,
    }


def _lmtd(dT1, dT2):
    if abs(dT1 - dT2) < 0.01:
        return (dT1 + dT2) / 2
    if dT1 <= 0 or dT2 <= 0:
        return max(dT1, dT2, 0.1)
    return (dT1 - dT2) / np.log(dT1 / dT2)


def _acc_face_area(Q_reject_btu_hr, T_cond, T_ambient, U=None):
    U = U or U_VALUES["acc"]
    dT = T_cond - T_ambient
    if dT <= 0:
        dT = 1
    return Q_reject_btu_hr / (U * dT)


def _hx_area(Q_btu_hr, lmtd_val, U):
    if lmtd_val <= 0:
        lmtd_val = 0.1
    return Q_btu_hr / (U * lmtd_val)


def _duct_segment_cost(segment):
    """Cost a single duct segment: $/ft * length * diameter factor."""
    dia_in = segment["diameter_in"]
    length = segment["length_ft"]
    return COST_FACTORS["duct_per_ft_per_in"] * dia_in * length


def calculate_costs_a(states, performance, inputs, duct_result=None) -> dict:
    """Component-by-component installed cost for Config A."""
    inp = {**_default_inputs(), **inputs}
    perf = performance
    m_dot = perf["m_dot_iso"]

    costs = {}

    # Turbine-generator
    costs["turbine_generator"] = perf["gross_power_kw"] * COST_FACTORS["turbine_per_kw"]

    # Isopentane pump
    pump_power_btu_hr = m_dot * perf["w_pump"]
    pump_hp = pump_power_btu_hr / 2544.43
    costs["iso_pump"] = pump_hp * COST_FACTORS["iso_pump_per_hp"]

    # Vaporizer (State 7 -> 1)
    Q_vap = m_dot * perf["q_vaporizer"]
    dT1_vap = inp["T_geo_in"] - states["1"].T
    dT2_vap = perf["T_brine_mid"] - states["7"].T
    lmtd_vap = _lmtd(dT1_vap, dT2_vap)
    vap_area = _hx_area(Q_vap, lmtd_vap, U_VALUES["vaporizer"])
    costs["vaporizer"] = vap_area * COST_FACTORS["vaporizer_per_ft2"]
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater (State 6 -> 7)
    Q_pre = m_dot * perf["q_preheater"]
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    costs["preheater"] = pre_area * COST_FACTORS["preheater_per_ft2"]
    costs["preheater_area_ft2"] = pre_area

    # Recuperator
    Q_recup = m_dot * perf["q_recup"]
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

    # ACC
    Q_cond = m_dot * perf["q_cond"]
    acc_area = _acc_face_area(Q_cond, perf["T_cond"], inp["T_ambient"])
    costs["acc"] = acc_area * COST_FACTORS["acc_per_ft2"]
    costs["acc_area_ft2"] = acc_area

    # Ductwork (segment-level)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    if duct_result:
        for seg in duct_result["segments"]:
            seg_cost = _duct_segment_cost(seg)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost

    # No Config B components
    costs["intermediate_hx"] = 0
    costs["propane_pump"] = 0
    costs["propane_loop_piping"] = 0

    component_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "intermediate_hx",
        "propane_pump", "propane_loop_piping",
    ]
    costs["total_installed"] = sum(costs[k] for k in component_keys)

    # Equipment count
    costs["equipment_count"] = 6  # turbine, pump, vaporizer, preheater, recup, ACC

    return costs


def calculate_costs_b(states, propane_states, performance, inputs, duct_result=None) -> dict:
    """Component-by-component installed cost for Config B."""
    inp = {**_default_inputs(), **inputs}
    perf = performance
    m_dot_iso = perf["m_dot_iso"]
    m_dot_prop = perf["m_dot_prop"]

    costs = {}

    # Turbine-generator
    costs["turbine_generator"] = perf["gross_power_kw"] * COST_FACTORS["turbine_per_kw"]

    # Isopentane pump
    pump_power_iso = m_dot_iso * perf["w_pump_iso"]
    pump_hp_iso = pump_power_iso / 2544.43
    costs["iso_pump"] = pump_hp_iso * COST_FACTORS["iso_pump_per_hp"]

    # Vaporizer
    Q_vap = m_dot_iso * perf["q_vaporizer"]
    dT1_vap = inp["T_geo_in"] - states["1"].T
    dT2_vap = perf["T_brine_mid"] - states["7"].T
    lmtd_vap = _lmtd(dT1_vap, dT2_vap)
    vap_area = _hx_area(Q_vap, lmtd_vap, U_VALUES["vaporizer"])
    costs["vaporizer"] = vap_area * COST_FACTORS["vaporizer_per_ft2"]
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater
    Q_pre = m_dot_iso * perf["q_preheater"]
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    costs["preheater"] = pre_area * COST_FACTORS["preheater_per_ft2"]
    costs["preheater_area_ft2"] = pre_area

    # Recuperator
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

    # Intermediate HX
    Q_intermediate = m_dot_iso * perf["q_cond_iso"]
    lmtd_int = inp.get("dt_approach_intermediate", 10)
    int_area = _hx_area(Q_intermediate, lmtd_int, U_VALUES["intermediate_hx"])
    costs["intermediate_hx"] = int_area * COST_FACTORS["hx_per_ft2"]
    costs["intermediate_hx_area_ft2"] = int_area

    # Propane ACC
    Q_prop_cond = m_dot_prop * (propane_states["A"].h - propane_states["B"].h)
    T_propane_cond = perf["T_propane_cond"]
    acc_area = _acc_face_area(Q_prop_cond, T_propane_cond, inp["T_ambient"])
    costs["acc"] = acc_area * COST_FACTORS["acc_per_ft2"]
    costs["acc_area_ft2"] = acc_area

    # Propane pump
    pump_power_prop = m_dot_prop * perf["w_pump_prop"]
    pump_hp_prop = pump_power_prop / 2544.43
    costs["propane_pump"] = pump_hp_prop * COST_FACTORS["pump_per_hp"]

    # Ductwork (segment-level)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    if duct_result:
        for seg in duct_result["segments"]:
            seg_cost = _duct_segment_cost(seg)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost

    # Propane loop piping
    costs["propane_loop_piping"] = COST_FACTORS["propane_loop_piping_lump"]

    component_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "intermediate_hx",
        "propane_pump", "propane_loop_piping",
    ]
    costs["total_installed"] = sum(costs[k] for k in component_keys)

    # Equipment count
    costs["equipment_count"] = 9  # turbine, iso pump, vap, pre, recup, ACC, IHX, prop pump, prop piping

    return costs


def construction_schedule_delta(duct_a):
    """
    Config B construction schedule delta relative to Config A (weeks).
    +3 weeks for IHX and propane system.
    -4 weeks if tailpipe > 60 inches (saves large duct fab/erection).
    """
    tp_dia = duct_a.get("tailpipe_diameter_in", 0)
    delta = 3  # IHX and propane system
    if tp_dia > 60:
        delta -= 4  # saves on large duct
    return delta


def lifecycle_cost(installed_cost, net_power_kw, inputs) -> dict:
    """NPV / lifecycle cost calculation. Electricity price in $/MWh."""
    inp = {**_default_inputs(), **inputs}
    price_per_mwh = inp["electricity_price"]  # $/MWh
    r = inp["discount_rate"]
    n = inp["project_life"]
    cf = inp["capacity_factor"]

    annual_energy_mwh = net_power_kw / 1000 * 8760 * cf
    annual_revenue = annual_energy_mwh * price_per_mwh

    if r > 0:
        annuity_factor = (1 - (1 + r) ** (-n)) / r
    else:
        annuity_factor = n

    npv_revenue = annual_revenue * annuity_factor
    net_npv = npv_revenue - installed_cost

    lcoe = installed_cost / (annual_energy_mwh * annuity_factor) if annual_energy_mwh > 0 else float("inf")

    specific_cost = installed_cost / net_power_kw if net_power_kw > 0 else float("inf")

    return {
        "installed_cost": installed_cost,
        "annual_energy_mwh": annual_energy_mwh,
        "annual_revenue": annual_revenue,
        "npv_revenue": npv_revenue,
        "net_npv": net_npv,
        "lcoe": lcoe,
        "annuity_factor": annuity_factor,
        "specific_cost_per_kw": specific_cost,
    }


def simple_payback(cost_a, cost_b, net_power_a_kw, net_power_b_kw, inputs):
    """Simple payback on cost delta. Returns years or None if Config B is cheaper."""
    inp = {**_default_inputs(), **inputs}
    cost_delta = cost_b - cost_a
    if cost_delta <= 0:
        return None  # Config B is cheaper

    power_delta_kw = net_power_b_kw - net_power_a_kw
    if power_delta_kw <= 0:
        # Config B costs more AND produces less power
        return float("inf")

    price_per_mwh = inp["electricity_price"]
    cf = inp["capacity_factor"]
    annual_saving = (power_delta_kw / 1000) * 8760 * cf * price_per_mwh
    if annual_saving <= 0:
        return float("inf")
    return cost_delta / annual_saving


def optimize_approach_temp(inputs, fp):
    """
    Find optimal intermediate HX approach dT (5-25 degF) for Config B.
    Returns sweep data including costs, LCOE, IHX cost.
    """
    from thermodynamics import solve_config_b, solve_config_a

    # Solve Config A once for reference
    try:
        result_a = solve_config_a(inputs, fp)
        costs_a = calculate_costs_a(
            result_a["states"], result_a["performance"], inputs,
            result_a.get("duct"),
        )
        lc_a = lifecycle_cost(costs_a["total_installed"],
                              result_a["performance"]["net_power_kw"], inputs)
        ref_power_a = result_a["performance"]["net_power_kw"]
        ref_cost_a = costs_a["total_installed"]
        ref_lcoe_a = lc_a["lcoe"]
    except Exception:
        ref_power_a = 0
        ref_cost_a = 0
        ref_lcoe_a = 0

    sweep_dts = np.linspace(5, 25, 21)
    sweep_results = []

    def objective(dt):
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result = solve_config_b(inp, fp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp, result.get("duct"),
            )
            lc = lifecycle_cost(costs["total_installed"],
                                result["performance"]["net_power_kw"], inp)
            return -lc["net_npv"]
        except Exception:
            return 1e12

    for dt in sweep_dts:
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result = solve_config_b(inp, fp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp, result.get("duct"),
            )
            lc = lifecycle_cost(costs["total_installed"],
                                result["performance"]["net_power_kw"], inp)
            sweep_results.append({
                "dt_approach": float(dt),
                "net_power_kw": result["performance"]["net_power_kw"],
                "installed_cost": costs["total_installed"],
                "intermediate_hx_cost": costs["intermediate_hx"],
                "net_npv": lc["net_npv"],
                "lcoe": lc["lcoe"],
            })
        except Exception:
            sweep_results.append({
                "dt_approach": float(dt),
                "net_power_kw": 0,
                "installed_cost": 0,
                "intermediate_hx_cost": 0,
                "net_npv": -1e12,
                "lcoe": float("inf"),
            })

    opt = minimize_scalar(objective, bounds=(5, 25), method="bounded")
    optimal_dt = opt.x

    return {
        "optimal_dt": optimal_dt,
        "sweep": sweep_results,
        "ref_power_a": ref_power_a,
        "ref_cost_a": ref_cost_a,
        "ref_lcoe_a": ref_lcoe_a,
    }
