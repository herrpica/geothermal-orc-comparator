"""
Equipment sizing and cost estimation for ORC Config A and Config B.
All costs in 2024 USD (installed). Uses parametric unit-cost factors
that can be overridden via the inputs dict (uc_* keys).
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq


# -- Unit cost factor defaults (2024 USD, installed) --------------------------
# These serve as reference defaults; user overrides via uc_* keys in inputs.

COST_FACTORS = {
    "vaporizer_per_ft2": 35,
    "preheater_per_ft2": 30,
    "recup_per_ft2": 25,
    "hx_per_ft2": 40,              # intermediate HX (pressure rated)
    "acc_per_ft2": 12,
    "turbine_per_kw": 1200,
    "iso_pump_per_hp": 1500,
    "pump_per_hp": 1500,
    "iso_duct_per_ft2": 180,       # $/ft2 of duct surface area
    "prop_pipe_per_ft2": 120,      # $/ft2 of propane pipe surface area
    "prop_piping_pct": 20,         # % of IHX cost for propane piping+pump
    "steel_per_lb": 4.50,
    "foundation_pct": 8,
    "engineering_pct": 12,
    "construction_mgmt_pct": 8,
    "contingency_pct": 15,
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
        "dt_pinch_acc_a": 15,
        "dt_pinch_acc_b": 15,
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
        # Hydraulic parameters (mirrored from thermodynamics.py)
        "f_darcy": 0.02,
        "dp_acc_tubes_a": 0.5,
        "dp_acc_headers_a": 0.3,
        "dp_recup_a": 0.3,
        "dp_ihx_iso": 0.5,
        "dp_recup_b": 0.3,
        "dp_tailpipe_iso_b": 0.3,
        "dp_acc_tubes_prop": 1.0,
        "dp_prop_headers": 0.5,
        "dp_ihx_prop": 0.5,
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


def _duct_segment_cost(segment, inp=None):
    """Cost a single duct segment using surface-area basis ($/ft2).

    surface_area = pi * (D_in / 12) * L_ft  (ft2)
    Diameter multiplier: >72" -> x1.7, >60" -> x1.4, else x1.0
    Propane segments use uc_prop_pipe_per_ft2 (no diameter multiplier).
    """
    dia_in = segment["diameter_in"]
    length = segment["length_ft"]
    fluid = segment.get("fluid", "iso")

    if inp is None:
        inp = {}

    surface_area = math.pi * (dia_in / 12) * length  # ft2

    if fluid == "propane":
        uc = inp.get("uc_prop_pipe_per_ft2", COST_FACTORS["prop_pipe_per_ft2"])
        return uc * surface_area
    else:
        uc = inp.get("uc_iso_duct_per_ft2", COST_FACTORS["iso_duct_per_ft2"])
        if dia_in > 72:
            dia_mult = 1.7
        elif dia_in > 60:
            dia_mult = 1.4
        else:
            dia_mult = 1.0
        return uc * surface_area * dia_mult


def _structural_steel_cost(segments, inp=None):
    """Structural steel cost for duct support structures.

    For each segment:
      support_spacing = 20 ft
      n_supports = max(1, ceil(L_ft / 20))
      support_weight_lb = 200 + 4 * D_in  (saddle + columns + bracing)
      beam_weight = 8.0 * L_ft  (lb, connecting beams)
      total_weight = n_supports * support_weight + beam_weight
      cost = total_weight * uc_steel_per_lb
    """
    if inp is None:
        inp = {}

    uc = inp.get("uc_steel_per_lb", COST_FACTORS["steel_per_lb"])
    total_cost = 0
    total_weight = 0

    for seg in segments:
        dia_in = seg["diameter_in"]
        length = seg["length_ft"]
        n_supports = max(1, math.ceil(length / 20))
        support_weight = 200 + 4 * dia_in
        beam_weight = 8.0 * length
        seg_weight = n_supports * support_weight + beam_weight
        total_weight += seg_weight
        total_cost += seg_weight * uc

    return total_cost, total_weight


def _apply_indirect_costs(costs, inp):
    """Apply foundation, E&P, CM, and contingency layers to costs dict.

    Expects costs dict to already have 'equipment_subtotal'.
    Mutates costs in-place and returns it.
    """
    equip_sub = costs["equipment_subtotal"]

    # Foundation
    fdn_pct = inp.get("uc_foundation_pct", COST_FACTORS["foundation_pct"])
    costs["foundation"] = equip_sub * fdn_pct / 100

    subtotal_with_fdn = equip_sub + costs["foundation"]
    costs["subtotal_with_foundation"] = subtotal_with_fdn

    # Engineering & procurement
    eng_pct = inp.get("uc_engineering_pct", COST_FACTORS["engineering_pct"])
    costs["engineering"] = subtotal_with_fdn * eng_pct / 100

    # Construction management
    cm_pct = inp.get("uc_construction_mgmt_pct", COST_FACTORS["construction_mgmt_pct"])
    costs["construction_mgmt"] = subtotal_with_fdn * cm_pct / 100

    total_before_cont = subtotal_with_fdn + costs["engineering"] + costs["construction_mgmt"]
    costs["total_before_contingency"] = total_before_cont

    # Contingency
    cont_pct = inp.get("uc_contingency_pct", COST_FACTORS["contingency_pct"])
    costs["contingency"] = total_before_cont * cont_pct / 100

    costs["total_installed"] = total_before_cont + costs["contingency"]
    return costs


def calculate_costs_a(states, performance, inputs, duct_result=None) -> dict:
    """Component-by-component installed cost for Config A."""
    inp = {**_default_inputs(), **inputs}
    perf = performance
    m_dot = perf["m_dot_iso"]

    costs = {}

    # Turbine-generator
    uc_turb = inp.get("uc_turbine_per_kw", COST_FACTORS["turbine_per_kw"])
    costs["turbine_generator"] = perf["gross_power_kw"] * uc_turb

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
    uc_vap = inp.get("uc_vaporizer_per_ft2", COST_FACTORS["vaporizer_per_ft2"])
    costs["vaporizer"] = vap_area * uc_vap
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater (State 6 -> 7)
    Q_pre = m_dot * perf["q_preheater"]
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    uc_pre = inp.get("uc_preheater_per_ft2", COST_FACTORS["preheater_per_ft2"])
    costs["preheater"] = pre_area * uc_pre
    costs["preheater_area_ft2"] = pre_area

    # Recuperator
    Q_recup = m_dot * perf["q_recup"]
    if Q_recup > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area = _hx_area(Q_recup, lmtd_recup, U_VALUES["recuperator"])
        uc_rec = inp.get("uc_recuperator_per_ft2", COST_FACTORS["recup_per_ft2"])
        costs["recuperator"] = recup_area * uc_rec
        costs["recuperator_area_ft2"] = recup_area
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # ACC
    Q_cond = m_dot * perf["q_cond"]
    acc_area = _acc_face_area(Q_cond, perf["T_cond"], inp["T_ambient"])
    uc_acc = inp.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"])
    costs["acc"] = acc_area * uc_acc
    costs["acc_area_ft2"] = acc_area

    # Ductwork (segment-level, surface area basis)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    segments = []
    if duct_result:
        segments = duct_result["segments"]
        for seg in segments:
            seg_cost = _duct_segment_cost(seg, inp)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost

    # Structural steel
    steel_cost, steel_weight = _structural_steel_cost(segments, inp)
    costs["structural_steel"] = steel_cost
    costs["structural_steel_weight_lb"] = steel_weight

    # No Config B components
    costs["intermediate_hx"] = 0
    costs["intermediate_hx_area_ft2"] = 0
    costs["propane_system"] = 0

    # Equipment subtotal
    equipment_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "structural_steel",
        "intermediate_hx", "propane_system",
    ]
    costs["equipment_subtotal"] = sum(costs[k] for k in equipment_keys)

    # Apply indirect cost layers
    _apply_indirect_costs(costs, inp)

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
    uc_turb = inp.get("uc_turbine_per_kw", COST_FACTORS["turbine_per_kw"])
    costs["turbine_generator"] = perf["gross_power_kw"] * uc_turb

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
    uc_vap = inp.get("uc_vaporizer_per_ft2", COST_FACTORS["vaporizer_per_ft2"])
    costs["vaporizer"] = vap_area * uc_vap
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater
    Q_pre = m_dot_iso * perf["q_preheater"]
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    uc_pre = inp.get("uc_preheater_per_ft2", COST_FACTORS["preheater_per_ft2"])
    costs["preheater"] = pre_area * uc_pre
    costs["preheater_area_ft2"] = pre_area

    # Recuperator
    Q_recup = m_dot_iso * perf["q_recup"]
    if Q_recup > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area = _hx_area(Q_recup, lmtd_recup, U_VALUES["recuperator"])
        uc_rec = inp.get("uc_recuperator_per_ft2", COST_FACTORS["recup_per_ft2"])
        costs["recuperator"] = recup_area * uc_rec
        costs["recuperator_area_ft2"] = recup_area
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # Intermediate HX
    Q_intermediate = m_dot_iso * perf["q_cond_iso"]
    lmtd_int = inp.get("dt_approach_intermediate", 10)
    int_area = _hx_area(Q_intermediate, lmtd_int, U_VALUES["intermediate_hx"])
    uc_ihx = inp.get("uc_ihx_per_ft2", COST_FACTORS["hx_per_ft2"])
    costs["intermediate_hx"] = int_area * uc_ihx
    costs["intermediate_hx_area_ft2"] = int_area

    # Propane ACC
    Q_prop_cond = m_dot_prop * (propane_states["A"].h - propane_states["B"].h)
    T_propane_cond = perf["T_propane_cond"]
    acc_area = _acc_face_area(Q_prop_cond, T_propane_cond, inp["T_ambient"])
    uc_acc = inp.get("uc_acc_per_ft2", COST_FACTORS["acc_per_ft2"])
    costs["acc"] = acc_area * uc_acc
    costs["acc_area_ft2"] = acc_area

    # Propane system (piping + pump) as % of IHX cost
    prop_pct = inp.get("uc_prop_piping_pct", COST_FACTORS["prop_piping_pct"])
    costs["propane_system"] = costs["intermediate_hx"] * prop_pct / 100

    # Ductwork (segment-level, surface area basis)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    segments = []
    if duct_result:
        segments = duct_result["segments"]
        for seg in segments:
            seg_cost = _duct_segment_cost(seg, inp)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost

    # Structural steel
    steel_cost, steel_weight = _structural_steel_cost(segments, inp)
    costs["structural_steel"] = steel_cost
    costs["structural_steel_weight_lb"] = steel_weight

    # Equipment subtotal
    equipment_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "structural_steel",
        "intermediate_hx", "propane_system",
    ]
    costs["equipment_subtotal"] = sum(costs[k] for k in equipment_keys)

    # Apply indirect cost layers
    _apply_indirect_costs(costs, inp)

    # Equipment count
    costs["equipment_count"] = 9  # turbine, iso pump, vap, pre, recup, ACC, IHX, prop pump, prop piping

    return costs


def construction_schedule_delta(duct_a):
    """
    Config B construction schedule delta relative to Config A (weeks).
    Returns dict with savings breakdown, adders, and net delta.
    Negative net = Config B is faster.
    """
    tp_dia = duct_a.get("tailpipe_diameter_in", 0)

    # Duct fabrication and erection savings
    if tp_dia > 72:
        duct_fab_savings = 6
    elif tp_dia > 60:
        duct_fab_savings = 4
    elif tp_dia > 48:
        duct_fab_savings = 3
    else:
        duct_fab_savings = 1

    # Welding vs flanged connection savings
    if tp_dia > 60:
        weld_savings = 3
    elif tp_dia > 48:
        weld_savings = 2
    else:
        weld_savings = 1

    # Structural steel savings
    if tp_dia > 60:
        steel_savings = 2
    else:
        steel_savings = 1

    total_savings = duct_fab_savings + weld_savings + steel_savings

    # Fixed adders for Config B
    ihx_install = 2
    propane_pressure_test = 2
    propane_safety_commissioning = 1
    total_adder = ihx_install + propane_pressure_test + propane_safety_commissioning

    net_delta = total_adder - total_savings

    return {
        "duct_fab_savings": duct_fab_savings,
        "weld_savings": weld_savings,
        "steel_savings": steel_savings,
        "total_savings": total_savings,
        "ihx_install": ihx_install,
        "propane_pressure_test": propane_pressure_test,
        "propane_safety_commissioning": propane_safety_commissioning,
        "total_adder": total_adder,
        "net_delta": net_delta,
        "tailpipe_diameter_in": tp_dia,
    }


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


def schedule_savings_npv(sched_info, net_power_kw, inputs):
    """
    NPV of early startup revenue from schedule savings.
    If Config B is faster by N weeks, the plant earns revenue N weeks earlier.
    Discount that revenue back to construction midpoint.
    """
    inp = {**_default_inputs(), **inputs}
    weeks_saved = -sched_info["net_delta"]  # positive = B is faster
    if weeks_saved <= 0:
        return 0.0

    price_per_mwh = inp["electricity_price"]
    cf = inp["capacity_factor"]
    r = inp["discount_rate"]
    n = inp["project_life"]

    # Revenue per week
    annual_energy_mwh = net_power_kw / 1000 * 8760 * cf
    weekly_revenue = annual_energy_mwh * price_per_mwh / 52

    # NPV of the extra weeks of revenue at the END of plant life
    # (plant starts earlier, so it generates revenue for weeks_saved extra weeks)
    # Discount from end of plant life back to start
    if r > 0:
        discount_factor = 1 / (1 + r) ** n
    else:
        discount_factor = 1.0

    npv = weeks_saved * weekly_revenue * discount_factor
    return npv


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


# ============================================================================
# EQUIPMENT SIZING TRADE-OFF CALCULATIONS
# ============================================================================

# ACC tube bundle model defaults
ACC_TUBE_DEFAULTS = {
    "f_tube": 0.025,
    "L_tube_ft": 20.0,
    "D_tube_in": 1.0,
    "N_rows": 6,
    "cost_per_ft2_row": 12.0,   # $/ft2 per row of face area
}


def hx_area_vs_pinch(Q_btu_hr, U, T_hot_in, T_hot_out, T_cold_in, T_cold_out,
                     pinch_range=None, cost_per_ft2=100):
    """
    Sweep pinch temperature for a heat exchanger and return area + cost curve.

    The exchanger has fixed duty Q.  As pinch tightens, LMTD shrinks so area
    grows.  We model the HX as counter-flow with the given terminal temps as
    baseline, then scale LMTD proportionally to the pinch variation.

    For each pinch value, we recompute LMTD from the hot/cold profile where
    the minimum approach (pinch) is forced to the sweep value.  The simplest
    correct approach: hold Q constant, hold one side's profile, and adjust the
    other side's outlet so the minimum dT equals the sweep pinch.

    In practice the ORC cycle changes when pinch changes, so this is an
    approximation at constant duty - good enough for the trade-off visual.

    Returns list of dicts: [{pinch, lmtd, area_ft2, cost}]
    """
    if pinch_range is None:
        pinch_range = np.linspace(5, 25, 21)

    # Baseline LMTD from actual terminal temps
    dT1_base = T_hot_in - T_cold_out
    dT2_base = T_hot_out - T_cold_in

    # Identify which end is the pinch (minimum dT)
    min_dT_base = min(dT1_base, dT2_base)
    if min_dT_base <= 0:
        min_dT_base = 1.0

    results = []
    for pinch in pinch_range:
        # Scale both terminal dTs so the minimum equals the sweep pinch
        scale = pinch / min_dT_base
        dT1 = max(dT1_base * scale, 0.1)
        dT2 = max(dT2_base * scale, 0.1)
        lmtd_val = _lmtd(dT1, dT2)

        area = _hx_area(Q_btu_hr, lmtd_val, U) if lmtd_val > 0 else 0
        cost = area * cost_per_ft2
        results.append({
            "pinch": float(pinch),
            "lmtd": lmtd_val,
            "area_ft2": area,
            "cost": cost,
        })

    return results


def acc_area_vs_pinch(Q_reject_btu_hr, T_ambient, pinch_range=None,
                      cost_per_ft2=None):
    """
    ACC area and cost vs ACC pinch (condensing approach to ambient).
    """
    if pinch_range is None:
        pinch_range = np.linspace(5, 25, 21)
    if cost_per_ft2 is None:
        cost_per_ft2 = COST_FACTORS["acc_per_ft2"]

    results = []
    for pinch in pinch_range:
        dT = max(pinch, 0.1)
        area = Q_reject_btu_hr / (U_VALUES["acc"] * dT)
        cost = area * cost_per_ft2
        results.append({
            "pinch": float(pinch),
            "area_ft2": area,
            "cost": cost,
        })
    return results


def duct_diameter_vs_dp(m_dot_lbs, rho, length_ft, f_darcy=0.02,
                        dp_range=None, uc_per_ft2=None):
    """
    Given allowable dP, solve for required duct diameter using Darcy-Weisbach.

    Darcy-Weisbach:  dP = f * (L/D) * 0.5 * rho * V^2    (lbf/ft2)
    With V = mdot / (rho * A),  A = pi/4 * D^2:
      dP = f * L * 8 * mdot^2 / (pi^2 * rho * D^5)       (lbf/ft2)

    Solving for D:
      D^5 = f * L * 8 * mdot^2 / (pi^2 * rho * dP_lbft2)
      D = (f * L * 8 * mdot^2 / (pi^2 * rho * dP_lbft2))^(1/5)

    Cost uses surface area: cost = uc * pi * (D_in/12) * L_ft * dia_multiplier
    """
    if dp_range is None:
        dp_range = np.linspace(0.1, 2.0, 20)
    if uc_per_ft2 is None:
        uc_per_ft2 = COST_FACTORS["iso_duct_per_ft2"]

    results = []
    for dp_psi in dp_range:
        dp_lbft2 = dp_psi * 144.0
        if dp_lbft2 <= 0 or rho <= 0 or m_dot_lbs <= 0:
            results.append({"dp_psi": float(dp_psi), "diameter_in": 0,
                            "velocity_fps": 0, "cost": 0})
            continue

        D_ft_5 = f_darcy * length_ft * 8 * m_dot_lbs**2 / (
            math.pi**2 * rho * dp_lbft2)
        D_ft = D_ft_5 ** 0.2
        D_in = D_ft * 12

        area_flow = math.pi / 4 * D_ft**2
        velocity = m_dot_lbs / (rho * area_flow) if area_flow > 0 else 0

        # Surface area costing
        surface_area = math.pi * (D_in / 12) * length_ft
        if D_in > 72:
            dia_mult = 1.7
        elif D_in > 60:
            dia_mult = 1.4
        else:
            dia_mult = 1.0
        cost = uc_per_ft2 * surface_area * dia_mult

        results.append({
            "dp_psi": float(dp_psi),
            "diameter_in": D_in,
            "velocity_fps": velocity,
            "cost": cost,
        })

    return results


def acc_tubes_vs_dp(m_dot_lbs, rho, dp_range=None, params=None):
    """
    ACC tube bundle model:
      dP_tubes = f_tube * (N_rows * L_tube / D_tube) * (rho * V^2 / 2)

    Given allowable dP, solve for max velocity, then required face area
    from mass flow.  Cost = face_area * N_rows * cost_per_ft2_row.

    Returns list of dicts with dp_psi, velocity_fps, face_area_ft2, cost.
    """
    p = {**ACC_TUBE_DEFAULTS, **(params or {})}
    f_tube = p["f_tube"]
    L_tube = p["L_tube_ft"]
    D_tube_ft = p["D_tube_in"] / 12.0
    N_rows = p["N_rows"]
    cost_rate = p["cost_per_ft2_row"]

    if dp_range is None:
        dp_range = np.linspace(0.1, 2.0, 20)

    K = f_tube * (N_rows * L_tube / D_tube_ft)

    results = []
    for dp_psi in dp_range:
        dp_lbft2 = dp_psi * 144.0
        if dp_lbft2 <= 0 or rho <= 0 or K <= 0:
            results.append({"dp_psi": float(dp_psi), "velocity_fps": 0,
                            "face_area_ft2": 0, "cost": 0})
            continue

        # V_max = sqrt(2 * dP / (K * rho))
        V_max = (2 * dp_lbft2 / (K * rho)) ** 0.5
        face_area = m_dot_lbs / (rho * V_max) if V_max > 0 else 0
        cost = face_area * N_rows * cost_rate

        results.append({
            "dp_psi": float(dp_psi),
            "velocity_fps": V_max,
            "face_area_ft2": face_area,
            "cost": cost,
        })

    return results


def sizing_tradeoff_sweep(inputs, fp, config="A"):
    """
    Combined trade-off sweep: tighten all pinch points and dP allowances
    together from loose (multiplier=2.0) to tight (multiplier=0.25).

    At each step, re-solve the cycle, re-cost, and compute net power.
    Returns list of dicts with:
      multiplier, total_cost, net_power_kw, delta_cost, delta_power_kw
    """
    from thermodynamics import solve_config_a, solve_config_b

    # Pinch and dP keys to scale
    if config == "A":
        pinch_keys = ["dt_pinch_vaporizer", "dt_pinch_preheater",
                      "dt_pinch_recup", "dt_pinch_acc_a"]
        dp_keys = ["dp_acc_tubes_a", "dp_acc_headers_a", "dp_recup_a"]
    else:
        pinch_keys = ["dt_pinch_vaporizer", "dt_pinch_preheater",
                      "dt_pinch_recup", "dt_approach_intermediate",
                      "dt_pinch_acc_b"]
        dp_keys = ["dp_ihx_iso", "dp_recup_b", "dp_tailpipe_iso_b",
                   "dp_acc_tubes_prop", "dp_ihx_prop"]

    # Baseline values
    base_vals = {}
    for k in pinch_keys + dp_keys:
        base_vals[k] = inputs.get(k, _default_inputs().get(k, 10))

    multipliers = np.linspace(0.25, 2.0, 15)
    results = []

    for mult in multipliers:
        inp_mod = {**inputs}
        for k in pinch_keys + dp_keys:
            inp_mod[k] = max(base_vals[k] * mult, 1.0 if k in pinch_keys else 0.05)

        try:
            if config == "A":
                r = solve_config_a(inp_mod, fp)
                c = calculate_costs_a(r["states"], r["performance"],
                                      inp_mod, r.get("duct"))
            else:
                r = solve_config_b(inp_mod, fp)
                c = calculate_costs_b(r["states"], r["propane_states"],
                                      r["performance"], inp_mod, r.get("duct"))
            results.append({
                "multiplier": float(mult),
                "total_cost": c["total_installed"],
                "net_power_kw": r["performance"]["net_power_kw"],
            })
        except Exception:
            results.append({
                "multiplier": float(mult),
                "total_cost": float("nan"),
                "net_power_kw": float("nan"),
            })

    # Compute deltas relative to baseline (mult=1.0)
    base_idx = min(range(len(multipliers)),
                   key=lambda i: abs(multipliers[i] - 1.0))
    base_cost = results[base_idx]["total_cost"]
    base_power = results[base_idx]["net_power_kw"]

    for r in results:
        r["delta_cost"] = r["total_cost"] - base_cost
        r["delta_power_kw"] = r["net_power_kw"] - base_power
        # Marginal cost of electricity: $ per additional kW capacity
        if r["delta_power_kw"] > 0 and r["delta_cost"] != 0:
            r["marginal_cost_per_kw"] = r["delta_cost"] / r["delta_power_kw"]
        else:
            r["marginal_cost_per_kw"] = float("nan")

    return results
