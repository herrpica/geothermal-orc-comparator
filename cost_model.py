"""
Equipment sizing and cost estimation for ORC Config A and Config B.
All costs in 2024 USD. Uses parametric unit-cost factors
that can be overridden via the inputs dict (uc_* keys).

Cost stack (calibrated to Turboden 93 MW lump-sum EPC project data):
  1. Equipment package (~$1,150/kW — TG set, HX, ACC, pumps, controls, electrical)
  2. Installation adders (BOP piping, civil, E&I, construction, engineering, commissioning)
  3. Contingency on all above (AACE Class 4)
  4. Gathering system (injection pumps, brine piping to wells)
  5. T&D (highlines, transformers, relays, interconnection)
  → Total installed = all surface facilities, wellhead to interconnect
"""

import math
import numpy as np
from scipy.optimize import minimize_scalar, brentq

# Parallel turbine/ACC trains sharing common vaporizer + preheater
N_TRAINS = 2

# -- Unit cost factor defaults (2024 USD) ------------------------------------
# Equipment unit costs calibrated to Turboden 93 MW lump-sum EPC project data.
# Installation adders calibrated to geothermal EPC field actuals.

COST_FACTORS = {
    # ── Equipment unit costs (produce ORC equipment package ~$1,150/kW) ──
    "vaporizer_per_ft2": 75,        # shell-and-tube, brine service, ASME VIII
    "preheater_per_ft2": 65,        # liquid-liquid, corrosion-resistant
    "recup_per_ft2": 55,            # welded plate or S&T, organic vapor-liquid
    "hx_per_ft2": 40,               # intermediate HX (pressure rated)
    "acc_per_ft2": 12,              # legacy — used by app.py sidebar only
    "acc_per_bay": 575000,           # A-frame w/ fans, motors, staging control, structure
    "turbine_per_kw": 340,          # TG package: turbine, gearbox, generator, lube, enclosure
    "iso_pump_per_hp": 600,         # API 610 process pump package
    "pump_per_hp": 600,
    "iso_duct_per_ft2": 70,         # field-erected duct, insulated
    "prop_pipe_per_ft2": 55,        # propane piping, insulated
    "prop_piping_pct": 20,          # % of IHX cost for propane piping+pump
    "steel_per_lb": 7.00,           # erected, primed, painted, fireproofed
    "wf_inventory_per_kw": 15,      # working fluid charge (isopentane ~$5/lb)
    "controls_instrumentation_per_kw": 50,  # DCS, PLCs, field instruments, SCADA
    "fan_control_per_kw": 0.15,              # staging control: PLC logic only (no VFDs)
    "electrical_equipment_per_kw": 65,      # MV switchgear, MCC, transformers, cable
    # ── Installation & construction adders (~$1,350/kW) ──
    "bop_piping_pct": 16,               # % of equipment — BOP piping & valves
    "civil_structural_per_kw": 260,     # $/kW gross — civil, foundations, structures
    "ei_installation_per_kw": 190,      # $/kW gross — E&I installation labor
    "construction_labor_per_kw": 285,   # $/kW gross — construction labor & CM
    "engineering_pct": 10,              # % of equipment — engineering & procurement
    "commissioning_per_kw": 65,         # $/kW gross — commissioning & startup
    "contingency_pct": 10,              # % of all above — AACE Class 4
    # ── Gathering & T&D (included in total installed cost) ──
    "gathering_per_kw": 860,            # $/kW net — injection pumps, brine piping to wells
    "td_per_kw": 355,                   # $/kW net — highlines, transformers, relays, interconnect
    # ── Hybrid wet/dry & dual pressure parametric factors ──
    "water_system_per_kw": 50,          # $/kW gross for hybrid wet/dry water system
    "hybrid_acc_bay_reduction": 0.17,   # 17% fewer ACC bays for hybrid
    "dual_pressure_hx_per_kw": 20,     # $/kW gross additional HX for dual-pressure
    "dual_pressure_schedule_weeks": 3,  # additional weeks for dual-pressure
    "dual_pressure_efficiency_gain": 0.03,  # +3% absolute efficiency gain
    # ── Legacy keys (backward compat for app.py sidebar uc_* overrides) ──
    "foundation_pct": 5,
    "construction_mgmt_pct": 5,
}

# ── Procurement Strategy Override Sets ──────────────────────────────────────
# Each strategy provides overrides to merge onto COST_FACTORS.
# Priority chain: user sidebar uc_* > strategy override > COST_FACTORS baseline
#
# oem_lump_sum: OEM integrated package + EPC lump-sum contractor (baseline)
# direct_lump_sum: Direct vendor purchase + EPC lump-sum contractor
# oem_self_perform: OEM integrated package + Owner T&M self-perform
# direct_self_perform: Direct vendor purchase + Owner T&M self-perform

PROCUREMENT_STRATEGIES = {
    "oem_lump_sum": {},  # Baseline — uses COST_FACTORS as-is

    "direct_lump_sum": {
        # Equipment: strip OEM integration margin (30-40% reduction)
        "turbine_per_kw": 200,
        "vaporizer_per_ft2": 50,
        "preheater_per_ft2": 43,
        "recup_per_ft2": 37,
        "hx_per_ft2": 27,
        "acc_per_bay": 347200,       # Worldwide validated bid (already direct)
        "iso_pump_per_hp": 420,
        "pump_per_hp": 420,
        "controls_instrumentation_per_kw": 40,
        "electrical_equipment_per_kw": 55,
        # Construction: lump-sum (same as baseline)
    },

    "oem_self_perform": {
        # Equipment: OEM package (same as baseline)
        # Construction: self-perform T&M (no contractor markup)
        "bop_piping_pct": 12,
        "civil_structural_per_kw": 90,
        "ei_installation_per_kw": 115,
        "construction_labor_per_kw": 185,
        "engineering_pct": 7,
        "commissioning_per_kw": 48,
        "contingency_pct": 9,
    },

    "direct_self_perform": {
        # Both direct equipment AND self-perform construction
        "turbine_per_kw": 200,
        "vaporizer_per_ft2": 50,
        "preheater_per_ft2": 43,
        "recup_per_ft2": 37,
        "hx_per_ft2": 27,
        "acc_per_bay": 347200,
        "iso_pump_per_hp": 420,
        "pump_per_hp": 420,
        "controls_instrumentation_per_kw": 40,
        "electrical_equipment_per_kw": 55,
        "bop_piping_pct": 12,
        "civil_structural_per_kw": 90,
        "ei_installation_per_kw": 115,
        "construction_labor_per_kw": 185,
        "engineering_pct": 7,
        "commissioning_per_kw": 48,
        "contingency_pct": 9,
    },
}

STRATEGY_LABELS = {
    "oem_lump_sum": "OEM + Lump-Sum",
    "direct_lump_sum": "Direct + Lump-Sum",
    "oem_self_perform": "OEM + Self-Perform",
    "direct_self_perform": "Direct + Self-Perform",
}

STRATEGY_SHORT_LABELS = {
    "oem_lump_sum": "OEM+LS",
    "direct_lump_sum": "DIR+LS",
    "oem_self_perform": "OEM+SP",
    "direct_self_perform": "DIR+SP",
}


def get_effective_cost_factors(strategy: str = "oem_lump_sum") -> dict:
    """Return COST_FACTORS merged with the given procurement strategy overrides.

    Priority chain: user sidebar uc_* override > strategy override > COST_FACTORS baseline.
    This function handles the strategy layer; uc_* overrides are applied in calculate_costs_*.
    """
    overrides = PROCUREMENT_STRATEGIES.get(strategy, {})
    if not overrides:
        return dict(COST_FACTORS)
    merged = dict(COST_FACTORS)
    merged.update(overrides)
    return merged


U_VALUES = {
    "acc": 80,               # effective U incl. fin enhancement (bare-tube basis)
    "intermediate_hx": 150,
    "recuperator": 60,       # organic vapor-liquid, compact design
    "vaporizer": 150,
    "preheater": 250,        # liquid-liquid service, high-pressure organic
}


def _default_inputs():
    return {
        "T_geo_in": 420,
        "m_dot_geo": 1100,
        "cp_brine": 1.0,
        "T_geo_out_min": 180,
        "T_ambient": 95,
        "dt_pinch_vaporizer": 8,
        "dt_pinch_preheater": 6,
        "dt_pinch_acc_a": 30,
        "dt_pinch_acc_b": 30,
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
        # ACC fan parameters
        "dT_air": 25,
        "fan_static_inwc": 0.75,
        "eta_fan": 0.78,
        "eta_motor": 0.95,
        "n_fan_bays": 0,           # 0 = auto-solve from airflow
        "fan_diameter_ft": 28,
        "W_aux_kw": 150,
    }


def _lmtd(dT1, dT2):
    if abs(dT1 - dT2) < 0.01:
        return (dT1 + dT2) / 2
    if dT1 <= 0 or dT2 <= 0:
        return max(dT1, dT2, 0.1)
    return (dT1 - dT2) / np.log(dT1 / dT2)


def _acc_face_area(Q_mmbtu_hr, T_cond, T_ambient, U=None):
    """ACC face area.  Q in MMBtu/hr, U in BTU/(hr·ft2·°F), returns ft2."""
    U = U or U_VALUES["acc"]
    dT = T_cond - T_ambient
    if dT <= 0:
        dT = 1
    return Q_mmbtu_hr * 1e6 / (U * dT)


def calculate_fan_power(Q_reject_mmbtu_hr, T_ambient, inputs):
    """ACC fan power from first principles.

    Returns dict with air properties, airflow, fan power, and fan count.
    """
    inp = {**_default_inputs(), **inputs}
    dT_air = inp["dT_air"]
    fan_static_inwc = inp["fan_static_inwc"]
    eta_fan = inp["eta_fan"]
    eta_motor = inp["eta_motor"]
    n_fan_override = inp["n_fan_bays"]
    fan_dia_ft = inp["fan_diameter_ft"]

    Cp_air = 0.24  # BTU/(lb·°F)
    rho_air = 0.0735 * (460 + 68) / (460 + T_ambient)  # lb/ft3

    # Air mass flow from heat balance: Q = m_dot_air * Cp * dT_air
    m_dot_air_lb_hr = Q_reject_mmbtu_hr * 1e6 / (Cp_air * dT_air)

    # Volumetric flow
    vol_flow_ft3s = m_dot_air_lb_hr / (rho_air * 3600)

    # Fan static pressure: convert in WC to lbf/ft2
    dP_fan_psf = fan_static_inwc * 5.192

    # Total fan shaft power: W = V_dot * dP / (eta_fan * eta_motor)
    W_fans_kw = vol_flow_ft3s * dP_fan_psf / (eta_fan * eta_motor * 745.7)

    # Fan sizing: each fan sweeps a circular area
    fan_area_each_ft2 = math.pi / 4 * fan_dia_ft ** 2
    face_vel_fpm = 400  # ft/min typical ACC fan face velocity
    n_fans_required = max(1, math.ceil(
        vol_flow_ft3s * 60 / (fan_area_each_ft2 * face_vel_fpm)
    ))
    n_fans_used = n_fan_override if n_fan_override > 0 else n_fans_required

    return {
        "rho_air": rho_air,
        "m_dot_air_lb_hr": m_dot_air_lb_hr,
        "vol_flow_ft3s": vol_flow_ft3s,
        "dP_fan_psf": dP_fan_psf,
        "W_fans_kw": W_fans_kw,
        "fan_area_each_ft2": fan_area_each_ft2,
        "n_fans_required": n_fans_required,
        "n_fans_used": n_fans_used,
    }


def pump_sizing(m_dot_lb_hr, rho_liquid, P_high_psia, P_low_psia, W_pump_btu_lb, eta_pump):
    """Pump sizing for equipment selection.

    Returns dict with flow_gpm, dP_psi, power_kw, power_hp.
    """
    # rho in lb/ft3; convert to SG for standard gpm formula: gpm = lb_hr / (SG * 500)
    SG = rho_liquid / 62.4 if rho_liquid > 0 else 1
    flow_gpm = m_dot_lb_hr / (SG * 500) if SG > 0 else 0
    dP_psi = P_high_psia - P_low_psia
    power_kw = m_dot_lb_hr * W_pump_btu_lb / 3412.14
    power_hp = power_kw * 1.341

    return {
        "flow_gpm": flow_gpm,
        "dP_psi": dP_psi,
        "power_kw": power_kw,
        "power_hp": power_hp,
    }


def compute_power_balance(perf, fan_result, inputs, config="A"):
    """Centralized power balance for an ORC configuration.

    Every power/efficiency metric flows from this single function.
    All values in kW unless noted.

    Config A: P_net = P_generator - W_iso_pump - W_fans - W_auxiliary
    Config B: P_net = P_generator - W_iso_pump - W_prop_pump - W_fans - W_auxiliary

    Generator efficiency (0.96) accounts for gearbox, generator, and
    transformer losses between turbine shaft and switchgear.
    """
    generator_efficiency = inputs.get("generator_efficiency", 0.96)
    P_shaft = perf["gross_power_kw"]
    P_gross = P_shaft * generator_efficiency  # electrical output at switchgear

    # ISO pump (both configs)
    if config == "B":
        W_iso_pump = perf["m_dot_iso"] * perf["w_pump_iso"] / 3412.14
    else:
        W_iso_pump = perf["m_dot_iso"] * perf["w_pump"] / 3412.14

    # Propane pump (Config B only)
    W_prop_pump_calc = 0.0
    if config == "B":
        W_prop_pump_calc = perf["m_dot_prop"] * perf["w_pump_prop"] / 3412.14
    thermosiphon = config == "B" and inputs.get("prop_thermosiphon", True)
    W_prop_pump = 0.0 if thermosiphon else W_prop_pump_calc

    # ACC fans
    W_fans = fan_result["W_fans_kw"]

    # Auxiliary
    W_auxiliary = inputs.get("W_aux_kw", 150)

    # Total parasitic and net power
    W_total_parasitic = W_iso_pump + W_prop_pump + W_fans + W_auxiliary
    P_net = P_gross - W_total_parasitic

    # Brine heat input (kW)
    Q_brine_kw = perf["m_dot_iso"] * perf["q_evap"] / 3412.14

    # Thermal efficiency: P_net / Q_brine_input
    eta_thermal = P_net / Q_brine_kw if Q_brine_kw > 0 else 0

    # Brine effectiveness: kW per (lb/s) of brine
    m_dot_geo = inputs.get("m_dot_geo", 200)
    brine_effectiveness = P_net / m_dot_geo if m_dot_geo > 0 else 0

    # Parasitic fraction
    parasitic_pct = W_total_parasitic / P_gross * 100 if P_gross > 0 else 0

    return {
        "P_shaft": P_shaft,
        "P_gross": P_gross,
        "generator_efficiency": generator_efficiency,
        "W_iso_pump": W_iso_pump,
        "W_prop_pump": W_prop_pump,
        "W_prop_pump_calc": W_prop_pump_calc,
        "W_fans": W_fans,
        "W_auxiliary": W_auxiliary,
        "W_total_parasitic": W_total_parasitic,
        "P_net": P_net,
        "Q_brine_kw": Q_brine_kw,
        "eta_thermal": eta_thermal,
        "brine_effectiveness": brine_effectiveness,
        "parasitic_pct": parasitic_pct,
        "thermosiphon": thermosiphon,
    }


def acc_area_with_air_rise(Q_reject_mmbtu_hr, T_cond, T_ambient, dT_air, U=None):
    """ACC area using proper LMTD with air inlet/outlet temperatures.

    Air enters at T_ambient and exits at T_ambient + dT_air.
    Returns area_ft2.
    """
    U = U or U_VALUES["acc"]
    dT_hot = T_cond - T_ambient           # hot end: condensing vs air inlet
    dT_cold = T_cond - (T_ambient + dT_air)  # cold end: condensing vs air outlet
    if dT_cold <= 0:
        dT_cold = 0.5
    lmtd_val = _lmtd(dT_hot, dT_cold)
    return Q_reject_mmbtu_hr * 1e6 / (U * lmtd_val)


def _hx_area(Q_mmbtu_hr, lmtd_val, U):
    """HX area.  Q in MMBtu/hr, U in BTU/(hr·ft2·°F), returns ft2."""
    if lmtd_val <= 0:
        lmtd_val = 0.1
    return Q_mmbtu_hr * 1e6 / (U * lmtd_val)


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


def _apply_indirect_costs(costs, inp, gross_power_kw=0, net_power_kw=None):
    """Apply installation adders and contingency to equipment costs.

    Cost stack (per user's vendor bid calibration):
      1. Equipment subtotal (from line items above)
      2. BOP piping & valves (% of equipment)
      3. Civil & structural ($/kW gross)
      4. E&I installation labor ($/kW gross)
      5. Construction labor & CM ($/kW gross)
      6. Engineering & procurement (% of equipment)
      7. Commissioning & startup ($/kW gross)
      8. Contingency (% of all above)

    Expects costs dict to already have 'equipment_subtotal'.
    Mutates costs in-place and returns it.

    If legacy uc_foundation_pct / uc_construction_mgmt_pct keys are present
    in inp (from app.py sidebar), uses the legacy percentage-based structure
    instead.
    """
    equip_sub = costs["equipment_subtotal"]
    costs["equipment_cost"] = equip_sub  # for vendor bid comparison
    gross_kw = max(gross_power_kw, 1)  # available in both branches

    # Detect legacy mode: if user has set foundation_pct or construction_mgmt_pct
    # via the sidebar, fall back to the old percentage-based structure.
    use_legacy = ("uc_foundation_pct" in inp or "uc_construction_mgmt_pct" in inp)

    if use_legacy:
        # Legacy percentage-based structure (backward compat with app.py sidebar)
        fdn_pct = inp.get("uc_foundation_pct", COST_FACTORS["foundation_pct"])
        costs["foundation"] = equip_sub * fdn_pct / 100
        costs["civil_structural"] = costs["foundation"]  # alias

        subtotal_with_fdn = equip_sub + costs["foundation"]
        costs["subtotal_with_foundation"] = subtotal_with_fdn

        eng_pct = inp.get("uc_engineering_pct", COST_FACTORS["engineering_pct"])
        costs["engineering"] = subtotal_with_fdn * eng_pct / 100

        cm_pct = inp.get("uc_construction_mgmt_pct", COST_FACTORS["construction_mgmt_pct"])
        costs["construction_mgmt"] = subtotal_with_fdn * cm_pct / 100
        costs["construction_labor"] = costs["construction_mgmt"]  # alias

        # Zero out new-stack items not in legacy mode
        costs["bop_piping"] = 0
        costs["ei_installation"] = 0
        costs["commissioning"] = 0

        total_before_cont = subtotal_with_fdn + costs["engineering"] + costs["construction_mgmt"]
        costs["total_before_contingency"] = total_before_cont

        cont_pct = inp.get("uc_contingency_pct", COST_FACTORS["contingency_pct"])
        costs["contingency"] = total_before_cont * cont_pct / 100
        costs["plant_installed"] = total_before_cont + costs["contingency"]
    else:
        # New installation cost stack (calibrated to vendor EPC data)
        # Uses strategy-aware cost factors (procurement_strategy in inp)
        strategy = inp.get("procurement_strategy", "oem_lump_sum")
        cf = get_effective_cost_factors(strategy)

        # BOP piping & valves (% of equipment)
        bop_pct = cf["bop_piping_pct"]
        costs["bop_piping"] = equip_sub * bop_pct / 100

        # Civil & structural ($/kW gross × kW = $)
        civil_per_kw = inp.get("uc_civil_structural_per_kw", cf["civil_structural_per_kw"])
        costs["civil_structural"] = civil_per_kw * gross_kw
        costs["foundation"] = costs["civil_structural"]  # legacy alias

        # E&I installation labor ($/kW gross × kW = $)
        ei_per_kw = inp.get("uc_ei_installation_per_kw", cf["ei_installation_per_kw"])
        costs["ei_installation"] = ei_per_kw * gross_kw

        # Construction labor & CM ($/kW gross × kW = $)
        labor_per_kw = cf["construction_labor_per_kw"]
        costs["construction_labor"] = labor_per_kw * gross_kw
        costs["construction_mgmt"] = costs["construction_labor"]  # legacy alias

        # Engineering & procurement (% of equipment)
        eng_pct = cf["engineering_pct"]
        costs["engineering"] = equip_sub * eng_pct / 100

        # Commissioning & startup ($/kW gross × kW = $)
        comm_per_kw = cf["commissioning_per_kw"]
        costs["commissioning"] = comm_per_kw * gross_kw

        total_before_cont = (
            equip_sub + costs["bop_piping"] + costs["civil_structural"]
            + costs["ei_installation"] + costs["construction_labor"]
            + costs["engineering"] + costs["commissioning"]
        )
        costs["total_before_contingency"] = total_before_cont

        # Contingency (% of all above)
        cont_pct = cf["contingency_pct"]
        costs["contingency"] = total_before_cont * cont_pct / 100
        costs["plant_installed"] = total_before_cont + costs["contingency"]

    # ── Gathering & T&D (wellhead-to-interconnect scope) ──────────
    # Use net power basis (deliverable capacity at interconnect)
    # cf may not exist in legacy mode — use strategy-aware factors or baseline
    if not use_legacy:
        strategy = inp.get("procurement_strategy", "oem_lump_sum")
        _cf = get_effective_cost_factors(strategy)
    else:
        _cf = COST_FACTORS
    net_kw = net_power_kw if net_power_kw and net_power_kw > 0 else gross_kw
    costs["gathering"] = _cf.get("gathering_per_kw", 860) * net_kw
    costs["td"] = _cf.get("td_per_kw", 355) * net_kw

    # Total installed = plant + gathering + T&D (all surface facilities)
    costs["total_installed"] = (
        costs["plant_installed"] + costs["gathering"] + costs["td"]
    )

    return costs


def calculate_costs_a(states, performance, inputs, duct_result=None) -> dict:
    """Component-by-component installed cost for Config A.

    Plant layout: 1 vaporizer + 1 preheater (total flow) feeding
    n_trains parallel turbine/recuperator/ACC trains.
    Per-train equipment sized at m_dot/n_trains, cost x n_trains.
    """
    inp = {**_default_inputs(), **inputs}
    n_trains = inp.get("n_trains", N_TRAINS)
    perf = performance
    m_dot = perf["m_dot_iso"]
    m_dot_train = m_dot / n_trains  # per-train mass flow

    # Get effective cost factors for the selected procurement strategy
    strategy = inp.get("procurement_strategy", "oem_lump_sum")
    cf = get_effective_cost_factors(strategy)

    costs = {}
    costs["procurement_strategy"] = strategy

    # Turbine-generator (per-train, x n_trains)
    uc_turb = inp.get("uc_turbine_per_kw", cf["turbine_per_kw"])
    costs["turbine_generator"] = perf["gross_power_kw"] * uc_turb  # N * (P/N * uc) = P * uc

    # Isopentane pump (per-train, x n_trains)
    pump_power_btu_hr = m_dot * perf["w_pump"]  # N * (m/N * w) = m * w
    pump_hp = pump_power_btu_hr / 2544.43
    costs["iso_pump"] = pump_hp * cf["iso_pump_per_hp"]

    # HX cost multiplier (applies to all heat exchangers)
    hx_mult = inp.get("uc_hx_multiplier", 1.0)

    # Vaporizer (State 7 -> 1) -- SHARED, sized at total flow
    Q_vap = m_dot * perf["q_vaporizer"] / 1e6          # MMBtu/hr
    dT1_vap = inp["T_geo_in"] - states["1"].T
    dT2_vap = perf["T_brine_mid"] - states["7"].T
    lmtd_vap = _lmtd(dT1_vap, dT2_vap)
    vap_area = _hx_area(Q_vap, lmtd_vap, U_VALUES["vaporizer"])
    uc_vap = inp.get("uc_vaporizer_per_ft2", cf["vaporizer_per_ft2"]) * hx_mult
    costs["vaporizer"] = vap_area * uc_vap
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater (State 6 -> 7) -- SHARED, sized at total flow
    Q_pre = m_dot * perf["q_preheater"] / 1e6          # MMBtu/hr
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    uc_pre = inp.get("uc_preheater_per_ft2", cf["preheater_per_ft2"]) * hx_mult
    costs["preheater"] = pre_area * uc_pre
    costs["preheater_area_ft2"] = pre_area

    # Recuperator (per-train, x n_trains)
    Q_recup_train = m_dot_train * perf["q_recup"] / 1e6  # MMBtu/hr per train
    if Q_recup_train > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area_train = _hx_area(Q_recup_train, lmtd_recup, U_VALUES["recuperator"])
        uc_rec = inp.get("uc_recuperator_per_ft2", cf["recup_per_ft2"]) * hx_mult
        costs["recuperator"] = recup_area_train * uc_rec * n_trains
        costs["recuperator_area_ft2"] = recup_area_train * n_trains  # plant total
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # ACC (per-train, x n_trains)
    Q_cond_train = m_dot_train * perf["q_cond"] / 1e6   # MMBtu/hr per train
    acc_area_train = _acc_face_area(Q_cond_train, perf["T_cond"], inp["T_ambient"])
    costs["acc_area_ft2"] = acc_area_train * n_trains  # plant total
    if "uc_acc_per_bay" in inp:
        # Per-bay model: use fan bay count from parasitic calc or estimate
        n_bays = inp.get("n_fan_bays_computed", 10)
        acc_per_bay = inp["uc_acc_per_bay"]
        costs["acc"] = n_bays * acc_per_bay
        costs["acc_n_bays"] = n_bays
    else:
        # Default per-bay costing uses strategy-aware cost factor
        n_bays = inp.get("n_fan_bays_computed", 10)
        costs["acc"] = n_bays * cf["acc_per_bay"]
        costs["acc_n_bays"] = n_bays

    # Ductwork (segments are per-train from thermo, cost x n_trains)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    segments = []
    if duct_result:
        segments = duct_result["segments"]
        for seg in segments:
            seg_cost = _duct_segment_cost(seg, inp)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost * n_trains

    # Structural steel (per-train segments, cost x n_trains)
    steel_cost_train, steel_weight_train = _structural_steel_cost(segments, inp)
    costs["structural_steel"] = steel_cost_train * n_trains
    costs["structural_steel_weight_lb"] = steel_weight_train * n_trains

    # No Config B components
    costs["intermediate_hx"] = 0
    costs["intermediate_hx_area_ft2"] = 0
    costs["propane_system"] = 0

    # Working fluid inventory (sized per gross kW)
    costs["wf_inventory"] = cf.get("wf_inventory_per_kw", 0) * perf["gross_power_kw"]

    # Controls & instrumentation (DCS, PLCs, field instruments, SCADA)
    costs["controls_instrumentation"] = cf.get("controls_instrumentation_per_kw", 0) * perf["gross_power_kw"]

    # Electrical equipment (MV switchgear, MCC, transformers, cable)
    costs["electrical_equipment"] = cf.get("electrical_equipment_per_kw", 0) * perf["gross_power_kw"]

    # Equipment subtotal
    equipment_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "structural_steel",
        "intermediate_hx", "propane_system",
        "wf_inventory", "controls_instrumentation", "electrical_equipment",
    ]
    costs["equipment_subtotal"] = sum(costs[k] for k in equipment_keys)

    # Apply installation & indirect cost layers
    _apply_indirect_costs(costs, inp, gross_power_kw=perf["gross_power_kw"])

    # Equipment count (n_trains trains: turbines, pumps, recups, ACCs + 1 vap + 1 pre)
    costs["equipment_count"] = 4 * n_trains + 2
    costs["n_trains"] = n_trains

    return costs


def calculate_costs_b(states, propane_states, performance, inputs, duct_result=None) -> dict:
    """Component-by-component installed cost for Config B.

    Plant layout: 1 vaporizer + 1 preheater (total flow) feeding
    n_trains parallel turbine/recuperator/IHX/ACC trains.
    Per-train equipment sized at m_dot/n_trains, cost x n_trains.
    """
    inp = {**_default_inputs(), **inputs}
    n_trains = inp.get("n_trains", N_TRAINS)
    perf = performance
    m_dot_iso = perf["m_dot_iso"]
    m_dot_prop = perf["m_dot_prop"]
    m_dot_iso_train = m_dot_iso / n_trains
    m_dot_prop_train = m_dot_prop / n_trains

    # Get effective cost factors for the selected procurement strategy
    strategy = inp.get("procurement_strategy", "oem_lump_sum")
    cf = get_effective_cost_factors(strategy)

    costs = {}
    costs["procurement_strategy"] = strategy

    # Turbine-generator (per-train, x n_trains)
    uc_turb = inp.get("uc_turbine_per_kw", cf["turbine_per_kw"])
    costs["turbine_generator"] = perf["gross_power_kw"] * uc_turb  # N * (P/N * uc) = P * uc

    # Isopentane pump (per-train, x n_trains)
    pump_power_iso = m_dot_iso * perf["w_pump_iso"]  # N * (m/N * w) = m * w
    pump_hp_iso = pump_power_iso / 2544.43
    costs["iso_pump"] = pump_hp_iso * cf["iso_pump_per_hp"]

    # HX cost multiplier (applies to all heat exchangers)
    hx_mult = inp.get("uc_hx_multiplier", 1.0)

    # Vaporizer -- SHARED, sized at total flow
    Q_vap = m_dot_iso * perf["q_vaporizer"] / 1e6          # MMBtu/hr
    dT1_vap = inp["T_geo_in"] - states["1"].T
    dT2_vap = perf["T_brine_mid"] - states["7"].T
    lmtd_vap = _lmtd(dT1_vap, dT2_vap)
    vap_area = _hx_area(Q_vap, lmtd_vap, U_VALUES["vaporizer"])
    uc_vap = inp.get("uc_vaporizer_per_ft2", cf["vaporizer_per_ft2"]) * hx_mult
    costs["vaporizer"] = vap_area * uc_vap
    costs["vaporizer_area_ft2"] = vap_area

    # Preheater -- SHARED, sized at total flow
    Q_pre = m_dot_iso * perf["q_preheater"] / 1e6          # MMBtu/hr
    dT1_pre = perf["T_brine_mid"] - states["7"].T
    dT2_pre = perf["T_geo_out_calc"] - states["6"].T
    lmtd_pre = _lmtd(dT1_pre, dT2_pre)
    pre_area = _hx_area(Q_pre, lmtd_pre, U_VALUES["preheater"])
    uc_pre = inp.get("uc_preheater_per_ft2", cf["preheater_per_ft2"]) * hx_mult
    costs["preheater"] = pre_area * uc_pre
    costs["preheater_area_ft2"] = pre_area

    # Recuperator (per-train, x n_trains)
    Q_recup_train = m_dot_iso_train * perf["q_recup"] / 1e6  # MMBtu/hr per train
    if Q_recup_train > 0:
        dT1_r = states["2"].T - states["6"].T
        dT2_r = states["3"].T - states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area_train = _hx_area(Q_recup_train, lmtd_recup, U_VALUES["recuperator"])
        uc_rec = inp.get("uc_recuperator_per_ft2", cf["recup_per_ft2"]) * hx_mult
        costs["recuperator"] = recup_area_train * uc_rec * n_trains
        costs["recuperator_area_ft2"] = recup_area_train * n_trains  # plant total
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # Intermediate HX (per-train, x n_trains)
    Q_int_train = m_dot_iso_train * perf["q_cond_iso"] / 1e6  # MMBtu/hr per train
    lmtd_int = inp.get("dt_approach_intermediate", 10)
    int_area_train = _hx_area(Q_int_train, lmtd_int, U_VALUES["intermediate_hx"])
    uc_ihx = inp.get("uc_ihx_per_ft2", cf["hx_per_ft2"]) * hx_mult
    costs["intermediate_hx"] = int_area_train * uc_ihx * n_trains
    costs["intermediate_hx_area_ft2"] = int_area_train * n_trains  # plant total

    # Propane ACC (per-train, x n_trains)
    Q_prop_train = m_dot_prop_train * (propane_states["A"].h - propane_states["B"].h) / 1e6
    T_propane_cond = perf["T_propane_cond"]
    acc_area_train = _acc_face_area(Q_prop_train, T_propane_cond, inp["T_ambient"])
    costs["acc_area_ft2"] = acc_area_train * n_trains  # plant total
    if "uc_acc_per_bay" in inp:
        n_bays = inp.get("n_fan_bays_computed", 10)
        acc_per_bay = inp["uc_acc_per_bay"]
        costs["acc"] = n_bays * acc_per_bay
        costs["acc_n_bays"] = n_bays
    else:
        # Default per-bay costing uses strategy-aware cost factor
        n_bays = inp.get("n_fan_bays_computed", 10)
        costs["acc"] = n_bays * cf["acc_per_bay"]
        costs["acc_n_bays"] = n_bays

    # Propane system (piping + pump) as % of IHX cost -- already plant total
    prop_pct = inp.get("uc_prop_piping_pct", cf["prop_piping_pct"])
    costs["propane_system"] = costs["intermediate_hx"] * prop_pct / 100

    # Ductwork (segments are per-train from thermo, cost x n_trains)
    costs["ductwork_segments"] = {}
    total_duct_cost = 0
    segments = []
    if duct_result:
        segments = duct_result["segments"]
        for seg in segments:
            seg_cost = _duct_segment_cost(seg, inp)
            costs["ductwork_segments"][seg["name"]] = seg_cost
            total_duct_cost += seg_cost
    costs["ductwork"] = total_duct_cost * n_trains

    # Structural steel (per-train segments, cost x n_trains)
    steel_cost_train, steel_weight_train = _structural_steel_cost(segments, inp)
    costs["structural_steel"] = steel_cost_train * n_trains
    costs["structural_steel_weight_lb"] = steel_weight_train * n_trains

    # Working fluid inventory (sized per gross kW)
    costs["wf_inventory"] = cf.get("wf_inventory_per_kw", 0) * perf["gross_power_kw"]

    # Controls & instrumentation (DCS, PLCs, field instruments, SCADA)
    costs["controls_instrumentation"] = cf.get("controls_instrumentation_per_kw", 0) * perf["gross_power_kw"]

    # Electrical equipment (MV switchgear, MCC, transformers, cable)
    costs["electrical_equipment"] = cf.get("electrical_equipment_per_kw", 0) * perf["gross_power_kw"]

    # Equipment subtotal
    equipment_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "structural_steel",
        "intermediate_hx", "propane_system",
        "wf_inventory", "controls_instrumentation", "electrical_equipment",
    ]
    costs["equipment_subtotal"] = sum(costs[k] for k in equipment_keys)

    # Apply installation & indirect cost layers
    _apply_indirect_costs(costs, inp, gross_power_kw=perf["gross_power_kw"])

    # Equipment count (n_trains: turbines, iso pumps, recups, IHX, ACC,
    #                  prop pumps + 1 vap + 1 pre + prop piping)
    costs["equipment_count"] = 6 * n_trains + 3
    costs["n_trains"] = n_trains

    return costs


def calculate_costs_dual_pressure(dual_result: dict, inputs: dict) -> dict:
    """Component-by-component installed cost for dual-pressure ORC (Config D).

    Two pressure stages (HP + LP) sharing brine and ACC. Equipment priced
    per-stage, installation applied once on combined total.

    HP cycle: recuperated (2 turbines, 2 pumps, 1 vaporizer, 1 preheater, 2 recuperators)
    LP cycle: basic (2 turbines, 2 pumps, 1 vaporizer, 1 preheater)
    Shared: ACC (combined heat rejection)
    """
    inp = {**_default_inputs(), **inputs}
    n_trains = inp.get("n_trains", N_TRAINS)
    perf = dual_result["performance"]
    hp_perf = dual_result["hp_performance"]
    lp_perf = dual_result["lp_performance"]
    hp_states = dual_result["hp_states"]
    lp_states = dual_result["lp_states"]

    # Get effective cost factors for the selected procurement strategy
    strategy = inp.get("procurement_strategy", "oem_lump_sum")
    cf = get_effective_cost_factors(strategy)

    costs = {}
    costs["procurement_strategy"] = strategy

    # HX cost multiplier
    hx_mult = inp.get("uc_hx_multiplier", 1.0)
    uc_turb = inp.get("uc_turbine_per_kw", cf["turbine_per_kw"])

    # ── HP Turbine-generator (per-train × n_trains) ──
    hp_tg_cost = hp_perf["gross_power_kw"] * uc_turb

    # ── LP Turbine-generator (per-train × n_trains) ──
    lp_tg_cost = lp_perf["gross_power_kw"] * uc_turb

    # Combined turbine_generator for downstream compat
    costs["turbine_generator"] = hp_tg_cost + lp_tg_cost

    # ── HP Iso pump (per-train × n_trains) ──
    hp_pump_power = hp_perf["m_dot_iso"] * hp_perf["w_pump"]  # BTU/hr
    hp_pump_hp = hp_pump_power / 2544.43
    hp_pump_cost = hp_pump_hp * cf["iso_pump_per_hp"]

    # ── LP Iso pump (per-train × n_trains) ──
    lp_pump_power = lp_perf["m_dot_iso"] * lp_perf["w_pump"]
    lp_pump_hp = lp_pump_power / 2544.43
    lp_pump_cost = lp_pump_hp * cf["iso_pump_per_hp"]

    costs["iso_pump"] = hp_pump_cost + lp_pump_cost

    # ── HP Vaporizer (SHARED, sized at HP total flow) ──
    Q_vap_hp = hp_perf["m_dot_iso"] * hp_perf["q_vaporizer"] / 1e6  # MMBtu/hr
    T_geo_in = inp["T_geo_in"]
    T_split = perf["T_split"]
    dT1_vap_hp = T_geo_in - hp_states["1"].T
    dT2_vap_hp = hp_perf["T_brine_mid"] - hp_states["7"].T
    lmtd_vap_hp = _lmtd(dT1_vap_hp, dT2_vap_hp)
    vap_area_hp = _hx_area(Q_vap_hp, lmtd_vap_hp, U_VALUES["vaporizer"])
    uc_vap = inp.get("uc_vaporizer_per_ft2", cf["vaporizer_per_ft2"]) * hx_mult

    # ── HP Preheater (SHARED, sized at HP total flow) ──
    Q_pre_hp = hp_perf["m_dot_iso"] * hp_perf["q_preheater"] / 1e6
    dT1_pre_hp = hp_perf["T_brine_mid"] - hp_states["7"].T
    dT2_pre_hp = T_split - hp_states["6"].T
    lmtd_pre_hp = _lmtd(dT1_pre_hp, dT2_pre_hp)
    pre_area_hp = _hx_area(Q_pre_hp, lmtd_pre_hp, U_VALUES["preheater"])
    uc_pre = inp.get("uc_preheater_per_ft2", cf["preheater_per_ft2"]) * hx_mult

    # ── LP Vaporizer (SHARED, sized at LP total flow) ──
    Q_vap_lp = lp_perf["m_dot_iso"] * lp_perf["q_vaporizer"] / 1e6
    dT1_vap_lp = T_split - lp_states["1"].T
    dT2_vap_lp = lp_perf["T_brine_mid"] - lp_states["7"].T
    lmtd_vap_lp = _lmtd(dT1_vap_lp, dT2_vap_lp)
    vap_area_lp = _hx_area(Q_vap_lp, lmtd_vap_lp, U_VALUES["vaporizer"])

    # ── LP Preheater (SHARED, sized at LP total flow) ──
    Q_pre_lp = lp_perf["m_dot_iso"] * lp_perf["q_preheater"] / 1e6
    dT1_pre_lp = lp_perf["T_brine_mid"] - lp_states["7"].T
    T_geo_out_min = inp["T_geo_out_min"]
    dT2_pre_lp = lp_perf["T_geo_out_calc"] - lp_states["6"].T
    lmtd_pre_lp = _lmtd(dT1_pre_lp, dT2_pre_lp)
    pre_area_lp = _hx_area(Q_pre_lp, lmtd_pre_lp, U_VALUES["preheater"])

    # Combined HX costs (HP + LP)
    costs["vaporizer"] = (vap_area_hp + vap_area_lp) * uc_vap
    costs["vaporizer_area_ft2"] = vap_area_hp + vap_area_lp
    costs["preheater"] = (pre_area_hp + pre_area_lp) * uc_pre
    costs["preheater_area_ft2"] = pre_area_hp + pre_area_lp

    # ── HP Recuperator (per-train × n_trains) ──
    hp_m_dot_train = hp_perf["m_dot_iso"] / n_trains
    Q_recup_train = hp_m_dot_train * hp_perf["q_recup"] / 1e6
    uc_rec = inp.get("uc_recuperator_per_ft2", cf["recup_per_ft2"]) * hx_mult
    if Q_recup_train > 0:
        dT1_r = hp_states["2"].T - hp_states["6"].T
        dT2_r = hp_states["3"].T - hp_states["5"].T
        lmtd_recup = _lmtd(dT1_r, dT2_r)
        recup_area_train = _hx_area(Q_recup_train, lmtd_recup, U_VALUES["recuperator"])
        costs["recuperator"] = recup_area_train * uc_rec * n_trains
        costs["recuperator_area_ft2"] = recup_area_train * n_trains
    else:
        costs["recuperator"] = 0
        costs["recuperator_area_ft2"] = 0

    # ── Shared ACC (combined HP + LP heat rejection) ──
    Q_reject_total = perf["Q_reject_mmbtu_hr"]
    T_cond = perf["T_cond"]
    # Use per-bay costing
    n_bays = inp.get("n_fan_bays_computed", 10)
    if "uc_acc_per_bay" in inp:
        costs["acc"] = n_bays * inp["uc_acc_per_bay"]
    else:
        costs["acc"] = n_bays * cf["acc_per_bay"]
    costs["acc_n_bays"] = n_bays
    acc_area_total = _acc_face_area(Q_reject_total, T_cond, inp["T_ambient"])
    costs["acc_area_ft2"] = acc_area_total

    # ── Ductwork (allowance for complex dual-exhaust merge) ──
    gross_kw = perf["gross_power_kw"]
    costs["ductwork"] = 15 * gross_kw  # $15/kW allowance
    costs["ductwork_segments"] = {}

    # ── Structural steel (1.3x single-pressure for more equipment support) ──
    # Estimate from gross power basis
    steel_per_lb = inp.get("uc_steel_per_lb", cf["steel_per_lb"])
    # Base estimate: ~0.5 lb/kW for single-pressure, 1.3x for dual
    base_steel_weight = 0.5 * gross_kw * 1.3
    costs["structural_steel"] = base_steel_weight * steel_per_lb
    costs["structural_steel_weight_lb"] = base_steel_weight

    # No propane intermediate components
    costs["intermediate_hx"] = 0
    costs["intermediate_hx_area_ft2"] = 0
    costs["propane_system"] = 0

    # Working fluid inventory (sized per gross kW — more fluid for two loops)
    costs["wf_inventory"] = cf.get("wf_inventory_per_kw", 0) * gross_kw * 1.3

    # Controls & instrumentation (more I/O points for dual-pressure)
    costs["controls_instrumentation"] = cf.get("controls_instrumentation_per_kw", 0) * gross_kw * 1.2

    # Electrical equipment
    costs["electrical_equipment"] = cf.get("electrical_equipment_per_kw", 0) * gross_kw

    # Equipment subtotal
    equipment_keys = [
        "turbine_generator", "iso_pump", "vaporizer", "preheater",
        "recuperator", "acc", "ductwork", "structural_steel",
        "intermediate_hx", "propane_system",
        "wf_inventory", "controls_instrumentation", "electrical_equipment",
    ]
    costs["equipment_subtotal"] = sum(costs[k] for k in equipment_keys)

    # Apply installation & indirect cost layers
    _apply_indirect_costs(costs, inp, gross_power_kw=gross_kw)

    # Equipment count: HP turbines, LP turbines, HP pumps, LP pumps,
    #                  1 HP vap, 1 HP pre, 1 LP vap, 1 LP pre, HP recups, ACC
    costs["equipment_count"] = 4 * n_trains + 5
    costs["n_trains"] = n_trains

    return costs


def construction_schedule(duct_a):
    """
    Absolute phase-based construction schedules for Config A and Config B.

    Config A is serial: civil -> equip -> duct fab/erect -> ACC -> commissioning.
    Config B parallelises: civil -> (power block || propane/ACC) -> tie-in -> commissioning.

    Returns Gantt-ready phase dicts with start/end weeks plus schedule savings.
    """
    tp_dia = duct_a.get("tailpipe_diameter_in", 0)

    # --- Config A: Serial construction ---
    # Duct phase duration depends on tailpipe diameter
    if tp_dia > 72:
        duct_weeks = 24
    elif tp_dia > 60:
        duct_weeks = 20
    elif tp_dia > 48:
        duct_weeks = 16
    else:
        duct_weeks = 10

    a_phases = []
    t = 0

    # Phase 1: Civil & foundations
    a_phases.append({"name": "Civil & foundations", "start": t, "end": t + 16,
                     "duration": 16, "color": "gray", "critical": True, "track": "main"})
    t += 16

    # Phase 2: Equipment delivery & setting
    a_phases.append({"name": "Equipment delivery & setting", "start": t, "end": t + 12,
                     "duration": 12, "color": "steelblue", "critical": True, "track": "main"})
    t += 12

    # Phase 3: Duct fabrication & erection
    duct_start = t
    a_phases.append({"name": "Duct fab & erection", "start": t, "end": t + duct_weeks,
                     "duration": duct_weeks, "color": "indianred", "critical": True, "track": "main"})
    t += duct_weeks

    # Phase 4: ACC structure & tubes
    a_phases.append({"name": "ACC structure & tubes", "start": t, "end": t + 10,
                     "duration": 10, "color": "indianred", "critical": True, "track": "main"})

    # Phase 5: E&I -- 50% overlap with phases 3+4, NOT on critical path
    ei_start_a = duct_start + (duct_weeks + 10) // 2
    a_phases.append({"name": "E&I", "start": ei_start_a, "end": ei_start_a + 10,
                     "duration": 10, "color": "orange", "critical": False, "track": "main"})
    t += 10  # ACC finishes

    # Phase 6: Commissioning
    a_phases.append({"name": "Commissioning", "start": t, "end": t + 8,
                     "duration": 8, "color": "darkblue", "critical": True, "track": "main"})
    t += 8

    a_total = t  # 16 + 12 + duct_weeks + 10 + 8

    # --- Config B: Parallel construction ---
    b_phases = []
    t = 0

    # Civil & foundations
    b_phases.append({"name": "Civil & foundations", "start": t, "end": t + 16,
                     "duration": 16, "color": "gray", "critical": True, "track": "main"})
    t = 16  # end of civil

    # --- Power block track (serial sub-phases, 22 wk total) ---
    pb_start = t
    pb_subs = [
        ("Equipment setting",       10, "steelblue"),
        ("IHX install & align",      3, "steelblue"),
        ("Short ISO ductwork",       3, "lightblue"),
        ("ISO pump & piping",        4, "lightblue"),
        ("Mechanical completion",    2, "steelblue"),
    ]
    pb_t = pb_start
    for name, dur, clr in pb_subs:
        b_phases.append({"name": name, "start": pb_t, "end": pb_t + dur,
                         "duration": dur, "color": clr, "critical": True, "track": "power_block"})
        pb_t += dur
    pb_end = pb_t  # 16 + 22 = 38

    # --- Propane / ACC track (parallel with power block) ---
    acc_t = pb_start  # starts same time as power block

    # ACC structure erection: 8 wk
    b_phases.append({"name": "ACC structure erection", "start": acc_t, "end": acc_t + 8,
                     "duration": 8, "color": "green", "critical": False, "track": "propane_acc"})
    # Propane piping fab: 4 wk concurrent with ACC structure
    b_phases.append({"name": "Propane piping fab", "start": acc_t, "end": acc_t + 4,
                     "duration": 4, "color": "lightgreen", "critical": False, "track": "propane_acc"})
    acc_t += 8  # after structure

    # ACC tube bundle install: 6 wk (after structure)
    b_phases.append({"name": "ACC tube bundle install", "start": acc_t, "end": acc_t + 6,
                     "duration": 6, "color": "green", "critical": False, "track": "propane_acc"})
    acc_t += 6  # wk 30

    # Propane piping erection: 4 wk (after tube bundles)
    b_phases.append({"name": "Propane piping erection", "start": acc_t, "end": acc_t + 4,
                     "duration": 4, "color": "lightgreen", "critical": False, "track": "propane_acc"})
    acc_t += 4  # wk 34

    # Propane pump install: 2 wk
    b_phases.append({"name": "Propane pump install", "start": acc_t, "end": acc_t + 2,
                     "duration": 2, "color": "lightgreen", "critical": False, "track": "propane_acc"})
    acc_t += 2  # wk 36

    parallel_end = max(pb_end, acc_t)  # max(38, 36) = 38

    # E&I: starts 4 wk before parallel ends (40% overlap with parallel), 10 wk total
    ei_overlap = 4
    ei_start_b = parallel_end - ei_overlap
    b_phases.append({"name": "E&I", "start": ei_start_b, "end": ei_start_b + 10,
                     "duration": 10, "color": "orange", "critical": False, "track": "main"})
    ei_remaining = 10 - ei_overlap  # 6 wk net contribution to critical path

    # Tie-in & integration: 3 wk after parallel
    tie_start = parallel_end
    b_phases.append({"name": "Tie-in & integration", "start": tie_start, "end": tie_start + 3,
                     "duration": 3, "color": "purple", "critical": True, "track": "main"})

    # Commissioning: after tie-in + E&I remaining (sequential on critical path)
    # Critical path: parallel(22) -> tie-in(3) -> E&I remaining(6) -> commissioning(8)
    comm_start = parallel_end + 3 + ei_remaining
    b_phases.append({"name": "Commissioning", "start": comm_start, "end": comm_start + 8,
                     "duration": 8, "color": "darkblue", "critical": True, "track": "main"})

    b_total = comm_start + 8  # 16 + 22 + 3 + 6 + 8 = 55

    savings = a_total - b_total

    # --- Config D: Dual-pressure construction ---
    # More equipment pads, 4 TG sets + 4 HX vessels, HP/LP piping in parallel
    d_phases = []
    t = 0

    # Civil & foundations (more equipment pads)
    d_phases.append({"name": "Civil & foundations", "start": t, "end": t + 18,
                     "duration": 18, "color": "gray", "critical": True, "track": "main"})
    t += 18

    # Equipment delivery (4 TG sets + 4 HX vessels)
    d_phases.append({"name": "Equipment delivery & setting", "start": t, "end": t + 14,
                     "duration": 14, "color": "steelblue", "critical": True, "track": "main"})
    t += 14

    # HP piping (parallel with LP piping — not on critical path unless longer)
    hp_pipe_start = t
    d_phases.append({"name": "HP piping & connections", "start": t, "end": t + 12,
                     "duration": 12, "color": "indianred", "critical": True, "track": "hp_piping"})

    # LP piping (parallel with HP piping)
    d_phases.append({"name": "LP piping & connections", "start": t, "end": t + 12,
                     "duration": 12, "color": "lightcoral", "critical": False, "track": "lp_piping"})
    t += 12  # max(HP, LP) = 12

    # ACC structure & tubes
    d_phases.append({"name": "ACC structure & tubes", "start": t, "end": t + 10,
                     "duration": 10, "color": "green", "critical": True, "track": "main"})

    # E&I — 50% overlap with piping + ACC
    ei_start_d = hp_pipe_start + 6
    d_phases.append({"name": "E&I", "start": ei_start_d, "end": ei_start_d + 10,
                     "duration": 10, "color": "orange", "critical": False, "track": "main"})
    t += 10  # ACC finishes

    # Commissioning
    d_phases.append({"name": "Commissioning", "start": t, "end": t + 8,
                     "duration": 8, "color": "darkblue", "critical": True, "track": "main"})
    t += 8

    d_total = t  # 18 + 14 + 12 + 10 + 8 = 62

    return {
        "config_a": {
            "total_weeks": a_total,
            "phases": a_phases,
        },
        "config_b": {
            "total_weeks": b_total,
            "phases": b_phases,
        },
        "config_d": {
            "total_weeks": d_total,
            "phases": d_phases,
        },
        "schedule_savings_weeks": savings,
        "schedule_savings_months": savings / 4.33,
        "tailpipe_diameter_in": tp_dia,
        "duct_phase_weeks": duct_weeks,
        "net_delta": -savings,  # backward compat (negative = B faster)
    }


def construction_cost_savings(sched_info, installed_cost_a, installed_cost_b, inputs):
    """
    Financial impact of Config B's shorter construction schedule.

    Three components:
    1. Overhead savings: fewer weeks of PM, CM, QC, safety, rentals
    2. Craft labor efficiency: parallel execution reduces idle/waiting time
    3. Financing savings: shorter draw period on construction loan

    Returns dict with component breakdown and total.
    """
    inp = {**_default_inputs(), **inputs}
    a_weeks = sched_info["config_a"]["total_weeks"]
    b_weeks = sched_info["config_b"]["total_weeks"]
    savings_weeks = sched_info["schedule_savings_weeks"]

    if savings_weeks <= 0:
        return {
            "overhead_savings": 0.0,
            "craft_labor_savings": 0.0,
            "financing_savings": 0.0,
            "total_construction_savings": 0.0,
            "schedule_savings_weeks": savings_weeks,
            "weekly_site_overhead": 0.0,
            "weekly_equip_rental": 0.0,
        }

    # --- 1. Overhead savings ---
    weekly_overhead = inp.get("weekly_site_overhead", 20_000)
    weekly_rental = inp.get("weekly_equip_rental", 15_000)
    overhead_savings = savings_weeks * (weekly_overhead + weekly_rental)

    # --- 2. Craft labor efficiency ---
    # Labor budget as fraction of the relevant config's installed cost
    craft_labor_pct = inp.get("craft_labor_pct", 15) / 100
    craft_labor_budget = installed_cost_a * craft_labor_pct
    waiting_pct = inp.get("craft_labor_waiting_pct", 15) / 100
    # Parallel execution reduces waiting time proportionally to schedule compression
    craft_labor_savings = craft_labor_budget * waiting_pct * (savings_weeks / a_weeks) if a_weeks > 0 else 0.0

    # --- 3. Financing savings ---
    loan_pct = inp.get("construction_loan_pct", 60) / 100
    loan_rate = inp.get("construction_loan_rate", 7) / 100
    # Use average of both configs' installed cost for loan sizing
    avg_installed = (installed_cost_a + installed_cost_b) / 2
    loan_amount = avg_installed * loan_pct
    financing_savings = loan_amount * loan_rate * (savings_weeks / 52)

    total = overhead_savings + craft_labor_savings + financing_savings

    return {
        "overhead_savings": overhead_savings,
        "craft_labor_savings": craft_labor_savings,
        "financing_savings": financing_savings,
        "total_construction_savings": total,
        "schedule_savings_weeks": savings_weeks,
        "weekly_site_overhead": weekly_overhead,
        "weekly_equip_rental": weekly_rental,
    }


# Backward-compatible alias
construction_schedule_delta = construction_schedule


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
    weeks_saved = sched_info["schedule_savings_weeks"]  # positive = B is faster
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

    # Solve Config A once for reference (with full power balance)
    try:
        result_a = solve_config_a(inputs, fp)
        costs_a = calculate_costs_a(
            result_a["states"], result_a["performance"], inputs,
            result_a.get("duct"),
        )
        fan_a = calculate_fan_power(
            result_a["performance"]["Q_reject_mmbtu_hr"],
            inputs.get("T_ambient", 95), inputs)
        pwr_a = compute_power_balance(result_a["performance"], fan_a, inputs, config="A")
        lc_a = lifecycle_cost(costs_a["total_installed"], pwr_a["P_net"], inputs)
        ref_power_a = pwr_a["P_net"]
        ref_cost_a = costs_a["total_installed"]
        ref_lcoe_a = lc_a["lcoe"]
    except Exception:
        ref_power_a = 0
        ref_cost_a = 0
        ref_lcoe_a = 0

    sweep_dts = np.linspace(5, 25, 21)
    sweep_results = []

    def _solve_b_with_balance(inp):
        """Solve Config B and return full power balance net power."""
        result = solve_config_b(inp, fp)
        ps = result["propane_states"]
        Q_rej_b = result["performance"]["m_dot_prop"] * (ps["A"].h - ps["B"].h) / 1e6
        fan_b = calculate_fan_power(Q_rej_b, inp.get("T_ambient", 95), inp)
        pwr_b = compute_power_balance(result["performance"], fan_b, inp, config="B")
        return result, pwr_b

    def objective(dt):
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result, pwr_b = _solve_b_with_balance(inp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp, result.get("duct"),
            )
            lc = lifecycle_cost(costs["total_installed"], pwr_b["P_net"], inp)
            return -lc["net_npv"]
        except Exception:
            return 1e12

    for dt in sweep_dts:
        try:
            inp = {**inputs, "dt_approach_intermediate": float(dt)}
            result, pwr_b = _solve_b_with_balance(inp)
            costs = calculate_costs_b(
                result["states"], result["propane_states"],
                result["performance"], inp, result.get("duct"),
            )
            lc = lifecycle_cost(costs["total_installed"], pwr_b["P_net"], inp)
            sweep_results.append({
                "dt_approach": float(dt),
                "net_power_kw": pwr_b["P_net"],
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


def hx_area_vs_pinch(Q_mmbtu_hr, U, T_hot_in, T_hot_out, T_cold_in, T_cold_out,
                     pinch_range=None, cost_per_ft2=100):
    """
    Sweep pinch temperature for a heat exchanger and return area + cost curve.

    Q in MMBtu/hr, U in BTU/(hr·ft2·°F).

    The exchanger has fixed duty Q.  As pinch tightens, LMTD shrinks so area
    grows.  We model the HX as counter-flow with the given terminal temps as
    baseline, then scale LMTD proportionally to the pinch variation.

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

        area = _hx_area(Q_mmbtu_hr, lmtd_val, U) if lmtd_val > 0 else 0
        cost = area * cost_per_ft2
        results.append({
            "pinch": float(pinch),
            "lmtd": lmtd_val,
            "area_ft2": area,
            "cost": cost,
        })

    return results


def acc_area_vs_pinch(Q_reject_mmbtu_hr, T_ambient, pinch_range=None,
                      cost_per_ft2=None):
    """
    ACC area and cost vs ACC pinch (condensing approach to ambient).
    Q in MMBtu/hr.
    """
    if pinch_range is None:
        pinch_range = np.linspace(5, 25, 21)
    if cost_per_ft2 is None:
        cost_per_ft2 = COST_FACTORS["acc_per_ft2"]

    results = []
    for pinch in pinch_range:
        dT = max(pinch, 0.1)
        area = Q_reject_mmbtu_hr * 1e6 / (U_VALUES["acc"] * dT)
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
                fan = calculate_fan_power(
                    r["performance"]["Q_reject_mmbtu_hr"],
                    inp_mod.get("T_ambient", 95), inp_mod)
                pwr = compute_power_balance(r["performance"], fan, inp_mod, config="A")
            else:
                r = solve_config_b(inp_mod, fp)
                c = calculate_costs_b(r["states"], r["propane_states"],
                                      r["performance"], inp_mod, r.get("duct"))
                ps = r["propane_states"]
                Q_rej = r["performance"]["m_dot_prop"] * (ps["A"].h - ps["B"].h) / 1e6
                fan = calculate_fan_power(Q_rej, inp_mod.get("T_ambient", 95), inp_mod)
                pwr = compute_power_balance(r["performance"], fan, inp_mod, config="B")
            results.append({
                "multiplier": float(mult),
                "total_cost": c["total_installed"],
                "net_power_kw": pwr["P_net"],
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
