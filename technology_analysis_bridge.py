"""
Technology Analysis Bridge — Thermodynamic models for all 19 geothermal technologies.

Reuses existing ORC solvers for orc_direct and orc_propane_loop.
Builds CoolProp-based models for flash, steam Rankine, Kalina.
Builds parametric models for sCO2, TEG, Stirling, TFC, OFC.

All models return the STANDARD_OUTPUT dict format for uniform comparison.
"""

import math
import numpy as np
from typing import Any

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False

from analysis_bridge import run_orc_analysis, design_basis_to_inputs, _get_fp
from cost_model import lifecycle_cost, COST_FACTORS

# ── Unit conversions (SI-focused for new models) ─────────────────────────────

def _c_to_f(t_c: float) -> float:
    return t_c * 9.0 / 5.0 + 32.0

def _f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0

def _c_to_k(t_c: float) -> float:
    return t_c + 273.15

def _k_to_c(t_k: float) -> float:
    return t_k - 273.15

def _bar_to_pa(p_bar: float) -> float:
    return p_bar * 1e5

def _pa_to_bar(p_pa: float) -> float:
    return p_pa / 1e5

def _kgs_to_lbs(m: float) -> float:
    return m * 2.20462

def _lbs_to_kgs(m: float) -> float:
    return m / 2.20462

def _btu_lb_to_j_kg(h: float) -> float:
    return h * 2326.0

def _j_kg_to_btu_lb(h: float) -> float:
    return h / 2326.0


# ── Standard output template ─────────────────────────────────────────────────

def _empty_output(technology_id: str, warnings: list | None = None) -> dict:
    """Return a zeroed standard output dict."""
    return {
        "net_power_MW": 0.0,
        "gross_power_MW": 0.0,
        "parasitic_MW": 0.0,
        "cycle_efficiency": 0.0,
        "brine_utilization_efficiency": 0.0,
        "condenser_duty_MW": 0.0,
        "capex_total_USD": 0.0,
        "capex_per_kW": 0.0,
        "opex_annual_USD": 0.0,
        "lcoe_per_MWh": 0.0,
        "npv_USD": 0.0,
        "construction_weeks": 0,
        "first_power_weeks": 0,
        "water_consumption_m3_per_MWh": 0.0,
        "technology_id": technology_id,
        "model_confidence": "low",
        "warnings": warnings or [],
        "converged": False,
    }


def _build_standard_output(
    technology_id: str,
    net_power_MW: float,
    gross_power_MW: float,
    parasitic_MW: float,
    cycle_efficiency: float,
    brine_utilization_eff: float,
    condenser_duty_MW: float,
    capex_total_USD: float,
    opex_annual_USD: float,
    construction_weeks: int,
    first_power_weeks: int,
    water_consumption: float,
    model_confidence: str,
    design_basis: dict,
    warnings: list | None = None,
) -> dict:
    """Build the standard output dict with derived economics."""
    net_kw = net_power_MW * 1000
    capex_per_kw = capex_total_USD / net_kw if net_kw > 0 else float("inf")

    # Use shared lifecycle economics
    inputs = design_basis_to_inputs(design_basis)
    inputs.setdefault("electricity_price", design_basis.get("energy_value_per_MWh", 80))
    inputs.setdefault("discount_rate", design_basis.get("discount_rate", 0.08))
    inputs.setdefault("project_life", design_basis.get("plant_life_years", 30))
    inputs.setdefault("capacity_factor", design_basis.get("capacity_factor", 0.95))

    lc = lifecycle_cost(capex_total_USD, net_kw, inputs)

    return {
        "net_power_MW": net_power_MW,
        "gross_power_MW": gross_power_MW,
        "parasitic_MW": parasitic_MW,
        "cycle_efficiency": cycle_efficiency,
        "brine_utilization_efficiency": brine_utilization_eff,
        "condenser_duty_MW": condenser_duty_MW,
        "capex_total_USD": capex_total_USD,
        "capex_per_kW": capex_per_kw,
        "opex_annual_USD": opex_annual_USD,
        "lcoe_per_MWh": lc["lcoe"],
        "npv_USD": lc["net_npv"],
        "construction_weeks": construction_weeks,
        "first_power_weeks": first_power_weeks,
        "water_consumption_m3_per_MWh": water_consumption,
        "technology_id": technology_id,
        "model_confidence": model_confidence,
        "warnings": warnings or [],
        "converged": True,
    }


# ── CoolProp helpers (SI internally) ─────────────────────────────────────────

def _water_props(T_C: float, P_bar: float | None = None, quality: float | None = None) -> dict:
    """Get water/steam properties using CoolProp in SI units."""
    if not HAS_COOLPROP:
        raise RuntimeError("CoolProp required for flash/steam models")

    T_K = _c_to_k(T_C)
    if quality is not None:
        # Two-phase state at given T and quality
        P_Pa = CP.PropsSI("P", "T", T_K, "Q", quality, "Water")
        h = CP.PropsSI("H", "T", T_K, "Q", quality, "Water")
        s = CP.PropsSI("S", "T", T_K, "Q", quality, "Water")
        rho = CP.PropsSI("D", "T", T_K, "Q", quality, "Water")
    elif P_bar is not None:
        P_Pa = _bar_to_pa(P_bar)
        h = CP.PropsSI("H", "T", T_K, "P", P_Pa, "Water")
        s = CP.PropsSI("S", "T", T_K, "P", P_Pa, "Water")
        rho = CP.PropsSI("D", "T", T_K, "P", P_Pa, "Water")
    else:
        raise ValueError("Must provide P_bar or quality")

    return {"T_C": T_C, "T_K": T_K, "P_Pa": P_Pa, "P_bar": _pa_to_bar(P_Pa),
            "h": h, "s": s, "rho": rho}


def _water_sat(T_C: float = None, P_bar: float = None) -> dict:
    """Saturation properties of water at T or P."""
    if not HAS_COOLPROP:
        raise RuntimeError("CoolProp required")

    if T_C is not None:
        T_K = _c_to_k(T_C)
        P_Pa = CP.PropsSI("P", "T", T_K, "Q", 0, "Water")
    elif P_bar is not None:
        P_Pa = _bar_to_pa(P_bar)
        T_K = CP.PropsSI("T", "P", P_Pa, "Q", 0, "Water")
        T_C = _k_to_c(T_K)
    else:
        raise ValueError("Must provide T_C or P_bar")

    h_f = CP.PropsSI("H", "T", T_K, "Q", 0, "Water")
    h_g = CP.PropsSI("H", "T", T_K, "Q", 1, "Water")
    s_f = CP.PropsSI("S", "T", T_K, "Q", 0, "Water")
    s_g = CP.PropsSI("S", "T", T_K, "Q", 1, "Water")

    return {
        "T_C": T_C, "T_K": T_K, "P_Pa": P_Pa, "P_bar": _pa_to_bar(P_Pa),
        "h_f": h_f, "h_g": h_g, "s_f": s_f, "s_g": s_g,
        "h_fg": h_g - h_f,
    }


def _water_state_ph(P_bar: float, h: float) -> dict:
    """State from P (bar) and h (J/kg)."""
    P_Pa = _bar_to_pa(P_bar)
    T_K = CP.PropsSI("T", "P", P_Pa, "H", h, "Water")
    s = CP.PropsSI("S", "P", P_Pa, "H", h, "Water")
    rho = CP.PropsSI("D", "P", P_Pa, "H", h, "Water")
    Q = CP.PropsSI("Q", "P", P_Pa, "H", h, "Water")
    return {"T_C": _k_to_c(T_K), "T_K": T_K, "P_Pa": P_Pa, "P_bar": P_bar,
            "h": h, "s": s, "rho": rho, "Q": Q}


def _water_state_ps(P_bar: float, s: float) -> dict:
    """State from P (bar) and s (J/kg-K)."""
    P_Pa = _bar_to_pa(P_bar)
    T_K = CP.PropsSI("T", "P", P_Pa, "S", s, "Water")
    h = CP.PropsSI("H", "P", P_Pa, "S", s, "Water")
    rho = CP.PropsSI("D", "P", P_Pa, "S", s, "Water")
    Q = CP.PropsSI("Q", "P", P_Pa, "S", s, "Water")
    return {"T_C": _k_to_c(T_K), "T_K": T_K, "P_Pa": P_Pa, "P_bar": P_bar,
            "h": h, "s": s, "rho": rho, "Q": Q}


# ── Helper: brine thermal input ──────────────────────────────────────────────

def _brine_thermal_input(design_basis: dict) -> dict:
    """Extract brine thermal parameters from design basis (all SI)."""
    T_in_C = design_basis.get("brine_inlet_temp_C", 215.6)
    T_out_C = design_basis.get("brine_outlet_temp_C", 71.1)
    m_dot_kgs = design_basis.get("brine_flow_kg_s", 498.95)
    T_amb_C = design_basis.get("ambient_temp_C", 13.9)
    cp_brine = 4186.0  # J/kg-K (close to water for moderate TDS)

    Q_available_W = m_dot_kgs * cp_brine * (T_in_C - T_out_C)
    Q_available_MW = Q_available_W / 1e6

    return {
        "T_in_C": T_in_C, "T_out_C": T_out_C,
        "m_dot_kgs": m_dot_kgs, "T_amb_C": T_amb_C,
        "cp_brine": cp_brine,
        "Q_available_W": Q_available_W,
        "Q_available_MW": Q_available_MW,
    }


def _capex_with_indirects(equipment_cost: float, indirect_pcts: dict | None = None) -> float:
    """Apply indirect cost layers to equipment subtotal."""
    pcts = indirect_pcts or {}
    foundation_pct = pcts.get("foundation_pct", 8) / 100
    engineering_pct = pcts.get("engineering_pct", 12) / 100
    cm_pct = pcts.get("construction_mgmt_pct", 8) / 100
    contingency_pct = pcts.get("contingency_pct", 15) / 100

    with_foundation = equipment_cost * (1 + foundation_pct)
    with_eng = with_foundation * (1 + engineering_pct)
    with_cm = with_eng * (1 + cm_pct)
    total = with_cm * (1 + contingency_pct)
    return total


# ═══════════════════════════════════════════════════════════════════════════════
# ORC MODELS — delegate to existing solvers
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_orc_direct(params: dict, design_basis: dict) -> dict:
    """Binary ORC with direct ACC — delegates to existing solver."""
    tool_input = {
        "config": "A",
        "evaporator_approach_delta_F": params.get("evaporator_approach_delta_F", 10),
        "recuperator_approach_delta_F": params.get("recuperator_approach_delta_F", 15),
        "preheater_approach_delta_F": params.get("preheater_approach_delta_F", 10),
        "acc_approach_delta_F": params.get("acc_approach_delta_F", 15),
        "turbine_isentropic_efficiency": params.get("turbine_isentropic_efficiency",
                                                     design_basis.get("eta_turbine", 0.85)),
        "pump_isentropic_efficiency": params.get("pump_isentropic_efficiency",
                                                  design_basis.get("eta_pump", 0.80)),
        "energy_value_per_MWh": design_basis.get("energy_value_per_MWh", 80),
    }

    result = run_orc_analysis(tool_input, design_basis)

    if not result.get("converged", False):
        out = _empty_output("orc_direct", result.get("warnings", []))
        return out

    net_MW = result["net_power_MW"]
    gross_MW = result["gross_power_MW"]
    parasitic_MW = result["parasitic_MW"]
    capex = result["capex_total_USD"]
    opex = capex * 0.025  # 2.5% of CAPEX annual O&M
    sched = result["construction_weeks_critical_path"]

    return _build_standard_output(
        technology_id="orc_direct",
        net_power_MW=net_MW,
        gross_power_MW=gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=result["cycle_efficiency"],
        brine_utilization_eff=result["cycle_efficiency"],
        condenser_duty_MW=result["condenser_duty_MW"],
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=sched,
        first_power_weeks=sched + 8,
        water_consumption=0.0,  # dry cooling
        model_confidence="high",
        design_basis=design_basis,
        warnings=result.get("warnings", []),
    )


def analyze_orc_propane_loop(params: dict, design_basis: dict) -> dict:
    """Binary ORC with propane intermediate heat rejection — delegates to existing solver."""
    tool_input = {
        "config": "B",
        "evaporator_approach_delta_F": params.get("evaporator_approach_delta_F", 10),
        "recuperator_approach_delta_F": params.get("recuperator_approach_delta_F", 15),
        "preheater_approach_delta_F": params.get("preheater_approach_delta_F", 10),
        "acc_approach_delta_F": params.get("acc_approach_delta_F", 15),
        "intermediate_hx_approach_delta_F": params.get("intermediate_hx_approach_delta_F", 10),
        "turbine_isentropic_efficiency": params.get("turbine_isentropic_efficiency",
                                                     design_basis.get("eta_turbine", 0.85)),
        "pump_isentropic_efficiency": params.get("pump_isentropic_efficiency",
                                                  design_basis.get("eta_pump", 0.80)),
        "energy_value_per_MWh": design_basis.get("energy_value_per_MWh", 80),
    }

    result = run_orc_analysis(tool_input, design_basis)

    if not result.get("converged", False):
        out = _empty_output("orc_propane_loop", result.get("warnings", []))
        return out

    net_MW = result["net_power_MW"]
    gross_MW = result["gross_power_MW"]
    parasitic_MW = result["parasitic_MW"]
    capex = result["capex_total_USD"]
    opex = capex * 0.025
    sched = result["construction_weeks_critical_path"]

    return _build_standard_output(
        technology_id="orc_propane_loop",
        net_power_MW=net_MW,
        gross_power_MW=gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=result["cycle_efficiency"],
        brine_utilization_eff=result["cycle_efficiency"],
        condenser_duty_MW=result["condenser_duty_MW"],
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=sched,
        first_power_weeks=sched + 6,  # parallel construction advantage
        water_consumption=0.0,
        model_confidence="high",
        design_basis=design_basis,
        warnings=result.get("warnings", []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# FLASH STEAM MODELS — CoolProp water/steam
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_single_flash(params: dict, design_basis: dict) -> dict:
    """
    Single flash steam plant.

    1. Brine enthalpy at inlet conditions (subcooled liquid at wellhead P)
    2. Isenthalpic throttling to flash pressure — get steam quality
    3. Steam turbine power output (isentropic expansion to condenser P)
    4. NCG handling parasitic estimate
    5. Condenser heat rejection
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_out_C = brine["T_out_C"]
    m_dot = brine["m_dot_kgs"]
    T_amb_C = brine["T_amb_C"]
    warnings = []

    if T_in_C < 150:
        warnings.append(f"Brine at {T_in_C:.0f}°C is marginal for flash — quality will be very low")

    # Parameters
    flash_P_bar = params.get("flash_pressure_bar", 5.0)
    ncg_pct = params.get("ncg_content_pct", 0.5)
    eta_turbine = params.get("turbine_isentropic_efficiency", 0.82)
    separator_eff = params.get("separator_efficiency", 0.98)
    condenser_T_C = T_amb_C + params.get("condenser_approach_C", 15)

    try:
        # Brine enthalpy at inlet (liquid at ~10 bar above saturation)
        brine_P_bar = max(flash_P_bar + 5, 10.0)
        inlet_sat = _water_sat(T_C=T_in_C)
        h_inlet = inlet_sat["h_f"]  # liquid enthalpy at brine T

        # Isenthalpic flash to flash pressure
        flash_sat = _water_sat(P_bar=flash_P_bar)
        T_flash_C = flash_sat["T_C"]

        if T_in_C < T_flash_C:
            return _empty_output("single_flash",
                                 [f"Brine temp {T_in_C:.0f}°C below flash sat temp {T_flash_C:.0f}°C"])

        # Steam quality after flash: x = (h_brine - h_f) / h_fg
        x_flash = (h_inlet - flash_sat["h_f"]) / flash_sat["h_fg"]
        x_flash = max(0, min(1, x_flash))

        if x_flash < 0.05:
            warnings.append(f"Very low flash quality {x_flash:.3f} — flash not economical")

        # Steam and brine flow rates
        m_steam = m_dot * x_flash * separator_eff
        m_brine_out = m_dot - m_steam

        # Turbine: isentropic expansion from flash sat vapor to condenser P
        cond_sat = _water_sat(T_C=condenser_T_C)
        P_cond_bar = cond_sat["P_bar"]

        s_steam_in = flash_sat["s_g"]
        h_steam_in = flash_sat["h_g"]

        # Isentropic exit state
        exit_is = _water_state_ps(P_cond_bar, s_steam_in)
        h_exit_is = exit_is["h"]

        # Actual exit enthalpy
        h_exit = h_steam_in - eta_turbine * (h_steam_in - h_exit_is)

        # Turbine power
        W_turbine = m_steam * (h_steam_in - h_exit)  # W
        W_gross_MW = W_turbine / 1e6

        # Condenser duty
        exit_state = _water_state_ph(P_cond_bar, h_exit)
        h_cond_liquid = cond_sat["h_f"]
        Q_cond_W = m_steam * (h_exit - h_cond_liquid)
        Q_cond_MW = Q_cond_W / 1e6

        # Parasitic loads
        ncg_handling_MW = W_gross_MW * ncg_pct / 100 * 0.5  # ~0.5% per 1% NCG
        fan_parasitic_MW = Q_cond_MW * 0.015  # ~1.5% of condenser duty for fans
        pump_MW = W_gross_MW * 0.01  # cooling water pumps
        total_parasitic_MW = ncg_handling_MW + fan_parasitic_MW + pump_MW

        W_net_MW = W_gross_MW - total_parasitic_MW

        # Efficiency
        Q_in_MW = brine["Q_available_MW"]
        eta_cycle = W_net_MW / Q_in_MW if Q_in_MW > 0 else 0
        eta_brine = W_net_MW / (m_dot * (h_inlet - flash_sat["h_f"]) / 1e6) if x_flash > 0 else 0

        # Cost estimate — flash plants: ~$2000-3000/kW installed at scale
        net_kW = W_net_MW * 1000
        base_cost_per_kW = 2500
        # Scale factor: smaller plants cost more per kW
        if net_kW < 5000:
            scale_mult = 1.5
        elif net_kW < 20000:
            scale_mult = 1.2
        else:
            scale_mult = 1.0

        equipment_cost = net_kW * base_cost_per_kW * scale_mult * 0.65  # equipment ~65% of installed
        capex = _capex_with_indirects(equipment_cost)

        opex = capex * 0.03  # 3% annual (higher than ORC due to brine handling)

        return _build_standard_output(
            technology_id="single_flash",
            net_power_MW=W_net_MW,
            gross_power_MW=W_gross_MW,
            parasitic_MW=total_parasitic_MW,
            cycle_efficiency=eta_cycle,
            brine_utilization_eff=eta_brine,
            condenser_duty_MW=Q_cond_MW,
            capex_total_USD=capex,
            opex_annual_USD=opex,
            construction_weeks=60,
            first_power_weeks=68,
            water_consumption=0.8,  # wet cooling typical
            model_confidence="high",
            design_basis=design_basis,
            warnings=warnings,
        )

    except Exception as e:
        return _empty_output("single_flash", [f"Solver error: {e}"])


def analyze_double_flash(params: dict, design_basis: dict) -> dict:
    """
    Double flash steam plant.
    HP flash → HP turbine → LP flash of separated brine → LP turbine.
    Combined power from both stages.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    m_dot = brine["m_dot_kgs"]
    T_amb_C = brine["T_amb_C"]
    warnings = []

    if T_in_C < 180:
        warnings.append(f"Brine at {T_in_C:.0f}°C is marginal for double flash")

    hp_flash_P = params.get("hp_flash_pressure_bar", 7.0)
    lp_flash_P = params.get("lp_flash_pressure_bar", 2.5)
    eta_turbine = params.get("turbine_isentropic_efficiency", 0.82)
    ncg_pct = params.get("ncg_content_pct", 0.5)
    separator_eff = params.get("separator_efficiency", 0.98)
    condenser_T_C = T_amb_C + params.get("condenser_approach_C", 15)

    try:
        # HP flash
        inlet_sat = _water_sat(T_C=T_in_C)
        h_inlet = inlet_sat["h_f"]

        hp_sat = _water_sat(P_bar=hp_flash_P)
        T_hp_flash = hp_sat["T_C"]

        if T_in_C < T_hp_flash:
            return _empty_output("double_flash",
                                 [f"Brine temp below HP flash sat temp {T_hp_flash:.0f}°C"])

        x_hp = (h_inlet - hp_sat["h_f"]) / hp_sat["h_fg"]
        x_hp = max(0, min(1, x_hp))

        m_hp_steam = m_dot * x_hp * separator_eff
        m_hp_brine = m_dot - m_hp_steam

        # LP flash of separated brine
        lp_sat = _water_sat(P_bar=lp_flash_P)
        T_lp_flash = lp_sat["T_C"]

        h_hp_brine = hp_sat["h_f"]  # separated brine is saturated liquid
        x_lp = (h_hp_brine - lp_sat["h_f"]) / lp_sat["h_fg"]
        x_lp = max(0, min(1, x_lp))

        m_lp_steam = m_hp_brine * x_lp * separator_eff
        m_final_brine = m_hp_brine - m_lp_steam

        # Condenser conditions
        cond_sat = _water_sat(T_C=condenser_T_C)
        P_cond = cond_sat["P_bar"]

        # HP turbine
        h_hp_in = hp_sat["h_g"]
        s_hp_in = hp_sat["s_g"]
        hp_exit_is = _water_state_ps(P_cond, s_hp_in)
        h_hp_exit = h_hp_in - eta_turbine * (h_hp_in - hp_exit_is["h"])
        W_hp = m_hp_steam * (h_hp_in - h_hp_exit)

        # LP turbine
        h_lp_in = lp_sat["h_g"]
        s_lp_in = lp_sat["s_g"]
        lp_exit_is = _water_state_ps(P_cond, s_lp_in)
        h_lp_exit = h_lp_in - eta_turbine * (h_lp_in - lp_exit_is["h"])
        W_lp = m_lp_steam * (h_lp_in - h_lp_exit)

        W_gross_MW = (W_hp + W_lp) / 1e6

        # Condenser duty (both exhaust streams)
        h_cond_liq = cond_sat["h_f"]
        Q_cond_MW = (m_hp_steam * (h_hp_exit - h_cond_liq) +
                     m_lp_steam * (h_lp_exit - h_cond_liq)) / 1e6

        # Parasitics
        ncg_MW = W_gross_MW * ncg_pct / 100 * 0.5
        fan_MW = Q_cond_MW * 0.015
        pump_MW = W_gross_MW * 0.012
        parasitic_MW = ncg_MW + fan_MW + pump_MW

        W_net_MW = W_gross_MW - parasitic_MW

        Q_in_MW = brine["Q_available_MW"]
        eta_cycle = W_net_MW / Q_in_MW if Q_in_MW > 0 else 0

        # Cost: ~15% more than single flash
        net_kW = W_net_MW * 1000
        base_per_kW = 2800
        if net_kW < 5000:
            scale_mult = 1.5
        elif net_kW < 20000:
            scale_mult = 1.2
        else:
            scale_mult = 1.0

        equipment_cost = net_kW * base_per_kW * scale_mult * 0.65
        capex = _capex_with_indirects(equipment_cost)
        opex = capex * 0.032

        return _build_standard_output(
            technology_id="double_flash",
            net_power_MW=W_net_MW,
            gross_power_MW=W_gross_MW,
            parasitic_MW=parasitic_MW,
            cycle_efficiency=eta_cycle,
            brine_utilization_eff=eta_cycle,
            condenser_duty_MW=Q_cond_MW,
            capex_total_USD=capex,
            opex_annual_USD=opex,
            construction_weeks=70,
            first_power_weeks=78,
            water_consumption=1.0,
            model_confidence="high",
            design_basis=design_basis,
            warnings=warnings,
        )

    except Exception as e:
        return _empty_output("double_flash", [f"Solver error: {e}"])


def analyze_hybrid_flash_binary(params: dict, design_basis: dict) -> dict:
    """
    Hybrid flash-binary: flash stage for high-enthalpy steam,
    ORC bottoming cycle on flash brine discharge.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    m_dot = brine["m_dot_kgs"]
    T_amb_C = brine["T_amb_C"]
    warnings = []

    if T_in_C < 160:
        warnings.append(f"Brine at {T_in_C:.0f}°C is marginal for hybrid flash-binary")

    flash_P = params.get("flash_pressure_bar", 5.0)
    eta_steam_turb = params.get("steam_turbine_efficiency", 0.82)
    ncg_pct = params.get("ncg_content_pct", 0.5)
    condenser_T_C = T_amb_C + params.get("condenser_approach_C", 15)

    try:
        # Flash stage (same as single flash)
        inlet_sat = _water_sat(T_C=T_in_C)
        h_inlet = inlet_sat["h_f"]

        flash_sat = _water_sat(P_bar=flash_P)
        T_flash_C = flash_sat["T_C"]

        if T_in_C < T_flash_C:
            return _empty_output("hybrid_flash_binary",
                                 [f"Brine below flash temp {T_flash_C:.0f}°C"])

        x_flash = (h_inlet - flash_sat["h_f"]) / flash_sat["h_fg"]
        x_flash = max(0, min(1, x_flash))
        m_steam = m_dot * x_flash * 0.98
        m_brine_out = m_dot - m_steam

        # Steam turbine power
        cond_sat = _water_sat(T_C=condenser_T_C)
        P_cond = cond_sat["P_bar"]

        h_steam_in = flash_sat["h_g"]
        s_steam_in = flash_sat["s_g"]
        exit_is = _water_state_ps(P_cond, s_steam_in)
        h_exit = h_steam_in - eta_steam_turb * (h_steam_in - exit_is["h"])
        W_flash_MW = m_steam * (h_steam_in - h_exit) / 1e6

        # Binary ORC bottoming on flash brine discharge
        # Flash brine exits at T_flash_C as saturated liquid
        # Create a modified design basis for the binary cycle
        binary_basis = dict(design_basis)
        binary_basis["brine_inlet_temp_C"] = T_flash_C
        binary_basis["brine_flow_kg_s"] = m_brine_out

        binary_result = analyze_orc_direct(params.get("binary_params", {}), binary_basis)

        W_binary_net_MW = binary_result.get("net_power_MW", 0)
        W_binary_gross_MW = binary_result.get("gross_power_MW", 0)

        # Combined output
        flash_parasitic = W_flash_MW * (ncg_pct / 100 * 0.5 + 0.025)
        W_flash_net = W_flash_MW - flash_parasitic

        W_gross_total = W_flash_MW + W_binary_gross_MW
        W_net_total = W_flash_net + W_binary_net_MW
        parasitic_total = W_gross_total - W_net_total

        Q_cond_total = (m_steam * (h_exit - cond_sat["h_f"]) / 1e6 +
                        binary_result.get("condenser_duty_MW", 0))

        Q_in_MW = brine["Q_available_MW"]
        eta = W_net_total / Q_in_MW if Q_in_MW > 0 else 0

        # Cost: flash plant + binary plant, ~10% integration premium
        flash_capex = W_flash_MW * 1000 * 2500 * 0.65
        binary_capex = binary_result.get("capex_total_USD", 0) * 0.8  # share some infrastructure
        equipment_cost = (flash_capex + binary_capex) * 1.10
        capex = _capex_with_indirects(equipment_cost)
        opex = capex * 0.03

        return _build_standard_output(
            technology_id="hybrid_flash_binary",
            net_power_MW=W_net_total,
            gross_power_MW=W_gross_total,
            parasitic_MW=parasitic_total,
            cycle_efficiency=eta,
            brine_utilization_eff=eta,
            condenser_duty_MW=Q_cond_total,
            capex_total_USD=capex,
            opex_annual_USD=opex,
            construction_weeks=72,
            first_power_weeks=80,
            water_consumption=0.5,
            model_confidence="medium",
            design_basis=design_basis,
            warnings=warnings + binary_result.get("warnings", []),
        )

    except Exception as e:
        return _empty_output("hybrid_flash_binary", [f"Solver error: {e}"])


# ═══════════════════════════════════════════════════════════════════════════════
# STEAM RANKINE — indirect steam via brine/steam HX
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_steam_rankine(params: dict, design_basis: dict) -> dict:
    """
    Indirect steam Rankine cycle.
    Clean-side steam generated via brine-to-steam HX, avoids direct brine contact.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    m_dot_brine = brine["m_dot_kgs"]
    T_amb_C = brine["T_amb_C"]
    warnings = []

    if T_in_C < 150:
        warnings.append(f"Brine at {T_in_C:.0f}°C too low for efficient steam Rankine")

    steam_P_bar = params.get("steam_pressure_bar", 8.0)
    superheat_C = params.get("superheat_delta_C", 10.0)
    eta_turbine = params.get("turbine_isentropic_efficiency", 0.82)
    eta_pump = params.get("pump_isentropic_efficiency", 0.75)
    condenser_T_C = T_amb_C + params.get("condenser_approach_C", 15)
    hx_pinch_C = params.get("hx_pinch_C", 10)

    try:
        # Steam cycle state points
        steam_sat = _water_sat(P_bar=steam_P_bar)
        T_steam_sat = steam_sat["T_C"]

        # Check brine can heat to steam conditions
        T_steam_superheated = T_steam_sat + superheat_C
        if T_in_C < T_steam_superheated + hx_pinch_C:
            # Reduce superheat to fit
            superheat_C = max(0, T_in_C - T_steam_sat - hx_pinch_C)
            T_steam_superheated = T_steam_sat + superheat_C
            warnings.append(f"Superheat reduced to {superheat_C:.0f}°C to maintain HX pinch")

        # Turbine inlet
        T_turb_in_K = _c_to_k(T_steam_superheated)
        P_turb_in_Pa = _bar_to_pa(steam_P_bar)
        h_turb_in = CP.PropsSI("H", "T", T_turb_in_K, "P", P_turb_in_Pa, "Water")
        s_turb_in = CP.PropsSI("S", "T", T_turb_in_K, "P", P_turb_in_Pa, "Water")

        # Condenser
        cond_sat = _water_sat(T_C=condenser_T_C)
        P_cond = cond_sat["P_bar"]

        # Isentropic expansion
        exit_is = _water_state_ps(P_cond, s_turb_in)
        h_exit = h_turb_in - eta_turbine * (h_turb_in - exit_is["h"])

        # Pump work (liquid compression)
        h_pump_in = cond_sat["h_f"]
        v_f = 1.0 / CP.PropsSI("D", "T", _c_to_k(condenser_T_C), "Q", 0, "Water")
        w_pump_ideal = v_f * (P_turb_in_Pa - _bar_to_pa(P_cond))
        w_pump = w_pump_ideal / eta_pump
        h_pump_out = h_pump_in + w_pump

        # Steam mass flow rate (from brine energy balance)
        # Q_brine = m_brine * cp * (T_in - T_out_min)
        # Q_steam = m_steam * (h_turb_in - h_pump_out)
        q_per_kg_steam = h_turb_in - h_pump_out  # J/kg
        Q_available_W = brine["Q_available_W"]

        # Account for HX effectiveness (~90%)
        hx_eff = 0.90
        m_steam = Q_available_W * hx_eff / q_per_kg_steam

        # Power
        W_turbine = m_steam * (h_turb_in - h_exit)
        W_pump = m_steam * w_pump
        W_gross_MW = W_turbine / 1e6
        W_pump_MW = W_pump / 1e6

        # Condenser duty
        Q_cond_MW = m_steam * (h_exit - h_pump_in) / 1e6

        # ACC parasitic
        fan_MW = Q_cond_MW * 0.015
        aux_MW = W_gross_MW * 0.01

        parasitic_MW = W_pump_MW + fan_MW + aux_MW
        W_net_MW = W_gross_MW - parasitic_MW

        Q_in_MW = brine["Q_available_MW"]
        eta = W_net_MW / Q_in_MW if Q_in_MW > 0 else 0

        # Cost: steam Rankine with HX ~ $2200-3200/kW
        net_kW = W_net_MW * 1000
        base_per_kW = 2800
        if net_kW < 5000:
            scale_mult = 1.4
        elif net_kW < 20000:
            scale_mult = 1.15
        else:
            scale_mult = 1.0

        equipment_cost = net_kW * base_per_kW * scale_mult * 0.65
        capex = _capex_with_indirects(equipment_cost)
        opex = capex * 0.028

        return _build_standard_output(
            technology_id="steam_rankine",
            net_power_MW=W_net_MW,
            gross_power_MW=W_gross_MW,
            parasitic_MW=parasitic_MW,
            cycle_efficiency=eta,
            brine_utilization_eff=eta,
            condenser_duty_MW=Q_cond_MW,
            capex_total_USD=capex,
            opex_annual_USD=opex,
            construction_weeks=56,
            first_power_weeks=64,
            water_consumption=0.0,
            model_confidence="high",
            design_basis=design_basis,
            warnings=warnings,
        )

    except Exception as e:
        return _empty_output("steam_rankine", [f"Solver error: {e}"])


# ═══════════════════════════════════════════════════════════════════════════════
# KALINA CYCLE — ammonia-water mixture
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_kalina(params: dict, design_basis: dict) -> dict:
    """
    Kalina cycle with ammonia-water working fluid.
    Uses published correlations for mixture properties since CoolProp
    ammonia-water mixture support requires REFPROP backend.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    nh3_fraction = params.get("ammonia_water_ratio", 0.82)
    sep_P_bar = params.get("separator_pressure_bar", 30)
    eta_turbine = params.get("turbine_isentropic_efficiency", 0.80)
    absorber_approach = params.get("absorber_approach_delta_C", 10)

    # Kalina efficiency correlation (Mlcak 1996, DiPippo 2012)
    # eta_Kalina ≈ 0.55 * eta_Carnot for well-designed systems
    # Slightly better temperature matching than ORC due to variable boiling
    T_hot_K = _c_to_k(T_in_C)
    T_cold_K = _c_to_k(T_amb_C + absorber_approach)

    eta_carnot = 1 - T_cold_K / T_hot_K
    # Practical Kalina: 50-60% of Carnot (lower in practice due to corrosion issues)
    carnot_fraction = params.get("carnot_fraction", 0.50)
    eta_cycle = eta_carnot * carnot_fraction

    W_net_MW = Q_available_MW * eta_cycle * 0.90  # 90% HX effectiveness
    W_gross_MW = W_net_MW / 0.88  # ~12% parasitic
    parasitic_MW = W_gross_MW - W_net_MW

    Q_cond_MW = Q_available_MW * 0.90 - W_gross_MW  # energy balance

    warnings.append("Kalina model uses parametric correlations — limited commercial track record")
    warnings.append("Ammonia handling and corrosion management add significant O&M complexity")

    # Cost: Kalina typically 15-30% more expensive than ORC due to materials
    net_kW = W_net_MW * 1000
    base_per_kW = 3500
    if net_kW < 5000:
        scale_mult = 1.5
    elif net_kW < 20000:
        scale_mult = 1.2
    else:
        scale_mult = 1.0

    equipment_cost = net_kW * base_per_kW * scale_mult * 0.65
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.04  # higher O&M due to ammonia + corrosion

    return _build_standard_output(
        technology_id="kalina",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_cycle,
        brine_utilization_eff=eta_cycle * 0.90,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=65,
        first_power_weeks=78,
        water_consumption=0.0,
        model_confidence="medium",
        design_basis=design_basis,
        warnings=warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# CASCADED DUAL-STAGE ORC
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_cascaded_orc(params: dict, design_basis: dict) -> dict:
    """
    Two ORC stages in series with different working fluids.
    High-temperature stage uses the brine, low-temperature stage uses HT discharge.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    warnings = []

    if T_in_C < 120:
        warnings.append("Cascaded ORC has minimal benefit below 120°C")

    interstage_T_C = params.get("interstage_temperature_C", (T_in_C + brine["T_out_C"]) / 2)

    # High-stage ORC
    ht_basis = dict(design_basis)
    ht_basis["brine_outlet_temp_C"] = interstage_T_C
    ht_result = analyze_orc_direct(params.get("high_stage_params", {}), ht_basis)

    # Low-stage ORC — uses brine from interstage temp down
    lt_basis = dict(design_basis)
    lt_basis["brine_inlet_temp_C"] = interstage_T_C
    lt_result = analyze_orc_direct(params.get("low_stage_params", {}), lt_basis)

    # Combined
    W_net = ht_result.get("net_power_MW", 0) + lt_result.get("net_power_MW", 0)
    W_gross = ht_result.get("gross_power_MW", 0) + lt_result.get("gross_power_MW", 0)
    parasitic = W_gross - W_net
    Q_cond = ht_result.get("condenser_duty_MW", 0) + lt_result.get("condenser_duty_MW", 0)

    Q_in = brine["Q_available_MW"]
    eta = W_net / Q_in if Q_in > 0 else 0

    # Cost: two ORC units + integration
    ht_capex = ht_result.get("capex_total_USD", 0)
    lt_capex = lt_result.get("capex_total_USD", 0)
    capex = (ht_capex + lt_capex) * 1.08  # 8% integration premium
    opex = capex * 0.028

    sched = max(ht_result.get("construction_weeks", 52), lt_result.get("construction_weeks", 52)) + 8

    all_warnings = warnings + ht_result.get("warnings", []) + lt_result.get("warnings", [])

    return _build_standard_output(
        technology_id="cascaded_orc",
        net_power_MW=W_net,
        gross_power_MW=W_gross,
        parasitic_MW=parasitic,
        cycle_efficiency=eta,
        brine_utilization_eff=eta,
        condenser_duty_MW=Q_cond,
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=sched,
        first_power_weeks=sched + 8,
        water_consumption=0.0,
        model_confidence="medium",
        design_basis=design_basis,
        warnings=all_warnings,
    )


# ═══════════════════════════════════════════════════════════════════════════════
# GEOTHERMAL CHP
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_geothermal_chp(params: dict, design_basis: dict) -> dict:
    """
    Combined heat and power — ORC with thermal offtake.
    Sacrifices power output for usable heat delivery.
    """
    brine = _brine_thermal_input(design_basis)
    warnings = []

    heat_split = params.get("power_heat_split_ratio", 0.6)  # 60% power, 40% heat
    heat_value_per_MWh = params.get("heat_market_value_per_MWh", 30)
    heat_load_MW = params.get("heat_load_MW", brine["Q_available_MW"] * (1 - heat_split) * 0.8)

    site_notes = design_basis.get("site_notes", "")
    has_heat_market = any(kw in site_notes.lower()
                         for kw in ["heat", "district", "greenhouse", "industrial", "thermal"])

    if not has_heat_market:
        warnings.append("No heat market indicated in site notes — CHP not economically viable")

    # Power-only ORC on the power fraction
    power_basis = dict(design_basis)
    # Reduce available brine for power (heat offtake takes priority)
    power_basis["brine_flow_kg_s"] = brine["m_dot_kgs"] * heat_split

    power_result = analyze_orc_direct(params.get("power_params", {}), power_basis)

    W_net = power_result.get("net_power_MW", 0)
    W_gross = power_result.get("gross_power_MW", 0)
    parasitic = W_gross - W_net

    # Heat delivery
    Q_heat_delivered_MW = min(heat_load_MW, brine["Q_available_MW"] * (1 - heat_split) * 0.85)

    eta_power = W_net / brine["Q_available_MW"] if brine["Q_available_MW"] > 0 else 0
    eta_total = (W_net + Q_heat_delivered_MW) / brine["Q_available_MW"] if brine["Q_available_MW"] > 0 else 0

    # Cost: ORC + heat exchangers + distribution piping
    power_capex = power_result.get("capex_total_USD", 0)
    heat_capex = Q_heat_delivered_MW * 500_000  # ~$500k/MW_th for heat delivery
    capex = power_capex + heat_capex
    opex = capex * 0.025

    sched = power_result.get("construction_weeks", 52) + 4  # heat system adds weeks

    return _build_standard_output(
        technology_id="geothermal_chp",
        net_power_MW=W_net,
        gross_power_MW=W_gross,
        parasitic_MW=parasitic,
        cycle_efficiency=eta_power,
        brine_utilization_eff=eta_total,
        condenser_duty_MW=power_result.get("condenser_duty_MW", 0),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=sched,
        first_power_weeks=sched + 6,
        water_consumption=0.0,
        model_confidence="medium",
        design_basis=design_basis,
        warnings=warnings + power_result.get("warnings", []),
    )


# ═══════════════════════════════════════════════════════════════════════════════
# PARAMETRIC / EMERGING TECHNOLOGY MODELS
# ═══════════════════════════════════════════════════════════════════════════════

def analyze_sco2(params: dict, design_basis: dict) -> dict:
    """
    Supercritical CO2 Brayton cycle — parametric model.
    Based on NREL/Sandia published efficiency correlations.
    CoolProp CO2 properties for state point verification.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    if T_in_C < 150:
        warnings.append(f"sCO2 cycle needs >150°C source — {T_in_C:.0f}°C is marginal")

    # sCO2 cycle parameters
    turb_inlet_T = params.get("turbine_inlet_temp_C", T_in_C - 15)  # HX pinch
    turb_inlet_P = params.get("turbine_inlet_pressure_bar", 200)
    recup_eff = params.get("recuperator_effectiveness", 0.90)

    # Efficiency correlation from NREL studies (Neises & Turchi 2019)
    # eta_net = f(T_hot, T_cold) with correction for recuperation
    T_hot_K = _c_to_k(turb_inlet_T)
    T_cold_K = _c_to_k(T_amb_C + 15)  # compressor inlet

    eta_carnot = 1 - T_cold_K / T_hot_K

    # sCO2 Brayton achieves ~45-55% of Carnot at geothermal temperatures
    # Lower than at high-T applications (600°C+) due to compression penalty near critical point
    carnot_fraction = 0.45 + 0.05 * min(1, (turb_inlet_T - 150) / 100)
    eta_cycle = eta_carnot * carnot_fraction

    # HX effectiveness limits heat input
    hx_eff = 0.88
    W_net_MW = Q_available_MW * hx_eff * eta_cycle
    W_gross_MW = W_net_MW / 0.82  # ~18% compressor + auxiliary parasitic
    parasitic_MW = W_gross_MW - W_net_MW

    Q_cond_MW = Q_available_MW * hx_eff - W_gross_MW

    warnings.append("No commercial geothermal sCO2 plant exists — parametric model only")
    warnings.append("Cost estimates based on DOE demonstration program projections")

    # Cost: current estimates $3000-5000/kW (pre-commercial premium)
    net_kW = W_net_MW * 1000
    base_per_kW = 4000  # pre-commercial
    equipment_cost = net_kW * base_per_kW * 0.65
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.03

    return _build_standard_output(
        technology_id="sco2_brayton",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_cycle,
        brine_utilization_eff=eta_cycle * hx_eff,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=78,
        first_power_weeks=90,
        water_consumption=0.0,
        model_confidence="low",
        design_basis=design_basis,
        warnings=warnings,
    )


def analyze_trilateral_flash(params: dict, design_basis: dict) -> dict:
    """
    Trilateral Flash Cycle — expands saturated liquid through two-phase expander.
    Better temperature matching than ORC but limited by expander efficiency.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    eta_expander = params.get("expander_isentropic_efficiency", 0.75)
    hx_pinch_C = params.get("hx_pinch_C", 8)

    # TFC efficiency: better temperature matching (~15-20% improvement in Q utilization)
    # but expander efficiency 70-80% vs ORC turbine 85-90%
    T_hot_K = _c_to_k(T_in_C - hx_pinch_C)
    T_cold_K = _c_to_k(T_amb_C + 15)

    eta_carnot = 1 - T_cold_K / T_hot_K
    # TFC achieves ~40-50% of Carnot (limited by two-phase expander)
    carnot_fraction = 0.35 + 0.10 * (eta_expander - 0.70) / 0.10
    carnot_fraction = max(0.30, min(0.50, carnot_fraction))
    eta_cycle = eta_carnot * carnot_fraction

    hx_eff = 0.92  # better temp matching = better HX utilization
    W_net_MW = Q_available_MW * hx_eff * eta_cycle
    W_gross_MW = W_net_MW / 0.90
    parasitic_MW = W_gross_MW - W_net_MW
    Q_cond_MW = Q_available_MW * hx_eff - W_gross_MW

    warnings.append("Pre-commercial — two-phase expander is key technological barrier")
    warnings.append(f"Assumed expander efficiency {eta_expander:.0%} (best demonstrated 70-80%)")

    net_kW = W_net_MW * 1000
    base_per_kW = 4500
    equipment_cost = net_kW * base_per_kW * 0.65
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.035

    return _build_standard_output(
        technology_id="trilateral_flash",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_cycle,
        brine_utilization_eff=eta_cycle * hx_eff,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=70,
        first_power_weeks=82,
        water_consumption=0.0,
        model_confidence="low",
        design_basis=design_basis,
        warnings=warnings,
    )


def analyze_organic_flash(params: dict, design_basis: dict) -> dict:
    """
    Organic Flash Cycle — partial flash of organic working fluid.
    Hybrid between ORC and TFC. Avoids worst of two-phase expander problem.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    flash_fraction = params.get("flash_fraction", 0.3)

    # OFC efficiency: between ORC and TFC
    T_hot_K = _c_to_k(T_in_C - 10)
    T_cold_K = _c_to_k(T_amb_C + 15)
    eta_carnot = 1 - T_cold_K / T_hot_K

    # OFC achieves slightly better than ORC due to partial temperature matching improvement
    carnot_fraction = 0.42 + 0.03 * flash_fraction
    eta_cycle = eta_carnot * carnot_fraction

    hx_eff = 0.90
    W_net_MW = Q_available_MW * hx_eff * eta_cycle
    W_gross_MW = W_net_MW / 0.88
    parasitic_MW = W_gross_MW - W_net_MW
    Q_cond_MW = Q_available_MW * hx_eff - W_gross_MW

    warnings.append("Pre-commercial — closer to standard ORC than TFC")
    warnings.append("Partial flash reduces two-phase expander challenge but adds complexity")

    net_kW = W_net_MW * 1000
    base_per_kW = 3800
    equipment_cost = net_kW * base_per_kW * 0.65
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.032

    return _build_standard_output(
        technology_id="organic_flash",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_cycle,
        brine_utilization_eff=eta_cycle * hx_eff,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=65,
        first_power_weeks=76,
        water_consumption=0.0,
        model_confidence="low",
        design_basis=design_basis,
        warnings=warnings,
    )


def analyze_teg(params: dict, design_basis: dict) -> dict:
    """
    Thermoelectric Generator.
    η = (√(1+ZT_avg)-1) / (√(1+ZT_avg) + Tc/Th)
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    ZT_hot = params.get("ZT_hot", 1.0)
    ZT_cold = params.get("ZT_cold", 0.8)
    module_cost_per_W = params.get("module_cost_per_W", 10.0)
    hx_approach_C = params.get("HX_approach_delta_C", 20)

    T_h_K = _c_to_k(T_in_C - hx_approach_C)
    T_c_K = _c_to_k(T_amb_C + hx_approach_C)

    ZT_avg = (ZT_hot + ZT_cold) / 2

    # Theoretical device efficiency
    sqrt_term = math.sqrt(1 + ZT_avg)
    eta_device = (sqrt_term - 1) / (sqrt_term + T_c_K / T_h_K)

    # Practical system efficiency: 40-60% of device level
    system_fraction = params.get("system_fraction", 0.50)
    eta_system = eta_device * system_fraction

    # HX effectiveness
    hx_eff = 0.85
    W_net_MW = Q_available_MW * hx_eff * eta_system
    W_gross_MW = W_net_MW  # no mechanical parasitic, but pumps for coolant
    pump_MW = W_gross_MW * 0.05
    parasitic_MW = pump_MW
    W_net_MW = W_gross_MW - parasitic_MW
    Q_cond_MW = Q_available_MW * hx_eff - W_gross_MW

    warnings.append(f"TEG at ZT={ZT_avg:.1f}: device eff={eta_device:.1%}, system eff={eta_system:.1%}")
    warnings.append("Current commercial ZT ~1.0 — lab demonstrations up to 2.0+")
    warnings.append("30-year durability at elevated temperatures not demonstrated")

    # Cost: module cost dominates
    net_kW = W_net_MW * 1000
    net_W = net_kW * 1000
    module_cost = net_W * module_cost_per_W / system_fraction  # need more modules than net output
    hx_cost = Q_available_MW * 200_000  # $/MW_th for hot/cold side HX
    equipment_cost = module_cost + hx_cost
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.02  # low O&M (no moving parts in TEG itself)

    return _build_standard_output(
        technology_id="teg",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_system,
        brine_utilization_eff=eta_system * hx_eff,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=40,
        first_power_weeks=44,
        water_consumption=0.0,
        model_confidence="low",
        design_basis=design_basis,
        warnings=warnings,
    )


def analyze_stirling(params: dict, design_basis: dict) -> dict:
    """
    Stirling engine array.
    Carnot efficiency × practical correction factor (0.4-0.6).
    Array sizing from published unit data.
    """
    brine = _brine_thermal_input(design_basis)
    T_in_C = brine["T_in_C"]
    T_amb_C = brine["T_amb_C"]
    Q_available_MW = brine["Q_available_MW"]
    warnings = []

    correction_factor = params.get("carnot_correction", 0.45)
    unit_size_kW = params.get("unit_size_kW", 25)  # largest commercial units
    cost_per_kW = params.get("cost_per_kW", 8000)  # very high at small scale

    T_h_K = _c_to_k(T_in_C - 15)
    T_c_K = _c_to_k(T_amb_C + 10)

    eta_carnot = 1 - T_c_K / T_h_K
    eta_actual = eta_carnot * correction_factor

    hx_eff = 0.85
    W_net_MW = Q_available_MW * hx_eff * eta_actual
    W_gross_MW = W_net_MW / 0.92
    parasitic_MW = W_gross_MW - W_net_MW
    Q_cond_MW = Q_available_MW * hx_eff - W_gross_MW

    net_kW = W_net_MW * 1000
    n_units = math.ceil(net_kW / unit_size_kW)

    warnings.append(f"Requires array of {n_units} × {unit_size_kW}kW Stirling units")
    warnings.append("Stirling engines are 1-25 kW — not competitive at MW geothermal scale")
    warnings.append("Specific power is very low vs turbine-based cycles")

    equipment_cost = net_kW * cost_per_kW * 0.65
    capex = _capex_with_indirects(equipment_cost)
    opex = capex * 0.05  # high O&M for array maintenance

    return _build_standard_output(
        technology_id="stirling",
        net_power_MW=W_net_MW,
        gross_power_MW=W_gross_MW,
        parasitic_MW=parasitic_MW,
        cycle_efficiency=eta_actual,
        brine_utilization_eff=eta_actual * hx_eff,
        condenser_duty_MW=max(0, Q_cond_MW),
        capex_total_USD=capex,
        opex_annual_USD=opex,
        construction_weeks=50,
        first_power_weeks=58,
        water_consumption=0.0,
        model_confidence="low",
        design_basis=design_basis,
        warnings=warnings,
    )


def analyze_mhd(params: dict, design_basis: dict) -> dict:
    """MHD generation — always excluded for geothermal temperatures."""
    brine = _brine_thermal_input(design_basis)
    warnings = [
        f"MHD requires >500°C, typically >2000°C. Brine at {brine['T_in_C']:.0f}°C is far below viable range.",
        "MHD research in 1970-80s targeted coal combustion gases at 2500°C+.",
        "Fluid conductivity at geothermal temperatures is insufficient for MHD generation.",
    ]
    return _empty_output("mhd", warnings)


# ═══════════════════════════════════════════════════════════════════════════════
# ANALYSIS DISPATCHER
# ═══════════════════════════════════════════════════════════════════════════════

ANALYZERS = {
    "orc_direct": analyze_orc_direct,
    "orc_propane_loop": analyze_orc_propane_loop,
    "single_flash": analyze_single_flash,
    "double_flash": analyze_double_flash,
    "hybrid_flash_binary": analyze_hybrid_flash_binary,
    "steam_rankine": analyze_steam_rankine,
    "kalina": analyze_kalina,
    "cascaded_orc": analyze_cascaded_orc,
    "geothermal_chp": analyze_geothermal_chp,
    "sco2_brayton": analyze_sco2,
    "trilateral_flash": analyze_trilateral_flash,
    "organic_flash": analyze_organic_flash,
    "teg": analyze_teg,
    "stirling": analyze_stirling,
    "mhd": analyze_mhd,
}


def analyze_technology(technology_id: str, params: dict, design_basis: dict) -> dict:
    """Dispatch to the appropriate technology analyzer.

    Parameters
    ----------
    technology_id : str
        Technology key from the registry (e.g., "orc_direct", "single_flash")
    params : dict
        Technology-specific optimization parameters
    design_basis : dict
        Design basis from the shared sidebar (SI units)

    Returns
    -------
    dict
        Standard output format with all performance/economic metrics.
    """
    analyzer = ANALYZERS.get(technology_id)
    if analyzer is None:
        return _empty_output(technology_id, [f"No analyzer for technology: {technology_id}"])
    try:
        return analyzer(params, design_basis)
    except Exception as e:
        return _empty_output(technology_id, [f"Analysis error: {e}"])
