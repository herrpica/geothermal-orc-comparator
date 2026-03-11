"""
Analysis bridge between the dialectic engine and the existing ORC solvers.

This module is the SINGLE integration point. It:
  1. Accepts the standardized tool input dict (run_orc_analysis schema)
  2. Converts units and maps keys to the existing inputs dict format
  3. Calls solve_config_a / solve_config_b + cost + parasitic + lifecycle
  4. Returns the standardized output dict

All existing thermodynamic logic lives in thermodynamics.py and cost_model.py.
This module NEVER duplicates that logic — it only translates interfaces.
"""

import uuid
from typing import Any

from fluid_properties import FluidProperties
from thermodynamics import solve_config_a, solve_config_b, solve_dual_pressure, validate_inputs
from cost_model import (
    calculate_costs_a,
    calculate_costs_b,
    calculate_costs_dual_pressure,
    calculate_fan_power,
    compute_power_balance,
    lifecycle_cost,
    construction_schedule,
    acc_area_with_air_rise,
    pump_sizing,
)

# ── Unit conversions ────────────────────────────────────────────────────────

def _c_to_f(t_c: float) -> float:
    return t_c * 9.0 / 5.0 + 32.0

def _f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0

def _kgs_to_lbs(m: float) -> float:
    return m * 2.20462

def _ft2_to_m2(a: float) -> float:
    return a * 0.092903

def _mmbtu_hr_to_mw(q: float) -> float:
    return q * 0.29307107

# ── Singleton fluid properties instance ─────────────────────────────────────

_fp_instance: FluidProperties | None = None

def _get_fp() -> FluidProperties:
    global _fp_instance
    if _fp_instance is None:
        _fp_instance = FluidProperties()
    return _fp_instance

# ── Design basis → internal inputs mapping ──────────────────────────────────

def design_basis_to_inputs(design_basis: dict) -> dict:
    """Convert design_basis dict (SI / user-facing) to internal inputs dict (imperial).

    Only maps the fixed resource/site/economic constraints — NOT the
    per-run tunable parameters (those come from run_orc_analysis input).
    """
    inp: dict[str, Any] = {}

    # Resource
    if "brine_inlet_temp_C" in design_basis:
        inp["T_geo_in"] = _c_to_f(design_basis["brine_inlet_temp_C"])
    if "brine_outlet_temp_C" in design_basis:
        inp["T_geo_out_min"] = _c_to_f(design_basis["brine_outlet_temp_C"])
    if "brine_flow_kg_s" in design_basis:
        inp["m_dot_geo"] = _kgs_to_lbs(design_basis["brine_flow_kg_s"])

    # Site
    if "ambient_temp_C" in design_basis:
        inp["T_ambient"] = _c_to_f(design_basis["ambient_temp_C"])

    # Economics
    if "energy_value_per_MWh" in design_basis:
        inp["electricity_price"] = design_basis["energy_value_per_MWh"]
    if "plant_life_years" in design_basis:
        inp["project_life"] = design_basis["plant_life_years"]
    if "discount_rate" in design_basis:
        inp["discount_rate"] = design_basis["discount_rate"]

    # Efficiency params (pass through — already in decimal)
    for key in ["eta_turbine", "eta_pump", "generator_efficiency"]:
        if key in design_basis:
            inp[key] = design_basis[key]

    # Pinch points (pass through — already in °F)
    if "dt_pinch_vaporizer" in design_basis:
        inp["dt_pinch_vaporizer"] = design_basis["dt_pinch_vaporizer"]
    if "dt_pinch_preheater" in design_basis:
        inp["dt_pinch_preheater"] = design_basis["dt_pinch_preheater"]
    if "dt_pinch_acc" in design_basis:
        inp["dt_pinch_acc_a"] = design_basis["dt_pinch_acc"]
        inp["dt_pinch_acc_b"] = design_basis["dt_pinch_acc"]

    # Cost overrides — only pass through when NO procurement strategy will be
    # specified.  When a strategy IS used (optimizer path), the strategy's cost
    # factors must control equipment and construction costs; sidebar defaults
    # would shadow them and collapse all four strategies to identical numbers.
    # The caller can still force overrides via tool_input if truly intended.
    inp["_design_basis_uc"] = {}
    for key in ["uc_turbine_per_kw", "uc_acc_per_bay", "uc_hx_multiplier",
                "uc_civil_structural_per_kw", "uc_ei_installation_per_kw"]:
        if key in design_basis:
            inp["_design_basis_uc"][key] = design_basis[key]

    return inp


def tool_input_to_inputs(tool_input: dict, config: str) -> dict:
    """Convert run_orc_analysis tool input to internal inputs dict keys.

    Approach temperatures are already in °F in the tool schema.
    Pressure drop fractions multiply the default equipment dP values.
    """
    inp: dict[str, Any] = {}

    # Heat exchanger approach temperatures (°F → internal keys)
    key_map = {
        "evaporator_approach_delta_F": "dt_pinch_vaporizer",
        "recuperator_approach_delta_F": "dt_pinch_recup",
        "preheater_approach_delta_F": "dt_pinch_preheater",
    }
    for src, dst in key_map.items():
        if src in tool_input:
            inp[dst] = tool_input[src]

    # ACC approach — maps to config-specific key
    if "acc_approach_delta_F" in tool_input:
        if config in ("A", "D"):
            inp["dt_pinch_acc_a"] = tool_input["acc_approach_delta_F"]
        else:
            inp["dt_pinch_acc_b"] = tool_input["acc_approach_delta_F"]

    # Intermediate HX approach (Config B only)
    if "intermediate_hx_approach_delta_F" in tool_input:
        inp["dt_approach_intermediate"] = tool_input["intermediate_hx_approach_delta_F"]

    # Turbomachinery
    if "turbine_isentropic_efficiency" in tool_input:
        inp["eta_turbine"] = tool_input["turbine_isentropic_efficiency"]
    if "pump_isentropic_efficiency" in tool_input:
        inp["eta_pump"] = tool_input["pump_isentropic_efficiency"]

    # Turbine trains (1, 2, or 3 parallel trains)
    if "turbine_trains" in tool_input:
        inp["n_trains"] = tool_input["turbine_trains"]

    # Pressure drop fractions — scale default equipment dP values
    # Default dP values (psi) from thermodynamics._default_inputs():
    iso_dp_defaults = {
        "A": {"dp_acc_tubes_a": 0.5, "dp_acc_headers_a": 0.3, "dp_recup_a": 0.3},
        "B": {"dp_ihx_iso": 0.5, "dp_recup_b": 0.3, "dp_tailpipe_iso_b": 0.3},
    }
    prop_dp_defaults = {
        "dp_acc_tubes_prop": 1.0, "dp_prop_headers": 0.5, "dp_ihx_prop": 0.5,
    }

    iso_frac = tool_input.get("isopentane_pressure_drop_fraction", 1.0)
    # Config D uses same defaults as Config A
    config_for_dp = "A" if config == "D" else config
    for key, default in iso_dp_defaults.get(config_for_dp, {}).items():
        inp[key] = default * iso_frac

    if config == "B":
        prop_frac = tool_input.get("propane_pressure_drop_fraction", 1.0)
        for key, default in prop_dp_defaults.items():
            inp[key] = default * prop_frac

    # Working fluid override
    if "working_fluid" in tool_input:
        inp["working_fluid"] = tool_input["working_fluid"]

    # Economics overrides
    if "energy_value_per_MWh" in tool_input:
        inp["electricity_price"] = tool_input["energy_value_per_MWh"]

    # Procurement strategy
    if "procurement_strategy" in tool_input:
        inp["procurement_strategy"] = tool_input["procurement_strategy"]

    return inp


# ── Main analysis runner ────────────────────────────────────────────────────

def run_orc_analysis(tool_input: dict, design_basis: dict) -> dict:
    """Execute full ORC analysis pipeline for one configuration.

    Parameters
    ----------
    tool_input : dict
        Standardized run_orc_analysis input (see dialectic spec).
    design_basis : dict
        Fixed design constraints from the user form.

    Returns
    -------
    dict
        Standardized output matching the dialectic spec schema.
    """
    config = tool_input.get("config", "A")
    fp = _get_fp()

    # Build merged inputs dict: defaults ← design_basis ← tool_input
    base_inputs = design_basis_to_inputs(design_basis)
    tunable_inputs = tool_input_to_inputs(tool_input, config)
    inputs = {**base_inputs, **tunable_inputs}

    # Apply sidebar uc_* overrides ONLY when no procurement strategy is active.
    # When a strategy is specified (optimizer path), the strategy's cost factors
    # must control costs — sidebar defaults would shadow them.
    stashed_uc = inputs.pop("_design_basis_uc", {})
    if "procurement_strategy" not in inputs:
        inputs.update(stashed_uc)

    # ── Validate ────────────────────────────────────────────────────────
    warnings_list = validate_inputs(inputs)

    # ── Solve thermodynamic cycle ───────────────────────────────────────
    try:
        if config == "A":
            result = solve_config_a(inputs, fp)
        elif config == "D":
            result = solve_dual_pressure(inputs, fp)
        else:
            result = solve_config_b(inputs, fp)
    except Exception as e:
        return {
            "converged": False,
            "warnings": warnings_list + [f"Solver error: {e}"],
            "net_power_MW": 0, "gross_power_MW": 0, "parasitic_MW": 0,
            "cycle_efficiency": 0, "capex_total_USD": 0, "capex_per_kW": 0,
            "npv_USD": 0, "lcoe_per_MWh": 0, "construction_weeks_critical_path": 0,
            "isopentane_condensing_temp_C": 0, "isopentane_evaporating_temp_C": 0,
            "propane_condensing_temp_C": 0, "condenser_duty_MW": 0,
            "intermediate_hx_area_m2": 0, "acc_area_m2": 0,
            "acc_fan_parasitic_MW": 0, "pump_parasitic_MW": 0,
        }

    states = result["states"]
    perf = result["performance"]
    duct = result["duct"]

    # ── Fan power & parasitic balance (compute BEFORE costs for bay count) ──
    Q_reject = perf["Q_reject_mmbtu_hr"]
    T_ambient = inputs.get("T_ambient", 95)
    fan_result = calculate_fan_power(Q_reject, T_ambient, inputs)

    # Inject computed fan bay count into inputs for per-bay ACC costing
    inputs["n_fan_bays_computed"] = fan_result["n_fans_used"]

    power_bal = compute_power_balance(perf, fan_result, inputs, config=config)

    # ── Pump sizing (for FEED package / equipment list) ──────────────────
    eta_pump = inputs.get("eta_pump", 0.82)
    if config == "B":
        # Config B has separate iso and propane pressure keys
        iso_pump = pump_sizing(
            perf["m_dot_iso"], states["4"].rho,
            perf.get("P_high_iso", perf["P_high"]),
            perf.get("P_low_iso", perf["P_low"]),
            perf.get("w_pump_iso", perf["w_pump"]),
            eta_pump,
        )
        propane_states = result["propane_states"]
        prop_pump = pump_sizing(
            perf["m_dot_prop"], propane_states["B"].rho,
            perf["P_prop_evap"], perf["P_prop_cond"],
            perf["w_pump_prop"], eta_pump,
        )
    else:
        # Config A and D share the same iso pump pattern
        iso_pump = pump_sizing(
            perf["m_dot_iso"], states["4"].rho,
            perf["P_high"], perf["P_low"],
            perf["w_pump"], eta_pump,
        )
        prop_pump = {"flow_gpm": 0, "dP_psi": 0, "power_kw": 0, "power_hp": 0}

    # ── Calculate costs ─────────────────────────────────────────────────
    if config == "A":
        costs = calculate_costs_a(states, perf, inputs, duct_result=duct)
    elif config == "D":
        costs = calculate_costs_dual_pressure(result, inputs)
    else:
        costs = calculate_costs_b(states, propane_states, perf, inputs, duct_result=duct)

    net_power_kw = power_bal["P_net"]
    gross_power_kw = power_bal["P_gross"]
    total_parasitic_kw = power_bal["W_total_parasitic"]

    # ── Lifecycle economics ─────────────────────────────────────────────
    lc = lifecycle_cost(costs["total_installed"], net_power_kw, inputs)

    # ── Construction schedule ───────────────────────────────────────────
    sched = construction_schedule(duct)
    if config == "A":
        critical_weeks = sched["config_a"]["total_weeks"]
    elif config == "D":
        critical_weeks = sched["config_d"]["total_weeks"]
    else:
        critical_weeks = sched["config_b"]["total_weeks"]

    # ── Pump parasitic breakdown ────────────────────────────────────────
    pump_parasitic_kw = power_bal["W_iso_pump"] + power_bal.get("W_prop_pump", 0)

    # ── ACC area (with air rise LMTD) ───────────────────────────────────
    dT_air = inputs.get("dT_air", 25)
    if config in ("A", "D"):
        T_cond_for_acc = perf["T_cond"]
    else:
        T_cond_for_acc = perf.get("T_propane_cond", perf["T_cond"])
    acc_area_ft2 = acc_area_with_air_rise(Q_reject, T_cond_for_acc, T_ambient, dT_air)

    # ── Check convergence ───────────────────────────────────────────────
    converged = perf.get("converged", True)
    if perf.get("vaporizer_pinch_violation", False):
        warnings_list.append("Vaporizer pinch violation detected")
        converged = False
    if perf.get("preheater_pinch_violation", False):
        warnings_list.append("Preheater pinch violation detected")
        converged = False
    if perf.get("brine_outlet_violation", False):
        warnings_list.append("Brine outlet below silica constraint")

    # ── Build standardized output ───────────────────────────────────────
    output = {
        # Performance
        "net_power_MW": net_power_kw / 1000,
        "gross_power_MW": gross_power_kw / 1000,
        "parasitic_MW": total_parasitic_kw / 1000,
        "cycle_efficiency": power_bal["eta_thermal"],

        # Key temperatures
        "isopentane_condensing_temp_C": _f_to_c(perf["T_cond"]),
        "isopentane_evaporating_temp_C": _f_to_c(perf["T_evap"]),
        "propane_condensing_temp_C": (
            _f_to_c(perf["T_propane_cond"]) if config == "B" else None
        ),

        # Equipment
        "condenser_duty_MW": _mmbtu_hr_to_mw(Q_reject),
        "intermediate_hx_area_m2": (
            _ft2_to_m2(costs.get("intermediate_hx_area_ft2", 0))
            if config == "B" else 0
        ),
        "acc_area_m2": _ft2_to_m2(acc_area_ft2),
        "acc_fan_parasitic_MW": fan_result["W_fans_kw"] / 1000,
        "pump_parasitic_MW": pump_parasitic_kw / 1000,

        # Economics
        "capex_total_USD": costs["total_installed"],
        "capex_per_kW": lc["specific_cost_per_kw"],
        "equipment_cost_USD": costs.get("equipment_cost", costs.get("equipment_subtotal", 0)),
        "equipment_per_kW": (
            costs.get("equipment_cost", 0) / net_power_kw
            if net_power_kw > 0 else 0
        ),
        "npv_USD": lc["net_npv"],
        "lcoe_per_MWh": lc["lcoe"],

        # Schedule
        "construction_weeks_critical_path": critical_weeks,

        # Status
        "converged": converged,
        "warnings": warnings_list,

        # ── Extended detail (not in spec but useful for debate context) ──
        "_detail": {
            "config": config,
            "working_fluid": inputs.get("working_fluid", "isopentane"),
            "procurement_strategy": inputs.get("procurement_strategy", "oem_lump_sum"),
            "n_trains": inputs.get("n_trains", 2),
            "T_cond_F": perf["T_cond"],
            "T_evap_F": perf["T_evap"],
            "P_high_psia": perf["P_high"],
            "P_low_psia": perf["P_low"],
            "pressure_ratio": perf["pressure_ratio"],
            "m_dot_iso_lb_hr": perf["m_dot_iso"],
            "Q_reject_mmbtu_hr": Q_reject,
            "parasitic_pct": power_bal["parasitic_pct"],
            "vaporizer_area_ft2": costs.get("vaporizer_area_ft2", 0),
            "preheater_area_ft2": costs.get("preheater_area_ft2", 0),
            "recuperator_area_ft2": costs.get("recuperator_area_ft2", 0),
            "acc_area_ft2": acc_area_ft2,
            "equipment_subtotal": costs.get("equipment_subtotal", 0),
            "ductwork_cost": costs.get("ductwork", 0),
            "structural_steel_cost": costs.get("structural_steel", 0),
            "power_balance": power_bal,
            "annual_energy_mwh": lc["annual_energy_mwh"],
            "annual_revenue": lc["annual_revenue"],
            "annuity_factor": lc["annuity_factor"],
            # Cost line items for optimizer BOM tracking
            "cost_turbine_generator": costs.get("turbine_generator", 0),
            "cost_iso_pump": costs.get("iso_pump", 0),
            "cost_vaporizer": costs.get("vaporizer", 0),
            "cost_preheater": costs.get("preheater", 0),
            "cost_recuperator": costs.get("recuperator", 0),
            "cost_acc": costs.get("acc", 0),
            "cost_ductwork": costs.get("ductwork", 0),
            "cost_structural_steel": costs.get("structural_steel", 0),
            "cost_intermediate_hx": costs.get("intermediate_hx", 0),
            "cost_propane_system": costs.get("propane_system", 0),
            "cost_bop_piping": costs.get("bop_piping", 0),
            "cost_civil_structural": costs.get("civil_structural", 0),
            "cost_ei_installation": costs.get("ei_installation", 0),
            "cost_construction_labor": costs.get("construction_labor", 0),
            "cost_foundation": costs.get("foundation", 0),
            "cost_engineering": costs.get("engineering", 0),
            "cost_construction_mgmt": costs.get("construction_mgmt", 0),
            "cost_commissioning": costs.get("commissioning", 0),
            "cost_contingency": costs.get("contingency", 0),
            "cost_wf_inventory": costs.get("wf_inventory", 0),
            "cost_controls_instrumentation": costs.get("controls_instrumentation", 0),
            "cost_electrical_equipment": costs.get("electrical_equipment", 0),
            "cost_equipment_subtotal": costs.get("equipment_subtotal", 0),
            "cost_plant_installed": costs.get("plant_installed", 0),
            "cost_gathering": costs.get("gathering", 0),
            "cost_td": costs.get("td", 0),
            "cost_total_installed": costs.get("total_installed", 0),
            "acc_n_bays": costs.get("acc_n_bays", fan_result.get("n_fans_used", 0)),
            "schedule_phases": (
                sched["config_a"]["phases"] if config == "A"
                else sched["config_d"]["phases"] if config == "D"
                else sched["config_b"]["phases"]
            ),
            # ── FEED package enrichment ───────────────────────────────
            # Serialized state points (T °F, P psia, h BTU/lb, rho lb/ft³, phase)
            "states": {k: {"T": sp.T, "P": sp.P, "h": sp.h, "rho": sp.rho,
                           "phase": sp.phase, "label": sp.label}
                       for k, sp in states.items()},
            # Duct segments
            "duct_segments": duct.get("segments", []),
            "duct_n_trains": duct.get("n_trains", 2),
            "duct_total_delta_P_psi": duct.get("total_delta_P_psi", 0),
            "duct_total_delta_T_cond_F": duct.get("total_delta_T_cond_F", 0),
            # Fan sizing
            "fan_n_fans_used": fan_result.get("n_fans_used", 0),
            "fan_W_fans_kw": fan_result.get("W_fans_kw", 0),
            "fan_area_each_ft2": fan_result.get("fan_area_each_ft2", 0),
            # Pump sizing (isopentane)
            "pump_iso_flow_gpm": iso_pump.get("flow_gpm", 0),
            "pump_iso_dP_psi": iso_pump.get("dP_psi", 0),
            "pump_iso_power_kw": iso_pump.get("power_kw", 0),
            "pump_iso_power_hp": iso_pump.get("power_hp", 0),
            # Structural steel weight
            "structural_steel_weight_lb": costs.get("structural_steel_weight_lb", 0),
            # Brine temperatures for stream table
            "T_geo_in_F": inputs.get("T_geo_in", 420),
            "T_geo_out_min_F": inputs.get("T_geo_out_min", 180),
            "m_dot_geo_lb_s": inputs.get("m_dot_geo", 1100),
            "T_ambient_F": T_ambient,
            "eta_turbine": inputs.get("eta_turbine", 0.82),
            "eta_pump": eta_pump,
        },
    }

    if config == "D":
        output["_detail"].update({
            "hp_gross_power_kw": perf.get("hp_gross_power_kw", 0),
            "lp_gross_power_kw": perf.get("lp_gross_power_kw", 0),
            "hp_net_power_kw": perf.get("hp_net_power_kw", 0),
            "lp_net_power_kw": perf.get("lp_net_power_kw", 0),
            "T_evap_hp_F": perf.get("T_evap_hp", 0),
            "T_evap_lp_F": perf.get("T_evap_lp", 0),
            "T_split_F": perf.get("T_split", 0),
            "P_high_hp_psia": perf.get("P_high_hp", 0),
            "P_high_lp_psia": perf.get("P_high_lp", 0),
            "hp_m_dot_iso_lb_hr": perf.get("hp_m_dot_iso", 0),
            "lp_m_dot_iso_lb_hr": perf.get("lp_m_dot_iso", 0),
            # FEED enrichment — dual-pressure HP/LP states
            "hp_states": {k: {"T": sp.T, "P": sp.P, "h": sp.h, "rho": sp.rho,
                              "phase": sp.phase, "label": sp.label}
                         for k, sp in result.get("hp_states", {}).items()},
            "lp_states": {k: {"T": sp.T, "P": sp.P, "h": sp.h, "rho": sp.rho,
                              "phase": sp.phase, "label": sp.label}
                         for k, sp in result.get("lp_states", {}).items()},
        })

    if config == "B":
        output["_detail"].update({
            "T_propane_cond_F": perf.get("T_propane_cond", 0),
            "T_propane_evap_F": perf.get("T_propane_evap", 0),
            "m_dot_prop_lb_hr": perf.get("m_dot_prop", 0),
            "intermediate_hx_area_ft2": costs.get("intermediate_hx_area_ft2", 0),
            "intermediate_hx_cost": costs.get("intermediate_hx", 0),
            "propane_system_cost": costs.get("propane_system", 0),
            "thermosiphon": power_bal.get("thermosiphon", False),
            # FEED enrichment — propane loop
            "prop_states": {k: {"T": sp.T, "P": sp.P, "h": sp.h, "rho": sp.rho,
                                "phase": sp.phase, "label": sp.label}
                           for k, sp in propane_states.items()},
            "pump_prop_flow_gpm": prop_pump.get("flow_gpm", 0),
            "pump_prop_dP_psi": prop_pump.get("dP_psi", 0),
            "pump_prop_power_kw": prop_pump.get("power_kw", 0),
            "pump_prop_power_hp": prop_pump.get("power_hp", 0),
        })

    return output


# ── Structural change proposal logger ───────────────────────────────────────

_structural_proposals: list[dict] = []

def propose_structural_change(proposal: dict) -> dict:
    """Log a structural change proposal from a debate bot.

    Does not run any analysis — just records the proposal with an ID.
    """
    proposal_id = f"SC-{len(_structural_proposals) + 1:03d}"
    record = {
        "proposal_id": proposal_id,
        **proposal,
    }
    _structural_proposals.append(record)
    return {"logged": True, "proposal_id": proposal_id}


def get_structural_proposals() -> list[dict]:
    """Return all logged structural change proposals."""
    return list(_structural_proposals)


def clear_structural_proposals():
    """Reset the proposal log (call before starting a new debate)."""
    _structural_proposals.clear()


# ── Tool definitions for Claude tool_use ────────────────────────────────────

RUN_ORC_ANALYSIS_TOOL = {
    "name": "run_orc_analysis",
    "description": (
        "Run a full ORC cycle analysis for Config A (direct air-cooled condenser) "
        "or Config B (propane intermediate heat rejection loop). Returns net power, "
        "capex, NPV, LCOE, schedule, and equipment sizing. Use this to generate "
        "quantitative evidence for your engineering arguments."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "config": {
                "type": "string",
                "enum": ["A", "B"],
                "description": "Configuration to analyze: A = direct ACC, B = propane loop",
            },
            "evaporator_approach_delta_F": {
                "type": "number",
                "description": "Vaporizer approach temperature difference (°F). Typical 5-15°F.",
            },
            "recuperator_approach_delta_F": {
                "type": "number",
                "description": "Recuperator pinch temperature difference (°F). Typical 10-25°F.",
            },
            "preheater_approach_delta_F": {
                "type": "number",
                "description": "Preheater approach temperature difference (°F). Typical 5-15°F.",
            },
            "acc_approach_delta_F": {
                "type": "number",
                "description": "ACC approach: isopentane-to-air (Config A) or propane-to-air (Config B) (°F).",
            },
            "intermediate_hx_approach_delta_F": {
                "type": "number",
                "description": "Intermediate HX isopentane-to-propane approach (°F). Config B only.",
            },
            "turbine_isentropic_efficiency": {
                "type": "number",
                "description": "Turbine isentropic efficiency (0-1). Typical 0.82-0.91.",
            },
            "pump_isentropic_efficiency": {
                "type": "number",
                "description": "Pump isentropic efficiency (0-1). Typical 0.70-0.84.",
            },
            "turbine_trains": {
                "type": "integer",
                "enum": [1, 2, 3],
                "description": "Number of parallel turbine/ACC trains (1, 2, or 3). Affects per-unit turbine sizing and ductwork.",
            },
            "isopentane_pressure_drop_fraction": {
                "type": "number",
                "description": "Multiplier on default isopentane circuit pressure drops. 1.0 = default.",
            },
            "propane_pressure_drop_fraction": {
                "type": "number",
                "description": "Multiplier on default propane circuit pressure drops. 1.0 = default. Config B only.",
            },
            "construction_cost_per_kW": {
                "type": "number",
                "description": "Reference construction cost ($/kW). Informational — model uses bottom-up costing.",
            },
            "energy_value_per_MWh": {
                "type": "number",
                "description": "Energy value for NPV/LCOE calculation ($/MWh). Overrides design basis if provided.",
            },
            "procurement_strategy": {
                "type": "string",
                "enum": ["oem_lump_sum", "direct_lump_sum", "oem_self_perform", "direct_self_perform"],
                "description": (
                    "Procurement strategy affecting cost factors. "
                    "oem_lump_sum = OEM package + EPC contractor (baseline ~$2,500/kW). "
                    "direct_lump_sum = direct vendor purchase + EPC contractor. "
                    "oem_self_perform = OEM package + owner T&M self-perform. "
                    "direct_self_perform = direct vendor + owner self-perform (lowest cost)."
                ),
            },
        },
        "required": ["config"],
    },
}

PROPOSE_STRUCTURAL_CHANGE_TOOL = {
    "name": "propose_structural_change",
    "description": (
        "Propose an engineering innovation or structural change outside the current "
        "model scope. This logs your proposal for the arbitrator to review. Use this "
        "when you identify an improvement that cannot be captured by adjusting "
        "run_orc_analysis parameters alone."
    ),
    "input_schema": {
        "type": "object",
        "properties": {
            "title": {
                "type": "string",
                "description": "Short name for the proposed change.",
            },
            "description": {
                "type": "string",
                "description": "Detailed engineering argument for the change.",
            },
            "affected_system": {
                "type": "string",
                "description": "Which part of the plant is affected.",
            },
            "estimated_npv_delta_pct": {
                "type": "number",
                "description": "Estimated NPV impact as percentage change. Positive = improvement.",
            },
            "estimated_schedule_delta_weeks": {
                "type": "number",
                "description": "Estimated schedule impact in weeks. Negative = faster.",
            },
            "estimated_cost_delta_pct": {
                "type": "number",
                "description": "Estimated cost impact as percentage change. Negative = savings.",
            },
            "confidence": {
                "type": "string",
                "enum": ["high", "medium", "low"],
                "description": "Confidence level in the estimate.",
            },
            "engineering_basis": {
                "type": "string",
                "description": "First-principles engineering justification.",
            },
            "requires_model_extension": {
                "type": "boolean",
                "description": "Whether this needs new modeling capabilities to validate.",
            },
        },
        "required": [
            "title", "description", "affected_system",
            "estimated_npv_delta_pct", "confidence", "engineering_basis",
            "requires_model_extension",
        ],
    },
}

TOOLS = [RUN_ORC_ANALYSIS_TOOL, PROPOSE_STRUCTURAL_CHANGE_TOOL]
