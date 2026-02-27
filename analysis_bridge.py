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
from thermodynamics import solve_config_a, solve_config_b, validate_inputs
from cost_model import (
    calculate_costs_a,
    calculate_costs_b,
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
        if config == "A":
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

    # turbine_trains: N_TRAINS is hardcoded at 2 in the existing solver.
    # We note it here but cannot change it without modifying thermodynamics.py.
    # The bridge will pass it through as metadata in the output.

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
    for key, default in iso_dp_defaults.get(config, {}).items():
        inp[key] = default * iso_frac

    if config == "B":
        prop_frac = tool_input.get("propane_pressure_drop_fraction", 1.0)
        for key, default in prop_dp_defaults.items():
            inp[key] = default * prop_frac

    # Economics overrides
    if "energy_value_per_MWh" in tool_input:
        inp["electricity_price"] = tool_input["energy_value_per_MWh"]

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

    # ── Validate ────────────────────────────────────────────────────────
    warnings_list = validate_inputs(inputs)

    # ── Solve thermodynamic cycle ───────────────────────────────────────
    try:
        if config == "A":
            result = solve_config_a(inputs, fp)
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

    # ── Calculate costs ─────────────────────────────────────────────────
    if config == "A":
        costs = calculate_costs_a(states, perf, inputs, duct_result=duct)
    else:
        propane_states = result["propane_states"]
        costs = calculate_costs_b(states, propane_states, perf, inputs, duct_result=duct)

    # ── Fan power & parasitic balance ───────────────────────────────────
    Q_reject = perf["Q_reject_mmbtu_hr"]
    T_ambient = inputs.get("T_ambient", 95)
    fan_result = calculate_fan_power(Q_reject, T_ambient, inputs)
    power_bal = compute_power_balance(perf, fan_result, inputs, config=config)

    net_power_kw = power_bal["P_net"]
    gross_power_kw = power_bal["P_gross"]
    total_parasitic_kw = power_bal["W_total_parasitic"]

    # ── Lifecycle economics ─────────────────────────────────────────────
    lc = lifecycle_cost(costs["total_installed"], net_power_kw, inputs)

    # ── Construction schedule ───────────────────────────────────────────
    sched = construction_schedule(duct)
    if config == "A":
        critical_weeks = sched["config_a"]["total_weeks"]
    else:
        critical_weeks = sched["config_b"]["total_weeks"]

    # ── Pump parasitic breakdown ────────────────────────────────────────
    pump_parasitic_kw = power_bal["W_iso_pump"] + power_bal.get("W_prop_pump", 0)

    # ── ACC area (with air rise LMTD) ───────────────────────────────────
    dT_air = inputs.get("dT_air", 25)
    if config == "A":
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
            "annual_energy_mwh": lc["annual_energy_mwh"],
            "annual_revenue": lc["annual_revenue"],
            "annuity_factor": lc["annuity_factor"],
        },
    }

    if config == "B":
        output["_detail"].update({
            "T_propane_cond_F": perf.get("T_propane_cond", 0),
            "T_propane_evap_F": perf.get("T_propane_evap", 0),
            "m_dot_prop_lb_hr": perf.get("m_dot_prop", 0),
            "intermediate_hx_area_ft2": costs.get("intermediate_hx_area_ft2", 0),
            "intermediate_hx_cost": costs.get("intermediate_hx", 0),
            "propane_system_cost": costs.get("propane_system", 0),
            "thermosiphon": power_bal.get("thermosiphon", False),
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
                "description": "Number of parallel turbine/ACC trains. Currently fixed at 2.",
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
