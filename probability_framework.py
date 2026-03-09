"""
Probability Framework — Monte Carlo NPV simulation with correlated EGS failure modes.

Core proposition: simpler technology has higher joint probability of achieving design
performance because fewer nodes must perform simultaneously. Every additional fluid system,
rotating equipment train, and specialty contractor scope is a probability node.

Joint probability = product of individual probabilities -- it falls fast as complexity grows.

For EGS: failure modes are correlated through the brine system. A scaling event
affects all heat exchangers simultaneously. NCG spikes propagate through every system
touching the working fluid.
"""

import math
import numpy as np
from typing import Any


# ── Base probability nodes for EGS resource ──────────────────────────────────

BASE_NODES = {
    "thermal_performance": 0.75,     # P(achieves >=95% design output)
    "flow_rate": 0.70,               # P(brine flow within 10% of design)
    "ncg_within_spec": 0.65,         # P(NCG within design assumption)
    "brine_chemistry": 0.70,         # P(scaling/corrosion within spec)
    "rotating_equipment": 0.90,      # P(availability >=95%)
    "commissioning_schedule": 0.60,  # P(within 8 weeks of mech complete)
    "year1_no_major_outage": 0.70,
    "year15_performance": 0.72,      # P(>=90% of year 1 performance)
    "om_cost_within_budget": 0.65,   # P(O&M within 120% of estimate)
    "specialist_availability": 0.80,
}

# EGS correlation factor per node (0=independent, 1=fully correlated through brine)
NODE_EGS_CORRELATION = {
    "thermal_performance": 0.80,
    "flow_rate": 0.90,
    "ncg_within_spec": 0.70,
    "brine_chemistry": 0.85,
    "rotating_equipment": 0.20,
    "commissioning_schedule": 0.40,
    "year1_no_major_outage": 0.60,
    "year15_performance": 0.75,
    "om_cost_within_budget": 0.50,
    "specialist_availability": 0.10,
}

# Technology-specific probability modifiers
TECHNOLOGY_MODIFIERS = {
    "orc_direct": {
        "thermal_performance": +0.05,
        "ncg_within_spec": -0.08,       # direct ACC accumulates NCG
        "commissioning_schedule": +0.08,
    },
    "orc_propane_loop": {
        "ncg_within_spec": +0.12,       # KEY ADVANTAGE -- closed isopentane circuit
        "year1_no_major_outage": +0.08, # clean scope boundaries
        "commissioning_schedule": +0.05,
    },
    "single_flash": {
        "ncg_within_spec": -0.15,       # direct brine flash, surface NCG handling
        "brine_chemistry": -0.15,       # direct brine exposure
        "om_cost_within_budget": -0.10,
    },
    "double_flash": {
        "ncg_within_spec": -0.18,
        "brine_chemistry": -0.18,
        "rotating_equipment": -0.05,     # two turbine stages
        "om_cost_within_budget": -0.12,
    },
    "hybrid_flash_binary": {
        "thermal_performance": +0.08,
        "ncg_within_spec": -0.20,       # flash NCG contaminates binary
        "year1_no_major_outage": -0.18, # two interacting systems
        "commissioning_schedule": -0.10,
    },
    "steam_rankine": {
        "thermal_performance": +0.03,
        "brine_chemistry": -0.10,       # brine-side HX scaling
        "commissioning_schedule": -0.05,
        "rotating_equipment": +0.05,    # mature steam turbine technology
    },
    "kalina": {
        "commissioning_schedule": -0.20,
        "om_cost_within_budget": -0.20,
        "specialist_availability": -0.20,
        "year1_no_major_outage": -0.15,
        "brine_chemistry": -0.10,       # ammonia corrosion adds to brine issues
    },
    "cascaded_orc": {
        "thermal_performance": +0.03,
        "commissioning_schedule": -0.10,
        "rotating_equipment": -0.05,    # two turbine trains
        "om_cost_within_budget": -0.05,
    },
    "geothermal_chp": {
        "thermal_performance": +0.02,
        "om_cost_within_budget": -0.05,
        "specialist_availability": -0.05,
    },
    "sco2_brayton": {
        "thermal_performance": -0.15,
        "commissioning_schedule": -0.25,
        "rotating_equipment": -0.15,
        "specialist_availability": -0.30,
        "om_cost_within_budget": -0.20,
        "year1_no_major_outage": -0.20,
    },
    "trilateral_flash": {
        "thermal_performance": -0.10,
        "rotating_equipment": -0.20,    # two-phase expander uncertainty
        "specialist_availability": -0.25,
        "commissioning_schedule": -0.15,
    },
    "organic_flash": {
        "thermal_performance": -0.08,
        "rotating_equipment": -0.12,
        "specialist_availability": -0.15,
        "commissioning_schedule": -0.10,
    },
    "stirling": {
        "rotating_equipment": -0.10,
        "specialist_availability": -0.25,
        "commissioning_schedule": -0.20,
        "om_cost_within_budget": -0.25,
        "year1_no_major_outage": -0.15,
    },
    "teg": {
        "thermal_performance": -0.20,
        "year15_performance": -0.25,    # degradation uncertainty
        "specialist_availability": -0.20,
        "om_cost_within_budget": -0.15,
    },
    "mhd": {
        "thermal_performance": -0.50,
        "rotating_equipment": -0.50,
        "specialist_availability": -0.50,
        "commissioning_schedule": -0.50,
    },
}


def _get_technology_nodes(technology_id: str) -> dict:
    """Get probability nodes for a specific technology (base + modifiers)."""
    nodes = dict(BASE_NODES)
    modifiers = TECHNOLOGY_MODIFIERS.get(technology_id, {})
    for node, delta in modifiers.items():
        nodes[node] = max(0.05, min(0.99, nodes[node] + delta))
    return nodes


def _generate_correlated_samples(
    nodes: dict,
    correlations: dict,
    n_iterations: int,
    rng: np.random.Generator,
) -> np.ndarray:
    """Generate correlated binary samples for probability nodes.

    Uses a common EGS factor approach: each node's outcome is driven partly
    by a shared EGS resource factor and partly by independent factors.

    Parameters
    ----------
    nodes : dict
        Node name -> probability of success
    correlations : dict
        Node name -> EGS correlation factor (0-1)
    n_iterations : int
        Number of Monte Carlo iterations
    rng : np.random.Generator
        Random number generator

    Returns
    -------
    np.ndarray
        Shape (n_iterations, n_nodes) with 0/1 outcomes
    """
    node_names = list(nodes.keys())
    n_nodes = len(node_names)

    # Common EGS factor (standard normal)
    z_common = rng.standard_normal(n_iterations)

    # Independent factors for each node
    z_independent = rng.standard_normal((n_iterations, n_nodes))

    outcomes = np.zeros((n_iterations, n_nodes), dtype=int)

    for j, name in enumerate(node_names):
        p = nodes[name]
        corr = correlations.get(name, 0.0)

        # Combine common and independent normal variates
        # z = sqrt(corr) * z_common + sqrt(1-corr) * z_independent
        z = math.sqrt(corr) * z_common + math.sqrt(1 - corr) * z_independent[:, j]

        # Convert to probability using normal CDF threshold
        # Threshold chosen so that P(z < threshold) = p
        from scipy.stats import norm
        threshold = norm.ppf(p)

        outcomes[:, j] = (z < threshold).astype(int)

    return outcomes


def run_monte_carlo(
    technology_id: str,
    design_results: dict,
    design_basis: dict,
    n_iterations: int = 5000,
    seed: int | None = None,
) -> dict:
    """Run Monte Carlo simulation for a single technology.

    For each iteration:
    1. Sample correlated node performance (EGS correlation drives covariance)
    2. Calculate output considering correlated failures
    3. Calculate NPV at that output level

    Parameters
    ----------
    technology_id : str
        Technology identifier
    design_results : dict
        Standard output from analyze_technology()
    design_basis : dict
        Design basis from sidebar
    n_iterations : int
        Monte Carlo iterations (default 5000)
    seed : int, optional
        Random seed for reproducibility

    Returns
    -------
    dict
        P10, P50, P90 NPV, output distributions, and node statistics
    """
    rng = np.random.default_rng(seed)

    nodes = _get_technology_nodes(technology_id)
    correlations = NODE_EGS_CORRELATION

    # Generate correlated outcomes
    outcomes = _generate_correlated_samples(nodes, correlations, n_iterations, rng)
    node_names = list(nodes.keys())

    # Design performance values
    design_net_MW = design_results.get("net_power_MW", 0)
    design_capex = design_results.get("capex_total_USD", 0)
    design_opex = design_results.get("opex_annual_USD", 0)
    design_sched_weeks = design_results.get("construction_weeks", 52)

    energy_price = design_basis.get("energy_value_per_MWh", 80)
    discount_rate = design_basis.get("discount_rate", 0.08)
    plant_life = design_basis.get("plant_life_years", 30)
    capacity_factor = design_basis.get("capacity_factor", 0.95)

    # Impact of each node failure on performance
    node_impact = {
        "thermal_performance": {"output_mult": 0.85},
        "flow_rate": {"output_mult": 0.88},
        "ncg_within_spec": {"output_mult": 0.92, "opex_mult": 1.15},
        "brine_chemistry": {"output_mult": 0.95, "opex_mult": 1.20},
        "rotating_equipment": {"output_mult": 0.93},
        "commissioning_schedule": {"schedule_add_weeks": 12},
        "year1_no_major_outage": {"output_mult": 0.80},  # one-time year 1 hit
        "year15_performance": {"late_output_mult": 0.88},
        "om_cost_within_budget": {"opex_mult": 1.30},
        "specialist_availability": {"schedule_add_weeks": 6, "opex_mult": 1.10},
    }

    npv_array = np.zeros(n_iterations)
    output_array = np.zeros(n_iterations)
    capex_array = np.zeros(n_iterations)

    for i in range(n_iterations):
        output_mult = 1.0
        opex_mult = 1.0
        schedule_delay_weeks = 0
        late_output_mult = 1.0

        for j, name in enumerate(node_names):
            if outcomes[i, j] == 0:  # node failure
                impact = node_impact.get(name, {})
                output_mult *= impact.get("output_mult", 1.0)
                opex_mult *= impact.get("opex_mult", 1.0)
                schedule_delay_weeks += impact.get("schedule_add_weeks", 0)
                late_output_mult *= impact.get("late_output_mult", 1.0)

        # Add random noise to represent unknown unknowns (+/- 5%)
        noise = 1.0 + rng.normal(0, 0.025)
        output_mult *= max(0.5, noise)

        # Compute NPV for this iteration
        actual_net_MW = design_net_MW * output_mult
        actual_capex = design_capex * (1 + schedule_delay_weeks * 0.002)  # cost overrun from delay
        actual_opex = design_opex * opex_mult
        actual_sched = design_sched_weeks + schedule_delay_weeks

        # Revenue calculation
        annual_energy_MWh = actual_net_MW * 1000 * 8760 * capacity_factor / 1000  # MWh
        annual_revenue = annual_energy_MWh * energy_price

        # NPV with two-phase output (early years at full, later years degraded)
        if discount_rate > 0:
            annuity_factor = (1 - (1 + discount_rate) ** (-plant_life)) / discount_rate
        else:
            annuity_factor = plant_life

        # Simple NPV: revenue * annuity - capex - PV of opex
        # Apply late-life degradation for years 15+
        avg_output_factor = 0.5 * (1.0 + late_output_mult)  # simplified blend
        npv = (annual_revenue * avg_output_factor * annuity_factor
               - actual_capex
               - actual_opex * annuity_factor)

        npv_array[i] = npv
        output_array[i] = actual_net_MW
        capex_array[i] = actual_capex

    # Statistics
    npv_p10 = float(np.percentile(npv_array, 10))
    npv_p50 = float(np.percentile(npv_array, 50))
    npv_p90 = float(np.percentile(npv_array, 90))

    output_p10 = float(np.percentile(output_array, 10))
    output_p50 = float(np.percentile(output_array, 50))
    output_p90 = float(np.percentile(output_array, 90))

    # Joint probability of all nodes succeeding
    all_success = np.all(outcomes == 1, axis=1)
    joint_probability = float(np.mean(all_success))

    # Node success rates (realized vs target)
    node_success_rates = {}
    for j, name in enumerate(node_names):
        node_success_rates[name] = {
            "target": nodes[name],
            "realized": float(np.mean(outcomes[:, j])),
        }

    return {
        "technology_id": technology_id,
        "n_iterations": n_iterations,
        "npv_p10": npv_p10,
        "npv_p50": npv_p50,
        "npv_p90": npv_p90,
        "npv_mean": float(np.mean(npv_array)),
        "npv_std": float(np.std(npv_array)),
        "output_p10_MW": output_p10,
        "output_p50_MW": output_p50,
        "output_p90_MW": output_p90,
        "joint_probability": joint_probability,
        "node_success_rates": node_success_rates,
        "node_probabilities": dict(nodes),
        "downside_risk": npv_p50 - npv_p10,
        "upside_potential": npv_p90 - npv_p50,
    }


def calculate_complexity_penalty(
    technology_id: str,
    reference_id: str,
    tech_mc: dict,
    ref_mc: dict,
) -> dict:
    """Calculate complexity penalty vs a reference technology.

    Two components:
    1. Expected value penalty: P50 difference vs reference
    2. Distribution penalty: cost of wider outcome variance
       = difference in downside risk: (P50 - P10) tech vs (P50 - P10) reference

    A technology can have higher P50 but negative complexity-adjusted score
    if downside risk is sufficiently larger than reference.

    Parameters
    ----------
    technology_id : str
        Technology being evaluated
    reference_id : str
        Baseline reference technology (typically orc_direct)
    tech_mc : dict
        Monte Carlo results for the technology
    ref_mc : dict
        Monte Carlo results for the reference

    Returns
    -------
    dict
        Complexity penalty breakdown
    """
    tech_p50 = tech_mc["npv_p50"]
    ref_p50 = ref_mc["npv_p50"]

    tech_downside = tech_mc["npv_p50"] - tech_mc["npv_p10"]
    ref_downside = ref_mc["npv_p50"] - ref_mc["npv_p10"]

    # Expected value difference (positive = technology leads)
    ev_delta = tech_p50 - ref_p50

    # Distribution penalty (positive = technology has wider downside)
    distribution_penalty = tech_downside - ref_downside

    # Risk-adjusted NPV advantage
    # Subtract a risk premium for wider distribution (1.5x penalty weight)
    risk_premium_factor = 1.5
    risk_adjusted_delta = ev_delta - risk_premium_factor * max(0, distribution_penalty)

    # Complexity score
    tech_nodes = _get_technology_nodes(technology_id)
    ref_nodes = _get_technology_nodes(reference_id)

    # Count number of nodes where technology is worse than reference
    disadvantaged_nodes = sum(1 for k in tech_nodes
                              if tech_nodes[k] < ref_nodes.get(k, 0.5))

    # Joint probability comparison
    joint_delta = tech_mc["joint_probability"] - ref_mc["joint_probability"]

    return {
        "technology_id": technology_id,
        "reference_id": reference_id,
        "ev_delta_USD": ev_delta,
        "distribution_penalty_USD": distribution_penalty,
        "risk_adjusted_delta_USD": risk_adjusted_delta,
        "disadvantaged_nodes": disadvantaged_nodes,
        "joint_probability_delta": joint_delta,
        "complexity_justified": risk_adjusted_delta > 0,
        "summary": (
            f"vs {reference_id}: EV delta ${ev_delta/1e6:+.1f}M, "
            f"downside penalty ${distribution_penalty/1e6:+.1f}M, "
            f"risk-adjusted ${risk_adjusted_delta/1e6:+.1f}M"
        ),
    }


def run_probability_analysis(
    optimization_results: dict,
    design_basis: dict,
    reference_id: str = "orc_direct",
    n_iterations: int = 5000,
    seed: int = 42,
) -> dict:
    """Run full probability analysis for all technologies.

    Parameters
    ----------
    optimization_results : dict
        {technology_id: standard_output_dict} for all viable technologies
    design_basis : dict
        Design basis from sidebar
    reference_id : str
        Baseline technology for complexity penalty calculation
    n_iterations : int
        Monte Carlo iterations per technology
    seed : int
        Base random seed (incremented per technology for reproducibility)

    Returns
    -------
    dict
        Full probability analysis results including Monte Carlo per technology,
        complexity penalties, and ranking.
    """
    mc_results = {}
    penalties = {}

    # Run Monte Carlo for each technology
    for i, (tech_id, results) in enumerate(optimization_results.items()):
        if not results.get("converged", False):
            continue
        mc_results[tech_id] = run_monte_carlo(
            tech_id, results, design_basis,
            n_iterations=n_iterations, seed=seed + i,
        )

    # Calculate complexity penalties
    ref_mc = mc_results.get(reference_id)
    if ref_mc:
        for tech_id, mc in mc_results.items():
            if tech_id != reference_id:
                penalties[tech_id] = calculate_complexity_penalty(
                    tech_id, reference_id, mc, ref_mc,
                )

    # Rank by risk-adjusted NPV
    ranking = sorted(
        mc_results.keys(),
        key=lambda t: mc_results[t]["npv_p50"],
        reverse=True,
    )

    risk_adjusted_ranking = sorted(
        mc_results.keys(),
        key=lambda t: (
            penalties[t]["risk_adjusted_delta_USD"]
            if t in penalties
            else mc_results[t]["npv_p50"]
        ),
        reverse=True,
    )

    return {
        "monte_carlo": mc_results,
        "complexity_penalties": penalties,
        "ranking_by_p50": ranking,
        "ranking_risk_adjusted": risk_adjusted_ranking,
        "reference_technology": reference_id,
        "n_iterations": n_iterations,
    }
