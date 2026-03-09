"""
Technology Synthesis — Arbitrator bot with weighted scoring and horizon sensitivity.

Combines dimension bot scores, optimization results, and probability analysis
into a final technology recommendation with confidence level.
"""

import json
import math
from typing import Any

from technology_registry import TECHNOLOGIES
from technology_analysis_bridge import analyze_technology, _build_standard_output
from cost_model import lifecycle_cost
from analysis_bridge import design_basis_to_inputs


# ── Synthesis Prompt ──────────────────────────────────────────────────────────

SYNTHESIS_PROMPT = """
You are the chief engineering arbitrator for a geothermal technology selection.
You have dimension scores from six specialist bots and optimized results from
{n} technology agents.

Design basis: {design_basis_json}
Objective weights: efficiency {eff_w:.0%} / cost {cost_w:.0%} / schedule {sched_w:.0%}
Plant horizon: {horizon_years} years

Dimension scores:
{dimension_scores_table}

Optimized performance results:
{performance_results_table}

Probability analysis:
{probability_summary}

Your tasks:
1. Calculate weighted scores for each technology using the objective weights
2. Surface where dimension bots agree and disagree -- name the tensions explicitly
3. Determine the recommended technology with confidence level (High/Medium/Low)
4. Identify the second-best option and under what conditions it would be preferred
5. Consider horizon sensitivity:

HORIZON SENSITIVITY TABLE:
For each of short (10yr), medium (20yr), long (30yr) horizons:
- Which technology leads on NPV at that horizon
- What the NPV gap is between first and second place
- Why the ranking changes between horizons
- At what horizon does the recommendation change

Present as:
Horizon  | Recommended  | NPV Leader | NPV Gap  | Driving Factor
10 years | [tech]       | [tech]     | $[x]M    | Capital/schedule dominates
20 years | [tech]       | [tech]     | $[x]M    | Balanced
30 years | [tech]       | [tech]     | $[x]M    | Efficiency compounds

The user's stated horizon is {horizon_years} years. Weight the recommendation
accordingly but present all three scenarios.

6. Surface top 3 structural innovations that could change the ranking if implemented
7. Identify what information gaps would most change the recommendation
8. Write a 5-6 sentence executive summary suitable for a leadership presentation

Respond with ONLY valid JSON:
{{
  "weighted_scores": {{"<technology_id>": <float>}},
  "recommended_technology": "<technology_id>",
  "recommendation_confidence": "High"|"Medium"|"Low",
  "second_best": "<technology_id>",
  "second_best_conditions": "<when would this be preferred>",
  "dimension_tensions": ["<string>"],
  "horizon_sensitivity": [
    {{"horizon_years": 10, "recommended": "<id>", "npv_leader": "<id>", "npv_gap_USD": <float>, "driving_factor": "<string>"}},
    {{"horizon_years": 20, "recommended": "<id>", "npv_leader": "<id>", "npv_gap_USD": <float>, "driving_factor": "<string>"}},
    {{"horizon_years": 30, "recommended": "<id>", "npv_leader": "<id>", "npv_gap_USD": <float>, "driving_factor": "<string>"}}
  ],
  "npv_vs_horizon_data": {{
    "<technology_id>": [{{"year": <int>, "npv_USD": <float>}}]
  }},
  "weight_sensitivity": "<at what weight does recommendation change>",
  "structural_innovations": [
    {{"title": "<string>", "description": "<string>", "impact": "<string>"}}
  ],
  "information_gaps": ["<string>"],
  "executive_summary": "<5-6 sentences>"
}}
"""


def _build_dimension_table(dimension_results: dict) -> str:
    """Format dimension scores as a readable table."""
    lines = []
    header = f"{'Technology':<25s}"
    for dim_id in dimension_results:
        dim_name = dimension_results[dim_id].get("dimension_name", dim_id)[:12]
        header += f" | {dim_name:>12s}"
    lines.append(header)
    lines.append("-" * len(header))

    # Collect all technology IDs
    tech_ids = set()
    for dim_result in dimension_results.values():
        tech_ids.update(dim_result.get("scores", {}).keys())

    for tech_id in sorted(tech_ids):
        row = f"{tech_id:<25s}"
        for dim_id in dimension_results:
            score = dimension_results[dim_id].get("scores", {}).get(tech_id, 0)
            row += f" | {score:>12.1f}"
        lines.append(row)

    return "\n".join(lines)


def _build_performance_table(optimization_results: dict) -> str:
    """Format optimization results as a readable table."""
    lines = []
    header = (f"{'Technology':<25s} | {'Net MW':>8s} | {'Eff':>6s} | "
              f"{'CAPEX $M':>9s} | {'$/kW':>7s} | {'NPV $M':>8s} | "
              f"{'LCOE':>6s} | {'Wks':>4s} | {'Conf':>6s}")
    lines.append(header)
    lines.append("-" * len(header))

    for tech_id, r in optimization_results.items():
        lines.append(
            f"{tech_id:<25s} | "
            f"{r.get('net_power_MW', 0):>8.2f} | "
            f"{r.get('cycle_efficiency', 0):>6.3f} | "
            f"${r.get('capex_total_USD', 0)/1e6:>8.1f} | "
            f"${r.get('capex_per_kW', 0):>6.0f} | "
            f"${r.get('npv_USD', 0)/1e6:>7.1f} | "
            f"${r.get('lcoe_per_MWh', 0):>5.1f} | "
            f"{r.get('construction_weeks', 0):>4d} | "
            f"{r.get('model_confidence', '?'):>6s}"
        )

    return "\n".join(lines)


def _build_probability_summary(probability_results: dict | None) -> str:
    """Format probability results as a readable summary."""
    if not probability_results:
        return "No probability analysis available."

    lines = []
    mc = probability_results.get("monte_carlo", {})
    penalties = probability_results.get("complexity_penalties", {})

    header = f"{'Technology':<25s} | {'P10 $M':>8s} | {'P50 $M':>8s} | {'P90 $M':>8s} | {'Joint P':>7s}"
    lines.append(header)
    lines.append("-" * len(header))

    for tech_id, r in mc.items():
        lines.append(
            f"{tech_id:<25s} | "
            f"${r['npv_p10']/1e6:>7.1f} | "
            f"${r['npv_p50']/1e6:>7.1f} | "
            f"${r['npv_p90']/1e6:>7.1f} | "
            f"{r['joint_probability']:>7.3f}"
        )

    if penalties:
        lines.append("\nComplexity Penalties:")
        for tech_id, p in penalties.items():
            lines.append(f"  {tech_id}: {p['summary']}")

    return "\n".join(lines)


def compute_horizon_npv(
    optimization_results: dict,
    design_basis: dict,
    horizons: list[int] | None = None,
) -> dict:
    """Compute NPV for each technology at multiple horizons.

    Parameters
    ----------
    optimization_results : dict
        {technology_id: standard_output}
    design_basis : dict
        Design basis from sidebar
    horizons : list[int], optional
        List of horizon years to evaluate (default: 5-35 in 5yr steps)

    Returns
    -------
    dict
        {technology_id: [{year, npv_USD}]} for charting
    """
    if horizons is None:
        horizons = [5, 10, 15, 20, 25, 30, 35]

    inputs = design_basis_to_inputs(design_basis)
    inputs.setdefault("electricity_price", design_basis.get("energy_value_per_MWh", 80))
    inputs.setdefault("discount_rate", design_basis.get("discount_rate", 0.08))
    inputs.setdefault("capacity_factor", design_basis.get("capacity_factor", 0.95))

    npv_data = {}

    for tech_id, result in optimization_results.items():
        if not result.get("converged", False):
            continue

        net_kW = result.get("net_power_MW", 0) * 1000
        capex = result.get("capex_total_USD", 0)

        points = []
        for h in horizons:
            horizon_inputs = dict(inputs)
            horizon_inputs["project_life"] = h
            lc = lifecycle_cost(capex, net_kW, horizon_inputs)
            points.append({
                "year": h,
                "npv_USD": lc["net_npv"],
            })

        npv_data[tech_id] = points

    return npv_data


def compute_weighted_scores(
    dimension_results: dict,
    weights: dict,
) -> dict:
    """Compute weighted scores from dimension bot results.

    Parameters
    ----------
    dimension_results : dict
        {dim_id: {scores: {tech_id: float}}}
    weights : dict
        Objective weights (efficiency, capital_cost, schedule)

    Returns
    -------
    dict
        {technology_id: weighted_score}
    """
    # Map objective weights to dimension weights
    dim_weights = {
        "thermal_efficiency": weights.get("efficiency", 0.4) * 0.6,
        "capital_complexity": weights.get("capital_cost", 0.4) * 0.5,
        "operations": weights.get("schedule", 0.2) * 0.3 + 0.05,
        "om_pl": weights.get("capital_cost", 0.4) * 0.3 + weights.get("efficiency", 0.4) * 0.2,
        "technology_risk": 0.15,  # always weighted — risk matters
        "probability": 0.10,     # Occam's razor weight
    }

    # Normalize weights
    total_w = sum(dim_weights.values())
    if total_w > 0:
        dim_weights = {k: v / total_w for k, v in dim_weights.items()}

    # Collect all technology IDs
    tech_ids = set()
    for dim_result in dimension_results.values():
        tech_ids.update(dim_result.get("scores", {}).keys())

    weighted = {}
    for tech_id in tech_ids:
        score = 0.0
        for dim_id, w in dim_weights.items():
            dim_score = dimension_results.get(dim_id, {}).get("scores", {}).get(tech_id, 5.0)
            score += w * dim_score
        weighted[tech_id] = score

    return weighted


def run_synthesis(
    optimization_results: dict,
    dimension_results: dict,
    probability_results: dict | None,
    design_basis: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run the synthesis arbitrator.

    Combines all analysis layers into a final recommendation.

    Parameters
    ----------
    optimization_results : dict
        {technology_id: standard_output}
    dimension_results : dict
        {dim_id: evaluation_result}
    probability_results : dict, optional
        Monte Carlo results
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    model : str
        Claude model to use

    Returns
    -------
    dict
        Full synthesis with recommendation, horizon sensitivity, and NPV chart data
    """
    import anthropic

    weights = design_basis.get("objective_weights", {
        "efficiency": 0.4, "capital_cost": 0.4, "schedule": 0.2,
    })
    horizon = design_basis.get("plant_life_years", 30)

    # Pre-compute NPV vs horizon data
    npv_horizon_data = compute_horizon_npv(optimization_results, design_basis)

    # Pre-compute weighted scores
    weighted_scores = compute_weighted_scores(dimension_results, weights)

    # Build tables
    dim_table = _build_dimension_table(dimension_results)
    perf_table = _build_performance_table(optimization_results)
    prob_summary = _build_probability_summary(probability_results)

    prompt = SYNTHESIS_PROMPT.format(
        n=len(optimization_results),
        design_basis_json=json.dumps(design_basis, indent=2, default=str),
        eff_w=weights.get("efficiency", 0.4),
        cost_w=weights.get("capital_cost", 0.4),
        sched_w=weights.get("schedule", 0.2),
        horizon_years=horizon,
        dimension_scores_table=dim_table,
        performance_results_table=perf_table,
        probability_summary=prob_summary,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=6000,
            system=(
                "You are a chief geothermal engineer making a technology selection "
                "recommendation. Be decisive, cite specific numbers, and present "
                "clear confidence levels. Always respond with valid JSON."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        synthesis = json.loads(text)

    except (json.JSONDecodeError, Exception) as e:
        # Fallback synthesis
        synthesis = _fallback_synthesis(
            optimization_results, dimension_results,
            probability_results, weighted_scores, design_basis,
        )
        synthesis["synthesis_error"] = str(e)

    # Always include computed data (not dependent on AI)
    synthesis["npv_vs_horizon_data"] = npv_horizon_data
    synthesis["computed_weighted_scores"] = weighted_scores

    # Ensure horizon sensitivity data exists
    if "horizon_sensitivity" not in synthesis:
        synthesis["horizon_sensitivity"] = _compute_horizon_sensitivity(
            npv_horizon_data, optimization_results,
        )

    return synthesis


def _compute_horizon_sensitivity(npv_data: dict, optimization_results: dict) -> list[dict]:
    """Compute horizon sensitivity from NPV data."""
    sensitivity = []

    for horizon in [10, 20, 30]:
        best_tech = None
        best_npv = -float("inf")
        second_tech = None
        second_npv = -float("inf")

        for tech_id, points in npv_data.items():
            # Find NPV at this horizon (interpolate if needed)
            npv = None
            for p in points:
                if p["year"] == horizon:
                    npv = p["npv_USD"]
                    break
            if npv is None:
                # Nearest available
                closest = min(points, key=lambda p: abs(p["year"] - horizon))
                npv = closest["npv_USD"]

            if npv > best_npv:
                second_tech = best_tech
                second_npv = best_npv
                best_tech = tech_id
                best_npv = npv
            elif npv > second_npv:
                second_tech = tech_id
                second_npv = npv

        gap = best_npv - second_npv if second_tech else 0

        if horizon <= 10:
            factor = "Capital recovery speed dominates"
        elif horizon <= 20:
            factor = "Balanced capital and efficiency"
        else:
            factor = "Efficiency compounds over long horizon"

        sensitivity.append({
            "horizon_years": horizon,
            "recommended": best_tech,
            "npv_leader": best_tech,
            "npv_gap_USD": gap,
            "driving_factor": factor,
        })

    return sensitivity


def _fallback_synthesis(
    optimization_results: dict,
    dimension_results: dict,
    probability_results: dict | None,
    weighted_scores: dict,
    design_basis: dict,
) -> dict:
    """Generate deterministic synthesis when AI fails."""
    # Rank by weighted score
    ranking = sorted(weighted_scores.keys(), key=lambda t: weighted_scores[t], reverse=True)

    recommended = ranking[0] if ranking else None
    second = ranking[1] if len(ranking) > 1 else None

    # Determine confidence
    if recommended and second:
        gap = weighted_scores[recommended] - weighted_scores[second]
        if gap > 1.5:
            confidence = "High"
        elif gap > 0.5:
            confidence = "Medium"
        else:
            confidence = "Low"
    else:
        confidence = "Low"

    return {
        "weighted_scores": weighted_scores,
        "recommended_technology": recommended,
        "recommendation_confidence": confidence,
        "second_best": second,
        "second_best_conditions": "If capital constraints are more binding.",
        "dimension_tensions": [],
        "horizon_sensitivity": [],
        "weight_sensitivity": "Not computed in fallback mode.",
        "structural_innovations": [],
        "information_gaps": [
            "AI synthesis unavailable -- using heuristic scoring only"
        ],
        "executive_summary": (
            f"Based on weighted scoring, {recommended} is the recommended technology "
            f"with {confidence} confidence. "
            f"{second} is the second-best option."
            if recommended else
            "Unable to determine recommendation."
        ),
    }
