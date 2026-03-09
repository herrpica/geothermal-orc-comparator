"""
Dimension Bots — Six specialized evaluation bots for technology comparison.

Each bot evaluates ALL technologies on its specific dimension and returns
scores 1-10 with narrative. Run sequentially so each sees prior bot results.
"""

import json
from typing import Any


# ── Bot Definitions ───────────────────────────────────────────────────────────

DIMENSION_BOTS = {

    "thermal_efficiency": {
        "name": "Thermal Efficiency Bot",
        "icon": "E",
        "system_prompt": """
You are a thermodynamics expert. Your ONLY concern is how efficiently each technology
converts available brine enthalpy to electricity.

Evaluate on: first-law efficiency, second-law (exergetic) efficiency, brine utilization,
temperature matching quality (T-Q diagram), entropy generation sources.

Score each technology 1-10. Cite temperatures, efficiencies, and thermodynamic reasoning.
You are NOT allowed to consider cost, schedule, or operational complexity.

Horizon context: {horizon_context}
        """
    },

    "capital_complexity": {
        "name": "Capital & Complexity Bot",
        "icon": "C",
        "system_prompt": """
You are a construction and capital cost expert. Your ONLY concern is first installed
cost and construction feasibility.

Evaluate on: total installed $/kW, equipment count, construction schedule to first power,
specialist contractor requirements, modular vs field-constructed ratio, equipment lead
times, vendor ecosystem competitiveness.

Score each technology 1-10. Be specific about what drives cost differences.
You are NOT allowed to consider efficiency or long-term operations.

Horizon context: {horizon_context}
        """
    },

    "operations": {
        "name": "Operations Bot",
        "icon": "O",
        "system_prompt": """
You are an operations expert with 20 years running geothermal plants. Your ONLY concern
is what it takes to run each technology reliably for 30 years with available workforce.

Evaluate on: operational complexity, staffing requirements and skill level, NCG sensitivity,
scaling and chemistry management, unplanned downtime risk, startup/shutdown complexity,
remote monitoring capability, working fluid safety.

Score each technology 1-10. Draw on specific operational experience.
You are NOT allowed to consider capital cost or efficiency.
        """
    },

    "om_pl": {
        "name": "O&M and P&L Bot",
        "icon": "$",
        "system_prompt": """
You are a financial analyst focused on 30-year project economics. Your ONLY concern
is the long-term P&L of each technology.

Evaluate on: annual O&M cost ($/kW-year), major overhaul cycles, working fluid replacement,
availability factor, revenue reliability over time, insurance cost premium, parasitic load
degradation trend, residual value at end of plant life, LCOE and NPV from model outputs.

Score each technology 1-10. Be specific about cost drivers and NPV differences.
You are NOT allowed to consider construction cost or efficiency.

Horizon context: {horizon_context}
        """
    },

    "technology_risk": {
        "name": "Technology Risk Bot",
        "icon": "!",
        "system_prompt": """
You are a technology risk assessor. Your ONLY concern is what can go wrong and how badly.
10 = lowest risk, 1 = highest risk.

Evaluate on: commercial maturity (reference plants, at what scale), performance vs design
track record, vendor stability, spare parts availability over 30 years, regulatory complexity,
technology stranding risk, unique failure modes, financing and insurance acceptance.

For emerging technologies be rigorous -- no commercial deployment is a real risk, quantify it.
Be specific about risk drivers.
        """
    },

    "probability": {
        "name": "Probability & Occam's Razor Bot",
        "icon": "P",
        "system_prompt": """
You are a probabilistic risk analyst applying Occam's razor to engineering design.

Core proposition: simpler technology has higher joint probability of achieving design
performance because fewer nodes must perform simultaneously. Every additional fluid system,
rotating equipment train, and specialty contractor scope is a probability node.
Joint probability = product of individual probabilities -- it falls fast as complexity grows.

For EGS specifically: failure modes are correlated through the brine system. A scaling event
affects all heat exchangers simultaneously. NCG spikes propagate through every system touching
the working fluid. Correlated failures compound -- the left tail of the outcome distribution
is much worse than independent probability analysis suggests.

For each technology estimate:
- Number of independent probability nodes
- Key correlated failure modes
- Joint probability of achieving >=95% of design performance
- P10 / P50 / P90 NPV distribution width (qualitative)
- Complexity penalty vs simplest viable option

Score each technology 1-10 on probability of achieving design performance.
Produce an Occam's Razor verdict: is the complexity of more sophisticated technologies
quantitatively justified, or does simplicity win on risk-adjusted NPV?
        """
    },
}


def _build_evaluation_prompt(
    dim_id: str,
    optimization_results: dict,
    prior_dimension_results: dict,
    probability_results: dict | None = None,
    design_basis: dict | None = None,
) -> str:
    """Build the evaluation prompt for a dimension bot."""
    design_basis = design_basis or {}
    horizon = design_basis.get("plant_life_years", 30)

    if horizon <= 12:
        horizon_context = ("SHORT HORIZON (<=12 years): Capital recovery speed dominates. "
                          "Favor lower CAPEX and faster schedule over long-term efficiency.")
    elif horizon <= 22:
        horizon_context = ("MEDIUM HORIZON (13-22 years): Balanced evaluation. "
                          "Efficiency gains compound enough to justify modest capital premium.")
    else:
        horizon_context = ("LONG HORIZON (23+ years): Efficiency compounds significantly. "
                          "Higher capital can be justified if it delivers sustained output.")

    dim_config = DIMENSION_BOTS[dim_id]
    system = dim_config["system_prompt"].format(horizon_context=horizon_context)

    # Build technology results summary
    tech_summaries = []
    for tech_id, result in optimization_results.items():
        summary = {
            "technology_id": tech_id,
            "net_power_MW": result.get("net_power_MW", 0),
            "gross_power_MW": result.get("gross_power_MW", 0),
            "parasitic_MW": result.get("parasitic_MW", 0),
            "cycle_efficiency": result.get("cycle_efficiency", 0),
            "capex_total_USD": result.get("capex_total_USD", 0),
            "capex_per_kW": result.get("capex_per_kW", 0),
            "opex_annual_USD": result.get("opex_annual_USD", 0),
            "lcoe_per_MWh": result.get("lcoe_per_MWh", 0),
            "npv_USD": result.get("npv_USD", 0),
            "construction_weeks": result.get("construction_weeks", 0),
            "model_confidence": result.get("model_confidence", "unknown"),
            "warnings": result.get("warnings", []),
        }
        tech_summaries.append(summary)

    content = f"Evaluate these technologies:\n\n{json.dumps(tech_summaries, indent=2, default=str)}"

    # Add prior dimension results
    if prior_dimension_results:
        content += "\n\nPrior dimension evaluations:\n"
        for prior_dim, prior_result in prior_dimension_results.items():
            prior_name = DIMENSION_BOTS[prior_dim]["name"]
            content += f"\n{prior_name}:\n"
            if "scores" in prior_result:
                for tid, score in prior_result["scores"].items():
                    content += f"  {tid}: {score:.1f}/10\n"
            if "key_differentiators" in prior_result:
                content += f"  Key: {prior_result['key_differentiators']}\n"

    # Add probability results if available
    if probability_results and dim_id == "probability":
        mc = probability_results.get("monte_carlo", {})
        content += "\n\nMonte Carlo simulation results:\n"
        for tech_id, mc_result in mc.items():
            content += (f"  {tech_id}: P10=${mc_result['npv_p10']/1e6:.1f}M, "
                       f"P50=${mc_result['npv_p50']/1e6:.1f}M, "
                       f"P90=${mc_result['npv_p90']/1e6:.1f}M, "
                       f"Joint P={mc_result['joint_probability']:.3f}\n")

    content += """

Respond with ONLY valid JSON:
{
  "scores": {"<technology_id>": <float 1-10>},
  "ranking": ["<technology_id from best to worst>"],
  "narrative": "<2-3 paragraphs>",
  "key_differentiators": ["<string>"],
  "surprises": ["<string>"]"""

    if dim_id == "probability":
        content += ',\n  "occams_verdict": "<paragraph on whether complexity is justified>"'

    content += "\n}"

    return system, content


def run_dimension_bot(
    dim_id: str,
    optimization_results: dict,
    prior_dimension_results: dict,
    probability_results: dict | None,
    design_basis: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run a single dimension bot evaluation.

    Parameters
    ----------
    dim_id : str
        Dimension bot identifier
    optimization_results : dict
        {technology_id: standard_output} for all technologies
    prior_dimension_results : dict
        {dim_id: result} for previously run dimension bots
    probability_results : dict, optional
        Monte Carlo probability results
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    model : str
        Claude model to use

    Returns
    -------
    dict
        Scores, ranking, narrative, and key differentiators
    """
    import anthropic

    system, content = _build_evaluation_prompt(
        dim_id, optimization_results, prior_dimension_results,
        probability_results, design_basis,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=3000,
            system=system,
            messages=[{"role": "user", "content": content}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)

        # Ensure scores are floats
        if "scores" in result:
            result["scores"] = {k: float(v) for k, v in result["scores"].items()}

        result["dimension_id"] = dim_id
        result["dimension_name"] = DIMENSION_BOTS[dim_id]["name"]
        return result

    except json.JSONDecodeError:
        return _fallback_dimension_scores(dim_id, optimization_results)
    except Exception as e:
        result = _fallback_dimension_scores(dim_id, optimization_results)
        result["error"] = str(e)
        return result


def _fallback_dimension_scores(dim_id: str, optimization_results: dict) -> dict:
    """Generate simple heuristic scores when AI evaluation fails."""
    scores = {}
    tech_ids = list(optimization_results.keys())

    for tech_id in tech_ids:
        r = optimization_results[tech_id]
        if dim_id == "thermal_efficiency":
            eff = r.get("cycle_efficiency", 0)
            scores[tech_id] = min(10, max(1, eff / 0.025))  # normalize ~0.10-0.25 to 4-10
        elif dim_id == "capital_complexity":
            cpkw = r.get("capex_per_kW", 5000)
            scores[tech_id] = min(10, max(1, 10 - cpkw / 1000))  # lower $/kW = higher score
        elif dim_id == "operations":
            conf = r.get("model_confidence", "low")
            scores[tech_id] = {"high": 8, "medium": 5, "low": 3}.get(conf, 3)
        elif dim_id == "om_pl":
            npv = r.get("npv_USD", 0)
            scores[tech_id] = min(10, max(1, npv / 30e6))
        elif dim_id == "technology_risk":
            conf = r.get("model_confidence", "low")
            scores[tech_id] = {"high": 9, "medium": 5, "low": 2}.get(conf, 2)
        elif dim_id == "probability":
            conf = r.get("model_confidence", "low")
            scores[tech_id] = {"high": 8, "medium": 5, "low": 3}.get(conf, 3)
        else:
            scores[tech_id] = 5.0

    ranking = sorted(scores.keys(), key=lambda t: scores[t], reverse=True)

    return {
        "dimension_id": dim_id,
        "dimension_name": DIMENSION_BOTS[dim_id]["name"],
        "scores": scores,
        "ranking": ranking,
        "narrative": "Heuristic scores (AI evaluation unavailable).",
        "key_differentiators": [],
        "surprises": [],
        "fallback": True,
    }


def run_all_dimension_bots(
    optimization_results: dict,
    probability_results: dict | None,
    design_basis: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
    on_bot_complete: Any = None,
) -> dict:
    """Run all six dimension bots sequentially (each sees prior results).

    Parameters
    ----------
    optimization_results : dict
        {technology_id: standard_output}
    probability_results : dict, optional
        Monte Carlo results
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    model : str
        Claude model to use
    on_bot_complete : callable, optional
        Callback(dim_id, result) for progress

    Returns
    -------
    dict
        {dim_id: evaluation_result} for all six bots
    """
    dim_results = {}
    bot_order = [
        "thermal_efficiency",
        "capital_complexity",
        "operations",
        "om_pl",
        "technology_risk",
        "probability",
    ]

    for dim_id in bot_order:
        result = run_dimension_bot(
            dim_id, optimization_results, dim_results,
            probability_results, design_basis, api_key, model,
        )
        dim_results[dim_id] = result

        if on_bot_complete:
            on_bot_complete(dim_id, result)

    return dim_results
