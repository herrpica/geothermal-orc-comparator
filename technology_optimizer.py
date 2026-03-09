"""
Technology Optimizer — Screening layer + per-technology optimization agents.

Stage 1: Screening Claude call reviews resource and screens all technologies.
Stage 2: Each viable technology gets an optimization agent that iterates
         design point -> analysis -> refine until convergence.
"""

import json
from typing import Any, Generator

from technology_registry import TECHNOLOGIES, TechnologyDefinition
from technology_analysis_bridge import analyze_technology

# ── Screening ─────────────────────────────────────────────────────────────────

SCREENING_PROMPT = """
You are a geothermal technology screening engineer. Review this resource and screen
all technologies for viability. For each return:

  VIABLE -- proceed to full optimization
  MARGINAL -- optimize with specific caveats
  EXCLUDE -- do not optimize, with specific technical reason

Exclusion must cite specific thermodynamic or practical constraint. Example:
"Single flash EXCLUDED -- brine at 135C produces steam quality of 0.08 at optimal
flash pressure of 3.5 bar, yielding <40% of ORC output at higher capital cost."

For CHP: only VIABLE if user notes a heat market in site notes.
For reference technologies (MHD): always EXCLUDE with brief technical note.

Resource:
{design_basis_json}

Technologies to evaluate:
{technology_list}

Return ONLY valid JSON:
{{
  "screening_results": {{
    "<technology_id>": {{
      "status": "VIABLE"|"MARGINAL"|"EXCLUDE",
      "reasoning": "<string>",
      "caveats": ["<string>"]
    }}
  }},
  "screening_narrative": "<2-3 paragraph overview>"
}}
"""


def _build_technology_list(design_basis: dict) -> str:
    """Build technology list string for the screening prompt."""
    lines = []
    brine_T = design_basis.get("brine_inlet_temp_C", 200)
    for tech in TECHNOLOGIES.values():
        in_range = "IN RANGE" if tech.viable_brine_temp_min_C <= brine_T <= tech.viable_brine_temp_max_C else "OUT OF RANGE"
        lines.append(
            f"- {tech.id}: {tech.name} ({tech.category}) "
            f"[{tech.viable_brine_temp_min_C:.0f}-{tech.viable_brine_temp_max_C:.0f}C] "
            f"[{in_range}] "
            f"Notes: {tech.screening_notes}"
        )
    return "\n".join(lines)


def run_screening(
    design_basis: dict,
    api_key: str,
    research_results: dict | None = None,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run technology screening against the resource.

    Parameters
    ----------
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    research_results : dict, optional
        Research monitor results for emerging technologies
    model : str
        Claude model to use

    Returns
    -------
    dict
        Screening results with status per technology and narrative
    """
    import anthropic

    # Build context
    basis_json = json.dumps(design_basis, indent=2, default=str)
    tech_list = _build_technology_list(design_basis)

    # Add research context if available
    if research_results:
        tech_list += "\n\nResearch Monitor Updates:\n"
        for tech_id, result in research_results.items():
            if result.get("baseline_change_flag"):
                tech_list += f"  {tech_id}: {result.get('narrative', 'No narrative')[:200]}\n"

    prompt = SCREENING_PROMPT.format(
        design_basis_json=basis_json,
        technology_list=tech_list,
    )

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=4000,
            system=(
                "You are a geothermal technology screening engineer with 25 years of experience. "
                "Be rigorous and specific in your screening decisions. "
                "Cite thermodynamic limits and commercial evidence."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        return json.loads(text)

    except json.JSONDecodeError:
        # Fallback: temperature-based screening
        return _fallback_screening(design_basis)
    except Exception as e:
        return _fallback_screening(design_basis, error=str(e))


def _fallback_screening(design_basis: dict, error: str | None = None) -> dict:
    """Deterministic fallback screening based on temperature ranges."""
    brine_T = design_basis.get("brine_inlet_temp_C", 200)
    site_notes = design_basis.get("site_notes", "")
    has_heat_market = any(kw in site_notes.lower()
                         for kw in ["heat", "district", "greenhouse", "industrial"])

    results = {}
    for tech in TECHNOLOGIES.values():
        if tech.id == "mhd":
            status = "EXCLUDE"
            reasoning = "MHD requires >500C. Geothermal brine far below viable range."
        elif tech.id == "geothermal_chp" and not has_heat_market:
            status = "EXCLUDE"
            reasoning = "No heat market indicated in site notes."
        elif brine_T < tech.viable_brine_temp_min_C:
            status = "EXCLUDE"
            reasoning = (f"Brine at {brine_T:.0f}C below minimum viable "
                        f"temperature {tech.viable_brine_temp_min_C:.0f}C.")
        elif brine_T > tech.viable_brine_temp_max_C:
            status = "EXCLUDE"
            reasoning = (f"Brine at {brine_T:.0f}C above maximum viable "
                        f"temperature {tech.viable_brine_temp_max_C:.0f}C.")
        elif tech.category == "research":
            status = "MARGINAL"
            reasoning = f"Research-stage technology. Include for comparison only."
        elif tech.category == "emerging":
            status = "MARGINAL"
            reasoning = f"Emerging technology. Pre-commercial — include with caveats."
        elif brine_T < tech.viable_brine_temp_min_C + 20:
            status = "MARGINAL"
            reasoning = f"Brine at {brine_T:.0f}C near lower limit of viable range."
        else:
            status = "VIABLE"
            reasoning = f"Resource within viable range for {tech.name}."

        results[tech.id] = {
            "status": status,
            "reasoning": reasoning,
            "caveats": [],
        }

    narrative = f"Deterministic screening at {brine_T:.0f}C brine temperature."
    if error:
        narrative += f" (AI screening failed: {error})"

    return {
        "screening_results": results,
        "screening_narrative": narrative,
    }


# ── Optimization Agents ──────────────────────────────────────────────────────

OPTIMIZATION_PROMPT = """
You are optimizing a {technology_name} for this geothermal resource.

Resource: {design_basis_json}
Objective weights: efficiency {eff_w:.0%} / cost {cost_w:.0%} / schedule {sched_w:.0%}
Plant horizon: {horizon_years} years  (THIS AFFECTS WHAT OPTIMAL MEANS)

{technology_expert_prompt}

Round {round_num}. Previous best NPV: ${previous_npv:,.0f}. Previous params: {previous_params}

This round:
1. Propose a specific design point with full engineering rationale
2. The system will run the analysis with those parameters
3. Identify which parameter has highest NPV sensitivity -- adjust it
4. If you see an innovation outside the current model, describe it as a structural_change
5. Converge when NPV improvement between rounds < 2% (max {max_rounds} rounds)

After convergence provide:
- Final optimized parameter set
- Key design decisions and why
- Main strengths and weaknesses for this resource
- One-paragraph technology narrative for the comparison table

Respond with ONLY valid JSON:
{{
  "parameters": {{<technology-specific params>}},
  "rationale": "<engineering reasoning>",
  "sensitivity_target": "<parameter to adjust next>",
  "structural_changes": [{{
    "title": "<string>",
    "description": "<string>",
    "estimated_npv_delta_pct": <float>
  }}],
  "converged": <true|false>,
  "convergence_reason": "<string if converged>",
  "technology_narrative": "<string if converged>"
}}
"""


def optimize_technology(
    technology_id: str,
    design_basis: dict,
    api_key: str,
    research_results: dict | None = None,
    max_rounds: int = 4,
    model: str = "claude-sonnet-4-20250514",
    on_round: Any = None,
) -> dict:
    """Run optimization agent for a single technology.

    The agent iterates: propose params -> analyze -> refine -> converge.

    Parameters
    ----------
    technology_id : str
        Technology to optimize
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    research_results : dict, optional
        Research findings for this technology
    max_rounds : int
        Maximum optimization rounds
    model : str
        Claude model to use
    on_round : callable, optional
        Callback(round_num, result) for progress updates

    Returns
    -------
    dict
        Best analysis result with optimization metadata
    """
    import anthropic

    tech = TECHNOLOGIES.get(technology_id)
    if tech is None:
        return {"error": f"Unknown technology: {technology_id}"}

    client = anthropic.Anthropic(api_key=api_key)

    weights = design_basis.get("objective_weights", {
        "efficiency": 0.4, "capital_cost": 0.4, "schedule": 0.2,
    })
    horizon = design_basis.get("plant_life_years", 30)

    best_result = None
    best_npv = -float("inf")
    best_params = {}
    all_rounds = []
    structural_changes = []

    previous_npv = 0
    previous_params = "{}"

    for round_num in range(1, max_rounds + 1):
        prompt = OPTIMIZATION_PROMPT.format(
            technology_name=tech.name,
            design_basis_json=json.dumps(design_basis, indent=2, default=str),
            eff_w=weights.get("efficiency", 0.4),
            cost_w=weights.get("capital_cost", 0.4),
            sched_w=weights.get("schedule", 0.2),
            horizon_years=horizon,
            technology_expert_prompt=tech.expert_prompt,
            round_num=round_num,
            previous_npv=previous_npv,
            previous_params=previous_params,
            max_rounds=max_rounds,
        )

        # Add research context
        if research_results and technology_id in research_results:
            research = research_results[technology_id]
            prompt += f"\n\nResearch update: {json.dumps(research, indent=2, default=str)}"

        try:
            response = client.messages.create(
                model=model,
                max_tokens=2000,
                system=(
                    f"You are a {tech.name} optimization engineer. "
                    "Propose specific numerical parameters for the design point. "
                    "Always respond with valid JSON."
                ),
                messages=[{"role": "user", "content": prompt}],
            )

            text = response.content[0].text.strip()
            if text.startswith("```"):
                text = text.split("```")[1]
                if text.startswith("json"):
                    text = text[4:]

            agent_response = json.loads(text)
            params = agent_response.get("parameters", {})

            # Run analysis
            result = analyze_technology(technology_id, params, design_basis)
            npv = result.get("npv_USD", 0)

            round_info = {
                "round": round_num,
                "params": params,
                "result_summary": {
                    "net_power_MW": result.get("net_power_MW", 0),
                    "capex_total_USD": result.get("capex_total_USD", 0),
                    "npv_USD": npv,
                    "lcoe_per_MWh": result.get("lcoe_per_MWh", 0),
                },
                "rationale": agent_response.get("rationale", ""),
                "converged": agent_response.get("converged", False),
            }
            all_rounds.append(round_info)

            if npv > best_npv:
                best_npv = npv
                best_result = result
                best_params = params

            # Collect structural changes
            for sc in agent_response.get("structural_changes", []):
                sc["technology_id"] = technology_id
                sc["round"] = round_num
                structural_changes.append(sc)

            if on_round:
                on_round(round_num, round_info)

            # Check convergence
            if agent_response.get("converged", False):
                break

            if round_num > 1 and previous_npv > 0:
                improvement = abs(npv - previous_npv) / abs(previous_npv)
                if improvement < 0.02:
                    break

            previous_npv = npv
            previous_params = json.dumps(params, indent=2)

        except json.JSONDecodeError:
            # If agent response isn't valid JSON, run with defaults
            result = analyze_technology(technology_id, {}, design_basis)
            if result.get("npv_USD", 0) > best_npv:
                best_result = result
                best_npv = result.get("npv_USD", 0)
            break
        except Exception as e:
            all_rounds.append({
                "round": round_num,
                "error": str(e),
            })
            break

    # Ensure we have at least a default result
    if best_result is None:
        best_result = analyze_technology(technology_id, {}, design_basis)

    best_result["optimization_metadata"] = {
        "rounds_completed": len(all_rounds),
        "best_params": best_params,
        "all_rounds": all_rounds,
        "structural_changes": structural_changes,
        "technology_narrative": (
            all_rounds[-1].get("technology_narrative", "")
            if all_rounds else ""
        ),
    }

    return best_result


def optimize_all_viable(
    viable_technology_ids: list[str],
    design_basis: dict,
    api_key: str,
    research_results: dict | None = None,
    max_rounds: int = 4,
    model: str = "claude-sonnet-4-20250514",
    on_technology_complete: Any = None,
    on_round: Any = None,
) -> dict:
    """Optimize all viable technologies sequentially.

    Parameters
    ----------
    viable_technology_ids : list[str]
        Technologies that passed screening
    design_basis : dict
        Design basis from sidebar
    api_key : str
        Anthropic API key
    research_results : dict, optional
        Research monitor results
    max_rounds : int
        Max optimization rounds per technology
    model : str
        Claude model to use
    on_technology_complete : callable, optional
        Callback(technology_id, result) for progress
    on_round : callable, optional
        Callback(round_num, round_info) for per-round progress

    Returns
    -------
    dict
        {technology_id: optimized_result} for all viable technologies
    """
    results = {}

    for tech_id in viable_technology_ids:
        result = optimize_technology(
            tech_id, design_basis, api_key,
            research_results=research_results,
            max_rounds=max_rounds,
            model=model,
            on_round=on_round,
        )
        results[tech_id] = result

        if on_technology_complete:
            on_technology_complete(tech_id, result)

    return results
