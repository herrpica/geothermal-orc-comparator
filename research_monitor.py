"""
Research Monitor — Literature search for emerging geothermal technologies.

For technologies with research_monitor=True, uses Claude to assess current
state-of-the-art, demonstrated performance, and timeline to commercial readiness.
"""

import json
from typing import Any

from technology_registry import TECHNOLOGIES, get_research_monitored


RESEARCH_PROMPT = """
Search recent technical literature for advances in {technology_name} relevant to
geothermal power generation at {temp_range_C} C, target scale {target_MW:.0f} MW.

Find and report:
1. Best demonstrated efficiency or ZT at this temperature range
   -- distinguish lab sample vs module vs system demonstration
2. Current installed cost per kW at demonstration or pilot scale
3. Most significant advance in last 12-18 months
4. Gap between best demonstrated and realistic commercial device performance
5. Any geothermal-specific deployments or studies
6. Realistic timeline to commercial readiness at this scale
7. Key remaining technical barriers

Update these parameters for the optimization model:
  peak_efficiency_demonstrated: float
  device_efficiency_realistic: float
  current_cost_per_kW: float
  maturity_level: str
  timeline_to_commercial: str

Cite sources. Flag if any finding significantly changes the baseline assessment.

Respond with ONLY valid JSON matching this schema:
{{
  "technology_id": "{technology_id}",
  "peak_efficiency_demonstrated": <float>,
  "device_efficiency_realistic": <float>,
  "current_cost_per_kW": <float>,
  "maturity_level": "<lab|prototype|pilot|demonstration|pre-commercial>",
  "timeline_to_commercial": "<string>",
  "key_advances": ["<string>"],
  "technical_barriers": ["<string>"],
  "geothermal_deployments": ["<string>"],
  "baseline_change_flag": <true|false>,
  "narrative": "<2-3 paragraph summary>",
  "sources": ["<string>"]
}}
"""


def build_research_prompt(technology_id: str, design_basis: dict) -> str:
    """Build the research prompt for a specific technology."""
    tech = TECHNOLOGIES.get(technology_id)
    if tech is None:
        raise ValueError(f"Unknown technology: {technology_id}")

    temp_range = f"{tech.viable_brine_temp_min_C:.0f}-{tech.viable_brine_temp_max_C:.0f}"
    target_MW = design_basis.get("net_power_target_MW", 50)

    return RESEARCH_PROMPT.format(
        technology_name=tech.name,
        temp_range_C=temp_range,
        target_MW=target_MW,
        technology_id=technology_id,
    )


def run_research_monitor(
    technology_id: str,
    design_basis: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run research monitor for a single emerging technology.

    Parameters
    ----------
    technology_id : str
        Technology to research
    design_basis : dict
        Design basis for context
    api_key : str
        Anthropic API key
    model : str
        Claude model to use

    Returns
    -------
    dict
        Research findings in standardized format
    """
    import anthropic

    tech = TECHNOLOGIES.get(technology_id)
    if tech is None:
        return {"error": f"Unknown technology: {technology_id}"}

    prompt = build_research_prompt(technology_id, design_basis)

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=model,
            max_tokens=2000,
            system=(
                "You are a geothermal energy technology researcher. "
                "Provide accurate, current technical assessments based on your knowledge "
                "of published research, conference proceedings, and industry reports. "
                "Be rigorous about distinguishing lab results from commercial reality."
            ),
            messages=[{"role": "user", "content": prompt}],
        )

        text = response.content[0].text.strip()
        # Extract JSON from response
        if text.startswith("```"):
            text = text.split("```")[1]
            if text.startswith("json"):
                text = text[4:]
        result = json.loads(text)
        result["technology_id"] = technology_id
        return result

    except json.JSONDecodeError:
        # If Claude doesn't return valid JSON, wrap the text response
        return {
            "technology_id": technology_id,
            "narrative": text if 'text' in dir() else "Failed to get response",
            "peak_efficiency_demonstrated": 0,
            "device_efficiency_realistic": 0,
            "current_cost_per_kW": 0,
            "maturity_level": "unknown",
            "timeline_to_commercial": "unknown",
            "key_advances": [],
            "technical_barriers": [],
            "geothermal_deployments": [],
            "baseline_change_flag": False,
            "sources": [],
            "parse_error": True,
        }
    except Exception as e:
        return {
            "technology_id": technology_id,
            "error": str(e),
            "peak_efficiency_demonstrated": 0,
            "device_efficiency_realistic": 0,
            "current_cost_per_kW": 0,
            "maturity_level": "unknown",
            "timeline_to_commercial": "unknown",
            "key_advances": [],
            "technical_barriers": [],
            "geothermal_deployments": [],
            "baseline_change_flag": False,
            "sources": [],
        }


def run_all_research_monitors(
    design_basis: dict,
    api_key: str,
    model: str = "claude-sonnet-4-20250514",
) -> dict:
    """Run research monitors for all emerging technologies.

    Returns dict keyed by technology_id with research findings.
    """
    results = {}
    for tech in get_research_monitored():
        results[tech.id] = run_research_monitor(tech.id, design_basis, api_key, model)
    return results


def update_model_parameters(
    technology_id: str,
    research_result: dict,
    current_params: dict,
) -> dict:
    """Update optimization parameters based on research findings.

    If research_result.baseline_change_flag is True, the research found
    something that significantly changes the baseline assessment. Update
    the relevant parameters.

    Returns updated params dict (copy, doesn't modify input).
    """
    updated = dict(current_params)

    if not research_result.get("baseline_change_flag", False):
        return updated

    # Update efficiency if research found better demonstrated values
    demo_eff = research_result.get("peak_efficiency_demonstrated", 0)
    realistic_eff = research_result.get("device_efficiency_realistic", 0)

    if technology_id == "teg" and demo_eff > 0:
        # Convert efficiency to ZT estimate: eta ~ (sqrt(1+ZT)-1)/(sqrt(1+ZT)+Tc/Th)
        # Rough inverse: ZT ~ (4*eta/(1-eta))^2 - 1 for small eta
        updated["ZT_hot"] = max(updated.get("ZT_hot", 1.0), demo_eff * 10)

    if technology_id == "trilateral_flash" and realistic_eff > 0:
        updated["expander_isentropic_efficiency"] = max(
            updated.get("expander_isentropic_efficiency", 0.75),
            realistic_eff
        )

    # Update cost if research found current data
    cost_per_kW = research_result.get("current_cost_per_kW", 0)
    if cost_per_kW > 0:
        updated["research_cost_per_kW"] = cost_per_kW

    return updated
