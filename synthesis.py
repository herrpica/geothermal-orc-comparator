"""
Synthesis module for the ORC Design Dialectic.

Takes a completed debate transcript and all analysis runs, calls Claude
as an impartial arbitrator to extract the converged design into structured
JSON output including optimized parameters and structural recommendations.

Also produces the deck_summary.json for leadership presentations.
"""

import json
import re
from typing import Any

import anthropic

from analysis_bridge import get_structural_proposals, run_orc_analysis
from debate_engine import (
    DebateState,
    analysis_runs_to_json,
    transcript_to_text,
    DEFAULT_MODEL,
)

# ── Synthesis system prompt ─────────────────────────────────────────────────

SYNTHESIS_SYSTEM = """You are an engineering arbitrator reviewing a design debate between two ORC \
engineers. Your job is to extract the optimal design from the evidence.

Review the full conversation and all analysis runs. Identify:
1. The parameter set that achieved the highest NPV-weighted score
2. Any concessions made by either party that constrain the design space
3. Structural recommendations proposed by either party with sufficient \
engineering basis to include

Return a single valid JSON object with this exact structure — no other text:

{
  "architecture": "A" | "B" | "hybrid",
  "architecture_rationale": str,
  "optimized_parameters": { [full input dict for run_orc_analysis] },
  "predicted_performance": {
    "net_power_MW": float,
    "gross_power_MW": float,
    "parasitic_MW": float,
    "cycle_efficiency": float,
    "capex_total_USD": float,
    "capex_per_kW": float,
    "npv_USD": float,
    "lcoe_per_MWh": float,
    "construction_weeks_critical_path": int
  },
  "structural_recommendations": [
    {
      "title": str,
      "description": str,
      "proposed_by": "PropaneFirst" | "Conventionalist",
      "contested": bool,
      "contest_reasoning": str | null,
      "estimated_npv_delta_pct": float,
      "confidence": str,
      "engineering_basis": str,
      "requires_model_extension": bool
    }
  ],
  "key_debate_outcomes": [str],
  "model_extension_recommendations": [str],
  "concessions": {
    "PropaneFirst": [str],
    "Conventionalist": [str]
  }
}"""


# ── Build context for the synthesis call ────────────────────────────────────

def _build_synthesis_context(state: DebateState) -> str:
    """Build the full context string for the synthesis Claude call."""
    parts = []

    # Design basis
    parts.append("## Design Basis")
    parts.append(json.dumps(state.design_basis, indent=2, default=str))

    # Full debate transcript
    parts.append("\n## Debate Transcript")
    parts.append(transcript_to_text(state))

    # All analysis runs with full results
    parts.append("\n## All Analysis Runs")
    runs = analysis_runs_to_json(state)
    for i, run in enumerate(runs):
        parts.append(f"\n### Run {i+1} — Round {run['round']}, {run['bot']}, Config {run['config']}")
        parts.append(f"Input: {json.dumps(run['input'], indent=2)}")
        # Exclude _detail from synthesis context to save tokens
        result_clean = {k: v for k, v in run['result'].items() if k != '_detail'}
        parts.append(f"Result: {json.dumps(result_clean, indent=2, default=str)}")

    # Structural proposals
    proposals = get_structural_proposals()
    if proposals:
        parts.append("\n## Structural Change Proposals")
        for p in proposals:
            parts.append(json.dumps(p, indent=2))

    # Convergence info
    if state.converged:
        parts.append(f"\n## Convergence: {state.convergence_reason}")
    parts.append(f"Total rounds completed: {state.current_round}")
    parts.append(f"Total analysis runs: {len(state.analysis_runs)}")

    # Objective weights
    parts.append(f"\n## Objective Weights: {json.dumps(state.objective_weights)}")

    return "\n".join(parts)


# ── JSON extraction ─────────────────────────────────────────────────────────

def _extract_json(text: str) -> dict:
    """Extract JSON object from Claude response text.

    Handles responses wrapped in markdown code fences or with
    surrounding text.
    """
    # Try direct parse first
    text = text.strip()
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

    # Try extracting from code fence
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try finding the outermost {} pair
    start = text.find("{")
    if start >= 0:
        depth = 0
        for i in range(start, len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    try:
                        return json.loads(text[start:i+1])
                    except json.JSONDecodeError:
                        break

    raise ValueError(f"Could not extract valid JSON from synthesis response:\n{text[:500]}")


# ── Main synthesis function ─────────────────────────────────────────────────

def synthesize_debate(
    state: DebateState,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> dict:
    """Run the synthesis arbitrator on a completed debate.

    Parameters
    ----------
    state : DebateState
        Completed debate state with transcript and analysis runs.
    api_key : str, optional
        Anthropic API key.
    model : str
        Claude model ID.

    Returns
    -------
    dict
        The converged_design JSON matching the synthesis schema.
    """
    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs)

    context = _build_synthesis_context(state)

    response = client.messages.create(
        model=model,
        max_tokens=8192,
        system=SYNTHESIS_SYSTEM,
        messages=[{"role": "user", "content": context}],
    )

    # Extract JSON from response
    response_text = ""
    for block in response.content:
        if block.type == "text":
            response_text += block.text

    synthesis = _extract_json(response_text)

    # Validate the synthesis by running the optimized parameters
    if "optimized_parameters" in synthesis:
        try:
            validation_result = run_orc_analysis(
                synthesis["optimized_parameters"],
                state.design_basis,
            )
            synthesis["validated_performance"] = {
                k: v for k, v in validation_result.items() if k != "_detail"
            }
        except Exception as e:
            synthesis["validation_error"] = str(e)

    return synthesis


# ── Deck summary builder ───────────────────────────────────────────────────

def build_deck_summary(
    synthesis: dict,
    state: DebateState,
) -> dict:
    """Build a structured summary for leadership deck generation.

    Returns a dict formatted for slide content extraction:
    - Quantitative comparison table data
    - Structural recommendations as bullet points
    - Key debate outcomes as talking points
    """
    # Gather best runs for each config
    config_a_runs = [r for r in state.analysis_runs if r.config == "A"]
    config_b_runs = [r for r in state.analysis_runs if r.config == "B"]

    def _best_run(runs):
        if not runs:
            return None
        return max(runs, key=lambda r: r.result.get("npv_USD", float("-inf")))

    best_a = _best_run(config_a_runs)
    best_b = _best_run(config_b_runs)

    # Comparison table
    comparison_table = {"metrics": []}
    metrics = [
        ("Net Power (MW)", "net_power_MW", ".1f"),
        ("Gross Power (MW)", "gross_power_MW", ".1f"),
        ("Parasitic Load (MW)", "parasitic_MW", ".1f"),
        ("Cycle Efficiency (%)", "cycle_efficiency", ".1%"),
        ("Total CAPEX ($M)", "capex_total_USD", ",.0f"),
        ("CAPEX per kW ($/kW)", "capex_per_kW", ",.0f"),
        ("NPV ($M)", "npv_USD", ",.0f"),
        ("LCOE ($/MWh)", "lcoe_per_MWh", ".1f"),
        ("Schedule (weeks)", "construction_weeks_critical_path", "d"),
    ]

    for label, key, fmt in metrics:
        row = {"metric": label}
        if best_a:
            val_a = best_a.result.get(key, 0)
            if "USD" in key or "capex" in key.lower():
                if "M" in label:
                    row["config_a"] = f"${val_a/1e6:{fmt}}"
                else:
                    row["config_a"] = f"${val_a:{fmt}}"
            elif "%" in label:
                row["config_a"] = f"{val_a:{fmt}}"
            else:
                row["config_a"] = f"{val_a:{fmt}}"
        if best_b:
            val_b = best_b.result.get(key, 0)
            if "USD" in key or "capex" in key.lower():
                if "M" in label:
                    row["config_b"] = f"${val_b/1e6:{fmt}}"
                else:
                    row["config_b"] = f"${val_b:{fmt}}"
            elif "%" in label:
                row["config_b"] = f"{val_b:{fmt}}"
            else:
                row["config_b"] = f"{val_b:{fmt}}"

        # Winner indicator
        if best_a and best_b:
            va = best_a.result.get(key, 0)
            vb = best_b.result.get(key, 0)
            # Higher is better for power/efficiency/NPV, lower for cost/schedule/LCOE
            if key in ("capex_total_USD", "capex_per_kW", "lcoe_per_MWh",
                       "construction_weeks_critical_path", "parasitic_MW"):
                row["winner"] = "A" if va < vb else "B"
            else:
                row["winner"] = "A" if va > vb else "B"

        comparison_table["metrics"].append(row)

    # Structural recommendations for slides
    structural_slides = []
    for rec in synthesis.get("structural_recommendations", []):
        structural_slides.append({
            "title": rec.get("title", ""),
            "bullet": rec.get("description", ""),
            "proposed_by": rec.get("proposed_by", ""),
            "confidence": rec.get("confidence", ""),
            "npv_impact": f"{rec.get('estimated_npv_delta_pct', 0):+.1f}%",
            "requires_further_study": rec.get("requires_model_extension", False),
        })

    # Talking points from debate outcomes
    talking_points = synthesis.get("key_debate_outcomes", [])

    # Concessions summary
    concessions = synthesis.get("concessions", {})

    deck = {
        "title": "ORC Design Optimization — Dialectic Analysis Results",
        "recommended_architecture": synthesis.get("architecture", "TBD"),
        "architecture_rationale": synthesis.get("architecture_rationale", ""),
        "comparison_table": comparison_table,
        "structural_recommendations": structural_slides,
        "talking_points": talking_points,
        "concessions": concessions,
        "debate_stats": {
            "total_rounds": state.current_round,
            "total_analysis_runs": len(state.analysis_runs),
            "convergence_reason": state.convergence_reason,
            "converged": state.converged,
        },
        "design_basis_summary": {
            k: v for k, v in state.design_basis.items()
            if k not in ("objective_weights", "max_rounds")
        },
        "model_extension_recommendations": synthesis.get(
            "model_extension_recommendations", []
        ),
    }

    return deck


# ── Output file generators ──────────────────────────────────────────────────

def save_converged_design(synthesis: dict, path: str = "converged_design.json"):
    """Save the full synthesis JSON to file."""
    with open(path, "w") as f:
        json.dump(synthesis, f, indent=2, default=str)
    return path


def save_deck_summary(deck: dict, path: str = "deck_summary.json"):
    """Save the deck summary JSON to file."""
    with open(path, "w") as f:
        json.dump(deck, f, indent=2, default=str)
    return path
