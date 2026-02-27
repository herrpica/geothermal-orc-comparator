"""
Proposal and constraint management for the ORC Design Dialectic.

Core data model for the stop-point decision system:
  - RefinementProposal: a bot's proposed parameter change with metrics
  - ConstraintState: all proposals organized by decision status
  - ConstraintManager: decides on proposals, builds prompt injections

Used by both the decision cards UI (decision_cards.py) and the
round-by-round debate flow in dialectic.py.
"""

import uuid
from dataclasses import dataclass, field
from typing import Any


# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class RefinementProposal:
    """A single parameter refinement proposed by a debate bot."""
    proposal_id: str
    round_num: int
    proposed_by: str                    # "PropaneFirst" | "Conventionalist"
    title: str
    engineering_argument: str
    parameter_changes: dict             # run_orc_analysis param key -> new value
    baseline_metrics: dict              # metrics BEFORE (from opponent's last run)
    proposed_metrics: dict              # metrics AFTER (from this bot's run)
    confidence: str = "medium"          # "high" | "medium" | "low"
    contested_by: str | None = None
    contest_argument: str | None = None
    status: str = "pending"             # pending|locked|soft|deferred|declined|modified
    modified_parameters: dict = field(default_factory=dict)
    defer_until_round: int | None = None


@dataclass
class ConstraintState:
    """All proposals organized by decision status. Serializable to session state."""
    locked: list[RefinementProposal] = field(default_factory=list)
    soft: list[RefinementProposal] = field(default_factory=list)
    declined: list[RefinementProposal] = field(default_factory=list)
    deferred: list[RefinementProposal] = field(default_factory=list)
    pending: list[RefinementProposal] = field(default_factory=list)


# ── Friendly parameter names for display ────────────────────────────────────

PARAM_DISPLAY_NAMES = {
    "config": "Configuration",
    "evaporator_approach_delta_F": "Evaporator approach (°F)",
    "recuperator_approach_delta_F": "Recuperator approach (°F)",
    "preheater_approach_delta_F": "Preheater approach (°F)",
    "acc_approach_delta_F": "ACC approach (°F)",
    "intermediate_hx_approach_delta_F": "IHX approach (°F)",
    "turbine_isentropic_efficiency": "Turbine efficiency",
    "pump_isentropic_efficiency": "Pump efficiency",
    "turbine_trains": "Turbine trains",
    "isopentane_pressure_drop_fraction": "Iso. dP fraction",
    "propane_pressure_drop_fraction": "Propane dP fraction",
    "energy_value_per_MWh": "Energy value ($/MWh)",
}


# ── Constraint manager ──────────────────────────────────────────────────────

class ConstraintManager:
    """Manages constraint lifecycle: add, decide, inject into prompts."""

    def __init__(self, state: ConstraintState | None = None):
        self.state = state or ConstraintState()

    def add_proposal(self, proposal: RefinementProposal):
        """Add a new pending proposal."""
        self.state.pending.append(proposal)

    def decide(
        self,
        proposal_id: str,
        decision: str,
        modified_params: dict | None = None,
        defer_round: int | None = None,
    ):
        """Apply a user decision to a pending proposal.

        Parameters
        ----------
        proposal_id : str
        decision : str
            One of: "locked", "soft", "declined", "deferred", "modified"
        modified_params : dict, optional
            For "modified" decision — overridden parameter values.
        defer_round : int, optional
            For "deferred" decision — re-present at this round number.
        """
        proposal = self._find_pending(proposal_id)
        if proposal is None:
            return

        self.state.pending.remove(proposal)
        proposal.status = decision

        if decision == "locked":
            self.state.locked.append(proposal)
        elif decision == "soft":
            self.state.soft.append(proposal)
        elif decision == "declined":
            self.state.declined.append(proposal)
        elif decision == "deferred":
            proposal.defer_until_round = defer_round
            self.state.deferred.append(proposal)
        elif decision == "modified":
            proposal.modified_parameters = modified_params or {}
            # Merge modified params into the parameter changes
            proposal.parameter_changes.update(proposal.modified_parameters)
            self.state.soft.append(proposal)

    def get_due_deferred(self, current_round: int) -> list[RefinementProposal]:
        """Return deferred proposals that are due at or before current_round."""
        due = []
        remaining = []
        for p in self.state.deferred:
            if p.defer_until_round is not None and p.defer_until_round <= current_round:
                p.status = "pending"
                due.append(p)
            else:
                remaining.append(p)
        self.state.deferred = remaining
        self.state.pending.extend(due)
        return due

    def build_constraint_prompt_block(self) -> str:
        """Build the constraint injection block for bot system prompts."""
        lines = []

        if self.state.locked:
            lines.append("\n=== FIXED CONSTRAINTS (User-locked, do NOT challenge) ===")
            for p in self.state.locked:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in p.parameter_changes.items()
                    if k != "config"
                )
                lines.append(f"- {p.title}: {params_str} (locked by user)")

        if self.state.soft:
            lines.append("\n=== WORKING ASSUMPTIONS (challenge only with >5% NPV evidence) ===")
            for p in self.state.soft:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in p.parameter_changes.items()
                    if k != "config"
                )
                mod_note = " [user-modified]" if p.modified_parameters else ""
                lines.append(f"- {p.title}: {params_str}{mod_note}")

        if self.state.declined:
            lines.append("\n=== DECLINED PROPOSALS (do NOT re-propose the same change) ===")
            for p in self.state.declined:
                params_str = ", ".join(
                    f"{k}={v}" for k, v in p.parameter_changes.items()
                    if k != "config"
                )
                lines.append(f"- {p.title}: {params_str}")

        if not lines:
            return ""
        return "\n".join(lines)

    def get_locked_parameters(self) -> dict:
        """Return merged dict of all locked parameter values.

        Later locks override earlier ones for the same key.
        """
        merged: dict[str, Any] = {}
        for p in self.state.locked:
            for k, v in p.parameter_changes.items():
                if k != "config":
                    merged[k] = v
        return merged

    def check_violations(self, proposed_params: dict) -> list[str]:
        """Check if proposed parameters violate locked constraints.

        Returns list of violation descriptions.
        """
        locked = self.get_locked_parameters()
        violations = []
        for k, locked_val in locked.items():
            if k in proposed_params and proposed_params[k] != locked_val:
                violations.append(
                    f"{PARAM_DISPLAY_NAMES.get(k, k)}: proposed {proposed_params[k]} "
                    f"but locked at {locked_val}"
                )
        return violations

    def _find_pending(self, proposal_id: str) -> RefinementProposal | None:
        for p in self.state.pending:
            if p.proposal_id == proposal_id:
                return p
        return None


# ── Proposal extraction from debate messages ────────────────────────────────

# Keys to ignore when diffing parameters (not engineering-tunable)
_IGNORE_KEYS = {"config", "construction_cost_per_kW"}

# Metrics to extract for comparison
_METRIC_KEYS = {
    "net_power_MW": ("Net Power", "MW", True),       # higher better
    "npv_USD": ("NPV", "$M", True),
    "capex_total_USD": ("CAPEX", "$M", False),        # lower better
    "construction_weeks_critical_path": ("Schedule", "wk", False),
}


def extract_proposals_from_message(
    msg: "DebateMessage",
    prev_result: dict | None,
    current_result: dict | None,
    round_num: int,
) -> list["RefinementProposal"]:
    """Extract RefinementProposals by comparing tool_input parameters between runs.

    Parameters
    ----------
    msg : DebateMessage
        The bot's message containing tool calls and results.
    prev_result : dict or None
        The opponent's last run_orc_analysis result (baseline).
    current_result : dict or None
        This bot's run_orc_analysis result (proposed).
    round_num : int
        Current debate round number.

    Returns
    -------
    list[RefinementProposal]
        Proposals for each meaningful parameter change detected.
    """
    if not msg.tool_calls or current_result is None:
        return []

    proposals = []

    # Find run_orc_analysis tool calls
    for tc in msg.tool_calls:
        if tc["name"] != "run_orc_analysis":
            continue

        tool_input = tc["input"]
        # Find the corresponding result
        matching_result = None
        for tr in msg.tool_results:
            if tr["tool_use_id"] == tc["id"]:
                matching_result = tr["result"]
                break

        if matching_result is None or not isinstance(matching_result, dict):
            continue
        if "net_power_MW" not in matching_result:
            continue

        # Extract parameter changes (skip config and ignored keys)
        param_changes = {
            k: v for k, v in tool_input.items()
            if k not in _IGNORE_KEYS
        }

        if not param_changes:
            continue

        # Check NPV significance (>1% difference)
        if prev_result and "npv_USD" in prev_result and "npv_USD" in matching_result:
            npv_prev = abs(prev_result.get("npv_USD", 0))
            npv_curr = abs(matching_result.get("npv_USD", 0))
            avg_npv = (npv_prev + npv_curr) / 2
            if avg_npv > 0:
                npv_delta_pct = abs(npv_curr - npv_prev) / avg_npv * 100
                if npv_delta_pct < 1.0:
                    continue  # Skip trivial changes

        # Build baseline and proposed metric dicts
        baseline_metrics = _extract_metrics(prev_result) if prev_result else {}
        proposed_metrics = _extract_metrics(matching_result)

        # Extract title and argument from bot text
        title, argument = _extract_title_argument(msg.content, param_changes)

        # Infer confidence from NPV improvement magnitude
        confidence = _infer_confidence(prev_result, matching_result)

        proposal = RefinementProposal(
            proposal_id=f"P-{round_num}-{uuid.uuid4().hex[:6]}",
            round_num=round_num,
            proposed_by=msg.bot,
            title=title,
            engineering_argument=argument,
            parameter_changes=param_changes,
            baseline_metrics=baseline_metrics,
            proposed_metrics=proposed_metrics,
            confidence=confidence,
        )
        proposals.append(proposal)

    return proposals


def _extract_metrics(result: dict) -> dict:
    """Extract the comparison metrics from an analysis result."""
    metrics = {}
    for key, (label, unit, _) in _METRIC_KEYS.items():
        if key in result:
            metrics[key] = result[key]
    return metrics


def _extract_title_argument(text: str, param_changes: dict) -> tuple[str, str]:
    """Extract a proposal title and engineering argument from bot text.

    Uses the first substantial sentence as the title; the full first
    paragraph as the argument.
    """
    if not text.strip():
        # Fallback: describe the parameter changes
        changes_desc = ", ".join(
            f"{PARAM_DISPLAY_NAMES.get(k, k)}={v}"
            for k, v in param_changes.items()
            if k != "config"
        )
        return f"Parameter adjustment: {changes_desc[:60]}", changes_desc

    # Split into paragraphs, find the first substantive one
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    # Title: first sentence of the first paragraph
    first_para = paragraphs[0] if paragraphs else text[:200]
    sentences = first_para.replace("**", "").split(". ")
    title = sentences[0].strip()
    if len(title) > 80:
        title = title[:77] + "..."

    # Argument: first 2 paragraphs, capped
    argument = "\n\n".join(paragraphs[:2])[:500]

    return title, argument


def _infer_confidence(prev_result: dict | None, current_result: dict) -> str:
    """Infer proposal confidence from NPV improvement magnitude."""
    if prev_result is None:
        return "medium"

    npv_prev = prev_result.get("npv_USD", 0)
    npv_curr = current_result.get("npv_USD", 0)
    avg = (abs(npv_prev) + abs(npv_curr)) / 2
    if avg == 0:
        return "low"

    delta_pct = abs(npv_curr - npv_prev) / avg * 100

    if delta_pct > 5:
        return "high"
    elif delta_pct > 2:
        return "medium"
    else:
        return "low"
