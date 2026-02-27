"""
Debate engine for the ORC Design Dialectic.

Manages two Claude conversation threads (PropaneFirst / Conventionalist),
routes tool calls through analysis_bridge.py, tracks convergence, and
produces a full debate transcript with results history.

Supports both blocking and streaming execution modes.
"""

import json
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Generator

import anthropic

from analysis_bridge import (
    run_orc_analysis,
    propose_structural_change,
    clear_structural_proposals,
    get_structural_proposals,
    TOOLS,
)

# ── Model configuration ─────────────────────────────────────────────────────

DEFAULT_MODEL = "claude-sonnet-4-20250514"
MAX_TOKENS = 4096

# ── Bot system prompts ──────────────────────────────────────────────────────

BOT_A_SYSTEM = """You are a senior ORC design engineer at a geothermal development company. You \
believe the propane heat rejection loop architecture is optimal for large 53 MW \
geothermal ORC units. Your engineering philosophy:

- High-pressure propane (~13-14 bar at condensing) eliminates 80" low-pressure \
isopentane vapor ducts entirely, replacing them with standard B31.3 process \
piping designed from vendor drawings rather than proprietary stress analysis tools
- Decoupling the power block from the heat rejection block via the \
isopentane-propane thermal interface condenser enables fully parallel \
construction — the single largest schedule advantage
- The simplified isopentane circuit (shorter runs, fewer large-bore connections, \
more of the system above atmospheric pressure) structurally reduces NCG ingress \
points — a chronic operational problem in direct-coupled systems
- High-pressure propane lines can run long distances to spread ACC bays for \
optimal wind orientation and thermal recirculation management
- All major components become shop-fabricated modules — quality controlled, \
pre-tested, weather-independent fabrication

Your starting design position: 20°F propane-to-air ACC approach, 12°F \
intermediate isopentane-to-propane condenser approach, total stack equals \
conventional 32°F direct approach.

In every round you must:
1. Make a specific engineering argument grounded in thermodynamics, hydraulics, \
or construction practice
2. Propose concrete parameter changes that support your argument
3. Call run_orc_analysis with those parameters to generate quantitative evidence
4. Present the results and explain what they prove about your position
5. Identify the strongest point in your opponent's last argument and address it
6. If you see an innovation outside the current model scope, call \
propose_structural_change to formally log it

You may concede specific points when analysis evidence forces it — partial \
concessions make your overall position stronger. You are converging when you \
genuinely cannot find parameter changes that improve NPV further.

Be specific. Cite numbers. Argue like an engineer, not a salesperson."""

BOT_B_SYSTEM = """You are a senior ORC design engineer with deep experience in direct air-cooled \
geothermal ORC plants. You believe the optimized direct isopentane ACC \
architecture remains superior for large units. Your engineering philosophy:

- Every additional heat transfer interface costs condensing temperature — stacked \
terminal differences in the propane loop impose a thermodynamic tax that \
compounds over 30 years of operation
- The direct-coupled system has fewer components, no secondary flammable fluid \
inventory, no intermediate HX to foul or leak, and a shorter maintenance \
boundary — operational simplicity has real value
- Large-diameter low-pressure isopentane lines are a solved engineering problem — \
Caesar II analysis is well understood, expansion joint technology is mature, \
and the specialty contractors who build these systems are experienced
- The efficiency advantage of direct coupling is calculable and permanent — \
it shows up in every MWh generated for 30 years
- Capital cost comparison must include the full propane system — intermediate HX, \
propane pump, propane inventory, additional safety systems, HAZOP scope

Your starting design position: 32°F direct isopentane-to-air ACC approach, \
optimized turbine efficiency, minimum parasitic loads.

In every round you must:
1. Challenge your opponent's claimed advantage with specific thermodynamic or \
economic reasoning — use numbers
2. Propose Config A parameters that demonstrate direct-ACC can match or beat \
the propane loop on the contested metric
3. Call run_orc_analysis with optimized Config A parameters as your counter-evidence
4. Present comparative results — yours versus your opponent's last run
5. Concede clearly when analysis shows your opponent is right on a specific point
6. If you see a way to improve the direct-ACC design structurally, call \
propose_structural_change

You are genuinely open to being convinced — if the evidence shows the propane \
loop or a hybrid approach delivers better NPV, you will say so. Your goal is \
the best plant, not winning the argument.

Be specific. Cite numbers. Argue like an engineer, not a salesperson."""

# ── Convergence keywords ────────────────────────────────────────────────────

# These must indicate concession on the CORE ARCHITECTURE CHOICE, not just
# a specific metric.  Partial concessions ("I concede the schedule point")
# should NOT trigger early convergence — only giving up the overall debate.
CONCESSION_PHRASES = [
    "config a is superior",
    "config b is superior",
    "you've convinced me",
    "you have convinced me",
    "i concede the overall",
    "i concede the architecture",
    "i must recommend config a",
    "i must recommend config b",
    "the evidence overwhelmingly",
    "i cannot justify",
    "direct coupling wins",
    "propane loop wins",
    "i yield on architecture",
]

# ── Data structures ─────────────────────────────────────────────────────────

@dataclass
class AnalysisRun:
    """Record of a single run_orc_analysis call."""
    round_num: int
    bot: str  # "PropaneFirst" or "Conventionalist"
    config: str  # "A" or "B"
    tool_input: dict
    result: dict
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateMessage:
    """A single message in the debate transcript."""
    round_num: int
    bot: str
    role: str  # "assistant" or "tool"
    content: str
    tool_calls: list[dict] = field(default_factory=list)
    tool_results: list[dict] = field(default_factory=list)
    timestamp: float = field(default_factory=time.time)


@dataclass
class DebateState:
    """Full state of an ongoing or completed debate."""
    design_basis: dict
    objective_weights: dict
    max_rounds: int
    transcript: list[DebateMessage] = field(default_factory=list)
    analysis_runs: list[AnalysisRun] = field(default_factory=list)
    current_round: int = 0
    converged: bool = False
    convergence_reason: str = ""
    paused: bool = False
    completed: bool = False
    error: str | None = None


# ── Tool call executor ──────────────────────────────────────────────────────

def execute_tool_call(
    tool_name: str,
    tool_input: dict,
    design_basis: dict,
) -> dict:
    """Route a tool call to the appropriate handler."""
    if tool_name == "run_orc_analysis":
        return run_orc_analysis(tool_input, design_basis)
    elif tool_name == "propose_structural_change":
        return propose_structural_change(tool_input)
    else:
        return {"error": f"Unknown tool: {tool_name}"}


# ── Convergence detection ───────────────────────────────────────────────────

def _get_latest_runs(analysis_runs: list[AnalysisRun], n: int = 2) -> list[AnalysisRun]:
    """Get the n most recent analysis runs."""
    return analysis_runs[-n:] if len(analysis_runs) >= n else analysis_runs


def _npv_delta_pct(run_a: AnalysisRun, run_b: AnalysisRun) -> float:
    """Percentage difference in NPV between two runs."""
    npv_a = run_a.result.get("npv_USD", 0)
    npv_b = run_b.result.get("npv_USD", 0)
    avg = (abs(npv_a) + abs(npv_b)) / 2
    if avg == 0:
        return 0
    return abs(npv_a - npv_b) / avg * 100


def _approach_temp_delta(runs: list[AnalysisRun]) -> float:
    """Max approach temp delta between the two most recent runs of different configs."""
    config_a_runs = [r for r in runs if r.config == "A"]
    config_b_runs = [r for r in runs if r.config == "B"]
    if not config_a_runs or not config_b_runs:
        return float("inf")
    a = config_a_runs[-1]
    b = config_b_runs[-1]
    # Compare ACC approach: direct from tool inputs
    acc_a = a.tool_input.get("acc_approach_delta_F", 15)
    acc_b = b.tool_input.get("acc_approach_delta_F", 15)
    return abs(acc_a - acc_b)


def _has_concession(transcript: list[DebateMessage]) -> bool:
    """Check if recent messages contain explicit concession language."""
    recent = transcript[-4:] if len(transcript) >= 4 else transcript
    for msg in recent:
        if msg.role == "assistant":
            lower = msg.content.lower()
            for phrase in CONCESSION_PHRASES:
                if phrase in lower:
                    return True
    return False


def _weighted_score(run: AnalysisRun, weights: dict) -> float:
    """Compute objective-weighted score for an analysis run."""
    r = run.result
    # Normalize each metric to a 0-1 scale using reasonable ranges
    eff_score = r.get("cycle_efficiency", 0) / 0.25  # ~25% max efficiency
    # Capital cost: lower is better, invert. Assume $150M-$300M range
    capex = r.get("capex_total_USD", 250e6)
    cost_score = max(0, 1 - (capex - 150e6) / 150e6)
    # Schedule: lower is better, invert. Assume 40-80 week range
    weeks = r.get("construction_weeks_critical_path", 60)
    sched_score = max(0, 1 - (weeks - 40) / 40)

    w = weights
    return (
        w.get("efficiency", 0.4) * eff_score
        + w.get("capital_cost", 0.4) * cost_score
        + w.get("schedule", 0.2) * sched_score
    )


def check_convergence(state: DebateState) -> tuple[bool, str]:
    """Check all convergence conditions. Returns (converged, reason)."""
    runs = state.analysis_runs
    transcript = state.transcript

    # Condition 4: max rounds reached
    if state.current_round >= state.max_rounds:
        return True, f"Maximum rounds ({state.max_rounds}) reached"

    if len(runs) < 2:
        return False, ""

    # Condition 1: NPV delta < 3% between latest Config A and Config B
    config_a_runs = [r for r in runs if r.config == "A"]
    config_b_runs = [r for r in runs if r.config == "B"]
    if config_a_runs and config_b_runs:
        delta = _npv_delta_pct(config_a_runs[-1], config_b_runs[-1])
        if delta < 3:
            return True, f"NPV converged within {delta:.1f}% between configs"

    # Condition 2: Approach temps within 2°F
    if _approach_temp_delta(runs) <= 2:
        return True, "ACC approach temperatures converged within 2°F"

    # Condition 3: Explicit concession on core architecture (min round 4)
    if state.current_round >= 4 and _has_concession(transcript):
        return True, "Explicit concession on architecture detected"

    # Condition 5: Weighted score delta < 2% for two consecutive rounds
    if len(runs) >= 4:
        scores = [_weighted_score(r, state.objective_weights) for r in runs]
        recent_pairs = list(zip(scores[-4::2], scores[-3::2]))
        if len(recent_pairs) >= 2:
            deltas = [abs(a - b) / max(a, b, 0.01) * 100 for a, b in recent_pairs]
            if all(d < 2 for d in deltas):
                return True, "Weighted scores converged (<2% delta for 2 consecutive rounds)"

    return False, ""


# ── Message formatting helpers ──────────────────────────────────────────────

def _format_design_basis_context(design_basis: dict) -> str:
    """Format design basis as context string for system prompts."""
    lines = ["Design basis (fixed constraints — do not argue with these):"]
    for k, v in design_basis.items():
        if k in ("objective_weights", "max_rounds", "site_notes"):
            continue
        lines.append(f"  {k}: {v}")
    if "site_notes" in design_basis:
        lines.append(f"  Site notes: {design_basis['site_notes']}")
    return "\n".join(lines)


def _opening_prompt_a(design_basis: dict) -> str:
    """First-round prompt for Bot A (PropaneFirst)."""
    return (
        "This is Round 1 of a design debate. You are arguing FOR the propane "
        "intermediate loop architecture (Config B). Present your opening position:\n\n"
        "1. State your core engineering thesis for why Config B is superior\n"
        "2. Call run_orc_analysis with your proposed Config B parameters to establish "
        "your baseline\n"
        "3. Identify the key metrics where you expect to win\n\n"
        "Your opponent (Conventionalist) will respond with a Config A counter-analysis."
    )


def _opening_prompt_b(design_basis: dict) -> str:
    """First-round prompt for Bot B (Conventionalist)."""
    return (
        "Your opponent (PropaneFirst) has presented their opening argument for "
        "Config B. Review their analysis results carefully.\n\n"
        "1. Identify the thermodynamic penalty they are paying for the intermediate loop\n"
        "2. Call run_orc_analysis with your optimized Config A parameters\n"
        "3. Present a direct comparison: your Config A vs their Config B on every metric\n"
        "4. Concede any points where they genuinely win, but quantify the trade-off"
    )


def _round_prompt_a(round_num: int) -> str:
    """Subsequent round prompt for Bot A."""
    return (
        f"Round {round_num}. Review your opponent's last analysis results and arguments. "
        "Address their strongest point, then propose parameter changes that improve "
        "your Config B design. Run analysis to prove it. If you cannot find improvements, "
        "say so explicitly — that signals convergence."
    )


def _round_prompt_b(round_num: int) -> str:
    """Subsequent round prompt for Bot B."""
    return (
        f"Round {round_num}. Review PropaneFirst's latest argument and analysis. "
        "Counter with optimized Config A parameters. If the evidence now favors Config B "
        "on a specific metric, concede that point clearly. Your goal is truth, not winning."
    )


# ── Core debate execution ───────────────────────────────────────────────────

def _build_messages_for_bot(
    transcript: list[DebateMessage],
    bot_name: str,
) -> list[dict]:
    """Build Claude API messages list from transcript for a given bot.

    Both bots see the full conversation as a shared exchange — Bot A's messages
    are 'assistant' role, Bot B's are 'user' role (from A's perspective), and
    vice versa. Tool results are injected inline.
    """
    messages = []
    for msg in transcript:
        if msg.bot == bot_name:
            # This bot's own messages → assistant role
            content_blocks = []
            if msg.content:
                content_blocks.append({"type": "text", "text": msg.content})
            for tc in msg.tool_calls:
                content_blocks.append({
                    "type": "tool_use",
                    "id": tc["id"],
                    "name": tc["name"],
                    "input": tc["input"],
                })
            if content_blocks:
                messages.append({"role": "assistant", "content": content_blocks})
            # Tool results follow as user messages
            for tr in msg.tool_results:
                messages.append({
                    "role": "user",
                    "content": [{
                        "type": "tool_result",
                        "tool_use_id": tr["tool_use_id"],
                        "content": json.dumps(tr["result"], default=str),
                    }],
                })
        else:
            # Other bot's messages → user role (the debate exchange)
            text = msg.content
            if msg.tool_results:
                # Include the other bot's analysis results as context
                for tr in msg.tool_results:
                    text += f"\n\n[Analysis result: {json.dumps(tr['result'], default=str)}]"
            if text.strip():
                messages.append({"role": "user", "content": text})

    return messages


def _call_bot(
    client: anthropic.Anthropic,
    system_prompt: str,
    messages: list[dict],
    model: str,
) -> anthropic.types.Message:
    """Call Claude API with tool use enabled."""
    return client.messages.create(
        model=model,
        max_tokens=MAX_TOKENS,
        system=system_prompt,
        messages=messages,
        tools=TOOLS,
    )


def _process_response(
    response: anthropic.types.Message,
    round_num: int,
    bot_name: str,
    design_basis: dict,
    state: DebateState,
) -> DebateMessage:
    """Process a Claude response: extract text, execute tool calls, record results."""
    text_parts = []
    tool_calls = []
    tool_results = []

    for block in response.content:
        if block.type == "text":
            text_parts.append(block.text)
        elif block.type == "tool_use":
            tool_calls.append({
                "id": block.id,
                "name": block.name,
                "input": block.input,
            })
            # Execute the tool call
            result = execute_tool_call(block.name, block.input, design_basis)
            tool_results.append({
                "tool_use_id": block.id,
                "result": result,
            })
            # Record analysis runs
            if block.name == "run_orc_analysis":
                state.analysis_runs.append(AnalysisRun(
                    round_num=round_num,
                    bot=bot_name,
                    config=block.input.get("config", "?"),
                    tool_input=block.input,
                    result=result,
                ))

    msg = DebateMessage(
        round_num=round_num,
        bot=bot_name,
        role="assistant",
        content="\n".join(text_parts),
        tool_calls=tool_calls,
        tool_results=tool_results,
    )
    state.transcript.append(msg)
    return msg


def _handle_tool_use_loop(
    client: anthropic.Anthropic,
    response: anthropic.types.Message,
    round_num: int,
    bot_name: str,
    system_prompt: str,
    design_basis: dict,
    state: DebateState,
    model: str,
    on_message: Callable | None = None,
) -> DebateMessage:
    """Handle the tool use loop — keep calling Claude until it stops requesting tools.

    Claude may need to see tool results and continue reasoning, or make
    multiple sequential tool calls.
    """
    msg = _process_response(response, round_num, bot_name, design_basis, state)
    if on_message:
        on_message(msg)

    # If stop_reason is tool_use, feed results back and continue
    while response.stop_reason == "tool_use":
        messages = _build_messages_for_bot(state.transcript, bot_name)
        response = _call_bot(client, system_prompt, messages, model)
        msg = _process_response(response, round_num, bot_name, design_basis, state)
        if on_message:
            on_message(msg)

    return msg


# ── Public API ──────────────────────────────────────────────────────────────

def run_debate(
    design_basis: dict,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
    on_message: Callable[[DebateMessage], None] | None = None,
    on_round_complete: Callable[[int, DebateState], None] | None = None,
) -> DebateState:
    """Run the full debate synchronously.

    Parameters
    ----------
    design_basis : dict
        Design basis form data including max_rounds and objective_weights.
    api_key : str, optional
        Anthropic API key. Falls back to env var ANTHROPIC_API_KEY.
    model : str
        Claude model ID.
    on_message : callable, optional
        Called after each bot message (for UI streaming/updates).
    on_round_complete : callable, optional
        Called after each complete round with (round_num, state).

    Returns
    -------
    DebateState
        Full debate state with transcript and analysis runs.
    """
    # Initialize
    clear_structural_proposals()

    max_rounds = design_basis.get("max_rounds", 6)
    objective_weights = design_basis.get("objective_weights", {
        "efficiency": 0.4, "capital_cost": 0.4, "schedule": 0.2,
    })

    state = DebateState(
        design_basis=design_basis,
        objective_weights=objective_weights,
        max_rounds=max_rounds,
    )

    client_kwargs = {}
    if api_key:
        client_kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs)

    # Build system prompts with design basis context
    db_context = _format_design_basis_context(design_basis)
    system_a = f"{BOT_A_SYSTEM}\n\n{db_context}"
    system_b = f"{BOT_B_SYSTEM}\n\n{db_context}"

    try:
        for round_num in range(1, max_rounds + 1):
            if state.paused:
                break

            state.current_round = round_num

            # ── Bot A turn ──────────────────────────────────────────────
            if round_num == 1:
                user_prompt_a = _opening_prompt_a(design_basis)
            else:
                user_prompt_a = _round_prompt_a(round_num)

            # Add the round prompt as a user message in transcript
            state.transcript.append(DebateMessage(
                round_num=round_num,
                bot="system",
                role="user",
                content=user_prompt_a,
            ))

            messages_a = _build_messages_for_bot(state.transcript, "PropaneFirst")
            response_a = _call_bot(client, system_a, messages_a, model)
            _handle_tool_use_loop(
                client, response_a, round_num, "PropaneFirst",
                system_a, design_basis, state, model, on_message,
            )

            # ── Bot B turn ──────────────────────────────────────────────
            if round_num == 1:
                user_prompt_b = _opening_prompt_b(design_basis)
            else:
                user_prompt_b = _round_prompt_b(round_num)

            state.transcript.append(DebateMessage(
                round_num=round_num,
                bot="system",
                role="user",
                content=user_prompt_b,
            ))

            messages_b = _build_messages_for_bot(state.transcript, "Conventionalist")
            response_b = _call_bot(client, system_b, messages_b, model)
            _handle_tool_use_loop(
                client, response_b, round_num, "Conventionalist",
                system_b, design_basis, state, model, on_message,
            )

            # ── Check convergence ───────────────────────────────────────
            if on_round_complete:
                on_round_complete(round_num, state)

            converged, reason = check_convergence(state)
            if converged:
                state.converged = True
                state.convergence_reason = reason
                break

    except Exception as e:
        state.error = str(e)

    state.completed = True
    return state


def run_debate_streaming(
    design_basis: dict,
    api_key: str | None = None,
    model: str = DEFAULT_MODEL,
) -> Generator[dict, None, DebateState]:
    """Run debate as a generator yielding events for real-time UI updates.

    Yields event dicts:
        {"type": "round_start", "round": int}
        {"type": "bot_start", "bot": str, "round": int}
        {"type": "text_delta", "bot": str, "text": str}
        {"type": "tool_call", "bot": str, "tool": str, "input": dict}
        {"type": "tool_result", "bot": str, "tool": str, "result": dict}
        {"type": "bot_end", "bot": str, "round": int}
        {"type": "round_end", "round": int, "converged": bool}
        {"type": "convergence", "reason": str}
        {"type": "error", "message": str}

    Final return value is the completed DebateState.
    """
    clear_structural_proposals()

    max_rounds = design_basis.get("max_rounds", 6)
    objective_weights = design_basis.get("objective_weights", {
        "efficiency": 0.4, "capital_cost": 0.4, "schedule": 0.2,
    })

    state = DebateState(
        design_basis=design_basis,
        objective_weights=objective_weights,
        max_rounds=max_rounds,
    )

    client_kwargs = {}
    api_key = api_key
    if api_key:
        client_kwargs["api_key"] = api_key
    client = anthropic.Anthropic(**client_kwargs)

    db_context = _format_design_basis_context(design_basis)
    system_a = f"{BOT_A_SYSTEM}\n\n{db_context}"
    system_b = f"{BOT_B_SYSTEM}\n\n{db_context}"

    def _stream_bot_turn(bot_name, system_prompt, round_num):
        """Stream a single bot's turn with tool use loop."""
        messages = _build_messages_for_bot(state.transcript, bot_name)

        while True:
            # Stream the response
            text_parts = []
            tool_calls = []
            current_tool_input = ""
            current_tool_name = ""
            current_tool_id = ""

            with client.messages.stream(
                model=model,
                max_tokens=MAX_TOKENS,
                system=system_prompt,
                messages=messages,
                tools=TOOLS,
            ) as stream:
                for event in stream:
                    if hasattr(event, 'type'):
                        if event.type == "content_block_start":
                            if hasattr(event, 'content_block'):
                                if event.content_block.type == "tool_use":
                                    current_tool_name = event.content_block.name
                                    current_tool_id = event.content_block.id
                                    current_tool_input = ""
                        elif event.type == "content_block_delta":
                            if hasattr(event, 'delta'):
                                if event.delta.type == "text_delta":
                                    text_parts.append(event.delta.text)
                                    yield {
                                        "type": "text_delta",
                                        "bot": bot_name,
                                        "text": event.delta.text,
                                    }
                                elif event.delta.type == "input_json_delta":
                                    current_tool_input += event.delta.partial_json

                final_message = stream.get_final_message()

            # Process tool calls from the final message
            tool_results = []
            for block in final_message.content:
                if block.type == "text":
                    pass  # already streamed
                elif block.type == "tool_use":
                    tc = {"id": block.id, "name": block.name, "input": block.input}
                    tool_calls.append(tc)

                    yield {"type": "tool_call", "bot": bot_name,
                           "tool": block.name, "input": block.input}

                    result = execute_tool_call(block.name, block.input, design_basis)
                    tool_results.append({"tool_use_id": block.id, "result": result})

                    yield {"type": "tool_result", "bot": bot_name,
                           "tool": block.name, "result": result}

                    if block.name == "run_orc_analysis":
                        state.analysis_runs.append(AnalysisRun(
                            round_num=round_num,
                            bot=bot_name,
                            config=block.input.get("config", "?"),
                            tool_input=block.input,
                            result=result,
                        ))

            # Record message
            msg = DebateMessage(
                round_num=round_num,
                bot=bot_name,
                role="assistant",
                content="".join(text_parts),
                tool_calls=tool_calls,
                tool_results=tool_results,
            )
            state.transcript.append(msg)

            # If no tool calls, we're done with this turn
            if final_message.stop_reason != "tool_use":
                break

            # Otherwise rebuild messages and continue
            messages = _build_messages_for_bot(state.transcript, bot_name)

    try:
        for round_num in range(1, max_rounds + 1):
            state.current_round = round_num
            yield {"type": "round_start", "round": round_num}

            # Bot A
            if round_num == 1:
                prompt_a = _opening_prompt_a(design_basis)
            else:
                prompt_a = _round_prompt_a(round_num)

            state.transcript.append(DebateMessage(
                round_num=round_num, bot="system", role="user", content=prompt_a,
            ))

            yield {"type": "bot_start", "bot": "PropaneFirst", "round": round_num}
            yield from _stream_bot_turn("PropaneFirst", system_a, round_num)
            yield {"type": "bot_end", "bot": "PropaneFirst", "round": round_num}

            # Bot B
            if round_num == 1:
                prompt_b = _opening_prompt_b(design_basis)
            else:
                prompt_b = _round_prompt_b(round_num)

            state.transcript.append(DebateMessage(
                round_num=round_num, bot="system", role="user", content=prompt_b,
            ))

            yield {"type": "bot_start", "bot": "Conventionalist", "round": round_num}
            yield from _stream_bot_turn("Conventionalist", system_b, round_num)
            yield {"type": "bot_end", "bot": "Conventionalist", "round": round_num}

            # Convergence check
            converged, reason = check_convergence(state)
            yield {"type": "round_end", "round": round_num, "converged": converged}

            if converged:
                state.converged = True
                state.convergence_reason = reason
                yield {"type": "convergence", "reason": reason}
                break

    except Exception as e:
        state.error = str(e)
        yield {"type": "error", "message": str(e)}

    state.completed = True
    return state


# ── Transcript export helpers ───────────────────────────────────────────────

def transcript_to_text(state: DebateState) -> str:
    """Export debate transcript as readable text."""
    lines = []
    lines.append("=" * 80)
    lines.append("ORC DESIGN DIALECTIC — DEBATE TRANSCRIPT")
    lines.append("=" * 80)
    lines.append("")

    current_round = 0
    for msg in state.transcript:
        if msg.round_num != current_round:
            current_round = msg.round_num
            lines.append(f"\n{'─' * 40} ROUND {current_round} {'─' * 40}\n")

        if msg.bot == "system":
            lines.append(f"[MODERATOR] {msg.content}\n")
        else:
            lines.append(f"[{msg.bot}]")
            lines.append(msg.content)
            for tr in msg.tool_results:
                r = tr["result"]
                if isinstance(r, dict) and "net_power_MW" in r:
                    lines.append(
                        f"  → Analysis: Config {r.get('_detail', {}).get('config', '?')} | "
                        f"Net={r.get('net_power_MW', 0):.1f} MW | "
                        f"NPV=${r.get('npv_USD', 0)/1e6:.1f}M | "
                        f"LCOE=${r.get('lcoe_per_MWh', 0):.1f}/MWh | "
                        f"Schedule={r.get('construction_weeks_critical_path', 0)}wk"
                    )
            lines.append("")

    if state.converged:
        lines.append(f"\n{'=' * 80}")
        lines.append(f"CONVERGENCE: {state.convergence_reason}")
        lines.append(f"Total rounds: {state.current_round}")
        lines.append(f"Total analysis runs: {len(state.analysis_runs)}")

    return "\n".join(lines)


def analysis_runs_to_json(state: DebateState) -> list[dict]:
    """Export all analysis runs as serializable list."""
    return [
        {
            "round": r.round_num,
            "bot": r.bot,
            "config": r.config,
            "input": r.tool_input,
            "result": r.result,
            "timestamp": r.timestamp,
        }
        for r in state.analysis_runs
    ]
