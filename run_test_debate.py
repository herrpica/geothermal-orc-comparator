"""Run a full 6-round test debate and print results."""

import json
import time
import sys
import os

# Force UTF-8 output on Windows to handle Unicode in Claude responses
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

from debate_engine import run_debate, transcript_to_text, analysis_runs_to_json
from synthesis import (
    synthesize_debate, build_deck_summary,
    save_converged_design, save_deck_summary,
)
from analysis_bridge import get_structural_proposals


def main():
    print("=" * 78)
    print("  6-ROUND ORC DESIGN DIALECTIC — FULL DEBATE")
    print("=" * 78)
    print()

    design_basis = {
        "brine_inlet_temp_C": 215.6,
        "brine_outlet_temp_C": 71.1,
        "brine_flow_kg_s": 498.95,
        "ambient_temp_C": 13.9,
        "site_notes": "High desert site, good wind exposure for ACC",
        "net_power_target_MW": 53.0,
        "energy_value_per_MWh": 80.0,
        "plant_life_years": 30,
        "discount_rate": 0.08,
        "max_rounds": 6,
        "objective_weights": {
            "efficiency": 0.4,
            "capital_cost": 0.4,
            "schedule": 0.2,
        },
    }

    def on_message(msg):
        bot = msg.bot
        if bot == "system":
            return
        print(f"\n  [{bot}]")
        if msg.content:
            for line in msg.content.split("\n"):
                print(f"  {line}")
        for tc in msg.tool_calls:
            inp = tc["input"]
            config = inp.get("config", "n/a")
            params = []
            if "acc_approach_delta_F" in inp:
                params.append(f"ACC={inp['acc_approach_delta_F']}F")
            if "evaporator_approach_delta_F" in inp:
                params.append(f"evap={inp['evaporator_approach_delta_F']}F")
            if "intermediate_hx_approach_delta_F" in inp:
                params.append(f"IHX={inp['intermediate_hx_approach_delta_F']}F")
            if "turbine_isentropic_efficiency" in inp:
                params.append(f"eta_t={inp['turbine_isentropic_efficiency']}")
            if tc["name"] == "propose_structural_change":
                params = [inp.get("title", "")]
            param_str = " ".join(params)
            print(f"    >> {tc['name']}(config={config}, {param_str})")
        for tr in msg.tool_results:
            r = tr["result"]
            if isinstance(r, dict) and "net_power_MW" in r:
                print(
                    f"    << Net={r['net_power_MW']:.1f}MW | "
                    f"Gross={r['gross_power_MW']:.1f}MW | "
                    f"Parasitic={r['parasitic_MW']:.1f}MW | "
                    f"Eff={r['cycle_efficiency']*100:.1f}%"
                )
                print(
                    f"       NPV=${r['npv_USD']/1e6:.1f}M | "
                    f"LCOE=${r['lcoe_per_MWh']:.1f}/MWh | "
                    f"CAPEX=${r['capex_total_USD']/1e6:.0f}M | "
                    f"Sched={r['construction_weeks_critical_path']}wk"
                )
                if r.get("warnings"):
                    print(f"       Warnings: {r['warnings']}")
            elif isinstance(r, dict) and "logged" in r:
                print(f"    << Structural proposal logged: {r['proposal_id']}")
        sys.stdout.flush()

    def on_round_complete(round_num, state):
        n = len(state.analysis_runs)
        print()
        print(f"  {'=' * 68}")
        print(f"  ROUND {round_num} COMPLETE — {n} total analysis runs")
        a_runs = [r for r in state.analysis_runs if r.config == "A"]
        b_runs = [r for r in state.analysis_runs if r.config == "B"]
        if a_runs and b_runs:
            a = a_runs[-1].result
            b = b_runs[-1].result
            print(
                f"  Latest A: Net={a.get('net_power_MW',0):.1f}MW  "
                f"NPV=${a.get('npv_USD',0)/1e6:.1f}M  "
                f"LCOE=${a.get('lcoe_per_MWh',0):.1f}"
            )
            print(
                f"  Latest B: Net={b.get('net_power_MW',0):.1f}MW  "
                f"NPV=${b.get('npv_USD',0)/1e6:.1f}M  "
                f"LCOE=${b.get('lcoe_per_MWh',0):.1f}"
            )
            avg_npv = (abs(a.get("npv_USD", 0)) + abs(b.get("npv_USD", 0))) / 2
            if avg_npv > 0:
                npv_delta = abs(a.get("npv_USD", 0) - b.get("npv_USD", 0)) / avg_npv * 100
                print(f"  NPV delta: {npv_delta:.1f}%")
        print(f"  {'=' * 68}")
        print()
        sys.stdout.flush()

    print("Design Basis:")
    print(f"  Brine: {design_basis['brine_inlet_temp_C']}C inlet, {design_basis['brine_flow_kg_s']} kg/s")
    print(f"  Ambient: {design_basis['ambient_temp_C']}C")
    print(f"  Target: {design_basis['net_power_target_MW']} MW net @ ${design_basis['energy_value_per_MWh']}/MWh")
    print(f"  Life: {design_basis['plant_life_years']}yr, Discount: {design_basis['discount_rate']*100}%")
    print(f"  Max rounds: {design_basis['max_rounds']}")
    w = design_basis["objective_weights"]
    print(f"  Weights: eff={w['efficiency']}, cost={w['capital_cost']}, sched={w['schedule']}")
    print()
    print("Starting debate...")
    print()

    t0 = time.time()
    state = run_debate(
        design_basis=design_basis,
        on_message=on_message,
        on_round_complete=on_round_complete,
    )
    elapsed = time.time() - t0

    print()
    print("=" * 78)
    print(f"  DEBATE COMPLETE — {elapsed:.0f}s elapsed")
    print("=" * 78)
    print(f"  Rounds: {state.current_round}/{state.max_rounds}")
    print(f"  Analysis runs: {len(state.analysis_runs)}")
    print(f"  Converged: {state.converged}")
    print(f"  Reason: {state.convergence_reason}")
    if state.error:
        print(f"  ERROR: {state.error}")
    print()

    # Full summary table
    print("COMPLETE ANALYSIS RUN HISTORY")
    hdr = f"  {'Rnd':<4} {'Bot':<20} {'Cfg':<5} {'Net MW':<9} {'Gross':<9} {'Para':<8} {'Eff%':<7} {'NPV $M':<10} {'LCOE':<8} {'CAPEX $M':<10} {'Wks':<5}"
    print(hdr)
    print("  " + "-" * (len(hdr) - 2))
    for run in state.analysis_runs:
        r = run.result
        print(
            f"  {run.round_num:<4} {run.bot:<20} {run.config:<5} "
            f"{r.get('net_power_MW',0):<9.1f} {r.get('gross_power_MW',0):<9.1f} "
            f"{r.get('parasitic_MW',0):<8.1f} {r.get('cycle_efficiency',0)*100:<7.1f} "
            f"{r.get('npv_USD',0)/1e6:<10.1f} {r.get('lcoe_per_MWh',0):<8.1f} "
            f"{r.get('capex_total_USD',0)/1e6:<10.0f} "
            f"{r.get('construction_weeks_critical_path',0):<5}"
        )
    print()

    # Structural proposals
    proposals = get_structural_proposals()
    if proposals:
        print(f"STRUCTURAL PROPOSALS ({len(proposals)} total)")
        for p in proposals:
            print(f"  [{p['proposal_id']}] {p.get('title','')} ({p.get('confidence','?')})")
            print(f"    {p.get('description','')[:150]}")
            npv_d = p.get("estimated_npv_delta_pct", 0)
            sched_d = p.get("estimated_schedule_delta_weeks", 0)
            ext = p.get("requires_model_extension", False)
            print(f"    NPV delta: {npv_d:+.1f}%  Sched: {sched_d:+.0f}wk  Model ext: {ext}")
        print()

    # Synthesis
    print("Running arbitrator synthesis...")
    t1 = time.time()
    synthesis = synthesize_debate(state)
    synth_time = time.time() - t1
    print(f"Synthesis complete in {synth_time:.0f}s")
    print()

    arch = synthesis.get("architecture", "?")
    print("=" * 78)
    print(f"  ARBITRATOR VERDICT: Config {arch}")
    print("=" * 78)
    print()

    rationale = synthesis.get("architecture_rationale", "")
    print("Rationale:")
    # Wrap at ~75 chars
    words = rationale.split()
    line = "  "
    for w in words:
        if len(line) + len(w) + 1 > 76:
            print(line)
            line = "  " + w
        else:
            line += " " + w if line.strip() else "  " + w
    if line.strip():
        print(line)
    print()

    # Optimized parameters
    opt = synthesis.get("optimized_parameters", {})
    if opt:
        print("OPTIMIZED PARAMETERS:")
        for k, v in opt.items():
            print(f"  {k}: {v}")
        print()

    # Validated performance
    validated = synthesis.get("validated_performance", {})
    if validated and "net_power_MW" in validated:
        print("VALIDATED PERFORMANCE (re-run through analysis engine):")
        print(f"  Net Power:    {validated.get('net_power_MW', 0):.1f} MW")
        print(f"  Gross Power:  {validated.get('gross_power_MW', 0):.1f} MW")
        print(f"  Parasitic:    {validated.get('parasitic_MW', 0):.1f} MW")
        print(f"  Efficiency:   {validated.get('cycle_efficiency', 0)*100:.1f}%")
        print(f"  NPV:          ${validated.get('npv_USD', 0)/1e6:.1f}M")
        print(f"  LCOE:         ${validated.get('lcoe_per_MWh', 0):.1f}/MWh")
        print(f"  CAPEX:        ${validated.get('capex_total_USD', 0)/1e6:.0f}M")
        print(f"  CAPEX/kW:     ${validated.get('capex_per_kW', 0):,.0f}/kW")
        print(f"  Schedule:     {validated.get('construction_weeks_critical_path', 0)} weeks")
        print(f"  Converged:    {validated.get('converged', False)}")
        print()

    # Key outcomes
    outcomes = synthesis.get("key_debate_outcomes", [])
    if outcomes:
        print("KEY DEBATE OUTCOMES:")
        for i, o in enumerate(outcomes, 1):
            print(f"  {i}. {o}")
        print()

    # Concessions
    concessions = synthesis.get("concessions", {})
    for bot, items in concessions.items():
        if items:
            print(f"{bot.upper()} CONCESSIONS:")
            for c in items:
                print(f"  - {c}")
            print()

    # Structural recommendations
    recs = synthesis.get("structural_recommendations", [])
    if recs:
        print(f"STRUCTURAL RECOMMENDATIONS ({len(recs)}):")
        for r in recs:
            contested = "CONTESTED" if r.get("contested") else "AGREED"
            conf = r.get("confidence", "?")
            print(f"  [{conf}] [{contested}] {r.get('title','')}")
            print(f"    By: {r.get('proposed_by','')}")
            desc = r.get("description", "")
            # Wrap description
            words = desc.split()
            line = "    "
            for w in words:
                if len(line) + len(w) + 1 > 76:
                    print(line)
                    line = "    " + w
                else:
                    line += " " + w if line.strip() else "    " + w
            if line.strip():
                print(line)
            if r.get("requires_model_extension"):
                print("    >> Requires model extension")
            print()

    # Model extension recs
    ext = synthesis.get("model_extension_recommendations", [])
    if ext:
        print("MODEL EXTENSION RECOMMENDATIONS:")
        for e in ext:
            print(f"  - {e}")
        print()

    # Save outputs
    save_converged_design(synthesis, "converged_design.json")
    deck = build_deck_summary(synthesis, state)
    save_deck_summary(deck, "deck_summary.json")

    transcript = transcript_to_text(state)
    with open("debate_transcript.txt", "w", encoding="utf-8") as f:
        f.write(transcript)

    print("OUTPUT FILES:")
    for fn in ["converged_design.json", "deck_summary.json", "debate_transcript.txt"]:
        size = os.path.getsize(fn)
        print(f"  {fn}: {size:,} bytes")

    total = elapsed + synth_time
    print(f"\n{'=' * 78}")
    print(f"  TOTAL TIME: {total:.0f}s (debate: {elapsed:.0f}s, synthesis: {synth_time:.0f}s)")
    print(f"{'=' * 78}")


if __name__ == "__main__":
    main()
