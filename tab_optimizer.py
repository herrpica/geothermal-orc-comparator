"""
Autonomous ORC Optimizer — Streamlit UI tab.

Five sub-tabs:
  1. Controls & Status — Start/Pause/Reset, live progress, best config
  2. Pareto Frontier — Scatter plot (efficiency vs $/kW), Pareto line
  3. Run History — Filterable results table
  4. Step Change Analysis — Incremental vs transformative cost pathways
  5. FEED Package — Preliminary Front-End Engineering Design deliverables

Execution model: ONE config per st.rerun() cycle for live UI updates.
"""

import time
from datetime import datetime, timezone

import streamlit as st
import pandas as pd

from optimizer_engine import (
    ResultStore,
    OptConfig,
    generate_search_space,
    total_search_space_size,
    run_single_config,
    update_pareto_frontier,
    reevaluate_targets,
    generate_report,
    generate_seed_batch,
    call_ai_optimizer,
    parse_ai_configs,
    TARGET_CAPEX_PER_KW,
    TARGET_SCHEDULE_WEEKS,
    TARGET_MIN_NET_MW,
    TARGET_MAX_NET_MW,
    WORKING_FLUIDS,
    PROCUREMENT_STRATEGIES,
    HEAT_REJECTIONS,
    AI_GUIDED_MAX_ROUNDS,
    AI_GUIDED_BATCH_SIZE,
    COMPLEXITY_PENALTIES,
    TURBINE_MARKET_LIMIT_MW,
)
from cost_model import STRATEGY_LABELS, STRATEGY_SHORT_LABELS
from step_change_analysis import render_step_change_subtab


def _init_session_state():
    """Initialize optimizer session state keys."""
    defaults = {
        "opt_running": False,
        "opt_paused": False,
        "opt_queue": [],
        "opt_report": None,
        "opt_total_configs": 0,
        # Timing & progress
        "opt_start_time": 0.0,
        "opt_config_times": [],       # list of durations (seconds)
        "opt_last_config_label": "",   # human-readable label of last run
        "opt_last_completed_at": "",   # ISO timestamp of last completion
        "opt_last_result_summary": "", # one-line summary of last result
        # AI-Guided mode
        "opt_mode": "ai_guided",
        "opt_ai_round": 0,
        "opt_ai_batch_counter": 0,
        "opt_ai_reasoning_log": [],
        "opt_ai_insights": [],
        "opt_ai_converged": False,
        "opt_selected_hr": None,        # heat rejections locked at Start
        # DBD update flow
        "dbd_update_proposal": None,
        "dbd_update_phase": "idle",      # idle|preview|generating|preview_diff|reviewing|applying|complete
        "dbd_update_review_index": 0,
        "dbd_update_error": None,
        # FEED package
        "opt_feed_run_id": None,
        "opt_feed_data": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val


def _get_store() -> ResultStore:
    """Get or create the result store, re-evaluating targets against current thresholds."""
    if "opt_store" not in st.session_state:
        store = ResultStore()
        reevaluate_targets(store)
        st.session_state["opt_store"] = store
    return st.session_state["opt_store"]


def _format_duration(seconds: float) -> str:
    """Format seconds into human-readable duration."""
    if seconds < 60:
        return f"{seconds:.1f}s"
    minutes = int(seconds // 60)
    secs = seconds % 60
    if minutes < 60:
        return f"{minutes}m {secs:.0f}s"
    hours = int(minutes // 60)
    mins = minutes % 60
    return f"{hours}h {mins}m"


def render_optimizer_tab(design_basis: dict):
    """Render the Autonomous Optimizer tab."""
    _init_session_state()
    store = _get_store()

    st.header("Autonomous ORC Optimizer")

    # ── Adjustable target constraints ──────────────────────────────────
    tc1, tc2, tc3, tc4 = st.columns(4)
    with tc1:
        user_capex = st.number_input(
            "Target installed cost ($/kW)",
            min_value=500, max_value=10000, value=int(TARGET_CAPEX_PER_KW), step=100,
            key="opt_target_capex",
        )
    with tc2:
        user_schedule = st.number_input(
            "Target schedule (weeks)",
            min_value=10, max_value=200, value=int(TARGET_SCHEDULE_WEEKS), step=1,
            key="opt_target_schedule",
        )
    with tc3:
        user_min_mw = st.number_input(
            "Min net power (MW)",
            min_value=1.0, max_value=200.0, value=float(TARGET_MIN_NET_MW), step=1.0,
            format="%.0f", key="opt_target_min_mw",
        )
    with tc4:
        user_max_mw = st.number_input(
            "Max net power (MW)",
            min_value=1.0, max_value=500.0, value=float(TARGET_MAX_NET_MW), step=1.0,
            format="%.0f", key="opt_target_max_mw",
        )

    # Apply user overrides to module-level constants so all logic uses them
    import optimizer_engine as _oe
    _oe.TARGET_CAPEX_PER_KW = float(user_capex)
    _oe.TARGET_SCHEDULE_WEEKS = int(user_schedule)
    _oe.TARGET_MIN_NET_MW = float(user_min_mw)
    _oe.TARGET_MAX_NET_MW = float(user_max_mw)

    # Re-evaluate all results against current thresholds
    reevaluate_targets(store)

    _n_units_display = st.session_state.get("opt_locked_n_units",
                           st.session_state.get("opt_n_units", 1)) or 1
    _units_label = f" | **{_n_units_display} identical units** (multi-unit discount)" if _n_units_display > 1 else ""
    st.caption(
        f"Target: **${user_capex:,.0f}/kW** installed cost | "
        f"**{user_schedule} weeks** schedule | "
        f"**{user_min_mw:.0f}–{user_max_mw:.0f} MW** net power band"
        f"{_units_label}"
    )

    # ── Inject multi-unit count into design_basis ──────────────────────
    # Use locked value (set at Start) during runs; fall back to widget value
    site_n_units = st.session_state.get("opt_locked_n_units",
                       st.session_state.get("opt_n_units", 1))
    if site_n_units and site_n_units > 1:
        design_basis = {**design_basis, "n_units": int(site_n_units)}

    # ── Single-config-per-rerun execution ──────────────────────────────
    if st.session_state["opt_running"] and not st.session_state["opt_paused"]:
        queue = st.session_state["opt_queue"]
        is_ai_mode = st.session_state["opt_mode"] == "ai_guided"

        if queue:
            # Pop exactly ONE config
            cfg = queue[0]
            st.session_state["opt_queue"] = queue[1:]

            # ── Show progress BEFORE running (visible while config computes) ──
            run_stats = store.stats()
            remaining = len(st.session_state["opt_queue"])
            total = st.session_state["opt_total_configs"]
            done = run_stats["total_runs"]
            pct = done / total if total > 0 else 0
            hits = run_stats["target_hits"]
            misses = run_stats["converged"] - hits
            failed = run_stats["failed"]

            if is_ai_mode:
                ai_round = st.session_state["opt_ai_round"]
                batch_counter = st.session_state["opt_ai_batch_counter"]
                st.progress(
                    pct,
                    text=f"**AI Round {ai_round}/{AI_GUIDED_MAX_ROUNDS}** — "
                         f"Config {batch_counter + 1}/{AI_GUIDED_BATCH_SIZE} | "
                         f"{done:,} total tested"
                )
            else:
                st.progress(pct, text=f"**{done:,}/{total:,}** ({pct*100:.1f}%) — {remaining:,} remaining")

            # Hits / Misses / Failed metrics
            pm1, pm2, pm3, pm4 = st.columns(4)
            pm1.metric("Hits", hits, help="Meet all constraints")
            pm2.metric("Misses", misses, help="Converged but missed a constraint")
            pm3.metric("Failed", failed, help="Did not converge")
            if done > 0:
                pm4.metric("Hit Rate", f"{hits/done*100:.0f}%")
            else:
                pm4.metric("Hit Rate", "—")

            # Timing
            config_times = st.session_state["opt_config_times"]
            if config_times:
                elapsed = time.time() - st.session_state["opt_start_time"]
                avg_time = sum(config_times) / len(config_times)
                if is_ai_mode:
                    st.caption(
                        f"Elapsed: {_format_duration(elapsed)} | "
                        f"Avg: {avg_time:.1f}s/config"
                    )
                else:
                    eta_seconds = avg_time * remaining
                    st.caption(
                        f"Elapsed: {_format_duration(elapsed)} | "
                        f"Avg: {avg_time:.1f}s/config | "
                        f"ETA: {_format_duration(eta_seconds)}"
                    )

            # Last result
            last_summary = st.session_state.get("opt_last_result_summary", "")
            if last_summary:
                st.caption(f"Last: {last_summary}")

            # Show latest AI insight if available
            if is_ai_mode:
                ai_insights = st.session_state.get("opt_ai_insights", [])
                if ai_insights:
                    st.caption(f"Claude: _{ai_insights[-1]}_")

            st.info(f"Running: **{cfg.label()}**")

            # ── Run the single config ──────────────────────────────────
            result = run_single_config(cfg, design_basis, store)

            # Record timing
            st.session_state["opt_config_times"].append(result.duration_seconds)
            st.session_state["opt_last_config_label"] = cfg.label()
            st.session_state["opt_last_completed_at"] = datetime.now(
                timezone.utc
            ).strftime("%H:%M:%S UTC")

            # One-line result summary
            if result.converged:
                cmplx_str = f"+${result.complexity_per_kW:,.0f}cmplx" if result.complexity_per_kW > 0 else ""
                st.session_state["opt_last_result_summary"] = (
                    f"#{result.run_id} {cfg.label()} — "
                    f"${result.total_adjusted_per_kW:,.0f}/kW adj "
                    f"(${result.capex_per_kW:,.0f}inst{cmplx_str}), "
                    f"{result.cycle_efficiency*100:.1f}% eff, "
                    f"{result.net_power_MW:.1f} MW, "
                    f"{result.duration_seconds:.1f}s"
                )
            else:
                err_short = result.error[:60] if result.error else "unknown"
                st.session_state["opt_last_result_summary"] = (
                    f"#{result.run_id} {cfg.label()} — FAILED ({err_short}) "
                    f"{result.duration_seconds:.1f}s"
                )

            # Update Pareto frontier every 10 runs
            if store.stats()["total_runs"] % 10 == 0:
                update_pareto_frontier(store)
                st.session_state["opt_report"] = generate_report(store)

            # ── AI-Guided: batch completion logic ──────────────────────
            if is_ai_mode:
                st.session_state["opt_ai_batch_counter"] += 1

                if st.session_state["opt_ai_batch_counter"] >= AI_GUIDED_BATCH_SIZE or not st.session_state["opt_queue"]:
                    # Batch complete — call Claude for next batch
                    ai_round = st.session_state["opt_ai_round"]
                    _running_hr = st.session_state.get("opt_selected_hr") or list(HEAT_REJECTIONS)
                    ai_response = call_ai_optimizer(store, ai_round, design_basis,
                                                     heat_rejections=_running_hr)

                    # Log reasoning
                    log_entry = {
                        "round": ai_round,
                        "summary": ai_response.get("round_summary", ""),
                        "configs_proposed": len(ai_response.get("configs", [])),
                        "insights": ai_response.get("insights", []),
                        "error": ai_response.get("error", ""),
                    }
                    st.session_state["opt_ai_reasoning_log"].append(log_entry)
                    st.session_state["opt_ai_insights"].extend(ai_response.get("insights", []))

                    # Check convergence or max rounds
                    if ai_response.get("converged") or ai_round >= AI_GUIDED_MAX_ROUNDS:
                        st.session_state["opt_ai_converged"] = True
                        update_pareto_frontier(store)
                        st.session_state["opt_report"] = generate_report(store)
                        st.session_state["opt_running"] = False
                        st.session_state["opt_paused"] = False
                        if store.stats()["total_runs"] > 0:
                            st.session_state["dbd_update_phase"] = "preview"
                        st.rerun()
                    else:
                        # Queue next batch
                        new_configs = parse_ai_configs(ai_response, store)
                        st.session_state["opt_queue"] = new_configs
                        st.session_state["opt_total_configs"] = store.stats()["total_runs"] + len(new_configs)
                        st.session_state["opt_ai_batch_counter"] = 0
                        st.session_state["opt_ai_round"] += 1
                        st.rerun()
                else:
                    # More configs in current batch
                    st.rerun()
            else:
                # ── Brute Force: continue or finish ────────────────────
                if st.session_state["opt_queue"]:
                    st.rerun()
                else:
                    # Final updates
                    update_pareto_frontier(store)
                    st.session_state["opt_report"] = generate_report(store)
                    st.session_state["opt_running"] = False
                    st.session_state["opt_paused"] = False
                    if store.stats()["total_runs"] > 0:
                        st.session_state["dbd_update_phase"] = "preview"
                    st.rerun()
        else:
            st.session_state["opt_running"] = False
            st.session_state["opt_paused"] = False
            if store.stats()["total_runs"] > 0:
                st.session_state["dbd_update_phase"] = "preview"

    # ── Sub-tabs ─────────────────────────────────────────────────────
    sub1, sub2, sub3, sub4, sub5 = st.tabs([
        "Controls & Status", "Pareto Frontier", "Run History",
        "Step Change Analysis", "FEED Package",
    ])

    with sub1:
        _render_controls(design_basis, store)

    with sub2:
        _render_pareto(store)

    with sub3:
        _render_history(store)

    with sub4:
        render_step_change_subtab(store)

    with sub5:
        _render_feed_subtab(store, design_basis)


def _render_controls(design_basis: dict, store: ResultStore):
    """Controls & Status sub-tab."""
    stats = store.stats()

    # ── Mode selector ──────────────────────────────────────────────
    mode_labels = ["AI-Guided", "Brute Force"]
    current_mode = st.session_state.get("opt_mode", "ai_guided")
    default_idx = 0 if current_mode == "ai_guided" else 1
    mode = st.radio(
        "Optimizer Mode", mode_labels,
        index=default_idx, horizontal=True, key="opt_mode_radio",
        disabled=st.session_state["opt_running"],
    )
    st.session_state["opt_mode"] = "ai_guided" if mode == "AI-Guided" else "brute_force"

    if st.session_state["opt_mode"] == "ai_guided":
        st.caption(
            f"Claude picks batches of {AI_GUIDED_BATCH_SIZE} configs; "
            f"stops when converged or {AI_GUIDED_MAX_ROUNDS} rounds "
            f"({AI_GUIDED_MAX_ROUNDS * AI_GUIDED_BATCH_SIZE} configs max)"
        )
    else:
        st.caption("Exhaustive sweep of all valid configurations")

    # ── Procurement strategy selector ─────────────────────────────
    strategy_options = {STRATEGY_LABELS[s]: s for s in PROCUREMENT_STRATEGIES}
    selected_labels = st.multiselect(
        "Procurement strategies to explore",
        options=list(strategy_options.keys()),
        default=list(strategy_options.keys()),
        key="opt_strategy_selector",
        disabled=st.session_state["opt_running"],
    )
    selected_strategies = [strategy_options[lbl] for lbl in selected_labels] or list(PROCUREMENT_STRATEGIES)

    HR_LABELS = {"direct_acc": "Direct ACC", "propane_intermediate": "Propane IHX", "hybrid_wet_dry": "Hybrid Wet/Dry"}
    hr_options = {HR_LABELS[hr]: hr for hr in HEAT_REJECTIONS}
    selected_hr_labels = st.multiselect(
        "Heat rejection types to explore",
        options=list(hr_options.keys()),
        default=list(hr_options.keys()),
        key="opt_hr_selector",
        disabled=st.session_state["opt_running"],
    )
    selected_hr = [hr_options[lbl] for lbl in selected_hr_labels] or list(HEAT_REJECTIONS)

    # ── Train count selector ─────────────────────────────────────
    TRAIN_LABELS = {1: "1 train", 2: "2 trains", 3: "3 trains"}
    train_options = list(TRAIN_LABELS.keys())
    selected_trains = st.multiselect(
        "Turbine train configurations",
        options=train_options,
        default=[2],
        format_func=lambda x: TRAIN_LABELS[x],
        key="opt_trains_selector",
        disabled=st.session_state["opt_running"],
    )
    selected_n_trains = sorted(selected_trains) or [2]

    # ── Multi-unit site selector ──────────────────────────────────
    n_units = st.number_input(
        "Identical ORC units at site",
        min_value=1, max_value=10, value=1, step=1,
        help="Number of identical ORC plants. 2+ units get bulk procurement, "
             "shared mobilization, and construction learning curve discounts.",
        key="opt_n_units",
        disabled=st.session_state["opt_running"],
    )

    # ── Action buttons ─────────────────────────────────────────────
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        start_label = "Start" if not st.session_state["opt_running"] else "Running..."
        if st.session_state["opt_mode"] == "ai_guided" and not st.session_state["opt_running"]:
            start_label = "Start (AI-Guided)"
        if st.button(
            start_label,
            disabled=st.session_state["opt_running"],
            type="primary",
            use_container_width=True,
        ):
            if st.session_state["opt_mode"] == "ai_guided":
                queue = generate_seed_batch(store, strategies=selected_strategies, heat_rejections=selected_hr, n_trains_options=selected_n_trains)
                st.session_state["opt_queue"] = queue
                st.session_state["opt_total_configs"] = len(queue) + stats["total_runs"]
                st.session_state["opt_ai_round"] = 1
                st.session_state["opt_ai_batch_counter"] = 0
                st.session_state["opt_ai_reasoning_log"] = []
                st.session_state["opt_ai_insights"] = []
                st.session_state["opt_ai_converged"] = False
            else:
                queue = generate_search_space(store, strategies=selected_strategies, heat_rejections=selected_hr, n_trains_options=selected_n_trains)
                st.session_state["opt_queue"] = queue
                st.session_state["opt_total_configs"] = len(queue) + stats["total_runs"]
            st.session_state["opt_running"] = True
            st.session_state["opt_paused"] = False
            st.session_state["opt_start_time"] = time.time()
            st.session_state["opt_config_times"] = []
            st.session_state["opt_selected_hr"] = selected_hr
            st.session_state["opt_locked_n_units"] = n_units  # lock at start
            st.rerun()

    with col2:
        if st.button(
            "Resume" if st.session_state["opt_paused"] else "Pause",
            disabled=not st.session_state["opt_running"],
            use_container_width=True,
        ):
            st.session_state["opt_paused"] = not st.session_state["opt_paused"]
            if not st.session_state["opt_paused"]:
                # Resuming — trigger next config
                st.rerun()

    with col3:
        if st.button("Reset", use_container_width=True):
            store.reset()
            st.session_state["opt_running"] = False
            st.session_state["opt_paused"] = False
            st.session_state["opt_queue"] = []
            st.session_state["opt_report"] = None
            st.session_state["opt_total_configs"] = 0
            st.session_state["opt_start_time"] = 0.0
            st.session_state["opt_config_times"] = []
            st.session_state["opt_last_config_label"] = ""
            st.session_state["opt_last_completed_at"] = ""
            st.session_state["opt_last_result_summary"] = ""
            # Clear AI-guided state
            st.session_state["opt_ai_round"] = 0
            st.session_state["opt_ai_batch_counter"] = 0
            st.session_state["opt_ai_reasoning_log"] = []
            st.session_state["opt_ai_insights"] = []
            st.session_state["opt_ai_converged"] = False
            # Clear DBD update state
            st.session_state["dbd_update_proposal"] = None
            st.session_state["dbd_update_phase"] = "idle"
            st.session_state["dbd_update_review_index"] = 0
            st.session_state["dbd_update_error"] = None
            st.rerun()

    with col4:
        if st.button("Export JSON", use_container_width=True,
                      disabled=stats["total_runs"] == 0):
            st.download_button(
                "Download Results",
                data=open(store.path, "r").read() if stats["total_runs"] > 0 else "[]",
                file_name="optimizer_results.json",
                mime="application/json",
            )

    # ── Paused banner (running progress is shown in execution block above sub-tabs) ──
    if st.session_state["opt_running"] and st.session_state["opt_paused"]:
        remaining = len(st.session_state["opt_queue"])
        done = stats["total_runs"]
        hits = stats["target_hits"]
        misses = stats["converged"] - hits
        st.warning(
            f"**PAUSED** — {done} completed, {remaining} remaining | "
            f"{hits} hits / {misses} misses / {stats['failed']} failed"
        )

    # ── AI convergence banner ─────────────────────────────────────
    if st.session_state.get("opt_ai_converged"):
        ai_round = st.session_state.get("opt_ai_round", 0)
        st.success(
            f"AI-Guided optimization complete — {ai_round} rounds, "
            f"{stats['total_runs']} configs tested, "
            f"{stats['target_hits']} target hits"
        )

    # ── Running status (persistent in Controls tab) ───────────────
    if st.session_state["opt_running"] and not st.session_state["opt_paused"]:
        is_ai = st.session_state["opt_mode"] == "ai_guided"
        done = stats["total_runs"]
        total = st.session_state["opt_total_configs"]
        pct = min(done / total, 1.0) if total > 0 else 0
        hits = stats["target_hits"]
        misses = stats["converged"] - hits

        if is_ai:
            ai_round = st.session_state.get("opt_ai_round", 1)
            batch_counter = st.session_state.get("opt_ai_batch_counter", 0)
            st.info(
                f"**RUNNING (AI-Guided)** — Round {ai_round}/{AI_GUIDED_MAX_ROUNDS}, "
                f"Config {batch_counter}/{AI_GUIDED_BATCH_SIZE} in batch | "
                f"{done} tested, {hits} hits, {misses} misses, {stats['failed']} failed"
            )
        else:
            remaining = len(st.session_state["opt_queue"])
            st.info(
                f"**RUNNING (Brute Force)** — {done:,}/{total:,} ({pct*100:.1f}%) | "
                f"{remaining:,} remaining | "
                f"{hits} hits, {misses} misses, {stats['failed']} failed"
            )

        config_times = st.session_state.get("opt_config_times", [])
        if config_times:
            elapsed = time.time() - st.session_state["opt_start_time"]
            avg_time = sum(config_times) / len(config_times)
            st.caption(f"Elapsed: {_format_duration(elapsed)} | Avg: {avg_time:.1f}s/config")

    # ── Last completed config ─────────────────────────────────────
    last_summary = st.session_state.get("opt_last_result_summary", "")
    last_time = st.session_state.get("opt_last_completed_at", "")
    if last_summary:
        st.caption(f"Last completed ({last_time}): {last_summary}")

    # ── Metrics row ───────────────────────────────────────────────
    st.subheader("Progress")
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("Total Runs", stats["total_runs"])
    m2.metric("Converged", stats["converged"])
    m3.metric("Failed", stats["failed"])
    m4.metric("Target Hits", stats["target_hits"])
    m5.metric("Pareto Points", stats["pareto_count"])

    # ── Timing stats ─────────────────────────────────────────────
    config_times = st.session_state.get("opt_config_times", [])
    if config_times:
        tc1, tc2, tc3, tc4 = st.columns(4)
        avg_t = sum(config_times) / len(config_times)
        tc1.metric("Avg Time", f"{avg_t:.1f}s")
        tc2.metric("Min Time", f"{min(config_times):.1f}s")
        tc3.metric("Max Time", f"{max(config_times):.1f}s")
        tc4.metric("Total Time", _format_duration(sum(config_times)))

    # ── Best config (by adjusted $/kW) ──────────────────────────
    best = store.get_best_adjusted()
    if not best:
        best = store.get_best_per_kw()
    if best:
        st.subheader("Current Best (by Adjusted $/kW)")
        bc1, bc2, bc3, bc4, bc5, bc6 = st.columns(6)
        bc1.metric("Adjusted $/kW", f"${best.total_adjusted_per_kW:,.0f}",
                    delta=f"${best.total_adjusted_per_kW - TARGET_CAPEX_PER_KW:+,.0f} vs target")
        bc2.metric("Installed $/kW", f"${best.capex_per_kW:,.0f}")
        bc3.metric("Complexity $/kW", f"${best.complexity_per_kW:,.0f}")
        bc4.metric("Efficiency", f"{best.cycle_efficiency*100:.1f}%")
        bc5.metric("Net Power", f"{best.net_power_MW:.1f} MW")
        bc6.metric("Schedule", f"{best.construction_weeks} wk",
                    delta=f"{best.construction_weeks - TARGET_SCHEDULE_WEEKS:+d} vs target")

        cfg = best.config
        hr_map = {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}
        strat_label = STRATEGY_SHORT_LABELS.get(best.procurement_strategy, best.procurement_strategy)
        st.caption(
            f"Config: **{cfg.get('working_fluid', '?')}** | "
            f"**{cfg.get('topology', '?')}** | "
            f"**{hr_map.get(cfg.get('heat_rejection', ''), cfg.get('heat_rejection', '?'))}** | "
            f"**{strat_label}** | "
            f"Vap pinch {cfg.get('vaporizer_pinch_F', '?')}F | "
            f"ACC approach {cfg.get('acc_approach_F', '?')}F"
        )

        # Show raw best if different from adjusted best
        raw_best = store.get_best_per_kw()
        if raw_best and raw_best.config_hash != best.config_hash:
            st.caption(
                f"Note: Lowest raw installed cost is "
                f"${raw_best.capex_per_kW:,.0f}/kW "
                f"({raw_best.config.get('working_fluid')}/{raw_best.config.get('topology')}/"
                f"{hr_map.get(raw_best.config.get('heat_rejection', ''), '?')}) "
                f"but adjusted to ${raw_best.total_adjusted_per_kW:,.0f}/kW with complexity penalty"
            )

        with st.expander("Detailed Breakdown"):
            _render_config_breakdown(best)

        # Export report
        report_text = _generate_text_report(best)
        st.download_button(
            "Export Best Config Report",
            data=report_text,
            file_name=f"orc_best_config_run{best.run_id}.txt",
            mime="text/plain",
        )

    # ── Top Configurations Report ──────────────────────────────────
    _render_top_configs_report(store, stats)

    # ── Report insights ───────────────────────────────────────────
    report = st.session_state.get("opt_report")
    if report and report.get("insights"):
        st.subheader("Insights")
        for insight in report["insights"]:
            st.markdown(f"- {insight}")

    # ── Fluid rankings ────────────────────────────────────────────
    if report and report.get("fluid_rankings"):
        st.subheader("Fluid Rankings")
        fr = report["fluid_rankings"]
        rows = []
        for fl, data in sorted(fr.items(), key=lambda x: x[1]["best_kw"]):
            rows.append({
                "Fluid": fl,
                "Runs": data["count"],
                "Best $/kW": f"${data['best_kw']:,.0f}",
                "Avg $/kW": f"${data['avg_kw']:,.0f}",
                "Best Eff": f"{data['best_eff']*100:.1f}%",
            })
        if rows:
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    # ── Strategy rankings ──────────────────────────────────────────
    if report and report.get("strategy_rankings"):
        st.subheader("Procurement Strategy Rankings")
        sr = report["strategy_rankings"]
        strat_rows = []
        for strat, data in sorted(sr.items(), key=lambda x: x[1]["best_kw"]):
            strat_rows.append({
                "Strategy": STRATEGY_LABELS.get(strat, strat),
                "Runs": data["count"],
                "Best $/kW": f"${data['best_kw']:,.0f}",
                "Avg $/kW": f"${data['avg_kw']:,.0f}",
            })
        if strat_rows:
            st.dataframe(pd.DataFrame(strat_rows), use_container_width=True, hide_index=True)

    # ── Claude Insights (AI-Guided mode) ─────────────────────────
    ai_insights = st.session_state.get("opt_ai_insights", [])
    if ai_insights:
        st.subheader("Claude Insights")
        for ins in ai_insights[-10:]:
            st.markdown(f"- {ins}")

    # ── Optimizer Reasoning Log (AI-Guided mode) ──────────────────
    reasoning_log = st.session_state.get("opt_ai_reasoning_log", [])
    if reasoning_log:
        with st.expander(f"Optimizer Reasoning Log ({len(reasoning_log)} rounds)"):
            for entry in reversed(reasoning_log):
                st.markdown(f"**Round {entry['round']}**: {entry['summary']}")
                st.caption(f"Proposed: {entry['configs_proposed']} configs")
                if entry.get("error"):
                    st.caption(f"Error: {entry['error']}")
                if entry.get("insights"):
                    for ins in entry["insights"]:
                        st.markdown(f"  - {ins}")
                st.divider()

    # ── DBD Update Flow (post-optimization) ──────────────────────
    _render_dbd_update_flow(design_basis, store)

    # ── Search space info ─────────────────────────────────────────
    with st.expander("Search Space Details"):
        total_valid = total_search_space_size(strategies=selected_strategies, heat_rejections=selected_hr, n_trains_options=selected_n_trains)
        tested = stats["total_runs"]
        trains_desc = "/".join(str(t) for t in selected_n_trains)
        st.write(f"Total valid configurations: **{total_valid:,}** ({len(selected_strategies)} strategies, {trains_desc}-train)")
        st.write(f"Already tested: **{tested:,}**")
        st.write(f"Remaining: **{total_valid - tested:,}**")


def _render_dbd_update_flow(design_basis: dict, store: ResultStore):
    """Post-optimization DBD update flow.

    Phases: idle → preview → generating → preview_diff → reviewing → applying → complete
    """
    phase = st.session_state.get("dbd_update_phase", "idle")
    if phase == "idle":
        return

    st.divider()
    st.subheader("Design Basis Document Updates")

    error = st.session_state.get("dbd_update_error")
    if error:
        st.error(f"Error: {error}")

    # ── Phase: preview ─────────────────────────────────────────────
    if phase == "preview":
        st.info("Optimization complete. Would you like to generate Design Basis Document update proposals?")
        pc1, pc2 = st.columns(2)
        with pc1:
            if st.button("Generate DBD Update Proposal", type="primary",
                         use_container_width=True, key="dbd_generate_btn"):
                st.session_state["dbd_update_phase"] = "generating"
                st.session_state["dbd_update_error"] = None
                st.rerun()
        with pc2:
            if st.button("Skip", use_container_width=True, key="dbd_skip_btn"):
                st.session_state["dbd_update_phase"] = "idle"
                st.rerun()

    # ── Phase: generating ──────────────────────────────────────────
    elif phase == "generating":
        with st.spinner("Generating DBD update proposals..."):
            try:
                from optimizer_engine import generate_dbd_update_proposal
                proposal = generate_dbd_update_proposal(store, design_basis)
                st.session_state["dbd_update_proposal"] = proposal
                st.session_state["dbd_update_phase"] = "preview_diff"
                st.session_state["dbd_update_review_index"] = 0
            except Exception as e:
                st.session_state["dbd_update_error"] = str(e)
                st.session_state["dbd_update_phase"] = "preview"
        st.rerun()

    # ── Phase: preview_diff ─────────────────────────────────────────
    elif phase == "preview_diff":
        proposal = st.session_state.get("dbd_update_proposal")
        if not proposal or not proposal.get("items"):
            st.warning("No update proposals generated.")
            if st.button("Done", key="dbd_no_proposals_done"):
                st.session_state["dbd_update_phase"] = "idle"
                st.rerun()
            return

        items = proposal["items"]
        st.write(f"**{len(items)} proposed changes:**")

        # Group items by section
        from collections import defaultdict
        by_section = defaultdict(list)
        for item in items:
            by_section[item.get("section", "unknown")].append(item)

        section_labels = {
            "section_3_equipment": "Section 3: Equipment",
            "section_6_info_requests": "Section 6: Info Requests",
            "section_7_kb_inventory": "Section 7: KB Inventory",
            "section_8_opt_history": "Section 8: Opt History",
        }

        for section_key, section_items in by_section.items():
            label = section_labels.get(section_key, section_key)
            st.markdown(f"**{label}** ({len(section_items)} changes)")
            for item in section_items:
                action = item.get("action", "?")
                desc = item.get("description", "")
                confidence = item.get("confidence", "?")
                evidence = item.get("evidence", "")
                requires_approval = item.get("requires_approval", False)

                # Confidence badge color
                conf_color = {"HIGH": "green", "MEDIUM": "orange", "LOW": "red"}.get(
                    confidence, "gray"
                )

                col1, col2 = st.columns([4, 1])
                with col1:
                    approval_tag = " (requires approval)" if requires_approval else ""
                    st.markdown(f"- **[{action}]** {desc}{approval_tag}")
                    if item.get("old_value") is not None:
                        old_str = str(item["old_value"])
                        if len(old_str) > 100:
                            old_str = old_str[:100] + "..."
                        st.caption(f"Old: {old_str}")
                    new_str = str(item.get("new_value", ""))
                    if len(new_str) > 200:
                        new_str = new_str[:200] + "..."
                    st.caption(f"New: {new_str}")
                    if evidence:
                        st.caption(f"Evidence: {evidence}")
                with col2:
                    st.markdown(f":{conf_color}[{confidence}]")

        st.divider()

        # Action buttons
        bc1, bc2, bc3 = st.columns(3)
        with bc1:
            if st.button("Accept All", type="primary", use_container_width=True,
                         key="dbd_accept_all"):
                for item in items:
                    item["decision"] = "accepted"
                st.session_state["dbd_update_phase"] = "applying"
                st.rerun()
        with bc2:
            if st.button("Review Each", use_container_width=True, key="dbd_review_each"):
                st.session_state["dbd_update_review_index"] = 0
                st.session_state["dbd_update_phase"] = "reviewing"
                st.rerun()
        with bc3:
            if st.button("Dismiss All", use_container_width=True, key="dbd_dismiss"):
                st.session_state["dbd_update_phase"] = "idle"
                st.session_state["dbd_update_proposal"] = None
                st.rerun()

    # ── Phase: reviewing ────────────────────────────────────────────
    elif phase == "reviewing":
        proposal = st.session_state.get("dbd_update_proposal")
        if not proposal or not proposal.get("items"):
            st.session_state["dbd_update_phase"] = "idle"
            return

        items = proposal["items"]
        idx = st.session_state.get("dbd_update_review_index", 0)

        if idx >= len(items):
            # All reviewed — proceed to apply
            st.session_state["dbd_update_phase"] = "applying"
            st.rerun()
            return

        item = items[idx]
        st.progress((idx + 1) / len(items), text=f"Reviewing item {idx + 1} of {len(items)}")

        action = item.get("action", "?")
        section = item.get("section", "?")
        desc = item.get("description", "")
        confidence = item.get("confidence", "?")
        evidence = item.get("evidence", "")
        requires = item.get("requires_approval", False)

        section_labels = {
            "section_3_equipment": "Section 3: Equipment",
            "section_6_info_requests": "Section 6: Info Requests",
            "section_7_kb_inventory": "Section 7: KB Inventory",
            "section_8_opt_history": "Section 8: Opt History",
        }
        st.markdown(f"**[{action}] {section_labels.get(section, section)}**")
        st.markdown(desc)
        if requires:
            st.warning("This change affects a user-approved section and requires your explicit approval.")
        st.caption(f"Confidence: {confidence} | Evidence: {evidence}")

        if item.get("old_value") is not None:
            st.text_area("Current value", value=str(item["old_value"]),
                         disabled=True, height=80, key=f"dbd_rev_old_{idx}")

        import json as _json
        new_val = item.get("new_value", "")
        new_val_str = _json.dumps(new_val, indent=2, default=str) if isinstance(new_val, (dict, list)) else str(new_val)

        modified_value = st.text_area(
            "Proposed value (edit to modify)",
            value=new_val_str,
            height=120,
            key=f"dbd_rev_new_{idx}",
        )

        rc1, rc2, rc3, rc4 = st.columns(4)
        with rc1:
            if st.button("Accept", type="primary", use_container_width=True,
                         key=f"dbd_rev_accept_{idx}"):
                item["decision"] = "accepted"
                st.session_state["dbd_update_review_index"] = idx + 1
                st.rerun()
        with rc2:
            if st.button("Modify & Accept", use_container_width=True,
                         key=f"dbd_rev_modify_{idx}"):
                try:
                    parsed_mod = _json.loads(modified_value)
                except (ValueError, _json.JSONDecodeError):
                    parsed_mod = modified_value
                item["new_value"] = parsed_mod
                item["decision"] = "modified"
                st.session_state["dbd_update_review_index"] = idx + 1
                st.rerun()
        with rc3:
            if st.button("Reject", use_container_width=True,
                         key=f"dbd_rev_reject_{idx}"):
                item["decision"] = "rejected"
                st.session_state["dbd_update_review_index"] = idx + 1
                st.rerun()
        with rc4:
            if st.button("Skip", use_container_width=True,
                         key=f"dbd_rev_skip_{idx}"):
                # Leave as pending (won't be applied)
                st.session_state["dbd_update_review_index"] = idx + 1
                st.rerun()

    # ── Phase: applying ─────────────────────────────────────────────
    elif phase == "applying":
        proposal = st.session_state.get("dbd_update_proposal")
        if not proposal or not proposal.get("items"):
            st.session_state["dbd_update_phase"] = "idle"
            return

        items = proposal["items"]
        accepted = [i for i in items if i.get("decision") in ("accepted", "modified")]

        if not accepted:
            st.info("No changes accepted. No updates applied.")
            if st.button("Done", key="dbd_apply_none_done"):
                st.session_state["dbd_update_phase"] = "idle"
                st.session_state["dbd_update_proposal"] = None
                st.rerun()
            return

        with st.spinner(f"Applying {len(accepted)} DBD updates..."):
            try:
                from design_basis_document import apply_dbd_updates
                from optimizer_engine import generate_session_summary

                updated_dbd, new_version = apply_dbd_updates(items)

                # Generate session summary
                summary = generate_session_summary(store, design_basis, items)
                proposal["session_summary"] = summary

                # Save session file
                import os
                sessions_dir = os.path.join(
                    os.path.dirname(__file__), "knowledge", "docs", "sessions"
                )
                os.makedirs(sessions_dir, exist_ok=True)
                ts = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
                session_file = os.path.join(sessions_dir, f"session_{ts}.txt")
                with open(session_file, "w", encoding="utf-8") as f:
                    f.write(summary)

                # Ingest into knowledge base
                try:
                    from knowledge.ingestion import ingest_text
                    ingest_text(
                        summary,
                        filename=f"session_{ts}.txt",
                        domain="economics_market",
                        tags=["optimizer_session", "design_basis_update"],
                    )
                except Exception:
                    pass  # Knowledge base ingestion is best-effort

                st.session_state["dbd_update_phase"] = "complete"
            except Exception as e:
                st.session_state["dbd_update_error"] = str(e)
                st.session_state["dbd_update_phase"] = "preview_diff"
        st.rerun()

    # ── Phase: complete ─────────────────────────────────────────────
    elif phase == "complete":
        proposal = st.session_state.get("dbd_update_proposal", {})
        items = proposal.get("items", [])
        accepted = [i for i in items if i.get("decision") in ("accepted", "modified")]
        rejected = [i for i in items if i.get("decision") == "rejected"]

        st.success(
            f"DBD updated successfully: {len(accepted)} changes applied, "
            f"{len(rejected)} rejected"
        )

        # Show session summary
        summary = proposal.get("session_summary", "")
        if summary:
            with st.expander("Session Summary", expanded=True):
                st.text(summary)

        if st.button("Done", type="primary", key="dbd_complete_done"):
            st.session_state["dbd_update_phase"] = "idle"
            st.session_state["dbd_update_proposal"] = None
            st.session_state["dbd_update_review_index"] = 0
            st.session_state["dbd_update_error"] = None
            st.rerun()


def _render_pareto(store: ResultStore):
    """Pareto Frontier sub-tab."""
    converged = store.get_converged()
    if not converged:
        st.info("No converged results yet. Start the optimizer to see the Pareto frontier.")
        return

    st.subheader("Pareto Frontier: Efficiency vs Cost")

    # Controls row
    pc1, pc2 = st.columns(2)
    with pc1:
        color_by = st.radio(
            "Color scatter plot by:", ["Fluid", "Strategy"],
            horizontal=True, key="opt_pareto_color_by",
        )
    with pc2:
        cost_axis = st.radio(
            "X-axis cost metric:", ["Adjusted $/kW", "Raw $/kW"],
            horizontal=True, key="opt_pareto_cost_axis",
        )

    use_adjusted = cost_axis == "Adjusted $/kW"
    x_col = "Adjusted $/kW" if use_adjusted else "Raw $/kW"

    # Build dataframe for plotting
    rows = []
    for r in converged:
        rows.append({
            "Efficiency (%)": r.cycle_efficiency * 100,
            "Adjusted $/kW": r.total_adjusted_per_kW,
            "Raw $/kW": r.capex_per_kW,
            "Complexity $/kW": r.complexity_per_kW,
            "Fluid": r.config.get("working_fluid", "?"),
            "Topology": r.config.get("topology", "?"),
            "Heat Rejection": {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}.get(r.config.get("heat_rejection", ""), "?"),
            "Strategy": STRATEGY_SHORT_LABELS.get(r.procurement_strategy, r.procurement_strategy),
            "Pareto": "Yes" if r.pareto_optimal else "No",
            "Net MW": r.net_power_MW,
            "NPV ($M)": r.npv_USD / 1e6,
        })

    df = pd.DataFrame(rows)

    # Scatter plot
    st.scatter_chart(
        df,
        x=x_col,
        y="Efficiency (%)",
        color=color_by,
        size=None,
    )

    # Target line reference
    st.caption(f"Target: ${TARGET_CAPEX_PER_KW:,.0f}/kW adjusted (vertical reference)")

    # Pareto-optimal configs table
    pareto = store.get_pareto()
    if pareto:
        st.subheader(f"Pareto-Optimal Configurations ({len(pareto)})")
        pareto_rows = []
        for r in sorted(pareto, key=lambda x: x.total_adjusted_per_kW):
            cfg = r.config
            # Turbine layout string
            n_tr = r.n_trains if r.n_trains else 2
            if r.lp_turbine_mw > 0 and r.n_turbine_units > 0:
                turb_str = f"{n_tr}x{r.hp_turbine_mw:.1f}+{n_tr}x{r.lp_turbine_mw:.1f}"
            elif r.n_turbine_units > 0:
                turb_str = f"{n_tr}x{r.hp_turbine_mw:.1f}"
            else:
                turb_str = "-"
            mkt_str = "OK" if r.turbine_market_ok else "OVER"
            pareto_rows.append({
                "Run": r.run_id,
                "Fluid": cfg.get("working_fluid", "?"),
                "Topology": cfg.get("topology", "?"),
                "HR": {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}.get(cfg.get("heat_rejection", ""), "?"),
                "Strategy": STRATEGY_SHORT_LABELS.get(r.procurement_strategy, r.procurement_strategy),
                "Trains": n_tr,
                "Eff (%)": f"{r.cycle_efficiency*100:.1f}",
                "Equip $/kW": f"${r.equipment_per_kW:,.0f}",
                "Inst $/kW": f"${r.capex_per_kW:,.0f}",
                "Cmplx $/kW": f"${r.complexity_per_kW:,.0f}",
                "Adj $/kW": f"${r.total_adjusted_per_kW:,.0f}",
                "Net MW": f"{r.net_power_MW:.2f}",
                "Turbines": turb_str,
                "Mkt": mkt_str,
                "NPV ($M)": f"{r.npv_USD/1e6:.1f}",
                "Schedule": f"{r.construction_weeks} wk",
                "Target": "HIT" if r.target_fit else "",
            })
        st.dataframe(pd.DataFrame(pareto_rows), use_container_width=True, hide_index=True)


def _render_history(store: ResultStore):
    """Run History sub-tab."""
    if not store.results:
        st.info("No results yet. Start the optimizer to see run history.")
        return

    st.subheader(f"Run History ({len(store.results)} total)")

    # Filters
    fc1, fc2, fc3, fc4 = st.columns(4)
    with fc1:
        fluid_filter = st.multiselect(
            "Filter by fluid", WORKING_FLUIDS,
            default=[], key="opt_hist_fluid_filter",
        )
    with fc2:
        strategy_filter = st.multiselect(
            "Filter by strategy",
            options=[STRATEGY_SHORT_LABELS[s] for s in PROCUREMENT_STRATEGIES],
            default=[], key="opt_hist_strategy_filter",
        )
        # Reverse-map short labels to strategy keys
        _short_to_key = {v: k for k, v in STRATEGY_SHORT_LABELS.items()}
        strategy_filter_keys = [_short_to_key[lbl] for lbl in strategy_filter]
    with fc3:
        conv_filter = st.selectbox(
            "Convergence", ["All", "Converged", "Failed"],
            key="opt_hist_conv_filter",
        )
    with fc4:
        pareto_only = st.checkbox("Pareto-optimal only", key="opt_hist_pareto_only")

    # Build filtered dataframe
    rows = []
    for r in store.results:
        cfg = r.config
        fl = cfg.get("working_fluid", "?")

        if fluid_filter and fl not in fluid_filter:
            continue
        if strategy_filter_keys and r.procurement_strategy not in strategy_filter_keys:
            continue
        if conv_filter == "Converged" and not r.converged:
            continue
        if conv_filter == "Failed" and r.converged:
            continue
        if pareto_only and not r.pareto_optimal:
            continue

        # Turbine layout
        n_tr = r.n_trains if r.n_trains else 2
        if r.converged and r.lp_turbine_mw > 0 and r.n_turbine_units > 0:
            turb_hist = f"{n_tr}x{r.hp_turbine_mw:.1f}+{n_tr}x{r.lp_turbine_mw:.1f}"
        elif r.converged and r.n_turbine_units > 0:
            turb_hist = f"{n_tr}x{r.hp_turbine_mw:.1f}"
        else:
            turb_hist = "-"
        rows.append({
            "Run": r.run_id,
            "Fluid": fl,
            "Topology": cfg.get("topology", "?"),
            "HR": {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}.get(cfg.get("heat_rejection", ""), "?"),
            "Strategy": STRATEGY_SHORT_LABELS.get(r.procurement_strategy, r.procurement_strategy),
            "Trains": n_tr,
            "Vap": cfg.get("vaporizer_pinch_F", "?"),
            "ACC": cfg.get("acc_approach_F", "?"),
            "Pre": cfg.get("preheater_pinch_F", "?"),
            "Recup": cfg.get("recuperator_pinch_F", "?"),
            "OK": "Y" if r.converged else "N",
            "Eff (%)": f"{r.cycle_efficiency*100:.1f}" if r.converged else "-",
            "Inst $/kW": f"${r.capex_per_kW:,.0f}" if r.converged else "-",
            "Cmplx": f"${r.complexity_per_kW:,.0f}" if r.converged else "-",
            "Adj $/kW": f"${r.total_adjusted_per_kW:,.0f}" if r.converged else "-",
            "Net MW": f"{r.net_power_MW:.2f}" if r.converged else "-",
            "Turbines": turb_hist,
            "Time": f"{r.duration_seconds:.1f}s" if r.duration_seconds else "-",
            "Pareto": "Y" if r.pareto_optimal else "",
            "Target": "HIT" if r.target_fit else "",
            "Error": r.error[:40] if r.error else "",
        })

    if rows:
        st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
    else:
        st.info("No results match the current filters.")

    # Detailed breakdown for selected run
    if store.results:
        run_ids = [r.run_id for r in store.results if r.converged]
        if run_ids:
            with st.expander("Detailed Breakdown (select run)"):
                sel_id = st.selectbox("Run ID", run_ids, key="opt_bom_run_id")
                sel = next((r for r in store.results if r.run_id == sel_id), None)
                if sel:
                    _render_config_breakdown(sel)

            # FEED Light Package trigger
            st.divider()
            st.subheader("FEED Light Package")
            feed_id = st.selectbox(
                "Select run for FEED package", run_ids, key="opt_feed_select",
            )
            if st.button("Generate FEED Package", type="primary", key="opt_feed_btn"):
                st.session_state["opt_feed_run_id"] = feed_id
                st.session_state["opt_feed_data"] = None  # clear cache
                st.rerun()


def _render_feed_subtab(store: ResultStore, design_basis: dict):
    """FEED Package sub-tab — renders full FEED Light for selected run."""
    from feed_package import render_feed_package

    feed_id = st.session_state.get("opt_feed_run_id")
    if feed_id is None:
        st.info("Select a run from the **Run History** tab and click 'Generate FEED Package'.")
        return
    result = next((r for r in store.results if r.run_id == feed_id), None)
    if result is None:
        st.warning(f"Run #{feed_id} not found in result store.")
        return
    if not result.converged:
        st.error(f"Run #{feed_id} did not converge — cannot generate FEED package.")
        return
    render_feed_package(result, design_basis)


def _render_top_configs_report(store: ResultStore, stats: dict):
    """Render Top 5 detailed report + full converged results table."""
    if stats["converged"] == 0:
        return

    converged = sorted(store.get_converged(), key=lambda r: r.total_adjusted_per_kW)
    hr_map = {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}
    topo_map = {"basic": "Basic", "recuperated": "Recup", "dual_pressure": "Dual-P"}

    # ── Top 5 Detailed Cards ─────────────────────────────────────
    top_n = converged[:5]
    st.subheader(f"Top {len(top_n)} Configurations (by Adjusted $/kW)")
    for rank, r in enumerate(top_n, 1):
        cfg = r.config
        strat_label = STRATEGY_SHORT_LABELS.get(r.procurement_strategy, r.procurement_strategy)
        topo_label = topo_map.get(cfg.get("topology", ""), cfg.get("topology", "?"))
        hr_label = hr_map.get(cfg.get("heat_rejection", ""), cfg.get("heat_rejection", "?"))
        fit_str = "TARGET HIT" if r.target_fit else "miss"

        with st.expander(
            f"#{rank}  {cfg.get('working_fluid','?')} / {topo_label} / {hr_label} / {strat_label}"
            f"  |  **${r.total_adjusted_per_kW:,.0f}/kW**  |  {r.net_power_MW:.1f} MW  |  {fit_str}",
            expanded=(rank == 1),
        ):
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Adjusted $/kW", f"${r.total_adjusted_per_kW:,.0f}")
            c2.metric("Installed $/kW", f"${r.capex_per_kW:,.0f}")
            c3.metric("Equipment $/kW", f"${r.equipment_per_kW:,.0f}")
            c4.metric("Complexity $/kW", f"${r.complexity_per_kW:,.0f}")

            c5, c6, c7, c8 = st.columns(4)
            c5.metric("Net Power", f"{r.net_power_MW:.1f} MW")
            c6.metric("Gross Power", f"{r.gross_power_MW:.1f} MW")
            c7.metric("Efficiency", f"{r.cycle_efficiency*100:.1f}%")
            c8.metric("Schedule", f"{r.construction_weeks} wk")

            c9, c10, c11, c12 = st.columns(4)
            c9.metric("Total Installed", f"${r.capex_total_USD:,.0f}")
            c10.metric("NPV", f"${r.npv_USD:,.0f}")
            c11.metric("LCOE", f"${r.lcoe_per_MWh:.1f}/MWh")
            c12.metric("Target Fit", fit_str)

            # Turbine layout line
            n_tr = r.n_trains if r.n_trains else 2
            if r.lp_turbine_mw > 0 and r.n_turbine_units > 0:
                mkt_icon = " OK" if r.turbine_market_ok else f" OVER {TURBINE_MARKET_LIMIT_MW:.0f}MW"
                turb_line = (
                    f"{n_tr}x HP @ {r.hp_turbine_mw:.1f} MW + "
                    f"{n_tr}x LP @ {r.lp_turbine_mw:.1f} MW "
                    f"({r.n_turbine_units} units, {n_tr} trains){mkt_icon}"
                )
            elif r.n_turbine_units > 0:
                mkt_icon = " OK" if r.turbine_market_ok else f" OVER {TURBINE_MARKET_LIMIT_MW:.0f}MW"
                turb_line = f"{n_tr}x @ {r.hp_turbine_mw:.1f} MW ({r.n_turbine_units} units, {n_tr} trains){mkt_icon}"
            else:
                turb_line = "-"
            st.markdown(f"**Turbine Layout:** {turb_line}")
            if r.n_units > 1:
                st.markdown(f"**Site:** {r.n_units} identical units — {r.multi_unit_savings_pct:.1f}% per-unit savings")

            st.markdown("**Pinch Points:**")
            st.caption(
                f"Vaporizer: {cfg.get('vaporizer_pinch_F','?')} F  |  "
                f"ACC approach: {cfg.get('acc_approach_F','?')} F  |  "
                f"Preheater: {cfg.get('preheater_pinch_F','?')} F  |  "
                f"Recuperator: {cfg.get('recuperator_pinch_F','?')} F"
            )

            # BOM breakdown if available
            if r.bom_per_kw:
                st.markdown("**BOM ($/kW):**")
                bom_items = sorted(r.bom_per_kw.items(), key=lambda x: -x[1])
                bom_df = pd.DataFrame(
                    [{"Component": k.replace("_", " ").title(), "$/kW": f"${v:,.0f}"} for k, v in bom_items]
                )
                st.dataframe(bom_df, use_container_width=True, hide_index=True)

    # ── Full Results Table ────────────────────────────────────────
    st.subheader("All Converged Results")
    rows = []
    for r in converged:
        cfg = r.config
        rows.append({
            "Rank": len(rows) + 1,
            "Fluid": cfg.get("working_fluid", "?"),
            "Topology": topo_map.get(cfg.get("topology", ""), "?"),
            "HR": hr_map.get(cfg.get("heat_rejection", ""), "?"),
            "Strategy": STRATEGY_SHORT_LABELS.get(r.procurement_strategy, "?"),
            "Trains": r.n_trains if r.n_trains else 2,
            "Adj $/kW": round(r.total_adjusted_per_kW),
            "Inst $/kW": round(r.capex_per_kW),
            "Equip $/kW": round(r.equipment_per_kW),
            "Cmplx $/kW": round(r.complexity_per_kW),
            "Net MW": round(r.net_power_MW, 1),
            "Gross MW": round(r.gross_power_MW, 1),
            "Eff %": round(r.cycle_efficiency * 100, 1),
            "Sched wk": r.construction_weeks,
            "NPV $": round(r.npv_USD),
            "LCOE $/MWh": round(r.lcoe_per_MWh, 1),
            "Target": "YES" if r.target_fit else "no",
            "Vap F": cfg.get("vaporizer_pinch_F", ""),
            "ACC F": cfg.get("acc_approach_F", ""),
            "Pre F": cfg.get("preheater_pinch_F", ""),
            "Rec F": cfg.get("recuperator_pinch_F", ""),
            "HP MW": r.hp_turbine_mw if r.hp_turbine_mw > 0 else "",
            "LP MW": r.lp_turbine_mw if r.lp_turbine_mw > 0 else "",
        })
    if rows:
        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True, height=400)

        # CSV download
        csv = df.to_csv(index=False)
        st.download_button(
            "Export All Results (CSV)",
            data=csv,
            file_name="optimizer_all_results.csv",
            mime="text/csv",
        )


def _render_config_breakdown(result):
    """Render detailed cost/performance breakdown for one OptResult."""
    bom = result.bom_per_kw or {}
    net_kw = result.net_power_MW * 1000 if result.net_power_MW > 0 else 1
    cfg = result.config

    # ── Procurement strategy label ──────────────────────────────────
    strat_label = STRATEGY_LABELS.get(result.procurement_strategy, result.procurement_strategy)
    st.markdown(f"**Procurement Strategy:** {strat_label}")

    # ── Complexity Penalty Breakdown ─────────────────────────────────
    if result.complexity_per_kW > 0:
        st.markdown("**Complexity Penalty Breakdown**")
        topo = cfg.get("topology", "basic")
        hr = cfg.get("heat_rejection", "direct_acc")
        penalty_items = []
        if topo == "recuperated":
            penalty_items.append(("Recuperator (thermal cycling, pinch degradation)", COMPLEXITY_PENALTIES["recuperator"]))
        elif topo == "dual_pressure":
            penalty_items.append(("Dual pressure (2x equipment, 2x controls)", COMPLEXITY_PENALTIES["dual_pressure"]))
        if hr == "propane_intermediate":
            penalty_items.append(("Propane intermediate (leak risk, regulatory)", COMPLEXITY_PENALTIES["propane_intermediate"]))
        elif hr == "hybrid_wet_dry":
            penalty_items.append(("Hybrid wet/dry (water treatment, freeze protection)", COMPLEXITY_PENALTIES["hybrid_wet_dry"]))
        for desc, val in penalty_items:
            st.markdown(f"- {desc}: **${val}/kW**")
        st.markdown(
            f"- **Total complexity: ${result.complexity_per_kW:,.0f}/kW** | "
            f"Adjusted total: ${result.total_adjusted_per_kW:,.0f}/kW "
            f"(installed ${result.capex_per_kW:,.0f} + complexity ${result.complexity_per_kW:,.0f})"
        )
    else:
        st.markdown("**Complexity:** $0/kW — minimum viable design (basic + direct ACC)")

    # ── A) Equipment Costs ──────────────────────────────────────────
    st.markdown("**Equipment Costs**")
    equip_items = [
        ("Turbine-Generator", "turbine_generator"),
        ("Vaporizer", "vaporizer"),
        ("Preheater", "preheater"),
        ("Recuperator", "recuperator"),
        ("ACC", "acc"),
        ("ISO Pump", "iso_pump"),
        ("Ductwork", "ductwork"),
        ("Structural Steel", "structural_steel"),
    ]
    if cfg.get("heat_rejection") == "propane_intermediate":
        equip_items.extend([
            ("Intermediate HX", "intermediate_hx"),
            ("Propane System", "propane_system"),
        ])
    equip_rows = []
    for label, key in equip_items:
        per_kw = bom.get(key, 0)
        total = per_kw * net_kw
        display_label = label
        if key == "acc" and result.acc_n_bays > 0:
            display_label = f"ACC ({result.acc_n_bays} bays)"
        equip_rows.append({"Component": display_label, "$/kW": f"${per_kw:,.0f}",
                           "Total $": f"${total:,.0f}"})
    equip_sub = bom.get("equipment_subtotal", 0)
    equip_rows.append({"Component": "Equipment Subtotal", "$/kW": f"${equip_sub:,.0f}",
                        "Total $": f"${equip_sub * net_kw:,.0f}"})
    st.dataframe(pd.DataFrame(equip_rows), use_container_width=True, hide_index=True)

    # ── B) Installation Costs ───────────────────────────────────────
    st.markdown("**Installation & Indirect Costs**")
    install_items = [
        ("BOP Piping & Valves", "bop_piping"),
        ("Civil & Structural", "civil_structural"),
        ("E&I Installation", "ei_installation"),
        ("Construction Labor", "construction_labor"),
        ("Engineering", "engineering"),
        ("Commissioning", "commissioning"),
        ("Contingency", "contingency"),
    ]
    install_rows = []
    for label, key in install_items:
        per_kw = bom.get(key, 0)
        total = per_kw * net_kw
        install_rows.append({"Category": label, "$/kW": f"${per_kw:,.0f}",
                              "Total $": f"${total:,.0f}"})
    total_installed_kw = result.capex_per_kW
    install_rows.append({"Category": "Total Installed", "$/kW": f"${total_installed_kw:,.0f}",
                          "Total $": f"${result.capex_total_USD:,.0f}"})
    st.dataframe(pd.DataFrame(install_rows), use_container_width=True, hide_index=True)

    st.caption("Working fluid inventory, controls/instruments, electrical MV/LV "
               "included in equipment subtotal and E&I installation (not itemized).")

    # ── C) Top 3 Cost Drivers ───────────────────────────────────────
    if bom:
        # Filter out subtotals
        driver_keys = [k for k in bom if k not in ("equipment_subtotal",)]
        sorted_drivers = sorted(driver_keys, key=lambda k: -bom.get(k, 0))[:3]
        if sorted_drivers and total_installed_kw > 0:
            st.markdown("**Top 3 Cost Drivers**")
            for k in sorted_drivers:
                per_kw = bom[k]
                pct = per_kw / total_installed_kw * 100 if total_installed_kw > 0 else 0
                st.markdown(f"- **{k.replace('_', ' ').title()}**: ${per_kw:,.0f}/kW ({pct:.1f}%)")

    # ── D) Thermal & Parasitic Breakdown ────────────────────────────
    st.markdown("**Thermal Performance & Parasitic Loads**")
    td = result.thermal_detail or {}
    pb = result.parasitic_breakdown or {}
    col_therm, col_para = st.columns(2)

    with col_therm:
        st.markdown("*Cycle Parameters*")
        st.markdown(f"- Gross power: {result.gross_power_MW:.2f} MW")
        st.markdown(f"- Net power: {result.net_power_MW:.2f} MW")
        st.markdown(f"- Cycle efficiency: {result.cycle_efficiency*100:.1f}%")
        if td.get("T_evap_F"):
            st.markdown(f"- T_evap: {td['T_evap_F']:.0f} °F")
        if td.get("T_cond_F"):
            st.markdown(f"- T_cond: {td['T_cond_F']:.0f} °F")
        if td.get("P_high_psia"):
            st.markdown(f"- P_high: {td['P_high_psia']:.0f} psia")
        if td.get("P_low_psia"):
            st.markdown(f"- P_low: {td['P_low_psia']:.1f} psia")
        if td.get("pressure_ratio"):
            st.markdown(f"- Pressure ratio: {td['pressure_ratio']:.1f}")
        if td.get("brine_effectiveness"):
            st.markdown(f"- Brine effectiveness: {td['brine_effectiveness']:.1f} kW/(lb/s)")

    with col_para:
        st.markdown("*Parasitic Loads*")
        if pb:
            para_items = [
                ("ISO Pump", pb.get("W_iso_pump_kw", 0)),
                ("Propane Pump", pb.get("W_prop_pump_kw", 0)),
                ("ACC Fans", pb.get("W_fans_kw", 0)),
                ("Auxiliary", pb.get("W_auxiliary_kw", 0)),
            ]
            gross_kw = pb.get("P_gross_kw", 1)
            for label, kw in para_items:
                if kw > 0:
                    pct = kw / gross_kw * 100 if gross_kw > 0 else 0
                    st.markdown(f"- {label}: {kw:,.0f} kW ({pct:.1f}%)")
            total_para = pb.get("W_total_parasitic_kw", 0)
            para_pct = pb.get("parasitic_pct", 0)
            st.markdown(f"- **Total parasitic: {total_para:,.0f} kW ({para_pct:.1f}%)**")
        else:
            st.caption("No parasitic data available for this run.")

    # ── E) Efficiency Limiters ──────────────────────────────────────
    if pb or td:
        st.markdown("**Primary Efficiency Limiters**")
        limiters = []
        if pb.get("parasitic_pct", 0) > 0:
            limiters.append(("Parasitic losses", pb["parasitic_pct"], "%"))
        gen_eff = pb.get("generator_efficiency", 0.96) if pb else 0.96
        if gen_eff < 1.0:
            gen_loss = (1 - gen_eff) * 100
            limiters.append(("Generator/mechanical losses", gen_loss, "%"))
        if td.get("T_evap_F") and td.get("T_cond_F"):
            T_hot_R = td["T_evap_F"] + 459.67
            T_cold_R = td["T_cond_F"] + 459.67
            carnot = 1 - T_cold_R / T_hot_R if T_hot_R > 0 else 0
            if carnot > 0 and result.cycle_efficiency > 0:
                carnot_frac = result.cycle_efficiency / carnot * 100
                carnot_gap = (1 - result.cycle_efficiency / carnot) * 100
                limiters.append(("Carnot deviation", carnot_gap, f"% (achieving {carnot_frac:.0f}% of Carnot)"))
        limiters.sort(key=lambda x: -x[1])
        for label, val, unit in limiters[:3]:
            st.markdown(f"- {label}: {val:.1f}{unit}")

    # ── F) Schedule Breakdown ───────────────────────────────────────
    phases = result.schedule_phases
    if phases:
        st.markdown("**Schedule Breakdown**")
        phase_rows = [{"Phase": p.get("name", "?"),
                       "Start (wk)": p.get("start", 0),
                       "End (wk)": p.get("end", 0),
                       "Duration (wk)": p.get("duration", 0),
                       "Critical": "Yes" if p.get("critical") else ""}
                      for p in phases]
        st.dataframe(pd.DataFrame(phase_rows), use_container_width=True, hide_index=True)
        st.markdown(f"**Total: {result.construction_weeks} weeks** "
                    f"(delta vs 70-wk target: {result.construction_weeks - 70:+d} weeks)")
        critical = [p for p in phases if p.get("critical")]
        if critical:
            longest = max(critical, key=lambda p: p.get("duration", 0))
            st.caption(f"Critical path bottleneck: {longest.get('name', '?')} ({longest.get('duration', 0)} wk)")
    elif result.construction_weeks > 0:
        st.markdown(f"**Schedule: {result.construction_weeks} weeks** "
                    f"(delta vs 70-wk target: {result.construction_weeks - 70:+d} weeks)")


def _generate_text_report(result) -> str:
    """Generate formatted text report for download."""
    cfg = result.config
    strat_label = STRATEGY_LABELS.get(result.procurement_strategy, result.procurement_strategy)
    lines = [
        "=" * 60,
        "ORC OPTIMIZER — CONFIGURATION REPORT",
        "=" * 60,
        "",
        f"Run ID: {result.run_id}",
        f"Working Fluid: {cfg.get('working_fluid', '?')}",
        f"Topology: {cfg.get('topology', '?')}",
        f"Heat Rejection: {cfg.get('heat_rejection', '?')}",
        f"Procurement Strategy: {strat_label}",
        f"Vaporizer Pinch: {cfg.get('vaporizer_pinch_F', '?')} °F",
        f"ACC Approach: {cfg.get('acc_approach_F', '?')} °F",
        f"Preheater Pinch: {cfg.get('preheater_pinch_F', '?')} °F",
        f"Recuperator Pinch: {cfg.get('recuperator_pinch_F', '?')} °F",
        "",
        "── PERFORMANCE ──",
        f"Gross Power:       {result.gross_power_MW:.2f} MW",
        f"Net Power:         {result.net_power_MW:.2f} MW",
        f"Cycle Efficiency:  {result.cycle_efficiency*100:.1f}%",
        f"NPV:               ${result.npv_USD:,.0f}",
        f"LCOE:              ${result.lcoe_per_MWh:.1f}/MWh",
        "",
        "── COSTS ──",
        f"Equipment $/kW:    ${result.equipment_per_kW:,.0f}",
        f"Installed $/kW:    ${result.capex_per_kW:,.0f}",
        f"Complexity $/kW:   ${result.complexity_per_kW:,.0f}",
        f"Adjusted $/kW:     ${result.total_adjusted_per_kW:,.0f}",
        f"Total Installed:   ${result.capex_total_USD:,.0f}",
        f"Target Hit:        {'YES' if result.target_fit else 'NO'}",
        "",
        "── SCHEDULE ──",
        f"Construction:      {result.construction_weeks} weeks",
        f"Schedule Target:   70 weeks",
        "",
    ]
    # BOM
    if result.bom_per_kw:
        lines.append("── BOM ($/kW) ──")
        for k, v in sorted(result.bom_per_kw.items(), key=lambda x: -x[1]):
            lines.append(f"  {k:30s}  ${v:,.0f}/kW")
        lines.append("")

    # Parasitic
    pb = result.parasitic_breakdown
    if pb:
        lines.append("── PARASITIC LOADS ──")
        for label, key in [("ISO Pump", "W_iso_pump_kw"), ("ACC Fans", "W_fans_kw"),
                           ("Auxiliary", "W_auxiliary_kw"), ("Total", "W_total_parasitic_kw")]:
            lines.append(f"  {label:20s}  {pb.get(key, 0):,.0f} kW")
        lines.append(f"  Parasitic %:         {pb.get('parasitic_pct', 0):.1f}%")
        lines.append("")

    if result.warnings:
        lines.append("── WARNINGS ──")
        for w in result.warnings:
            lines.append(f"  - {w}")
        lines.append("")

    lines.append("Generated by ORC Workbench Autonomous Optimizer")
    return "\n".join(lines)
