"""
Step Change Analysis — maps incremental vs transformative cost-reduction pathways.

Three scenarios:
  A (Incremental): Current optimizer best — what's achievable today
  B (Methodology): Same thermodynamics, different execution strategy
  C (Technology):  Different fundamental technology choices

Output: Step Change Map chart + itemized breakdowns + investment prioritization.
"""

from dataclasses import dataclass, field
from typing import Optional

import streamlit as st
import pandas as pd

from optimizer_engine import ResultStore, TARGET_CAPEX_PER_KW


# ── Data Model ────────────────────────────────────────────────────────────

@dataclass
class StepChangeItem:
    name: str                       # e.g. "Modular skid fabrication"
    description: str
    delta_per_kW: float             # $/kW savings (negative = cheaper)
    delta_schedule_weeks: float     # schedule impact (negative = faster)
    probability_pct: float          # 0-100, chance of achieving
    proving_cost_USD: float         # investment to prove/validate
    basis: str                      # source/justification
    status: str                     # "proven" / "probable" / "speculative"
    category: str                   # "methodology" / "technology"
    # Technology items only — downside risk
    risk_description: str = ""
    risk_delta_per_kW: float = 0.0  # $/kW penalty if it fails


# ── Scenario B: Methodology Items ─────────────────────────────────────────

METHODOLOGY_ITEMS: list[StepChangeItem] = [
    StepChangeItem(
        name="Modular skid fabrication (TG + pump)",
        description="Factory-assemble turbine-generator and pump skids; ship as "
                    "tested modules. Eliminates field fit-up labor and reduces "
                    "commissioning scope.",
        delta_per_kW=-50,
        delta_schedule_weeks=-4,
        probability_pct=85,
        proving_cost_USD=200_000,
        basis="Ormat modular practice; confirmed by 3 EPC contractors",
        status="proven",
        category="methodology",
    ),
    StepChangeItem(
        name="Factory acceptance testing (FAT)",
        description="Full-load string test at vendor shop. Catches integration "
                    "issues before shipment, reducing field rework by ~60%.",
        delta_per_kW=-20,
        delta_schedule_weeks=-2,
        probability_pct=90,
        proving_cost_USD=50_000,
        basis="Standard practice for gas turbines; adapting to ORC",
        status="proven",
        category="methodology",
    ),
    StepChangeItem(
        name="Parallel unit construction (units 2-3)",
        description="Stagger identical units on 8-week offsets. Shared crew "
                    "learning curve, bulk material orders, single mobilization.",
        delta_per_kW=-25,
        delta_schedule_weeks=-8,
        probability_pct=75,
        proving_cost_USD=100_000,
        basis="Multi-unit geothermal precedent (Puna, Olkaria)",
        status="probable",
        category="methodology",
    ),
    StepChangeItem(
        name="Frame contract — ACC (8 units w/ Worldwide)",
        description="Lock in ACC supply for 8 identical units. Volume discount "
                    "on coils, fans, and structural steel.",
        delta_per_kW=-40,
        delta_schedule_weeks=0,
        probability_pct=80,
        proving_cost_USD=25_000,
        basis="Worldwide Cooling budgetary quote for 8-unit program",
        status="probable",
        category="methodology",
    ),
    StepChangeItem(
        name="Frame contract — HX (8 units w/ Precision)",
        description="Lock in heat exchanger supply for 8 identical units. "
                    "Standardized tube bundles and shell designs.",
        delta_per_kW=-20,
        delta_schedule_weeks=0,
        probability_pct=80,
        proving_cost_USD=25_000,
        basis="Precision Heat Exchangers indicative pricing",
        status="probable",
        category="methodology",
    ),
    StepChangeItem(
        name="Dedicated logistics & staging",
        description="Pre-position laydown yard near site. Coordinate heavy haul "
                    "windows to avoid weather delays.",
        delta_per_kW=-12,
        delta_schedule_weeks=-2,
        probability_pct=85,
        proving_cost_USD=50_000,
        basis="EPC contractor estimate for remote geothermal sites",
        status="proven",
        category="methodology",
    ),
    StepChangeItem(
        name="Standardized design package (reuse engineering)",
        description="One-time detailed engineering for standard plant; reuse "
                    "across all units with site-specific civils only.",
        delta_per_kW=-28,
        delta_schedule_weeks=0,
        probability_pct=95,
        proving_cost_USD=0,
        basis="Engineering cost allocation across 8+ identical units",
        status="proven",
        category="methodology",
    ),
]


# ── Scenario C: Technology Items ──────────────────────────────────────────

TECHNOLOGY_ITEMS: list[StepChangeItem] = [
    StepChangeItem(
        name="Transcritical iC4 cycle",
        description="Isobutane above critical point — 5-8% efficiency gain from "
                    "better temperature glide matching. Requires high-pressure HX.",
        delta_per_kW=-100,
        delta_schedule_weeks=0,
        probability_pct=40,
        proving_cost_USD=500_000,
        basis="Literature: Walraven et al. 2015; Atlas Copco pilot data",
        status="speculative",
        category="technology",
        risk_description="HX premium for >40 bar design pressure; 6-month "
                         "qualification delay if ASME U2 stamp required",
        risk_delta_per_kW=50,
    ),
    StepChangeItem(
        name="Direct-drive turbine (no gearbox)",
        description="Eliminate gearbox losses (+0.5% efficiency) and maintenance. "
                    "Permanent-magnet generator at turbine speed.",
        delta_per_kW=-30,
        delta_schedule_weeks=-2,
        probability_pct=60,
        proving_cost_USD=200_000,
        basis="GE sPower concept; Turboden R&D roadmap",
        status="probable",
        category="technology",
        risk_description="Single-speed operation limits off-design flexibility; "
                         "generator cost premium partially offsets gearbox savings",
        risk_delta_per_kW=10,
    ),
    StepChangeItem(
        name="PCHE vs STHE (printed circuit heat exchangers)",
        description="Compact diffusion-bonded HX — 5x surface density, 80% "
                    "smaller footprint. Reduces civil and structural costs.",
        delta_per_kW=-45,
        delta_schedule_weeks=-3,
        probability_pct=50,
        proving_cost_USD=300_000,
        basis="Heatric catalogue; geothermal pilot at Reykjanes",
        status="speculative",
        category="technology",
        risk_description="Fouling risk with high-silica brine; limited field "
                         "repair options; single-source supply chain",
        risk_delta_per_kW=25,
    ),
    StepChangeItem(
        name="Underground heat rejection (partial)",
        description="Reject heat to shallow ground loops or mine workings — "
                    "reduce or eliminate ACC entirely.",
        delta_per_kW=-150,
        delta_schedule_weeks=4,
        probability_pct=20,
        proving_cost_USD=1_000_000,
        basis="Conceptual; analogous to district heating ground source",
        status="speculative",
        category="technology",
        risk_description="Thermal interference after 5-10 years; permitting "
                         "uncertainty; site-specific geology required",
        risk_delta_per_kW=80,
    ),
    StepChangeItem(
        name="Higher brine flow per well (+20%)",
        description="Increase production rate per well — fixed surface costs "
                    "spread over more thermal input = lower $/kW.",
        delta_per_kW=-200,
        delta_schedule_weeks=0,
        probability_pct=35,
        proving_cost_USD=2_000_000,
        basis="Reservoir modeling; analogous to Coso and Salton Sea uprates",
        status="speculative",
        category="technology",
        risk_description="Wellbore integrity and scaling risk at higher rates; "
                         "may require workover or new slim wells",
        risk_delta_per_kW=100,
    ),
]


# ── Scenario Builder ──────────────────────────────────────────────────────

def build_step_change_scenarios(store: ResultStore) -> dict:
    """Build the three scenarios from optimizer data + predefined items.

    Returns dict with keys: baseline_per_kW, scenario_a, scenario_b, scenario_c.
    """
    best = store.get_best_adjusted()
    stats = store.stats()

    # Scenario A — current optimizer best
    if best is not None:
        baseline = best.total_adjusted_per_kW
        scenario_a = {
            "baseline_per_kW": baseline,
            "net_power_MW": best.net_power_MW,
            "efficiency": best.cycle_efficiency,
            "config": best.config,
            "strategy": best.procurement_strategy,
            "equipment_per_kW": best.equipment_per_kW,
            "capex_per_kW": best.capex_per_kW,
            "complexity_per_kW": best.complexity_per_kW,
            "total_runs": stats["total_runs"],
            "converged": stats["converged"],
            "target_hits": stats["target_hits"],
            "pareto_count": stats["pareto_count"],
        }
    else:
        baseline = TARGET_CAPEX_PER_KW  # fallback
        scenario_a = None

    # Scenario B — methodology
    meth_total_delta = sum(it.delta_per_kW for it in METHODOLOGY_ITEMS)
    meth_ev = sum(it.delta_per_kW * it.probability_pct / 100 for it in METHODOLOGY_ITEMS)
    meth_proving = sum(it.proving_cost_USD for it in METHODOLOGY_ITEMS)
    meth_schedule = sum(it.delta_schedule_weeks for it in METHODOLOGY_ITEMS)

    scenario_b = {
        "items": METHODOLOGY_ITEMS,
        "total_delta": meth_total_delta,
        "expected_delta": meth_ev,
        "projected_per_kW": baseline + meth_ev,
        "total_proving_cost": meth_proving,
        "total_schedule_delta": meth_schedule,
    }

    # Scenario C — technology
    tech_total_delta = sum(it.delta_per_kW for it in TECHNOLOGY_ITEMS)
    tech_ev = sum(it.delta_per_kW * it.probability_pct / 100 for it in TECHNOLOGY_ITEMS)
    tech_proving = sum(it.proving_cost_USD for it in TECHNOLOGY_ITEMS)

    scenario_c = {
        "items": TECHNOLOGY_ITEMS,
        "total_delta": tech_total_delta,
        "expected_delta": tech_ev,
        "projected_per_kW": baseline + tech_ev,
        "total_proving_cost": tech_proving,
    }

    return {
        "baseline_per_kW": baseline,
        "scenario_a": scenario_a,
        "scenario_b": scenario_b,
        "scenario_c": scenario_c,
    }


# ── Renderer ──────────────────────────────────────────────────────────────

def render_step_change_subtab(store: ResultStore):
    """Render the Step Change Analysis sub-tab inside the optimizer."""

    scenarios = build_step_change_scenarios(store)
    baseline = scenarios["baseline_per_kW"]
    sa = scenarios["scenario_a"]
    sb = scenarios["scenario_b"]
    sc = scenarios["scenario_c"]
    target = TARGET_CAPEX_PER_KW

    # ── 1. Summary Row ────────────────────────────────────────────────
    st.subheader("Scenario Summary")

    c1, c2, c3 = st.columns(3)

    with c1:
        st.markdown("**A — Incremental (Optimizer)**")
        if sa:
            _color_metric(
                f"${sa['capex_per_kW']:,.0f}/kW installed",
                sa["capex_per_kW"], target,
            )
            st.caption(
                f"{sa['net_power_MW']:.1f} MW net | "
                f"{sa['efficiency']*100:.1f}% eff | "
                f"{sa['strategy']}"
            )
        else:
            st.info("Run optimizer first to set baseline.")

    with c2:
        st.markdown("**B — Methodology**")
        proj_b = sb["projected_per_kW"]
        _color_metric(
            f"${proj_b:,.0f}/kW projected",
            proj_b, target,
        )
        st.caption(
            f"EV delta: ${sb['expected_delta']:+,.0f}/kW | "
            f"Proving: ${sb['total_proving_cost']:,.0f}"
        )

    with c3:
        st.markdown("**C — Technology**")
        proj_c = sc["projected_per_kW"]
        _color_metric(
            f"${proj_c:,.0f}/kW projected",
            proj_c, target,
        )
        st.caption(
            f"EV delta: ${sc['expected_delta']:+,.0f}/kW | "
            f"Proving: ${sc['total_proving_cost']:,.0f}"
        )

    st.divider()

    # ── 2. Step Change Map ────────────────────────────────────────────
    st.subheader("Step Change Map")
    _render_step_change_map(baseline, target)

    st.divider()

    # ── 3. Detail Expanders ───────────────────────────────────────────
    _render_scenario_a_detail(sa, baseline, target)
    _render_scenario_b_detail(sb, baseline)
    _render_scenario_c_detail(sc, baseline)

    st.divider()

    # ── 4. Investment Prioritization ──────────────────────────────────
    _render_investment_priority()


def _color_metric(label: str, value: float, target: float):
    """Display a metric with color coding relative to target."""
    if value <= target:
        st.success(label)
    elif value <= target * 1.10:
        st.warning(label)
    else:
        st.error(label)


def _render_step_change_map(baseline: float, target: float):
    """Scatter chart: X = probability, Y = projected $/kW, color = scenario."""
    rows = []
    for item in METHODOLOGY_ITEMS:
        ev_delta = item.delta_per_kW * item.probability_pct / 100
        rows.append({
            "Name": item.name,
            "Probability (%)": item.probability_pct,
            "Projected $/kW": baseline + ev_delta,
            "Proving Cost ($K)": item.proving_cost_USD / 1000,
            "Scenario": "B — Methodology",
        })
    for item in TECHNOLOGY_ITEMS:
        ev_delta = item.delta_per_kW * item.probability_pct / 100
        rows.append({
            "Name": item.name,
            "Probability (%)": item.probability_pct,
            "Projected $/kW": baseline + ev_delta,
            "Proving Cost ($K)": item.proving_cost_USD / 1000,
            "Scenario": "C — Technology",
        })

    df = pd.DataFrame(rows)

    if df.empty:
        st.info("No items to display.")
        return

    # Plotly scatter for bubble chart with target line
    try:
        import plotly.express as px
        import plotly.graph_objects as go

        fig = px.scatter(
            df,
            x="Probability (%)",
            y="Projected $/kW",
            size="Proving Cost ($K)",
            color="Scenario",
            hover_name="Name",
            color_discrete_map={
                "B — Methodology": "#2196F3",
                "C — Technology": "#FF9800",
            },
            size_max=40,
        )
        # Target line
        fig.add_hline(
            y=target, line_dash="dash", line_color="green",
            annotation_text=f"${target:,.0f}/kW target",
            annotation_position="top left",
        )
        # Baseline line
        fig.add_hline(
            y=baseline, line_dash="dot", line_color="red",
            annotation_text=f"${baseline:,.0f}/kW baseline",
            annotation_position="top right",
        )
        fig.update_layout(
            xaxis_title="Probability of Achievement (%)",
            yaxis_title="Projected Installed $/kW",
            height=450,
            xaxis=dict(range=[0, 100]),
        )
        st.plotly_chart(fig, use_container_width=True)
    except ImportError:
        # Fallback: Streamlit native scatter
        st.scatter_chart(
            df,
            x="Probability (%)",
            y="Projected $/kW",
            color="Scenario",
            size="Proving Cost ($K)",
        )

    st.caption(
        "Execute all incremental items (certain value) WHILE running "
        "1-2 step changes (uncertain but transformative). "
        "Bubble size = proving cost."
    )


def _render_scenario_a_detail(sa: Optional[dict], baseline: float, target: float):
    """Scenario A expander — optimizer results summary."""
    with st.expander("Scenario A — Incremental (Optimizer Results)", expanded=False):
        if sa is None:
            st.info(
                "No optimizer results available. Run the optimizer on the "
                "Controls tab to establish Scenario A baseline."
            )
            return

        a1, a2, a3, a4 = st.columns(4)
        a1.metric("Best Adjusted $/kW", f"${baseline:,.0f}")
        a2.metric("Equipment $/kW", f"${sa['equipment_per_kW']:,.0f}")
        a3.metric("Net Power", f"{sa['net_power_MW']:.1f} MW")
        a4.metric("Cycle Efficiency", f"{sa['efficiency']*100:.1f}%")

        cfg = sa["config"]
        st.markdown(
            f"**Best config:** {cfg.get('working_fluid', '?')} / "
            f"{cfg.get('topology', '?')} / {cfg.get('heat_rejection', '?')} | "
            f"Strategy: {sa['strategy']}"
        )
        st.markdown(
            f"**Search stats:** {sa['total_runs']} runs, "
            f"{sa['converged']} converged, "
            f"{sa['target_hits']} hit target, "
            f"{sa['pareto_count']} Pareto-optimal"
        )

        headroom = baseline - target
        if headroom > 0:
            st.warning(
                f"Current best is ${headroom:,.0f}/kW above the "
                f"${target:,.0f}/kW target. Methodology and technology "
                f"scenarios below show paths to close this gap."
            )
        else:
            st.success(
                f"Current best is ${-headroom:,.0f}/kW below the "
                f"${target:,.0f}/kW target. Methodology and technology "
                f"scenarios can drive costs even lower."
            )


def _render_scenario_b_detail(sb: dict, baseline: float):
    """Scenario B expander — methodology items table."""
    with st.expander("Scenario B — Methodology Items", expanded=False):
        rows = []
        for item in sb["items"]:
            rows.append({
                "Item": item.name,
                "$/kW Delta": f"${item.delta_per_kW:+,.0f}",
                "Schedule (wk)": f"{item.delta_schedule_weeks:+.0f}" if item.delta_schedule_weeks else "—",
                "Probability": f"{item.probability_pct:.0f}%",
                "Proving Cost": f"${item.proving_cost_USD:,.0f}",
                "Status": item.status,
            })

        # Expected value row
        ev = sb["expected_delta"]
        rows.append({
            "Item": "EXPECTED VALUE (prob-weighted)",
            "$/kW Delta": f"${ev:+,.0f}",
            "Schedule (wk)": f"{sb['total_schedule_delta']:+.0f}",
            "Probability": "—",
            "Proving Cost": f"${sb['total_proving_cost']:,.0f}",
            "Status": "—",
        })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        st.markdown(
            f"**If all items achieved:** ${sb['total_delta']:+,.0f}/kW "
            f"= **${baseline + sb['total_delta']:,.0f}/kW** projected"
        )
        st.markdown(
            f"**Probability-weighted (EV):** ${ev:+,.0f}/kW "
            f"= **${baseline + ev:,.0f}/kW** projected"
        )


def _render_scenario_c_detail(sc: dict, baseline: float):
    """Scenario C expander — technology items with prize and risk."""
    with st.expander("Scenario C — Technology Items", expanded=False):
        rows = []
        for item in sc["items"]:
            rows.append({
                "Technology": item.name,
                "Prize ($/kW)": f"${item.delta_per_kW:+,.0f}",
                "Risk ($/kW)": f"+${item.risk_delta_per_kW:,.0f}" if item.risk_delta_per_kW else "—",
                "Probability": f"{item.probability_pct:.0f}%",
                "Proving Cost": f"${item.proving_cost_USD:,.0f}",
                "Status": item.status,
            })

        ev = sc["expected_delta"]
        rows.append({
            "Technology": "EXPECTED VALUE (prob-weighted)",
            "Prize ($/kW)": f"${ev:+,.0f}",
            "Risk ($/kW)": "—",
            "Probability": "—",
            "Proving Cost": f"${sc['total_proving_cost']:,.0f}",
            "Status": "—",
        })

        df = pd.DataFrame(rows)
        st.dataframe(df, use_container_width=True, hide_index=True)

        # Detailed risk/reward for each item
        for item in sc["items"]:
            st.markdown(f"**{item.name}**")
            r1, r2 = st.columns(2)
            with r1:
                st.markdown(f"Prize: {item.description}")
            with r2:
                if item.risk_description:
                    st.markdown(f"Risk: {item.risk_description}")

        st.markdown(
            f"**If all items achieved:** ${sc['total_delta']:+,.0f}/kW "
            f"= **${baseline + sc['total_delta']:,.0f}/kW** projected"
        )
        st.markdown(
            f"**Probability-weighted (EV):** ${ev:+,.0f}/kW "
            f"= **${baseline + ev:,.0f}/kW** projected"
        )


def _render_investment_priority():
    """Rank all B + C items by risk-adjusted return."""
    st.subheader("Investment Prioritization")
    st.caption("Ranked by risk-adjusted return: |$/kW savings| x probability / proving cost")

    all_items = METHODOLOGY_ITEMS + TECHNOLOGY_ITEMS
    ranked = []
    for item in all_items:
        if item.proving_cost_USD > 0:
            roi = abs(item.delta_per_kW) * (item.probability_pct / 100) / (item.proving_cost_USD / 1000)
        else:
            # Free items get infinite ROI — show at top
            roi = float("inf")

        ev_savings = abs(item.delta_per_kW) * item.probability_pct / 100
        ranked.append({
            "Rank": 0,
            "Item": item.name,
            "Category": item.category.title(),
            "EV Savings ($/kW)": f"${ev_savings:,.1f}",
            "Proving Cost": f"${item.proving_cost_USD:,.0f}" if item.proving_cost_USD > 0 else "Free",
            "ROI ($/kW per $K invested)": f"{roi:.1f}" if roi != float("inf") else "Free",
            "Probability": f"{item.probability_pct:.0f}%",
            "Status": item.status,
            "_sort": roi,
        })

    ranked.sort(key=lambda r: r["_sort"], reverse=True)
    for i, r in enumerate(ranked):
        r["Rank"] = i + 1

    df = pd.DataFrame(ranked)
    df = df.drop(columns=["_sort"])
    st.dataframe(df, use_container_width=True, hide_index=True)
