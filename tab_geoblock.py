"""
GeoBlock Component Catalog — Streamlit UI Tab
================================================

5 sub-tabs:
  1. Configuration — define condition matrix and run
  2. Results Matrix — heatmaps and tables
  3. Standardization — per-component spec analysis
  4. Component Catalog & Assembly Grammar
  5. Commercial Summary
"""

import os
import time

import streamlit as st
import pandas as pd

from geoblock_engine import (
    generate_condition_matrix,
    run_condition_point,
    init_csv,
    append_csv,
    load_results,
    analyze_standardization,
    build_assembly_grammar,
    generate_commercial_summary,
    RESULTS_CSV,
    DEFAULT_BRINE_RANGE,
    DEFAULT_AMBIENT_RANGE,
    DEFAULT_MW_RANGE,
    DEFAULT_MAX_OVERSIZE_PCT,
    DEFAULT_MAX_SPECS,
    COMPONENT_DEFS,
    _range_inclusive,
)


def _init_state():
    """Initialize session state for GeoBlock tab."""
    defaults = {
        "gb_running": False,
        "gb_progress": 0,
        "gb_total": 0,
        "gb_current_label": "",
        "gb_run_complete": False,
        "gb_config": {},
        "gb_base_net_mw": 0.0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_geoblock_tab(design_basis: dict):
    """Main entry point — renders the GeoBlock Component Catalog tab."""
    _init_state()

    st.markdown("## GeoBlock Component Catalog")
    st.markdown(
        "Run the thermodynamic optimizer across a matrix of operating conditions, "
        "then analyze results for **component standardization** and **frame contract sizing**."
    )

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📐 Configuration",
        "📊 Results Matrix",
        "🔬 Standardization",
        "📦 Component Catalog",
        "💰 Commercial Summary",
    ])

    with tab1:
        _render_config_tab(design_basis)

    # Load results for display tabs
    df = load_results()

    with tab2:
        _render_results_tab(df)

    with tab3:
        _render_standardization_tab(df)

    with tab4:
        _render_catalog_tab(df)

    with tab5:
        _render_commercial_tab(df, design_basis)


# ── Sub-tab 1: Configuration ──────────────────────────────────────────────────

def _render_config_tab(design_basis: dict):
    st.markdown("### Condition Matrix Definition")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**Brine Inlet Temperature**")
        brine_min = st.number_input("Min (°F)", 300, 600, DEFAULT_BRINE_RANGE[0], 5, key="gb_brine_min")
        brine_max = st.number_input("Max (°F)", 300, 600, DEFAULT_BRINE_RANGE[1], 5, key="gb_brine_max")
        brine_step = st.number_input("Step (°F)", 5, 50, DEFAULT_BRINE_RANGE[2], 5, key="gb_brine_step")

    with col2:
        st.markdown("**Ambient Temperature**")
        amb_min = st.number_input("Min (°F)", -40, 130, DEFAULT_AMBIENT_RANGE[0], 5, key="gb_amb_min")
        amb_max = st.number_input("Max (°F)", -40, 130, DEFAULT_AMBIENT_RANGE[1], 5, key="gb_amb_max")
        amb_step = st.number_input("Step (°F)", 5, 40, DEFAULT_AMBIENT_RANGE[2], 5, key="gb_amb_step")

    with col3:
        st.markdown("**Net Output Target**")
        mw_min = st.number_input("Min (MW)", 10, 200, DEFAULT_MW_RANGE[0], 5, key="gb_mw_min")
        mw_max = st.number_input("Max (MW)", 10, 200, DEFAULT_MW_RANGE[1], 5, key="gb_mw_max")
        mw_step = st.number_input("Step (MW)", 1, 25, DEFAULT_MW_RANGE[2], 1, key="gb_mw_step")

    # Compute matrix size
    brine_vals = _range_inclusive(brine_min, brine_max, brine_step)
    amb_vals = _range_inclusive(amb_min, amb_max, amb_step)
    mw_vals = _range_inclusive(mw_min, mw_max, mw_step)
    matrix_size = len(brine_vals) * len(amb_vals) * len(mw_vals)

    st.info(
        f"**{len(brine_vals)}** brine temps × **{len(amb_vals)}** ambients × "
        f"**{len(mw_vals)}** MW targets = **{matrix_size}** condition combinations"
    )

    st.markdown("---")
    st.markdown("### Fixed Parameters")
    fcol1, fcol2, fcol3 = st.columns(3)

    with fcol1:
        topology = st.selectbox(
            "Cycle topology",
            ["recuperated", "basic", "dual_pressure"],
            index=0,
            key="gb_topology",
        )

    with fcol2:
        st.text_input("Working fluid", "isopentane (iC5)", disabled=True, key="gb_fluid_display")
        st.text_input("Procurement strategy", "Direct + Self-Perform", disabled=True, key="gb_strat_display")

    with fcol3:
        n_trains = st.selectbox("Turbine trains", [1, 2, 3], index=1, key="gb_n_trains")
        max_oversize = st.slider(
            "Max oversize tolerance (%)",
            10, 50, int(DEFAULT_MAX_OVERSIZE_PCT), 5,
            key="gb_max_oversize",
        )

    st.markdown("---")

    # Run control
    if st.session_state.gb_running:
        progress = st.session_state.gb_progress
        total = st.session_state.gb_total
        pct = progress / total if total > 0 else 0
        st.progress(pct, text=f"Running condition {progress}/{total}: {st.session_state.gb_current_label}")

        if progress < total:
            _run_next_condition(design_basis)
        else:
            st.session_state.gb_running = False
            st.session_state.gb_run_complete = True
            st.rerun()
    else:
        bcol1, bcol2 = st.columns([1, 3])
        with bcol1:
            if st.button("▶ Run Matrix", type="primary", key="gb_start"):
                _start_matrix_run(
                    design_basis,
                    (brine_min, brine_max, brine_step),
                    (amb_min, amb_max, amb_step),
                    (mw_min, mw_max, mw_step),
                    topology, n_trains,
                )
        with bcol2:
            if st.session_state.gb_run_complete:
                st.success("Matrix run complete! Check Results and Standardization tabs.")

    # Show existing results count
    if os.path.exists(RESULTS_CSV):
        df = load_results()
        n_conv = len(df[df["converged"] == True]) if len(df) > 0 else 0
        st.caption(f"Existing results: {len(df)} runs ({n_conv} converged)")


def _start_matrix_run(design_basis, brine_range, amb_range, mw_range, topology, n_trains):
    """Initialize matrix run."""
    matrix = generate_condition_matrix(brine_range, amb_range, mw_range)

    # Estimate base net MW from a reference run
    base_mw = _estimate_base_mw(design_basis, topology, n_trains)

    # Lock config
    st.session_state.gb_config = {
        "brine_range": brine_range,
        "ambient_range": amb_range,
        "mw_range": mw_range,
        "topology": topology,
        "n_trains": n_trains,
        "procurement_strategy": "direct_self_perform",
    }
    st.session_state.gb_base_net_mw = base_mw
    st.session_state.gb_matrix = [
        {"brine_inlet_F": p.brine_inlet_F, "ambient_F": p.ambient_F, "target_MW": p.target_MW}
        for p in matrix
    ]
    st.session_state.gb_progress = 0
    st.session_state.gb_total = len(matrix)
    st.session_state.gb_running = True
    st.session_state.gb_run_complete = False

    # Initialize CSV
    init_csv()
    st.rerun()


def _estimate_base_mw(design_basis, topology, n_trains):
    """Run a single reference condition to get baseline MW for scaling."""
    from analysis_bridge import run_orc_analysis

    config = "D" if topology == "dual_pressure" else "A"
    tool_input = {
        "config": config,
        "working_fluid": "isopentane",
        "procurement_strategy": "direct_self_perform",
        "turbine_trains": n_trains,
    }
    if topology == "recuperated":
        tool_input["recuperator_approach_delta_F"] = design_basis.get("dt_pinch_recuperator", 15)
    elif topology == "basic":
        tool_input["recuperator_approach_delta_F"] = 999.0
    else:
        tool_input["recuperator_approach_delta_F"] = design_basis.get("dt_pinch_recuperator", 15)

    try:
        output = run_orc_analysis(tool_input, design_basis)
        return output.get("net_power_MW", 50.0)
    except Exception:
        return 50.0  # fallback


def _run_next_condition(design_basis):
    """Run the next condition in the queue (one per rerun cycle)."""
    idx = st.session_state.gb_progress
    matrix = st.session_state.gb_matrix
    cfg = st.session_state.gb_config

    if idx >= len(matrix):
        return

    point_dict = matrix[idx]
    from geoblock_engine import ConditionPoint
    point = ConditionPoint(**point_dict)

    st.session_state.gb_current_label = (
        f"Brine {point.brine_inlet_F:.0f}°F, Amb {point.ambient_F:.0f}°F, {point.target_MW:.0f} MW"
    )

    row = run_condition_point(
        point=point,
        base_design_basis=design_basis,
        topology=cfg["topology"],
        procurement_strategy=cfg["procurement_strategy"],
        base_net_mw=st.session_state.gb_base_net_mw,
        n_trains=cfg["n_trains"],
    )
    append_csv(row)

    st.session_state.gb_progress = idx + 1
    st.rerun()


# ── Sub-tab 2: Results Matrix ─────────────────────────────────────────────────

def _render_results_tab(df: pd.DataFrame):
    if len(df) == 0:
        st.info("No results yet. Run the condition matrix from the Configuration tab.")
        return

    converged = df[df["converged"] == True]
    st.markdown(f"### Results: {len(converged)} converged / {len(df)} total")

    # ── Heatmaps ──────────────────────────────────────────────────────
    mw_targets = sorted(converged["target_MW"].unique())

    if len(mw_targets) > 0:
        st.markdown("#### $/kW Heatmap by Condition")
        mw_select = st.selectbox("Target MW", mw_targets, key="gb_hm_mw")

        subset = converged[converged["target_MW"] == mw_select]
        if len(subset) > 1:
            try:
                pivot = subset.pivot_table(
                    index="brine_inlet_F", columns="ambient_F",
                    values="capex_per_kW", aggfunc="mean",
                )
                pivot = pivot.sort_index(ascending=False)

                # Color-code with st.dataframe styling
                st.dataframe(
                    pivot.style.background_gradient(cmap="RdYlGn_r", axis=None)
                    .format("{:.0f}"),
                    use_container_width=True,
                )
            except Exception as e:
                st.warning(f"Could not generate heatmap: {e}")

            # Net MW achieved heatmap
            st.markdown("#### Net MW Achieved")
            try:
                pivot_mw = subset.pivot_table(
                    index="brine_inlet_F", columns="ambient_F",
                    values="net_MW", aggfunc="mean",
                )
                pivot_mw = pivot_mw.sort_index(ascending=False)
                st.dataframe(
                    pivot_mw.style.background_gradient(cmap="YlGn", axis=None)
                    .format("{:.1f}"),
                    use_container_width=True,
                )
            except Exception:
                pass

    # ── Non-converged conditions ──────────────────────────────────────
    non_conv = df[df["converged"] == False]
    if len(non_conv) > 0:
        st.warning(f"{len(non_conv)} conditions did not converge (shown as gaps in heatmaps)")
        with st.expander("Non-converged conditions"):
            st.dataframe(
                non_conv[["brine_inlet_F", "ambient_F", "target_MW"]],
                use_container_width=True,
            )

    # ── Full results table ────────────────────────────────────────────
    st.markdown("#### Full Results Table")
    display_cols = [
        "brine_inlet_F", "ambient_F", "target_MW", "net_MW", "gross_MW",
        "cycle_efficiency", "capex_per_kW", "equipment_per_kW", "lcoe",
        "turbine_mw_each", "vaporizer_n_shells", "preheater_n_shells",
        "acc_n_bays", "pump_hp_each",
    ]
    avail = [c for c in display_cols if c in converged.columns]
    st.dataframe(converged[avail], use_container_width=True)

    # CSV download
    csv_bytes = converged.to_csv(index=False).encode()
    st.download_button("📥 Download CSV", csv_bytes, "geoblock_results.csv", "text/csv")


# ── Sub-tab 3: Standardization ────────────────────────────────────────────────

def _render_standardization_tab(df: pd.DataFrame):
    if len(df) == 0 or len(df[df["converged"] == True]) == 0:
        st.info("No converged results yet. Run the condition matrix first.")
        return

    max_oversize = st.session_state.get("gb_max_oversize", DEFAULT_MAX_OVERSIZE_PCT)

    std_results = analyze_standardization(df, max_oversize_pct=max_oversize)
    if not std_results:
        st.warning("No standardization results.")
        return

    # Store for other tabs
    st.session_state["gb_std_results"] = std_results

    st.markdown("### Component Standardization Analysis")
    st.caption(f"Max oversize tolerance: {max_oversize:.0f}%")

    for sr in std_results:
        if sr.recommendation == "skip":
            continue

        with st.expander(f"**{sr.component_type.upper()}** — {sr.n_specs} spec(s) recommended", expanded=True):
            # Summary metrics
            mcol1, mcol2, mcol3, mcol4 = st.columns(4)
            mcol1.metric("Specs needed", sr.n_specs)
            mcol2.metric("Avg waste", f"{sr.total_waste_pct:.1f}%")
            mcol3.metric("1-spec coverage", f"{sr.one_spec_coverage_full:.0f}%")
            mcol4.metric("1-spec ≤30% oversize", f"{sr.one_spec_coverage_5pct:.0f}%")

            # Reasoning
            st.markdown(f"**Analysis:** {sr.reasoning}")

            # Per-spec details
            for spec in sr.specs:
                st.markdown(f"---")
                scol1, scol2, scol3 = st.columns(3)
                scol1.markdown(f"**{spec.spec_id}**")
                scol2.markdown(
                    f"Design: **{spec.design_rating:.1f} {spec.sizing_unit}** "
                    f"(range {spec.min_rating:.1f}–{spec.max_rating:.1f})"
                )
                scol3.markdown(f"Coverage: **{spec.coverage_pct:.0f}%** ({len(spec.condition_ids)} conditions)")

                if spec.n_shells > 0:
                    st.markdown(
                        f"  HX shells: **{spec.n_shells}** × ~{spec.shell_dia_in:.0f}\" OD"
                    )

                if spec.vendors:
                    st.markdown(f"  Vendors: {', '.join(spec.vendors)} | Lead: {spec.lead_weeks} weeks")

            # Distribution chart
            converged = df[df["converged"] == True]
            if sr.sizing_param in converged.columns:
                vals = converged[converged[sr.sizing_param] > 0][sr.sizing_param]
                if len(vals) > 3:
                    st.markdown("**Sizing Distribution**")
                    chart_df = pd.DataFrame({"value": vals})
                    st.bar_chart(chart_df["value"].value_counts().sort_index())


# ── Sub-tab 4: Component Catalog & Assembly Grammar ───────────────────────────

def _render_catalog_tab(df: pd.DataFrame):
    if len(df) == 0 or len(df[df["converged"] == True]) == 0:
        st.info("No results yet.")
        return

    std_results = st.session_state.get("gb_std_results")
    if not std_results:
        # Re-run analysis
        max_oversize = st.session_state.get("gb_max_oversize", DEFAULT_MAX_OVERSIZE_PCT)
        std_results = analyze_standardization(df, max_oversize_pct=max_oversize)
        st.session_state["gb_std_results"] = std_results

    if not std_results:
        st.warning("Run standardization analysis first.")
        return

    # ── Component Catalog Cards ───────────────────────────────────────
    st.markdown("### Component Catalog")

    for sr in std_results:
        if sr.recommendation == "skip":
            continue
        for spec in sr.specs:
            st.markdown(f"""
<div style="border:1px solid #444; border-radius:8px; padding:16px; margin:8px 0; background:#f8f9fa;">
<h4 style="margin:0 0 8px 0; color:#1A1A2E;">{spec.spec_id}: {spec.component_type.replace('_',' ').title()}</h4>
<table style="width:100%; color:#1A1A2E;">
<tr><td><b>Design rating:</b></td><td>{spec.design_rating:.1f} {spec.sizing_unit}</td>
    <td><b>Operating range:</b></td><td>{spec.min_rating:.1f} – {spec.max_rating:.1f} {spec.sizing_unit}</td></tr>
<tr><td><b>Coverage:</b></td><td>{spec.coverage_pct:.0f}% of conditions ({len(spec.condition_ids)} points)</td>
    <td><b>Design margin:</b></td><td>{spec.margin_pct:.0f}%</td></tr>
{"<tr><td><b>Shell arrangement:</b></td><td>" + str(spec.n_shells) + " shells × " + str(int(spec.shell_dia_in)) + '" OD</td><td></td><td></td></tr>' if spec.n_shells > 0 else ""}
<tr><td><b>Preferred vendors:</b></td><td colspan="3">{', '.join(spec.vendors) if spec.vendors else 'TBD'}</td></tr>
<tr><td><b>Lead time:</b></td><td>{spec.lead_weeks} weeks</td>
    <td><b>Confidence:</b></td><td>{'HIGH' if spec.lead_weeks > 0 else 'TBD'}</td></tr>
</table>
</div>
""", unsafe_allow_html=True)

    # ── Assembly Grammar ──────────────────────────────────────────────
    st.markdown("### Assembly Grammar — Configuration Lookup Table")

    rules = build_assembly_grammar(df, std_results)
    if not rules:
        st.info("No assembly rules generated.")
        return

    # Build display table
    rows = []
    for rule in rules:
        row = {
            "Brine Band": rule.brine_band,
            "Ambient Band": rule.ambient_band,
            "MW Band": rule.mw_band,
            "Est. $/kW": f"${rule.estimated_capex_per_kw:,.0f}",
            "Conditions": rule.condition_count,
        }
        # Add component spec columns
        for sr in std_results:
            if sr.recommendation != "skip":
                label = sr.component_type.replace("_", " ").title()
                row[label] = rule.component_specs.get(sr.component_type, "—")
        rows.append(row)

    grammar_df = pd.DataFrame(rows)
    st.dataframe(grammar_df, use_container_width=True)

    # Download
    csv_bytes = grammar_df.to_csv(index=False).encode()
    st.download_button("📥 Download Assembly Grammar", csv_bytes, "assembly_grammar.csv", "text/csv")


# ── Sub-tab 5: Commercial Summary ────────────────────────────────────────────

def _render_commercial_tab(df: pd.DataFrame, design_basis: dict):
    if len(df) == 0 or len(df[df["converged"] == True]) == 0:
        st.info("No results yet.")
        return

    std_results = st.session_state.get("gb_std_results")
    if not std_results:
        max_oversize = st.session_state.get("gb_max_oversize", DEFAULT_MAX_OVERSIZE_PCT)
        std_results = analyze_standardization(df, max_oversize_pct=max_oversize)

    st.markdown("### Frame Contract & Commercial Summary")

    st.markdown("""
<div style="border:2px solid #ffa726; border-radius:8px; padding:12px; margin:8px 0; background:#fff8e1;">
<b style="color:#e65100;">FERVO ENERGY — GEOBLOCK COMPONENT REQUIREMENTS</b><br>
<span style="color:#555;">Enhanced Geothermal ORC Surface Facilities Program</span>
</div>
""", unsafe_allow_html=True)

    pcol1, pcol2, pcol3 = st.columns(3)
    with pcol1:
        program_mw = st.number_input("Total program capacity (MW)", 100, 5000, 400, 50, key="gb_program_mw")
    with pcol2:
        delivery_years = st.number_input("Delivery period (years)", 1, 20, 5, 1, key="gb_delivery_years")
    with pcol3:
        unit_size_mw = st.number_input("Typical unit size (MW)", 20, 100, 50, 5, key="gb_unit_size")

    n_units = max(1, round(program_mw / unit_size_mw))
    st.info(f"**{n_units}** ORC units × **{unit_size_mw} MW** each = **{n_units * unit_size_mw} MW** program")

    summaries = generate_commercial_summary(std_results, program_mw, delivery_years, unit_size_mw)
    if summaries:
        st.markdown("#### Component Procurement Schedule")
        summary_df = pd.DataFrame(summaries)
        display_rename = {
            "component": "Component",
            "spec_id": "Spec ID",
            "design_rating": "Rating",
            "qty_per_unit": "Qty/Unit",
            "total_program_qty": "Program Total",
            "annual_qty": "Annual Qty",
            "vendors": "Vendors",
            "lead_weeks": "Lead (wk)",
            "coverage_pct": "Coverage %",
        }
        st.dataframe(
            summary_df.rename(columns=display_rename),
            use_container_width=True,
        )

        # Vendor inquiry template
        st.markdown("#### Vendor Inquiry Document")
        st.markdown(
            "Use this as a starting point for RFQ/RFI to frame contract vendors. "
            "Quantities are estimates based on the condition matrix analysis."
        )

        inquiry_lines = [
            "FERVO ENERGY — GEOBLOCK COMPONENT REQUIREMENTS",
            "=" * 50,
            f"Program: Enhanced geothermal ORC surface facilities",
            f"Anticipated program size: {program_mw} MW ({n_units} units)",
            f"Delivery period: {delivery_years} years",
            "",
        ]
        for s in summaries:
            inquiry_lines.extend([
                f"COMPONENT: {s['component'].replace('_', ' ').title()} ({s['spec_id']})",
                f"  Standard specification: {s['design_rating']}",
                f"  Estimated program quantity: {s['total_program_qty']} units",
                f"  Estimated annual quantity: {s['annual_qty']:.0f} units/year",
                f"  Preferred vendors: {s['vendors']}",
                f"  Lead time at volume: {s['lead_weeks']} weeks (confirm)",
                f"  Coverage: {s['coverage_pct']:.0f}% of operating conditions",
                "",
                "  Next step: Request vendor confirmation of:",
                "    [ ] Capability to manufacture to specification",
                "    [ ] Pricing at program volume",
                "    [ ] Lead time at steady-state production",
                "    [ ] Any design modifications required",
                "",
            ])

        inquiry_text = "\n".join(inquiry_lines)
        st.text_area("Vendor Inquiry (copy or download)", inquiry_text, height=400, key="gb_inquiry")
        st.download_button(
            "📥 Download Vendor Inquiry",
            inquiry_text.encode(),
            "geoblock_vendor_inquiry.txt",
            "text/plain",
        )

    # DBD integration
    st.markdown("---")
    if st.button("📋 Write Catalog to Design Basis Document", key="gb_dbd_update"):
        _update_dbd_with_catalog(std_results, program_mw, delivery_years, n_units)
        st.success("Design Basis Document updated with GeoBlock catalog data.")


def _update_dbd_with_catalog(std_results, program_mw, delivery_years, n_units):
    """Write catalog summary into DBD Section 9."""
    try:
        from design_basis_document import load_dbd, save_dbd

        dbd = load_dbd()
        catalog_data = {
            "program_mw": program_mw,
            "delivery_years": delivery_years,
            "n_units": n_units,
            "components": [],
        }
        for sr in std_results:
            if sr.recommendation == "skip":
                continue
            for spec in sr.specs:
                catalog_data["components"].append({
                    "spec_id": spec.spec_id,
                    "component": spec.component_type,
                    "design_rating": f"{spec.design_rating:.1f} {spec.sizing_unit}",
                    "n_shells": spec.n_shells,
                    "shell_dia_in": spec.shell_dia_in,
                    "coverage_pct": spec.coverage_pct,
                    "vendors": spec.vendors,
                    "lead_weeks": spec.lead_weeks,
                })

        dbd["section_9_geoblock_catalog"] = catalog_data
        save_dbd(dbd)
    except Exception as e:
        st.error(f"Could not update DBD: {e}")
