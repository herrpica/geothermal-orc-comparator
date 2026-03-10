"""
ORC Workbench — Combined Analysis + Dialectic Engine + Technology Selection

Single Streamlit app with three tabs:
  1. ORC Analysis — full thermodynamic / cost comparison tool
  2. Design Dialectic — AI-driven debate on optimal configuration
  3. Technology Selection — AI-powered comparison of all viable technologies

Shared sidebar for overlapping inputs (brine conditions, economics).
Both standalone files (app.py, dialectic.py) continue to work independently.

Launch: streamlit run main.py
"""

import streamlit as st

from app import build_analysis_sidebar, render_analysis_tab
from dialectic import build_dialectic_sidebar, render_dialectic_tab
from tab_technology import render_technology_tab
from tab_optimizer import render_optimizer_tab
from tab_knowledge import render_knowledge_tab
from design_basis_document import render_dbd_tab

# ── Page config ────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="ORC Workbench",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Sidebar styling ───────────────────────────────────────────────────────

st.markdown("""
<style>
/* Light background + dark text for main content area */
.stApp {
    background-color: #FFFFFF;
}
/* Main content text: dark for readability on white background */
[data-testid="stMainBlockContainer"] {
    color: #1A1A2E !important;
}
[data-testid="stMainBlockContainer"] p,
[data-testid="stMainBlockContainer"] span,
[data-testid="stMainBlockContainer"] label,
[data-testid="stMainBlockContainer"] li,
[data-testid="stMainBlockContainer"] td,
[data-testid="stMainBlockContainer"] th,
[data-testid="stMainBlockContainer"] h1,
[data-testid="stMainBlockContainer"] h2,
[data-testid="stMainBlockContainer"] h3,
[data-testid="stMainBlockContainer"] h4 {
    color: #1A1A2E !important;
}
/* Metric values and deltas */
[data-testid="stMetricValue"],
[data-testid="stMetricLabel"] {
    color: #1A1A2E !important;
}
/* Tab labels: ensure visible on white background */
.stTabs [data-baseweb="tab-list"] button [data-testid="stMarkdownContainer"] p {
    color: #444444 !important;
    font-weight: 500;
}
.stTabs [data-baseweb="tab-list"] [aria-selected="true"] [data-testid="stMarkdownContainer"] p {
    color: #1A1A2E !important;
    font-weight: 700;
}
/* Dark sidebar background for contrast */
[data-testid="stSidebar"] {
    background-color: #1e1e2f;
}
/* Sidebar text: target specific text elements, not buttons/links */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] span,
[data-testid="stSidebar"] label,
[data-testid="stSidebar"] .stMarkdown {
    color: #e0e0e0 !important;
}
/* Header accent */
[data-testid="stSidebar"] h2 {
    color: #ffa726 !important;
}
/* Sidebar headings */
[data-testid="stSidebar"] h4 {
    color: #90caf9 !important;
}
/* Expander headers slightly brighter */
[data-testid="stSidebar"] [data-testid="stExpander"] summary span {
    color: #90caf9 !important;
}
/* Number input fields: dark bg, white text */
[data-testid="stSidebar"] input {
    background-color: #2a2a3d !important;
    color: #ffffff !important;
    border-color: #555 !important;
}
/* Ensure sidebar buttons remain visible */
[data-testid="stSidebar"] button {
    color: inherit;
}
/* Checkbox labels */
[data-testid="stSidebar"] [data-testid="stCheckbox"] label span {
    color: #e0e0e0 !important;
}
</style>
""", unsafe_allow_html=True)

# ── Shared sidebar ─────────────────────────────────────────────────────────

with st.sidebar:
    st.header("Shared Inputs")

    with st.expander("Resource", expanded=True):
        T_geo_in = st.number_input(
            "Brine inlet temperature (°F)",
            min_value=300, max_value=600, value=420, step=5,
            key="shared_T_geo_in",
        )
        m_dot_geo = st.number_input(
            "Brine mass flow rate (lb/s)",
            min_value=200, max_value=2000, value=1100, step=10,
            key="shared_m_dot_geo",
        )
        T_geo_out_min = st.number_input(
            "Min brine outlet temperature (°F)",
            min_value=160, max_value=250, value=180, step=5,
            key="shared_T_geo_out_min",
            help="Silica/scaling constraint (160°F floor)",
        )
        if T_geo_out_min < 180:
            st.warning("Below silica saturation limit — scaling risk. "
                       "Requires anti-scaling treatment or brine chemistry verification.")

    with st.expander("Site"):
        T_ambient = st.number_input(
            "Ambient dry bulb temperature (°F)",
            min_value=-40, value=57, step=5,
            key="shared_T_ambient",
        )

    with st.expander("Cycle"):
        eta_turbine_pct = st.number_input(
            "Turbine isentropic efficiency (%)",
            min_value=70, max_value=95, value=90, step=1,
            key="shared_eta_turbine",
        )
        eta_turbine = eta_turbine_pct / 100.0
        eta_pump_pct = st.number_input(
            "Pump isentropic efficiency (%)",
            min_value=60, max_value=90, value=82, step=1,
            key="shared_eta_pump",
        )
        eta_pump = eta_pump_pct / 100.0
        generator_efficiency_pct = st.number_input(
            "Generator + mechanical efficiency (%)",
            min_value=90, max_value=99, value=96, step=1,
            key="shared_generator_efficiency",
        )
        generator_efficiency = generator_efficiency_pct / 100.0

    with st.expander("Pinch Points"):
        dt_pinch_vaporizer = st.number_input(
            "Vaporizer pinch (°F)",
            min_value=4, max_value=20, value=14, step=1,
            key="shared_dt_pinch_vaporizer",
        )
        dt_pinch_preheater = st.number_input(
            "Preheater pinch (°F)",
            min_value=3, max_value=15, value=14, step=1,
            key="shared_dt_pinch_preheater",
        )
        dt_pinch_acc = st.number_input(
            "ACC approach (°F)",
            min_value=10, max_value=45, value=22, step=1,
            key="shared_dt_pinch_acc",
        )

    with st.expander("Cost Parameters"):
        uc_turbine_per_kw = st.number_input(
            "Turbine cost ($/kW gross)",
            min_value=50, max_value=500, value=150, step=10,
            key="shared_uc_turbine_per_kw",
        )
        uc_acc_per_bay = st.number_input(
            "ACC cost ($/bay)",
            min_value=200000, max_value=600000, value=347200, step=10000,
            key="shared_uc_acc_per_bay",
        )
        uc_hx_multiplier = st.number_input(
            "HX cost multiplier",
            min_value=0.7, max_value=2.0, value=1.0, step=0.1, format="%.1f",
            key="shared_uc_hx_multiplier",
        )
        uc_civil_structural_per_kw = st.number_input(
            "Civil & structural ($/kW)",
            min_value=100, max_value=400, value=175, step=25,
            key="shared_uc_civil_structural_per_kw",
        )
        uc_ei_installation_per_kw = st.number_input(
            "E&I installation ($/kW)",
            min_value=50, max_value=300, value=125, step=25,
            key="shared_uc_ei_installation_per_kw",
        )

    with st.expander("Economics"):
        electricity_price = st.number_input(
            "Electricity price ($/MWh)",
            min_value=1, value=105, step=5,
            key="shared_electricity_price",
        )
        discount_rate = st.number_input(
            "Discount rate (%)",
            min_value=0, value=11, step=1,
            key="shared_discount_rate",
        )
        project_life = st.number_input(
            "Plant life (years)",
            min_value=1, value=30, step=1,
            key="shared_project_life",
        )
        capacity_factor = st.number_input(
            "Capacity factor (%)",
            min_value=1, value=95, step=1,
            key="shared_capacity_factor",
        )

    st.markdown("---")

    st.checkbox(
        "Apply dialectic constraints to comparison",
        key="shared_apply_constraints",
        help="Pass locked constraints from the dialectic debate into comparison analysis",
    )

    st.markdown("---")

shared_inputs = {
    "T_geo_in": T_geo_in,
    "T_geo_out_min": T_geo_out_min,
    "m_dot_geo": m_dot_geo,
    "T_ambient": T_ambient,
    "eta_turbine": eta_turbine,
    "eta_pump": eta_pump,
    "generator_efficiency": generator_efficiency,
    "dt_pinch_vaporizer": dt_pinch_vaporizer,
    "dt_pinch_preheater": dt_pinch_preheater,
    "dt_pinch_acc": dt_pinch_acc,
    "uc_turbine_per_kw": uc_turbine_per_kw,
    "uc_acc_per_bay": uc_acc_per_bay,
    "uc_hx_multiplier": uc_hx_multiplier,
    "uc_civil_structural_per_kw": uc_civil_structural_per_kw,
    "uc_ei_installation_per_kw": uc_ei_installation_per_kw,
    "electricity_price": electricity_price,
    "discount_rate": discount_rate,
    "project_life": project_life,
    "capacity_factor": capacity_factor,
}

# ── Tabs ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "🔧 ORC Analysis", "⚗️ Design Dialectic", "🔬 Technology Selection",
    "🔍 Autonomous Optimizer", "📚 Knowledge Base", "📋 Design Basis"])

with tab1:
    analysis_inputs = build_analysis_sidebar(shared_inputs)
    try:
        render_analysis_tab(analysis_inputs)
    except Exception as exc:
        # st.stop() raises StopException — catch it to avoid killing other tabs
        if type(exc).__name__ == "StopException":
            pass
        else:
            st.error(f"ORC Analysis error: {exc}")

# Build dialectic sidebar once (used by tabs 2-6)
design_basis = build_dialectic_sidebar(shared_inputs)

with tab2:
    render_dialectic_tab(design_basis)

with tab3:
    render_technology_tab(design_basis)

with tab4:
    render_optimizer_tab(design_basis)

with tab5:
    render_knowledge_tab(design_basis)

with tab6:
    render_dbd_tab(design_basis)
