"""
ORC Workbench — Combined Analysis + Dialectic Engine

Single Streamlit app with two tabs:
  1. ORC Analysis — full thermodynamic / cost comparison tool
  2. Design Dialectic — AI-driven debate on optimal configuration

Shared sidebar for overlapping inputs (brine conditions, economics).
Both standalone files (app.py, dialectic.py) continue to work independently.

Launch: streamlit run main.py
"""

import streamlit as st

from app import build_analysis_sidebar, render_analysis_tab
from dialectic import build_dialectic_sidebar, render_dialectic_tab
from comparison_tab import render_comparison_tab

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
            min_value=1, value=420, step=5,
            key="shared_T_geo_in",
        )
        m_dot_geo = st.number_input(
            "Brine mass flow rate (lb/s)",
            min_value=1, value=1100, step=10,
            key="shared_m_dot_geo",
        )
        T_geo_out_min = st.number_input(
            "Min brine outlet temperature (°F)",
            min_value=1, value=160, step=5,
            key="shared_T_geo_out_min",
            help="Silica/scaling constraint",
        )

    with st.expander("Site"):
        T_ambient = st.number_input(
            "Ambient dry bulb temperature (°F)",
            min_value=-40, value=57, step=5,
            key="shared_T_ambient",
        )

    with st.expander("Cycle"):
        eta_turbine = st.number_input(
            "Turbine isentropic efficiency",
            min_value=0.01, value=0.91, step=0.01, format="%.2f",
            key="shared_eta_turbine",
        )
        eta_pump = st.number_input(
            "Pump isentropic efficiency",
            min_value=0.01, value=0.84, step=0.01, format="%.2f",
            key="shared_eta_pump",
        )

    with st.expander("Economics"):
        electricity_price = st.number_input(
            "Electricity price ($/MWh)",
            min_value=1, value=35, step=5,
            key="shared_electricity_price",
        )
        discount_rate = st.number_input(
            "Discount rate (%)",
            min_value=0, value=8, step=1,
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
    "electricity_price": electricity_price,
    "discount_rate": discount_rate,
    "project_life": project_life,
    "capacity_factor": capacity_factor,
}

# ── Tabs ───────────────────────────────────────────────────────────────────

tab1, tab2, tab3 = st.tabs(["🔧 ORC Analysis", "⚗️ Design Dialectic", "🔬 Design Comparison"])

with tab1:
    analysis_inputs = build_analysis_sidebar(shared_inputs)
    render_analysis_tab(analysis_inputs)

with tab2:
    design_basis = build_dialectic_sidebar(shared_inputs)
    render_dialectic_tab(design_basis)

with tab3:
    render_comparison_tab(design_basis)
