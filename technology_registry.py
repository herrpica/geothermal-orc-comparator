"""
Technology Registry — Definitions for all 19 geothermal power conversion technologies.

Each technology has metadata for screening, analysis routing, and AI prompt context.
Categories:
  - commercial: Multiple reference plants, >100 MW installed globally
  - commercial_limited: <10 reference plants or niche applications
  - emerging: Pilot/demonstration stage, pre-commercial
  - research: Lab-scale or theoretical only
"""

from dataclasses import dataclass, field


@dataclass
class TechnologyDefinition:
    id: str
    name: str
    category: str        # "commercial" | "commercial_limited" | "emerging" | "research"
    viable_brine_temp_min_C: float
    viable_brine_temp_max_C: float
    typical_efficiency_range: tuple
    dominant_variables: list
    model_type: str      # "thermodynamic" | "parametric" | "reference"
    research_monitor: bool
    expert_prompt: str
    screening_notes: str


TECHNOLOGIES = {

    "orc_direct": TechnologyDefinition(
        id="orc_direct",
        name="Binary ORC -- Direct ACC",
        category="commercial",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=230,
        typical_efficiency_range=(0.10, 0.16),
        dominant_variables=["working_fluid", "turbine_efficiency",
                            "acc_approach_delta", "recuperation", "turbine_trains"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are a senior ORC design engineer with deep experience in binary geothermal plants.
You have detailed knowledge of commercial offerings from Ormat, TAS Energy, Turboden,
and ElectraTherm.

For this resource optimize:
- Working fluid selection (isopentane, isobutane, R134a, n-pentane) based on brine
  temperature and ambient conditions
- Recuperator inclusion decision -- worthwhile above ~140C brine
- Turbine staging -- single vs dual admission
- ACC approach temperature tradeoff vs fan parasitic load
- Number of parallel trains for reliability and constructability

Argue from thermodynamic first principles and commercial experience. Be specific
about why each choice is made for this resource.
        """,
        screening_notes="Viable across widest temperature range. Default commercial choice."
    ),

    "orc_propane_loop": TechnologyDefinition(
        id="orc_propane_loop",
        name="Binary ORC -- Propane Heat Rejection Loop",
        category="commercial",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=230,
        typical_efficiency_range=(0.09, 0.15),
        dominant_variables=["working_fluid", "turbine_efficiency",
                            "propane_acc_approach", "intermediate_hx_approach",
                            "parallel_construction_weeks_saved"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are a senior ORC design engineer specializing in split heat rejection architectures.
You understand the propane loop concept deeply: high-pressure propane replaces
low-pressure isopentane vapor ducts, enabling parallel construction, simplified
isopentane circuit, structural NCG reduction, and flexible ACC layout.

For this resource optimize:
- Approach temperature allocation: propane ACC vs intermediate HX
- Propane operating pressure and condensing temperature
- Parallel construction schedule benefit -- quantify weeks saved
- NCG reduction benefit -- estimate operational value over 30 years
- Shop fabrication advantage -- cost and quality impact

The thermodynamic tax of stacked temperature differences is real. Quantify it
and argue whether the construction and operational advantages justify it for
this specific resource and project context.
        """,
        screening_notes="Same viable range as direct ORC. Evaluate construction economics."
    ),

    "single_flash": TechnologyDefinition(
        id="single_flash",
        name="Single Flash Steam",
        category="commercial",
        viable_brine_temp_min_C=150, viable_brine_temp_max_C=320,
        typical_efficiency_range=(0.12, 0.20),
        dominant_variables=["flash_pressure_bar", "separator_efficiency",
                            "ncg_content_pct", "brine_chemistry_tds"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are an expert in flash steam geothermal plant design with knowledge of plants
at Geysers, Wairakei, Cerro Prieto, and Salton Sea.

For this resource optimize:
- Flash pressure for maximum power -- typically 4-8 bar
- Evaluate NCG content impact on turbine performance and handling system
- Assess scaling risk at flash conditions given brine chemistry
- Size the brine disposal system post-flash
- Evaluate direct contact vs surface condenser

Be direct about the lower temperature limit -- below 150C flash quality degrades
rapidly and ORC becomes preferable. If this resource is marginal for flash, say so
with specific numbers.
        """,
        screening_notes="Exclude below 150C. Screen for NCG and scaling risk."
    ),

    "double_flash": TechnologyDefinition(
        id="double_flash",
        name="Double Flash Steam",
        category="commercial",
        viable_brine_temp_min_C=180, viable_brine_temp_max_C=320,
        typical_efficiency_range=(0.15, 0.24),
        dominant_variables=["hp_flash_pressure_bar", "lp_flash_pressure_bar",
                            "hp_lp_split_ratio", "combined_vs_separate_turbines"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are an expert in double flash geothermal plant optimization. Double flash adds
15-25% more output than single flash by recovering additional energy from separated brine.

Optimize HP and LP flash pressures simultaneously for maximum combined output.
Evaluate whether combined admission turbine or separate HP/LP turbines are optimal
at this scale. Quantify incremental capital vs single flash and whether NPV justifies it.

If brine temperature is below 180C be explicit that efficiency gains diminish.
        """,
        screening_notes="Exclude below 180C. Compare NPV increment vs single flash."
    ),

    "hybrid_flash_binary": TechnologyDefinition(
        id="hybrid_flash_binary",
        name="Hybrid Flash-Binary",
        category="commercial",
        viable_brine_temp_min_C=160, viable_brine_temp_max_C=300,
        typical_efficiency_range=(0.14, 0.22),
        dominant_variables=["flash_pressure_bar", "binary_working_fluid",
                            "binary_turbine_inlet_temp_C", "flash_to_binary_integration"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are an expert in hybrid flash-binary geothermal plants, familiar with Ormat's
integrated plant designs in Nevada, Kenya, and Indonesia.

Flash stage handles high-enthalpy brine above flash temperature. Separated brine
discharge feeds binary ORC to recover remaining heat.

Optimize flash pressure for combined output, binary cycle conditions for the flash
brine discharge temperature, and evaluate whether total NPV exceeds either technology alone.
        """,
        screening_notes="Viable above 160C. Strongest when brine has high enthalpy."
    ),

    "steam_rankine": TechnologyDefinition(
        id="steam_rankine",
        name="Indirect Steam Rankine",
        category="commercial",
        viable_brine_temp_min_C=150, viable_brine_temp_max_C=250,
        typical_efficiency_range=(0.11, 0.18),
        dominant_variables=["steam_pressure_bar", "superheat_delta_C",
                            "condenser_pressure_bar", "feedwater_heating_stages"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are an expert in steam Rankine cycles applied to geothermal via clean-side HX.
Evaluate steam pressure, superheat decision, feedwater heating stages, and
brine-side HX design to manage scaling.

Steam Rankine can outperform ORC above 180C if scaling is managed.
Be honest about added complexity vs ORC.
        """,
        screening_notes="Viable above 150C. Compare efficiency gain vs ORC complexity."
    ),

    "kalina": TechnologyDefinition(
        id="kalina",
        name="Kalina Cycle",
        category="commercial_limited",
        viable_brine_temp_min_C=100, viable_brine_temp_max_C=220,
        typical_efficiency_range=(0.10, 0.16),
        dominant_variables=["ammonia_water_ratio", "separator_pressure_bar",
                            "absorber_approach_delta_C", "recuperation_configuration"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
You are an expert in Kalina cycle thermodynamics with knowledge of commercial
deployments at Husavik Iceland and Unterhaching Germany.

Be completely honest about the commercial track record -- Kalina has underperformed
expectations in most deployments due to corrosion, maintenance complexity of ammonia
handling, and difficulty achieving design performance. The efficiency advantage over
modern ORC is smaller than originally claimed. Do not oversell this technology.
        """,
        screening_notes="Be honest about limited commercial success vs ORC."
    ),

    "cascaded_orc": TechnologyDefinition(
        id="cascaded_orc",
        name="Cascaded Dual-Stage ORC",
        category="commercial_limited",
        viable_brine_temp_min_C=120, viable_brine_temp_max_C=230,
        typical_efficiency_range=(0.12, 0.17),
        dominant_variables=["high_stage_working_fluid", "low_stage_working_fluid",
                            "interstage_temperature_C", "high_low_power_split"],
        model_type="thermodynamic",
        research_monitor=False,
        expert_prompt="""
Two binary ORC stages in series with different working fluids optimized for
different temperature ranges. Most beneficial above 160C where a single
working fluid struggles to match the full temperature glide.

Optimize the working fluid pairing, interstage temperature, and evaluate whether
the efficiency gain justifies added complexity vs single-stage ORC.
        """,
        screening_notes="Most beneficial above 160C. Evaluate complexity premium."
    ),

    "geothermal_chp": TechnologyDefinition(
        id="geothermal_chp",
        name="Geothermal CHP -- Power + Heat",
        category="commercial_limited",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=230,
        typical_efficiency_range=(0.08, 0.13),
        dominant_variables=["power_heat_split_ratio", "heat_supply_temp_C",
                            "heat_load_MW", "heat_market_value_per_MWh",
                            "distance_to_heat_load_km"],
        model_type="parametric",
        research_monitor=False,
        expert_prompt="""
CHP sacrifices some power output to deliver usable heat -- district heating,
greenhouse agriculture, industrial process heat. Only analyze if there is a
credible heat offtaker. Evaluate whether total revenue (power + heat) exceeds
power-only configuration at this resource.
        """,
        screening_notes="Only viable with confirmed local heat market."
    ),

    "sco2_brayton": TechnologyDefinition(
        id="sco2_brayton",
        name="Supercritical CO2 Brayton Cycle",
        category="emerging",
        viable_brine_temp_min_C=150, viable_brine_temp_max_C=250,
        typical_efficiency_range=(0.14, 0.20),
        dominant_variables=["turbine_inlet_pressure_bar", "turbine_inlet_temp_C",
                            "recuperator_effectiveness", "cycle_topology"],
        model_type="parametric",
        research_monitor=True,
        expert_prompt="""
sCO2 advantages: very compact turbomachinery, high efficiency potential, no phase change.
No commercial geothermal sCO2 plant exists yet. Propose a design point based on
current demonstrated performance. Flag the technology readiness gaps explicitly.
Cite specific demonstration projects (Sandia, NREL, Southwest Research Institute).
        """,
        screening_notes="Emerging -- no commercial geothermal deployment. Flag maturity gap."
    ),

    "trilateral_flash": TechnologyDefinition(
        id="trilateral_flash",
        name="Trilateral Flash Cycle",
        category="emerging",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=170,
        typical_efficiency_range=(0.08, 0.13),
        dominant_variables=["working_fluid", "expander_isentropic_efficiency",
                            "flash_point_quality", "two_phase_expander_type"],
        model_type="parametric",
        research_monitor=True,
        expert_prompt="""
TFC expands saturated liquid through a two-phase expander -- better temperature
matching than ORC but the two-phase expander is the fundamental challenge.
Current best demonstrated isentropic efficiency 70-80% vs ORC turbines at 85-90%.
Calculate whether better temperature matching delivers more output despite lower
expander efficiency. Be honest -- TFC has been studied for decades and remains
pre-commercial primarily because the two-phase expander problem is harder than it looks.
        """,
        screening_notes="Pre-commercial. Two-phase expander is key barrier."
    ),

    "organic_flash": TechnologyDefinition(
        id="organic_flash",
        name="Organic Flash Cycle",
        category="emerging",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=180,
        typical_efficiency_range=(0.09, 0.14),
        dominant_variables=["working_fluid", "flash_pressure_bar",
                            "vapor_quality_at_flash", "liquid_recovery_configuration"],
        model_type="parametric",
        research_monitor=True,
        expert_prompt="""
Hybrid between ORC and TFC -- partially flashes the working fluid. Avoids the
worst of the two-phase expander problem. Propose optimal flash fraction and
evaluate whether efficiency gain over standard ORC justifies added complexity.
        """,
        screening_notes="Pre-commercial but closer to ORC than TFC."
    ),

    "stirling": TechnologyDefinition(
        id="stirling",
        name="Stirling Engine",
        category="research",
        viable_brine_temp_min_C=100, viable_brine_temp_max_C=300,
        typical_efficiency_range=(0.15, 0.30),
        dominant_variables=["hot_side_temp_C", "cold_side_temp_C",
                            "working_gas_pressure_bar", "engine_configuration"],
        model_type="parametric",
        research_monitor=True,
        expert_prompt="""
Theoretically Carnot-efficient, no working fluid phase change. Critical limitation:
specific power is very low -- best commercial units (Infinia, Kockums) are 1-25 kW.
Calculate the array required for this resource's target output and the cost premium
vs ORC. Be direct that Stirling is not competitive at geothermal plant scale today.
        """,
        screening_notes="Research stage at MW scale. Will likely be dismissed on cost/scale."
    ),

    "teg": TechnologyDefinition(
        id="teg",
        name="Thermoelectric Generator",
        category="research",
        viable_brine_temp_min_C=80, viable_brine_temp_max_C=300,
        typical_efficiency_range=(0.03, 0.08),
        dominant_variables=["ZT_value", "hot_side_temp_C", "cold_side_temp_C",
                            "module_cost_per_W", "HX_approach_delta_C"],
        model_type="parametric",
        research_monitor=True,
        expert_prompt="""
TEG efficiency governed by ZT = (S^2 * sigma / kappa) * T. Current commercial ZT ~1.0.

For this resource:
1. Research best demonstrated ZT at the relevant temperature range (not room temp)
2. Calculate device-level efficiency -- typically 40-60% of theoretical
3. Calculate required module area and cost at current $/W
4. Identify what ZT is needed to compete with ORC at this resource
5. Report materials closest to that threshold: lead telluride (ZT~2.2 at 500-900K),
   half-Heusler (ZT~1.5), skutterudites (ZT~1.7), GeTe-based (ZT~2.0+)

Be rigorous about the lab-to-device gap and 30-year durability at elevated temperatures.
        """,
        screening_notes="Research stage for bulk power. Calculate crossover ZT explicitly."
    ),

    "mhd": TechnologyDefinition(
        id="mhd",
        name="Magnetohydrodynamic Generation",
        category="research",
        viable_brine_temp_min_C=500, viable_brine_temp_max_C=3000,
        typical_efficiency_range=(0.10, 0.30),
        dominant_variables=["fluid_conductivity", "magnetic_field_strength"],
        model_type="reference",
        research_monitor=False,
        expert_prompt="""
MHD requires temperatures far above geothermal range. Dismiss with specific
temperature and conductivity numbers. Note the 1970s-80s coal research history
and why it doesn't apply here.
        """,
        screening_notes="Always excluded. Include to bound comparison."
    ),
}


# ── Convenience accessors ─────────────────────────────────────────────────────

def get_technology(tech_id: str) -> TechnologyDefinition | None:
    """Get a technology definition by ID."""
    return TECHNOLOGIES.get(tech_id)


def get_viable_technologies(brine_temp_C: float) -> list[TechnologyDefinition]:
    """Return technologies whose viable range includes the given brine temperature."""
    return [t for t in TECHNOLOGIES.values()
            if t.viable_brine_temp_min_C <= brine_temp_C <= t.viable_brine_temp_max_C]


def get_technologies_by_category(category: str) -> list[TechnologyDefinition]:
    """Return all technologies in a given category."""
    return [t for t in TECHNOLOGIES.values() if t.category == category]


def get_research_monitored() -> list[TechnologyDefinition]:
    """Return technologies with research_monitor=True."""
    return [t for t in TECHNOLOGIES.values() if t.research_monitor]


def technology_summary_table() -> list[dict]:
    """Return a summary list for display in the UI."""
    rows = []
    for t in TECHNOLOGIES.values():
        rows.append({
            "id": t.id,
            "name": t.name,
            "category": t.category,
            "temp_range": f"{t.viable_brine_temp_min_C:.0f}-{t.viable_brine_temp_max_C:.0f} C",
            "efficiency": f"{t.typical_efficiency_range[0]:.0%}-{t.typical_efficiency_range[1]:.0%}",
            "model_type": t.model_type,
            "research_monitor": t.research_monitor,
        })
    return rows
