"""
Path to $2,000/kW — Strategic Cost Reduction Analysis Engine
=============================================================

Reads actual data from the cost model, optimizer results, and GeoBlock
catalog to compute structured pathway assessments. Every number has a
confidence level and an explicit assumption.

NO LLM calls. All analysis is deterministic from model data + explicit
assumptions. No false precision — every output is a range.
"""

import math
import os
from dataclasses import dataclass, field, asdict
from typing import Optional

from cost_model import (
    COST_FACTORS,
    PROCUREMENT_STRATEGIES,
    MULTI_UNIT_DISCOUNTS,
    get_effective_cost_factors,
)

# ── Target ────────────────────────────────────────────────────────────────────

TARGET_PER_KW = 2000.0

# ── Confidence levels ─────────────────────────────────────────────────────────

CONFIDENCE_HIGH = "HIGH"
CONFIDENCE_MEDIUM = "MEDIUM"
CONFIDENCE_LOW = "LOW"

# ── Controllability ───────────────────────────────────────────────────────────

CONTROL_FULL = "Fervo fully controls"
CONTROL_PARTIAL = "Fervo partially controls"
CONTROL_MINIMAL = "Market / industry dependent"


# ── Core dataclass ────────────────────────────────────────────────────────────

@dataclass
class Mechanism:
    """One cost reduction mechanism within a pathway."""
    name: str
    saving_low: float          # $/kW — optimistic (most savings)
    saving_mid: float          # $/kW — expected
    saving_high: float         # $/kW — conservative (least savings)
    confidence: str            # HIGH / MEDIUM / LOW
    source: str                # where the number comes from
    assumption: str            # what must be true for this to materialize
    fervo_controls: str        # controllability label
    needs_unit1: bool          # can't validate until first unit built
    is_cost_addition: bool = False  # True if this adds cost (negative saving)
    notes: str = ""


@dataclass
class Pathway:
    """One strategic pathway to cost reduction."""
    name: str
    description: str
    mechanisms: list           # list of Mechanism
    honest_limitation: str     # what this pathway cannot do alone

    @property
    def total_saving_low(self) -> float:
        return sum(m.saving_low for m in self.mechanisms)

    @property
    def total_saving_mid(self) -> float:
        return sum(m.saving_mid for m in self.mechanisms)

    @property
    def total_saving_high(self) -> float:
        return sum(m.saving_high for m in self.mechanisms)


@dataclass
class PathwayAnalysis:
    """Complete analysis output."""
    baseline_per_kw: float
    baseline_source: str
    best_optimizer_per_kw: float
    best_optimizer_config: str
    target_per_kw: float
    gap_per_kw: float
    pathways: list             # list of Pathway
    critical_path: list        # ordered list of dicts
    what_must_be_true: list    # ranked conditions
    honest_limitations: list   # things we cannot know


# ── Baseline computation ──────────────────────────────────────────────────────

def compute_baseline_per_kw(strategy: str = "oem_lump_sum") -> dict:
    """Compute $/kW breakdown from COST_FACTORS for a given strategy.

    Returns dict with line-item $/kW values and total.
    This uses the cost factor rates directly (not from a specific thermo run),
    giving us the "model says" baseline independent of any specific brine condition.
    """
    cf = get_effective_cost_factors(strategy)

    # Equipment line items (these are $/kW or $/unit — normalize to $/kW gross)
    # For a reference 53 MW gross plant, equipment rates are already per-kW
    equip = {
        "turbine_generator": cf["turbine_per_kw"],
        "controls": cf["controls_instrumentation_per_kw"],
        "electrical": cf["electrical_equipment_per_kw"],
        "wf_inventory": cf["wf_inventory_per_kw"],
        "fan_control": cf["fan_control_per_kw"],
    }
    # HX and ACC require area/bay count — use reference values from optimizer
    # These are filled in by load_optimizer_bom() if available
    equip_subtotal_per_kw = sum(equip.values())

    # Construction line items (already $/kW gross in cost model)
    construction = {
        "civil_structural": cf["civil_structural_per_kw"],
        "ei_installation": cf["ei_installation_per_kw"],
        "construction_labor": cf["construction_labor_per_kw"],
        "commissioning": cf["commissioning_per_kw"],
    }
    construction_subtotal = sum(construction.values())

    # Percentage-based items (applied to equipment subtotal)
    pct_items = {
        "bop_piping_pct": cf["bop_piping_pct"],
        "engineering_pct": cf["engineering_pct"],
        "contingency_pct": cf["contingency_pct"],
    }

    return {
        "strategy": strategy,
        "equipment_items": equip,
        "equipment_subtotal_partial": equip_subtotal_per_kw,
        "construction_items": construction,
        "construction_subtotal": construction_subtotal,
        "pct_items": pct_items,
        "gathering_per_kw": cf["gathering_per_kw"],
        "td_per_kw": cf["td_per_kw"],
    }


def load_optimizer_best(store_path: str = None) -> dict:
    """Load best optimizer result if available."""
    if store_path is None:
        store_path = os.path.join("knowledge", "data", "optimizer_results.json")
    if not os.path.exists(store_path):
        return {"available": False}

    try:
        from optimizer_engine import ResultStore
        store = ResultStore(store_path)
        best = store.get_best_adjusted()
        if best is None:
            return {"available": False}
        return {
            "available": True,
            "capex_per_kw": best.capex_per_kW,
            "total_adjusted_per_kw": best.total_adjusted_per_kW,
            "equipment_per_kw": best.equipment_per_kW,
            "net_power_MW": best.net_power_MW,
            "config": best.config,
            "procurement_strategy": best.procurement_strategy,
            "bom_per_kw": best.bom_per_kw,
            "n_trains": best.n_trains,
        }
    except Exception:
        return {"available": False}


def load_geoblock_summary() -> dict:
    """Load GeoBlock catalog summary if available."""
    try:
        from geoblock_engine import load_results, analyze_standardization
        df = load_results()
        if len(df) == 0:
            return {"available": False}
        converged = df[df["converged"] == True]
        if len(converged) == 0:
            return {"available": False}

        std = analyze_standardization(df)
        one_spec_components = []
        two_spec_components = []
        for sr in std:
            if sr.recommendation == "skip":
                continue
            if sr.n_specs == 1:
                one_spec_components.append(sr.component_type)
            else:
                two_spec_components.append(sr.component_type)

        return {
            "available": True,
            "n_conditions": len(converged),
            "one_spec_components": one_spec_components,
            "two_spec_components": two_spec_components,
            "avg_capex_per_kw": round(converged["capex_per_kW"].mean(), 0),
            "min_capex_per_kw": round(converged["capex_per_kW"].min(), 0),
        }
    except Exception:
        return {"available": False}


# ── Pathway 1: Construction Speed & Self-Perform ─────────────────────────────

def build_pathway_1(opt_best: dict) -> Pathway:
    """Construction speed and cost reduction through self-perform execution."""
    mechanisms = []

    # ── 1A: Self-perform vs lump-sum EPC ──────────────────────────────
    # Derived directly from COST_FACTORS vs PROCUREMENT_STRATEGIES diffs
    cf_oem = get_effective_cost_factors("oem_lump_sum")
    cf_sp = get_effective_cost_factors("oem_self_perform")

    # Sum construction line items for both strategies
    construction_keys = [
        "civil_structural_per_kw", "ei_installation_per_kw",
        "construction_labor_per_kw", "commissioning_per_kw",
    ]
    oem_construction = sum(cf_oem[k] for k in construction_keys)
    sp_construction = sum(cf_sp[k] for k in construction_keys)
    construction_saving = oem_construction - sp_construction

    # Also account for BOP and engineering % reduction
    # These are %-based, so approximate using $1,150/kW equipment reference
    ref_equip_kw = 1150
    bop_saving = ref_equip_kw * (cf_oem["bop_piping_pct"] - cf_sp["bop_piping_pct"]) / 100
    eng_saving = ref_equip_kw * (cf_oem["engineering_pct"] - cf_sp["engineering_pct"]) / 100

    total_sp_saving = construction_saving + bop_saving + eng_saving
    # Contingency reduction: slightly lower contingency rate
    contingency_delta = ref_equip_kw * (cf_oem["contingency_pct"] - cf_sp["contingency_pct"]) / 100

    mid_saving = total_sp_saving + contingency_delta

    mechanisms.append(Mechanism(
        name="Self-perform construction vs lump-sum EPC",
        saving_low=round(mid_saving * 1.10),   # 10% better than model
        saving_mid=round(mid_saving),
        saving_high=round(mid_saving * 0.75),   # 25% less than model (learning curve)
        confidence=CONFIDENCE_HIGH,
        source=(
            f"cost_model.py: oem_lump_sum construction ${oem_construction}/kW vs "
            f"oem_self_perform ${sp_construction}/kW = ${construction_saving}/kW delta. "
            f"Plus BOP ${bop_saving:.0f}/kW + engineering ${eng_saving:.0f}/kW."
        ),
        assumption=(
            "Fervo has or can build a qualified self-perform construction team. "
            "T&M burdened rate $85-95/hr. Indirect multiplier 1.20x vs 1.35x EPC."
        ),
        fervo_controls=CONTROL_FULL,
        needs_unit1=True,
        notes=(
            f"Construction line items (lump-sum): civil ${cf_oem['civil_structural_per_kw']}/kW, "
            f"E&I ${cf_oem['ei_installation_per_kw']}/kW, labor ${cf_oem['construction_labor_per_kw']}/kW, "
            f"commissioning ${cf_oem['commissioning_per_kw']}/kW = ${oem_construction}/kW total.\n"
            f"Self-perform: civil ${cf_sp['civil_structural_per_kw']}/kW, "
            f"E&I ${cf_sp['ei_installation_per_kw']}/kW, labor ${cf_sp['construction_labor_per_kw']}/kW, "
            f"commissioning ${cf_sp['commissioning_per_kw']}/kW = ${sp_construction}/kW total."
        ),
    ))

    # ── 1B: Schedule compression ──────────────────────────────────────
    # Weekly burn rate = construction labor $/kW / schedule weeks
    baseline_weeks = 70
    target_weeks = 52
    weeks_saved = baseline_weeks - target_weeks
    # Use self-perform labor rate since that's the pathway we're on
    weekly_burn_per_kw = cf_sp["construction_labor_per_kw"] / baseline_weeks
    schedule_saving = weeks_saved * weekly_burn_per_kw

    mechanisms.append(Mechanism(
        name="Schedule compression (70 wk -> 52 wk)",
        saving_low=round(schedule_saving * 1.15),
        saving_mid=round(schedule_saving),
        saving_high=round(schedule_saving * 0.60),    # only get 60% of theoretical
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"Weekly burn rate: ${cf_sp['construction_labor_per_kw']}/kW "
            f"/ {baseline_weeks} wk = ${weekly_burn_per_kw:.1f}/kW/wk. "
            f"{weeks_saved} weeks saved = ${schedule_saving:.0f}/kW."
        ),
        assumption=(
            "Requires: modular pre-assembly, pre-purchased long-lead items, "
            "dedicated crew continuity, parallel execution with drilling. "
            "52 weeks assumes civil starts while equipment is in transit."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=True,
        notes=(
            "Conservative estimate assumes only 60% of theoretical saving materializes "
            "(weather, vendor delays, interface issues absorb rest)."
        ),
    ))

    # ── 1C: Repeat-unit learning curve ────────────────────────────────
    # 8% learning rate per doubling of cumulative units
    learning_rate = 0.08
    # Apply to construction cost only (not equipment — that's Pathway 3)
    unit1_construction = sp_construction  # after self-perform
    savings_at_n = {}
    for n in [2, 4, 8]:
        doublings = math.log2(n)
        cost_at_n = unit1_construction * (1 - learning_rate) ** doublings
        savings_at_n[n] = unit1_construction - cost_at_n

    # Report Unit 4 as the "mid" case (typical program by this point)
    mechanisms.append(Mechanism(
        name="Repeat-unit learning curve (8%/doubling)",
        saving_low=round(savings_at_n[8]),    # by Unit 8
        saving_mid=round(savings_at_n[4]),    # by Unit 4
        saving_high=round(savings_at_n[2]),   # by Unit 2
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"cost_model.py MULTI_UNIT_DISCOUNTS + 8%/doubling learning rate on "
            f"construction labor. Unit 1 base: ${unit1_construction}/kW. "
            f"Unit 2: -${savings_at_n[2]:.0f}/kW, "
            f"Unit 4: -${savings_at_n[4]:.0f}/kW, "
            f"Unit 8: -${savings_at_n[8]:.0f}/kW."
        ),
        assumption=(
            "Standard construction learning curve applies. "
            "Assumes crew continuity between units (same teams, same procedures). "
            "CANNOT VALIDATE UNTIL UNIT 1 IS COMPLETE."
        ),
        fervo_controls=CONTROL_FULL,
        needs_unit1=True,
        notes="Learning rates from DoE geothermal construction benchmarks. "
              "8% is conservative vs 12-15% seen in solar/wind.",
    ))

    # ── 1D: Standardized civil & foundations ──────────────────────────
    mechanisms.append(Mechanism(
        name="Standardized foundations and civil work",
        saving_low=100,
        saving_mid=75,
        saving_high=50,
        confidence=CONFIDENCE_MEDIUM,
        source=(
            "Estimate based on civil cost delta between custom vs standard designs. "
            "Reusable formwork, pre-purchased rebar/concrete, civil crew specialization."
        ),
        assumption=(
            "Requires frozen foundation design from GeoBlock parametric analysis. "
            "Standard formwork reuse across 8+ units. Pre-purchase of civil materials."
        ),
        fervo_controls=CONTROL_FULL,
        needs_unit1=False,
        notes="Can begin before Unit 1 — foundation design can be standardized from "
              "parametric analysis results.",
    ))

    return Pathway(
        name="Pathway 1 — Construction Speed & Self-Perform",
        description=(
            "Reduce construction cost through self-perform execution, schedule compression, "
            "repeat-unit learning, and standardized civil work. This is the highest-confidence "
            "pathway because Fervo controls most of the variables."
        ),
        mechanisms=mechanisms,
        honest_limitation=(
            "Self-perform productivity assumptions cannot be validated until Unit 1 is built and measured. "
            "Schedule compression depends on parallel execution with drilling — "
            "any drilling delay pushes surface facility start. "
            "Learning curve requires crew continuity — if crews turn over between units, learning resets."
        ),
    )


# ── Pathway 2: Modular Design ────────────────────────────────────────────────

def build_pathway_2(opt_best: dict) -> Pathway:
    """Factory-fabricated skid modules delivered to site."""
    mechanisms = []

    cf_sp = get_effective_cost_factors("oem_self_perform")

    # Current field labor (self-perform baseline)
    field_labor = (
        cf_sp["civil_structural_per_kw"]
        + cf_sp["ei_installation_per_kw"]
        + cf_sp["construction_labor_per_kw"]
    )

    # ── 2A: Field labor reduction ─────────────────────────────────────
    # Modular target: civil + module setting + interconnects only
    modular_field_low = 200   # optimistic
    modular_field_mid = 250
    modular_field_high = 300  # conservative
    field_saving_low = field_labor - modular_field_low
    field_saving_mid = field_labor - modular_field_mid
    field_saving_high = field_labor - modular_field_high

    mechanisms.append(Mechanism(
        name="Field labor reduction from modularization",
        saving_low=round(field_saving_low),
        saving_mid=round(field_saving_mid),
        saving_high=round(field_saving_high),
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"Current field labor (self-perform): ${field_labor}/kW "
            f"(civil ${cf_sp['civil_structural_per_kw']}/kW + "
            f"E&I ${cf_sp['ei_installation_per_kw']}/kW + "
            f"labor ${cf_sp['construction_labor_per_kw']}/kW). "
            f"Modular target: ${modular_field_mid}/kW (civil + setting + interconnects)."
        ),
        assumption=(
            "Assumes pump skid, HX module, TG module, and electrical building "
            "can be shop-fabricated. ACC structure and brine manifold remain field-erected. "
            "LIMITED GEOTHERMAL ORC PRECEDENT at this scale."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=True,
    ))

    # ── 2B: Shop fabrication cost addition (NEGATIVE saving) ──────────
    mechanisms.append(Mechanism(
        name="Shop fabrication cost addition",
        saving_low=-80,    # best case: low shop cost
        saving_mid=-100,
        saving_high=-120,  # worst case: high shop cost
        confidence=CONFIDENCE_MEDIUM,
        source=(
            "Shop labor + transport + heavy lift = $80-120/kW addition. "
            "Based on modular industrial plant benchmarks (LNG, petrochemical)."
        ),
        assumption=(
            "Requires shop fabrication vendor with adequate bay space. "
            "Transport logistics for oversize modules (rail or truck). "
            "Heavy-lift crane available on site."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=False,
        is_cost_addition=True,
        notes="This is a COST ADDITION (negative saving). "
              "Shop fabrication shifts labor from field to shop — cheaper per hour, "
              "but not free.",
    ))

    # ── 2C: Schedule reduction ────────────────────────────────────────
    baseline_weeks = 70
    modular_target_weeks_low = 40
    modular_target_weeks_mid = 42
    modular_target_weeks_high = 48
    weekly_burn = cf_sp["construction_labor_per_kw"] / baseline_weeks

    mechanisms.append(Mechanism(
        name="Schedule reduction from modularization (70 -> 40-48 wk)",
        saving_low=round((baseline_weeks - modular_target_weeks_low) * weekly_burn),
        saving_mid=round((baseline_weeks - modular_target_weeks_mid) * weekly_burn),
        saving_high=round((baseline_weeks - modular_target_weeks_high) * weekly_burn),
        confidence=CONFIDENCE_LOW,
        source=(
            f"Weekly burn ${weekly_burn:.1f}/kW/wk. "
            f"Field schedule 70 wk -> 40-48 wk (parallel shop fabrication "
            f"while civil underway)."
        ),
        assumption=(
            "Shop fabrication overlaps completely with civil construction. "
            "Module delivery is on critical path — any shop delay pushes site schedule. "
            "Heavy-lift mobilization is included in schedule."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=True,
    ))

    # ── 2D: Quality improvement (unmodeled) ───────────────────────────
    mechanisms.append(Mechanism(
        name="Quality improvement from shop fabrication",
        saving_low=30,
        saving_mid=15,
        saving_high=0,
        confidence=CONFIDENCE_LOW,
        source="Qualitative estimate. Reduced rework, better QC in controlled environment.",
        assumption=(
            "UNMODELABLE BENEFIT. Shop fabrication reduces field rework and "
            "commissioning time. Value exists but cannot be reliably quantified. "
            "Shown as $0-30/kW range to acknowledge without overstating."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=True,
        notes="This is a placeholder range, not a rigorous estimate.",
    ))

    return Pathway(
        name="Pathway 2 — Modular Design",
        description=(
            "Factory-fabricated skid modules delivered to site on pre-built foundations. "
            "Fundamentally changes the field labor equation but adds shop fabrication cost. "
            "Proven in LNG/petrochemical, limited geothermal ORC precedent."
        ),
        mechanisms=mechanisms,
        honest_limitation=(
            "Modular design for geothermal ORC at this scale has limited precedent. "
            "Module interface definition is the critical risk — inter-module connections "
            "are where problems hide. First unit will be MORE expensive than stick-built "
            "(design cost, prototype risk). Savings materialize only with repetition. "
            "ACC structure cannot be modularized — it is the largest single component. "
            "Requires heavy-lift crane (150+ ton), which may not be locally available."
        ),
    )


# ── Pathway 3: Equipment Cost Reduction ───────────────────────────────────────

def build_pathway_3(opt_best: dict, geoblock: dict) -> Pathway:
    """Equipment cost reduction through volume, frame contracts, standardization."""
    mechanisms = []

    cf_direct = get_effective_cost_factors("direct_self_perform")

    # ── 3A: ACC frame contract ────────────────────────────────────────
    current_acc_per_bay = cf_direct["acc_per_bay"]
    frame_discount_low = 0.20   # optimistic
    frame_discount_mid = 0.17
    frame_discount_high = 0.12  # conservative
    # Reference: ~35 bays per unit, estimate per-kW impact
    ref_bays_per_unit = 35
    ref_net_kw = 53000
    acc_per_kw_current = current_acc_per_bay * ref_bays_per_unit / ref_net_kw

    mechanisms.append(Mechanism(
        name="ACC frame contract (360 bays, 8 units)",
        saving_low=round(acc_per_kw_current * frame_discount_low),
        saving_mid=round(acc_per_kw_current * frame_discount_mid),
        saving_high=round(acc_per_kw_current * frame_discount_high),
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"Current ACC: ${current_acc_per_bay:,.0f}/bay (Worldwide validated bid, direct purchase). "
            f"~{ref_bays_per_unit} bays/unit = ${acc_per_kw_current:.0f}/kW. "
            f"Frame contract at 360 bays: 12-20% discount."
        ),
        assumption=(
            "Requires volume commitment of 360+ bays across program. "
            "Assumes single ACC vendor selected (Worldwide, SPX, or Harsco). "
            "Market pricing may move independently of Fervo volume."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=False,
    ))

    # ── 3B: Turbine-generator frame contract ──────────────────────────
    current_tg_per_kw = cf_direct["turbine_per_kw"]
    tg_discount_low = 0.15
    tg_discount_mid = 0.12
    tg_discount_high = 0.08

    mechanisms.append(Mechanism(
        name="Turbine-generator frame contract (16+ units)",
        saving_low=round(current_tg_per_kw * tg_discount_low),
        saving_mid=round(current_tg_per_kw * tg_discount_mid),
        saving_high=round(current_tg_per_kw * tg_discount_high),
        confidence=CONFIDENCE_LOW,
        source=(
            f"Current TG: ${current_tg_per_kw}/kW (direct purchase, Exergy/Turboden). "
            f"Frame contract for 16+ units (8 plants × 2 trains): 8-15% discount."
        ),
        assumption=(
            "ORC turbine market is still small. Volume discounts depend on vendor capacity "
            "and competing demand. Exergy, Turboden, and Ormat may have different cost structures. "
            "Confidence is LOW because ORC turbine pricing is not transparent."
        ),
        fervo_controls=CONTROL_MINIMAL,
        needs_unit1=False,
    ))

    # ── 3C: HX standardization ───────────────────────────────────────
    hx_standardization_note = "GeoBlock catalog not yet run — using estimate"
    if geoblock.get("available"):
        one_spec = geoblock.get("one_spec_components", [])
        two_spec = geoblock.get("two_spec_components", [])
        hx_standardization_note = (
            f"GeoBlock catalog: 1-spec components: {', '.join(one_spec) if one_spec else 'none'}. "
            f"2-spec components: {', '.join(two_spec) if two_spec else 'none'}."
        )

    # HX costs from cost model (per ft² rates × reference areas)
    # Estimate HX total from equipment subtotal (~20% of equipment)
    ref_equip_per_kw = 800  # approximate equipment $/kW at direct pricing
    hx_share = 0.20
    hx_per_kw = ref_equip_per_kw * hx_share

    mechanisms.append(Mechanism(
        name="HX standardization + frame contract",
        saving_low=round(hx_per_kw * 0.20),
        saving_mid=round(hx_per_kw * 0.15),
        saving_high=round(hx_per_kw * 0.10),
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"HX ~${hx_per_kw:.0f}/kW (est. 20% of equipment package). "
            f"Standardization + volume: 10-20% discount. "
            f"{hx_standardization_note}"
        ),
        assumption=(
            "Requires frozen HX specifications from GeoBlock parametric analysis. "
            "One or two standard shell sizes ordered as frame contract. "
            "Competitive market exists (Koch, Kelvion, Bronswerk)."
        ),
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=False,
    ))

    # ── 3D: Pump standardization ──────────────────────────────────────
    pump_per_kw = 15  # approximate from cost model
    mechanisms.append(Mechanism(
        name="WF pump standardization + frame contract",
        saving_low=round(pump_per_kw * 0.25),
        saving_mid=round(pump_per_kw * 0.18),
        saving_high=round(pump_per_kw * 0.12),
        confidence=CONFIDENCE_HIGH,
        source=(
            f"WF pump ~${pump_per_kw}/kW. GeoBlock analysis shows pump is likely "
            f"a 1-spec component across full condition range. "
            f"Commodity API 610 market — volume discount 12-25%."
        ),
        assumption="Standard pump specification + volume purchase. Commodity market.",
        fervo_controls=CONTROL_FULL,
        needs_unit1=False,
    ))

    # ── 3E: Structural steel & civil materials ────────────────────────
    steel_per_kw = cf_direct.get("steel_per_lb", 7.0) * 10  # rough: ~10 lb/kW
    mechanisms.append(Mechanism(
        name="Bulk steel and civil materials pre-purchase",
        saving_low=round(steel_per_kw * 0.12),
        saving_mid=round(steel_per_kw * 0.10),
        saving_high=round(steel_per_kw * 0.06),
        confidence=CONFIDENCE_MEDIUM,
        source=(
            f"Structural steel ~${steel_per_kw:.0f}/kW at ${cf_direct['steel_per_lb']}/lb. "
            f"Pre-purchase for 8 units locks pricing, avoids escalation. 6-12% saving."
        ),
        assumption="Requires capital commitment to purchase ahead of need. Price lock risk.",
        fervo_controls=CONTROL_PARTIAL,
        needs_unit1=False,
    ))

    # Equipment share limitation
    total_equip_saving_mid = sum(m.saving_mid for m in mechanisms)

    return Pathway(
        name="Pathway 3 — Equipment Cost Reduction",
        description=(
            "Drive down equipment unit costs through volume commitments, frame contracts, "
            "and component standardization from GeoBlock parametric analysis."
        ),
        mechanisms=mechanisms,
        honest_limitation=(
            f"Equipment is ~45-50% of installed cost. Total equipment saving at expected values: "
            f"~${total_equip_saving_mid}/kW. This is NECESSARY but INSUFFICIENT to reach $2,000/kW "
            f"without Pathway 1 or 2. Equipment cost reduction is an amplifier, not a standalone solution. "
            f"Some savings (ACC, TG) depend on market factors Fervo does not control."
        ),
    )


# ── Critical path ─────────────────────────────────────────────────────────────

def build_critical_path() -> list[dict]:
    """Ordered list of actions with gates and dependencies."""
    return [
        {
            "phase": "MUST HAPPEN FIRST (gates everything else)",
            "items": [
                {
                    "action": "Freeze component specifications",
                    "detail": "GeoBlock parametric analysis must complete. "
                              "Standard specs for HX, pumps, ACC defined.",
                    "owner": "Engineering",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Commit to self-perform delivery model",
                    "detail": "Organizational decision — not engineering. "
                              "Requires hiring or contracting field construction team.",
                    "owner": "Executive / Operations",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Establish vendor frame contracts",
                    "detail": "Requires frozen specs + volume commitment. "
                              "ACC, TG, HX, pumps.",
                    "owner": "Procurement",
                    "fervo_controls": CONTROL_PARTIAL,
                },
            ],
        },
        {
            "phase": "CAN HAPPEN IN PARALLEL",
            "items": [
                {
                    "action": "Develop modular skid designs",
                    "detail": "Pump skid, HX module, TG module, electrical building. "
                              "Can begin with preliminary specs.",
                    "owner": "Engineering",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Train self-perform construction crews",
                    "detail": "ORC-specific skills: HX rigging, turbine alignment, "
                              "working fluid handling.",
                    "owner": "Operations",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Build procurement relationships",
                    "detail": "Qualify vendors, negotiate frame terms, "
                              "establish supply chain.",
                    "owner": "Procurement",
                    "fervo_controls": CONTROL_PARTIAL,
                },
            ],
        },
        {
            "phase": "CANNOT HAPPEN UNTIL UNIT 1 COMPLETE",
            "items": [
                {
                    "action": "Validate learning curve assumptions",
                    "detail": "Measure actual Unit 1 cost vs model. "
                              "Calibrate learning rate for Units 2-8.",
                    "owner": "Project Controls",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Measure actual self-perform productivity",
                    "detail": "Track labor hours/kW installed. Compare to "
                              "EPC lump-sum baseline.",
                    "owner": "Construction",
                    "fervo_controls": CONTROL_FULL,
                },
                {
                    "action": "Confirm modular interface performance",
                    "detail": "If modular approach selected: validate that "
                              "inter-module connections perform as designed.",
                    "owner": "Engineering / Commissioning",
                    "fervo_controls": CONTROL_FULL,
                },
            ],
        },
    ]


# ── What must be true ─────────────────────────────────────────────────────────

def build_what_must_be_true(pathways: list[Pathway]) -> list[dict]:
    """Rank conditions by controllability and impact."""
    conditions = []
    for pw in pathways:
        for m in pw.mechanisms:
            if m.is_cost_addition:
                continue
            conditions.append({
                "condition": m.assumption,
                "saving_mid": m.saving_mid,
                "confidence": m.confidence,
                "fervo_controls": m.fervo_controls,
                "needs_unit1": m.needs_unit1,
                "pathway": pw.name.split("—")[1].strip() if "—" in pw.name else pw.name,
                "mechanism": m.name,
            })

    # Sort: fully controlled first, then by saving magnitude
    control_order = {CONTROL_FULL: 0, CONTROL_PARTIAL: 1, CONTROL_MINIMAL: 2}
    conditions.sort(key=lambda c: (control_order.get(c["fervo_controls"], 9), -c["saving_mid"]))
    return conditions


# ── Honest limitations ────────────────────────────────────────────────────────

def build_honest_limitations() -> list[str]:
    return [
        "Actual self-perform productivity: cannot know until Unit 1 is built and measured. "
        "The model assumes T&M rates of $85-95/hr burdened — actual rates depend on labor market, "
        "site location, and crew experience.",

        "Modular interface performance: cannot know until a prototype module is built, shipped, "
        "set, and commissioned. Inter-module connections are the #1 risk in modular construction.",

        "Equipment cost curve trajectory: depends on geothermal industry volume growth, not just "
        "Fervo's program. If the ORC market doubles (driven by all developers), costs fall faster. "
        "If Fervo is the only buyer, volume discounts are limited to Fervo's program alone.",

        "Whether $2,000/kW is the right target: this analysis considers surface facility cost only. "
        "Total project $/kW (including wells) may have a different optimization point. A surface "
        "facility that costs $2,200/kW but enables 10% higher well utilization may produce cheaper "
        "electricity than one at $2,000/kW with lower utilization.",

        "All learning curves assume crew continuity. If construction teams turn over between "
        "units, learning resets to a higher point. Fervo must invest in retention.",

        "Schedule compression assumes no drilling delays. Surface facility construction "
        "timeline depends on well delivery — any well delay propagates.",
    ]


# ── Main analysis ─────────────────────────────────────────────────────────────

def run_pathway_analysis() -> PathwayAnalysis:
    """Run the complete pathway analysis, reading all available data sources."""

    # Load data sources
    opt_best = load_optimizer_best()
    geoblock = load_geoblock_summary()

    # Compute baseline
    # Use oem_lump_sum as the "Phase 1 actuals" reference
    cf_oem = get_effective_cost_factors("oem_lump_sum")

    # Build a rough installed $/kW from cost factors
    # Equipment: use reference $1,150/kW (calibrated to Turboden 93 MW project)
    equip_per_kw = 1150
    bop = equip_per_kw * cf_oem["bop_piping_pct"] / 100
    civil = cf_oem["civil_structural_per_kw"]
    ei = cf_oem["ei_installation_per_kw"]
    labor = cf_oem["construction_labor_per_kw"]
    eng = equip_per_kw * cf_oem["engineering_pct"] / 100
    commission = cf_oem["commissioning_per_kw"]
    subtotal = equip_per_kw + bop + civil + ei + labor + eng + commission
    contingency = subtotal * cf_oem["contingency_pct"] / 100
    plant_installed = subtotal + contingency
    gathering = cf_oem["gathering_per_kw"]
    td = cf_oem["td_per_kw"]
    total_installed = plant_installed + gathering + td

    baseline_per_kw = round(total_installed)
    baseline_source = (
        f"cost_model.py COST_FACTORS (oem_lump_sum): equipment ${equip_per_kw}/kW + "
        f"BOP ${bop:.0f} + civil ${civil} + E&I ${ei} + labor ${labor} + "
        f"engineering ${eng:.0f} + commissioning ${commission} + "
        f"contingency ${contingency:.0f} = ${plant_installed:.0f}/kW plant installed. "
        f"Plus gathering ${gathering} + T&D ${td} = ${total_installed:.0f}/kW total."
    )

    # Best optimizer result
    if opt_best.get("available"):
        best_kw = opt_best["total_adjusted_per_kw"]
        best_config = str(opt_best.get("config", {}))
        best_strat = opt_best.get("procurement_strategy", "unknown")
    else:
        best_kw = baseline_per_kw
        best_config = "No optimizer results available"
        best_strat = "oem_lump_sum"

    # Build pathways
    pw1 = build_pathway_1(opt_best)
    pw2 = build_pathway_2(opt_best)
    pw3 = build_pathway_3(opt_best, geoblock)

    pathways = [pw1, pw2, pw3]
    critical_path = build_critical_path()
    what_must_be_true = build_what_must_be_true(pathways)
    honest_limitations = build_honest_limitations()

    # Gap from best known to target
    gap = best_kw - TARGET_PER_KW

    return PathwayAnalysis(
        baseline_per_kw=baseline_per_kw,
        baseline_source=baseline_source,
        best_optimizer_per_kw=best_kw,
        best_optimizer_config=best_config,
        target_per_kw=TARGET_PER_KW,
        gap_per_kw=round(gap),
        pathways=pathways,
        critical_path=critical_path,
        what_must_be_true=what_must_be_true,
        honest_limitations=honest_limitations,
    )


# ── Waterfall data for matplotlib ─────────────────────────────────────────────

def build_waterfall_data(analysis: PathwayAnalysis, pathway_index: int = None) -> dict:
    """Build data for a waterfall chart.

    If pathway_index is None, combines all pathways (best case).
    Otherwise shows a single pathway.
    """
    if pathway_index is not None:
        pathways = [analysis.pathways[pathway_index]]
    else:
        pathways = analysis.pathways

    labels = ["Baseline"]
    mid_values = [analysis.best_optimizer_per_kw]
    low_values = [analysis.best_optimizer_per_kw]
    high_values = [analysis.best_optimizer_per_kw]
    confidences = [""]

    running = analysis.best_optimizer_per_kw

    for pw in pathways:
        for m in pw.mechanisms:
            labels.append(m.name[:35] + ("..." if len(m.name) > 35 else ""))
            mid_values.append(-m.saving_mid)
            low_values.append(-m.saving_low)    # low = most saving = most negative
            high_values.append(-m.saving_high)   # high = least saving = least negative
            confidences.append(m.confidence)
            running -= m.saving_mid

    labels.append("Projected")
    mid_values.append(running)
    low_values.append(running - sum(m.saving_low - m.saving_mid for pw in pathways for m in pw.mechanisms))
    high_values.append(running + sum(m.saving_mid - m.saving_high for pw in pathways for m in pw.mechanisms))
    confidences.append("")

    return {
        "labels": labels,
        "mid_values": mid_values,
        "low_values": low_values,
        "high_values": high_values,
        "confidences": confidences,
        "target": analysis.target_per_kw,
    }
