"""
GeoBlock Component Catalog Engine
==================================

Runs the existing thermodynamic/cost model across a matrix of operating
conditions (brine temp × ambient temp × target MW), then analyzes results
for component standardization.

NO new physics — all thermo/cost calls delegate to analysis_bridge.run_orc_analysis().
"""

import csv
import hashlib
import math
import os
import time
from dataclasses import dataclass, field, asdict
from itertools import product
from typing import Optional

import pandas as pd

from feed_package import EQUIP_REF, _hx_shells, _hx_shell_dia_in

# ── Paths ─────────────────────────────────────────────────────────────────────

DATA_DIR = os.path.join("knowledge", "data")
RESULTS_CSV = os.path.join(DATA_DIR, "geoblock_results.csv")

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_BRINE_RANGE = (420, 480, 10)      # min, max, step °F
DEFAULT_AMBIENT_RANGE = (40, 100, 20)     # min, max, step °F
DEFAULT_MW_RANGE = (40, 60, 5)            # min, max, step MW
DEFAULT_MAX_OVERSIZE_PCT = 25.0
DEFAULT_MAX_SPECS = 3
MW_SCALING_TOLERANCE = 0.10               # ±10% — iterate m_dot if beyond
MW_SCALING_MAX_ITER = 2

# Reference condition for m_dot scaling (matches main.py defaults)
REF_M_DOT_LB_S = 1100.0

# ── CSV columns ───────────────────────────────────────────────────────────────

CSV_COLUMNS = [
    # Condition
    "condition_id", "brine_inlet_F", "ambient_F", "target_MW",
    "m_dot_geo_lb_s", "topology", "procurement_strategy",
    # Convergence
    "converged", "iterations",
    # Performance
    "net_MW", "gross_MW", "cycle_efficiency", "capex_per_kW", "lcoe",
    "capex_total_USD", "equipment_per_kW",
    # Turbine
    "turbine_n_units", "turbine_mw_each",
    "T_evap_F", "P_high_psia", "T_cond_F", "P_low_psia",
    # Vaporizer
    "vaporizer_area_ft2", "vaporizer_duty_mmbtu",
    "vaporizer_n_shells", "vaporizer_shell_dia_in",
    # Preheater
    "preheater_area_ft2", "preheater_duty_mmbtu",
    "preheater_n_shells", "preheater_shell_dia_in",
    # Recuperator
    "recuperator_area_ft2",
    # ACC
    "acc_n_bays", "acc_area_ft2", "acc_fan_kw",
    # WF Pump
    "pump_n_pumps", "pump_gpm_each", "pump_head_ft", "pump_hp_each",
    # IHX (Config B only)
    "ihx_area_ft2",
    # Structural
    "structural_steel_lb",
    # WF flow
    "m_dot_iso_lb_hr",
    # Cost line items
    "cost_turbine", "cost_vaporizer", "cost_preheater",
    "cost_recuperator", "cost_acc", "cost_pump",
    # Duration
    "run_duration_s",
]


# ── Dataclasses ───────────────────────────────────────────────────────────────

@dataclass
class ConditionPoint:
    """One cell in the operating-conditions matrix."""
    brine_inlet_F: float
    ambient_F: float
    target_MW: float

    @property
    def condition_id(self) -> str:
        key = f"{self.brine_inlet_F}|{self.ambient_F}|{self.target_MW}"
        return hashlib.sha256(key.encode()).hexdigest()[:12]


@dataclass
class ComponentSpec:
    """A standardized component specification covering a range of conditions."""
    component_type: str          # "turbine", "vaporizer", "preheater", etc.
    spec_id: str                 # "TG-S1", "VAP-S1", "VAP-S2"
    sizing_param: str            # "mw_each", "area_ft2", "n_bays", "hp_each"
    sizing_unit: str             # "MW", "ft²", "bays", "HP"
    min_rating: float
    max_rating: float
    design_rating: float         # max + margin
    margin_pct: float            # design margin applied
    n_shells: int = 0            # for HX types
    shell_dia_in: float = 0.0    # for HX types
    condition_ids: list = field(default_factory=list)
    coverage_pct: float = 0.0
    # Vendor info (populated from EQUIP_REF)
    vendors: list = field(default_factory=list)
    lead_weeks: int = 0
    estimated_unit_cost: float = 0.0
    cost_unit: str = "$/unit"
    # Sensitivity
    crossover_value: float = 0.0       # condition value at crossover
    crossover_param: str = ""          # "brine_inlet_F" or "ambient_F"
    crossover_sensitivity_band: float = 0.0  # ±°F movement at ±10% cost


@dataclass
class StandardizationResult:
    """Per-component standardization analysis output."""
    component_type: str
    sizing_param: str
    sizing_unit: str
    n_specs: int
    specs: list                  # list of ComponentSpec
    one_spec_coverage_full: float   # % of conditions at full perf with 1 spec
    one_spec_coverage_5pct: float   # % within 5% penalty
    total_waste_pct: float          # average oversize across all conditions
    recommendation: str             # "one_spec" or "two_specs" or "three_specs"
    reasoning: str


@dataclass
class AssemblyRule:
    """One row of the assembly grammar lookup table."""
    brine_band: str
    ambient_band: str
    mw_band: str
    component_specs: dict        # component_type -> spec_id
    estimated_capex_per_kw: float
    condition_count: int         # how many matrix points match


# ── Matrix generation ─────────────────────────────────────────────────────────

def generate_condition_matrix(
    brine_range: tuple = DEFAULT_BRINE_RANGE,
    ambient_range: tuple = DEFAULT_AMBIENT_RANGE,
    mw_range: tuple = DEFAULT_MW_RANGE,
) -> list[ConditionPoint]:
    """Cartesian product of user-defined operating condition ranges."""
    brine_vals = _range_inclusive(*brine_range)
    ambient_vals = _range_inclusive(*ambient_range)
    mw_vals = _range_inclusive(*mw_range)
    return [
        ConditionPoint(b, a, m)
        for b, a, m in product(brine_vals, ambient_vals, mw_vals)
    ]


def _range_inclusive(lo, hi, step):
    """Generate values from lo to hi inclusive, with given step."""
    vals = []
    v = lo
    while v <= hi + 1e-9:
        vals.append(round(v, 2))
        v += step
    return vals


# ── Single condition runner ───────────────────────────────────────────────────

def _f_to_c(t_f: float) -> float:
    return (t_f - 32.0) * 5.0 / 9.0


def _lbs_to_kgs(m: float) -> float:
    return m / 2.20462


def run_condition_point(
    point: ConditionPoint,
    base_design_basis: dict,
    topology: str,
    procurement_strategy: str,
    base_net_mw: float,
    n_trains: int = 2,
) -> dict:
    """Run one condition through analysis_bridge.run_orc_analysis().

    Returns a flat dict matching CSV_COLUMNS, ready for CSV append.
    """
    from analysis_bridge import run_orc_analysis

    t_start = time.time()

    # Scale m_dot to approximate target MW
    if base_net_mw > 0:
        scale = point.target_MW / base_net_mw
    else:
        scale = 1.0
    m_dot_lb_s = REF_M_DOT_LB_S * scale

    # Build design basis overrides
    db = dict(base_design_basis)
    db["brine_inlet_temp_C"] = _f_to_c(point.brine_inlet_F)
    db["ambient_temp_C"] = _f_to_c(point.ambient_F)
    db["brine_flow_kg_s"] = _lbs_to_kgs(m_dot_lb_s)

    # Build tool_input
    config = "A" if topology in ("recuperated", "basic") else "D"
    tool_input = {
        "config": config,
        "working_fluid": "isopentane",
        "procurement_strategy": procurement_strategy,
        "turbine_trains": n_trains,
    }
    if topology == "recuperated":
        tool_input["recuperator_approach_delta_F"] = db.get("dt_pinch_recuperator", 15)
    elif topology == "basic":
        tool_input["recuperator_approach_delta_F"] = 999.0
    elif topology == "dual_pressure":
        tool_input["config"] = "D"
        tool_input["recuperator_approach_delta_F"] = db.get("dt_pinch_recuperator", 15)

    # Pass through pinch points from design basis
    for src, dst in [("dt_pinch_vaporizer", "evaporator_approach_delta_F"),
                     ("dt_pinch_preheater", "preheater_approach_delta_F"),
                     ("dt_pinch_acc", "acc_approach_delta_F")]:
        if src in db:
            tool_input[dst] = db[src]

    # First run
    output = run_orc_analysis(tool_input, db)
    achieved_mw = output.get("net_power_MW", 0)
    iterations = 1

    # Iterate m_dot if MW off by more than tolerance
    if output.get("converged", False) and achieved_mw > 0:
        for _ in range(MW_SCALING_MAX_ITER):
            ratio = point.target_MW / achieved_mw
            if abs(ratio - 1.0) <= MW_SCALING_TOLERANCE:
                break
            m_dot_lb_s *= ratio
            db["brine_flow_kg_s"] = _lbs_to_kgs(m_dot_lb_s)
            output = run_orc_analysis(tool_input, db)
            achieved_mw = output.get("net_power_MW", 0)
            iterations += 1
            if achieved_mw <= 0:
                break

    duration = time.time() - t_start
    detail = output.get("_detail", {})
    power_bal = detail.get("power_balance", {})

    # Equipment sizing extraction
    vap_area = detail.get("vaporizer_area_ft2", 0)
    pre_area = detail.get("preheater_area_ft2", 0)
    recup_area = detail.get("recuperator_area_ft2", 0)
    acc_bays = detail.get("acc_n_bays", 0)
    acc_area = detail.get("acc_area_ft2", 0)

    gross_mw = output.get("gross_power_MW", 0)
    turb_mw_each = gross_mw / n_trains if gross_mw > 0 and n_trains > 0 else 0

    # Pump sizing
    pump_gpm = detail.get("pump_iso_flow_gpm", 0)
    pump_hp = detail.get("pump_iso_power_hp", 0)
    pump_dp = detail.get("pump_iso_dP_psi", 0)
    # Estimate pump head from dP and fluid density (~35 lb/ft³ for isopentane)
    rho_iso = 35.0
    pump_head_ft = pump_dp * 144.0 / rho_iso if pump_dp > 0 else 0
    # Number of pumps: 1 running + 1 spare per train up to 3000 gpm each
    n_pumps = max(1, math.ceil(pump_gpm / 3000)) if pump_gpm > 0 else 0
    gpm_each = pump_gpm / n_pumps if n_pumps > 0 else 0

    # Vaporizer duty from Q_reject + net (rough: Q_in ≈ gross / efficiency)
    eff = output.get("cycle_efficiency", 0)
    q_in_mmbtu = (gross_mw * 1000 / 0.29307107) if eff > 0 and gross_mw > 0 else 0
    q_reject_mmbtu = detail.get("Q_reject_mmbtu_hr", 0)
    # Better: preheater + vaporizer duty = Q_in to cycle
    vap_duty = q_in_mmbtu * 0.55 if q_in_mmbtu > 0 else 0    # ~55% of heat input
    pre_duty = q_in_mmbtu * 0.45 if q_in_mmbtu > 0 else 0    # ~45% of heat input

    row = {
        "condition_id": point.condition_id,
        "brine_inlet_F": point.brine_inlet_F,
        "ambient_F": point.ambient_F,
        "target_MW": point.target_MW,
        "m_dot_geo_lb_s": round(m_dot_lb_s, 1),
        "topology": topology,
        "procurement_strategy": procurement_strategy,
        "converged": output.get("converged", False),
        "iterations": iterations,
        "net_MW": round(output.get("net_power_MW", 0), 3),
        "gross_MW": round(gross_mw, 3),
        "cycle_efficiency": round(eff, 4),
        "capex_per_kW": round(output.get("capex_per_kW", 0), 0),
        "lcoe": round(output.get("lcoe_per_MWh", 0), 2),
        "capex_total_USD": round(output.get("capex_total_USD", 0), 0),
        "equipment_per_kW": round(output.get("equipment_per_kW", 0), 0),
        # Turbine
        "turbine_n_units": n_trains,
        "turbine_mw_each": round(turb_mw_each, 2),
        "T_evap_F": round(detail.get("T_evap_F", 0), 1),
        "P_high_psia": round(detail.get("P_high_psia", 0), 1),
        "T_cond_F": round(detail.get("T_cond_F", 0), 1),
        "P_low_psia": round(detail.get("P_low_psia", 0), 1),
        # Vaporizer
        "vaporizer_area_ft2": round(vap_area, 0),
        "vaporizer_duty_mmbtu": round(vap_duty, 1),
        "vaporizer_n_shells": _hx_shells(vap_area),
        "vaporizer_shell_dia_in": round(_hx_shell_dia_in(vap_area), 0),
        # Preheater
        "preheater_area_ft2": round(pre_area, 0),
        "preheater_duty_mmbtu": round(pre_duty, 1),
        "preheater_n_shells": _hx_shells(pre_area),
        "preheater_shell_dia_in": round(_hx_shell_dia_in(pre_area), 0),
        # Recuperator
        "recuperator_area_ft2": round(recup_area, 0),
        # ACC
        "acc_n_bays": acc_bays,
        "acc_area_ft2": round(acc_area, 0),
        "acc_fan_kw": round(detail.get("fan_W_fans_kw", 0), 0),
        # WF Pump
        "pump_n_pumps": n_pumps,
        "pump_gpm_each": round(gpm_each, 0),
        "pump_head_ft": round(pump_head_ft, 0),
        "pump_hp_each": round(pump_hp / n_pumps if n_pumps > 0 else 0, 0),
        # IHX
        "ihx_area_ft2": round(detail.get("intermediate_hx_area_ft2", 0), 0),
        # Structural
        "structural_steel_lb": round(detail.get("structural_steel_weight_lb", 0), 0),
        # WF flow
        "m_dot_iso_lb_hr": round(detail.get("m_dot_iso_lb_hr", 0), 0),
        # Cost line items
        "cost_turbine": round(detail.get("cost_turbine_generator", 0), 0),
        "cost_vaporizer": round(detail.get("cost_vaporizer", 0), 0),
        "cost_preheater": round(detail.get("cost_preheater", 0), 0),
        "cost_recuperator": round(detail.get("cost_recuperator", 0), 0),
        "cost_acc": round(detail.get("cost_acc", 0), 0),
        "cost_pump": round(detail.get("cost_iso_pump", 0), 0),
        # Duration
        "run_duration_s": round(duration, 2),
    }
    return row


# ── CSV I/O ───────────────────────────────────────────────────────────────────

def init_csv(path: str = RESULTS_CSV):
    """Create CSV with header row (overwrites existing)."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writeheader()


def append_csv(row: dict, path: str = RESULTS_CSV):
    """Append one result row to CSV."""
    with open(path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CSV_COLUMNS)
        writer.writerow({k: row.get(k, "") for k in CSV_COLUMNS})


def load_results(path: str = RESULTS_CSV) -> pd.DataFrame:
    """Load results CSV into a DataFrame."""
    if not os.path.exists(path):
        return pd.DataFrame(columns=CSV_COLUMNS)
    df = pd.read_csv(path)
    # Ensure boolean column
    if "converged" in df.columns:
        df["converged"] = df["converged"].astype(str).str.lower().isin(["true", "1"])
    return df


# ── Standardization analysis ──────────────────────────────────────────────────

# Component definitions: (csv_column for sizing, unit, component_type, cost_column)
COMPONENT_DEFS = [
    ("turbine_mw_each", "MW", "turbine", "cost_turbine"),
    ("vaporizer_area_ft2", "ft²", "vaporizer", "cost_vaporizer"),
    ("preheater_area_ft2", "ft²", "preheater", "cost_preheater"),
    ("recuperator_area_ft2", "ft²", "recuperator", "cost_recuperator"),
    ("acc_n_bays", "bays", "acc", "cost_acc"),
    ("pump_hp_each", "HP", "pump", "cost_pump"),
]


def find_standard_specs(
    values: list[float],
    condition_ids: list[str],
    component_type: str,
    sizing_param: str,
    sizing_unit: str,
    costs: list[float],
    conditions_df: pd.DataFrame,
    max_oversize_pct: float = DEFAULT_MAX_OVERSIZE_PCT,
    max_specs: int = DEFAULT_MAX_SPECS,
) -> StandardizationResult:
    """Find minimum number of standard specs to cover all conditions.

    Algorithm:
    1. Try N=1: design_rating = max(values) * 1.10
       If min(values) / design_rating > (1 - max_oversize_pct/100), one spec works.
    2. If not, try N=2: find optimal split minimizing total oversize waste.
    3. Cap at N=3.

    Also computes crossover sensitivity: how much does the crossover move
    if component costs change by ±10%?
    """
    if not values or all(v <= 0 for v in values):
        return StandardizationResult(
            component_type=component_type,
            sizing_param=sizing_param,
            sizing_unit=sizing_unit,
            n_specs=0,
            specs=[],
            one_spec_coverage_full=0,
            one_spec_coverage_5pct=0,
            total_waste_pct=0,
            recommendation="skip",
            reasoning=f"No valid {component_type} data across conditions.",
        )

    # Filter out zero values (e.g. recuperator when basic topology)
    valid_idx = [i for i, v in enumerate(values) if v > 0]
    if not valid_idx:
        return StandardizationResult(
            component_type=component_type, sizing_param=sizing_param,
            sizing_unit=sizing_unit, n_specs=0, specs=[],
            one_spec_coverage_full=0, one_spec_coverage_5pct=0,
            total_waste_pct=0, recommendation="skip",
            reasoning=f"All {component_type} values are zero (not applicable for this topology).",
        )

    vals = [values[i] for i in valid_idx]
    cids = [condition_ids[i] for i in valid_idx]
    costs_valid = [costs[i] for i in valid_idx] if costs else [0] * len(valid_idx)
    n = len(vals)

    sorted_pairs = sorted(zip(vals, cids, costs_valid), key=lambda x: x[0])
    sorted_vals = [p[0] for p in sorted_pairs]
    sorted_cids = [p[1] for p in sorted_pairs]
    sorted_costs = [p[2] for p in sorted_pairs]

    margin = 0.10
    v_max = max(sorted_vals)
    v_min = min(sorted_vals)

    # ── Try N=1 ───────────────────────────────────────────────────────
    design_1 = v_max * (1 + margin)
    oversize_at_min = 1.0 - (v_min / design_1) if design_1 > 0 else 0
    oversize_at_min_pct = oversize_at_min * 100

    # Coverage stats for one spec
    one_full = sum(1 for v in sorted_vals if (design_1 - v) / design_1 <= max_oversize_pct / 100) / n * 100
    one_5pct = sum(1 for v in sorted_vals if (design_1 - v) / design_1 <= 0.30) / n * 100

    if oversize_at_min_pct <= max_oversize_pct:
        # One spec works
        ref = EQUIP_REF.get(component_type, {})
        spec = ComponentSpec(
            component_type=component_type,
            spec_id=f"{component_type.upper()[:3]}-S1",
            sizing_param=sizing_param,
            sizing_unit=sizing_unit,
            min_rating=v_min,
            max_rating=v_max,
            design_rating=round(design_1, 2),
            margin_pct=margin * 100,
            condition_ids=list(sorted_cids),
            coverage_pct=100.0,
            vendors=ref.get("vendors", []),
            lead_weeks=ref.get("lead_wk", 0),
        )
        # Add shell info for HX types
        if sizing_unit == "ft²" and design_1 > 0:
            spec.n_shells = _hx_shells(design_1)
            spec.shell_dia_in = round(_hx_shell_dia_in(design_1), 0)

        waste = sum(design_1 - v for v in sorted_vals) / (design_1 * n) * 100 if design_1 > 0 else 0

        return StandardizationResult(
            component_type=component_type,
            sizing_param=sizing_param,
            sizing_unit=sizing_unit,
            n_specs=1,
            specs=[spec],
            one_spec_coverage_full=one_full,
            one_spec_coverage_5pct=one_5pct,
            total_waste_pct=round(waste, 1),
            recommendation="one_spec",
            reasoning=(
                f"One standard {component_type} spec (design rating {design_1:.1f} {sizing_unit}) "
                f"covers all {n} conditions with max oversize {oversize_at_min_pct:.0f}% "
                f"(within {max_oversize_pct:.0f}% tolerance). Average waste {waste:.1f}%."
            ),
        )

    # ── Try N=2 — find optimal split ──────────────────────────────────
    best_split = _find_best_split(sorted_vals, sorted_cids, sorted_costs,
                                  margin, conditions_df, component_type,
                                  sizing_param, sizing_unit)

    if max_specs >= 2 and best_split is not None:
        spec_lo, spec_hi, crossover_info = best_split
        waste = _total_waste([spec_lo, spec_hi], sorted_vals)

        return StandardizationResult(
            component_type=component_type,
            sizing_param=sizing_param,
            sizing_unit=sizing_unit,
            n_specs=2,
            specs=[spec_lo, spec_hi],
            one_spec_coverage_full=one_full,
            one_spec_coverage_5pct=one_5pct,
            total_waste_pct=round(waste, 1),
            recommendation="two_specs",
            reasoning=crossover_info,
        )

    # Fallback: single spec with warning
    ref = EQUIP_REF.get(component_type, {})
    spec = ComponentSpec(
        component_type=component_type,
        spec_id=f"{component_type.upper()[:3]}-S1",
        sizing_param=sizing_param, sizing_unit=sizing_unit,
        min_rating=v_min, max_rating=v_max,
        design_rating=round(design_1, 2), margin_pct=margin * 100,
        condition_ids=list(sorted_cids), coverage_pct=100.0,
        vendors=ref.get("vendors", []), lead_weeks=ref.get("lead_wk", 0),
    )
    waste = sum(design_1 - v for v in sorted_vals) / (design_1 * n) * 100 if design_1 > 0 else 0
    return StandardizationResult(
        component_type=component_type, sizing_param=sizing_param,
        sizing_unit=sizing_unit, n_specs=1, specs=[spec],
        one_spec_coverage_full=one_full, one_spec_coverage_5pct=one_5pct,
        total_waste_pct=round(waste, 1), recommendation="one_spec_with_penalty",
        reasoning=(
            f"One spec covers all conditions but max oversize is {oversize_at_min_pct:.0f}% "
            f"(exceeds {max_oversize_pct:.0f}% target). Consider two specs if budget allows."
        ),
    )


def _find_best_split(
    sorted_vals, sorted_cids, sorted_costs, margin,
    conditions_df, component_type, sizing_param, sizing_unit,
):
    """Find optimal split point for two specs. Returns (spec_lo, spec_hi, reasoning) or None."""
    n = len(sorted_vals)
    if n < 4:
        return None

    ref = EQUIP_REF.get(component_type, {})
    best_waste = float("inf")
    best_i = n // 2

    for i in range(2, n - 1):
        lo_design = sorted_vals[i - 1] * (1 + margin)
        hi_design = sorted_vals[-1] * (1 + margin)
        waste_lo = sum(lo_design - v for v in sorted_vals[:i]) / (lo_design * i) if lo_design > 0 else 0
        waste_hi = sum(hi_design - v for v in sorted_vals[i:]) / (hi_design * (n - i)) if hi_design > 0 else 0
        total = (waste_lo * i + waste_hi * (n - i)) / n
        if total < best_waste:
            best_waste = total
            best_i = i

    # Build specs
    lo_vals = sorted_vals[:best_i]
    hi_vals = sorted_vals[best_i:]
    lo_cids = sorted_cids[:best_i]
    hi_cids = sorted_cids[best_i:]

    lo_design = max(lo_vals) * (1 + margin)
    hi_design = max(hi_vals) * (1 + margin)

    prefix = component_type.upper()[:3]
    spec_lo = ComponentSpec(
        component_type=component_type,
        spec_id=f"{prefix}-S1",
        sizing_param=sizing_param, sizing_unit=sizing_unit,
        min_rating=min(lo_vals), max_rating=max(lo_vals),
        design_rating=round(lo_design, 2), margin_pct=margin * 100,
        condition_ids=list(lo_cids),
        coverage_pct=round(len(lo_cids) / n * 100, 1),
        vendors=ref.get("vendors", []), lead_weeks=ref.get("lead_wk", 0),
    )
    spec_hi = ComponentSpec(
        component_type=component_type,
        spec_id=f"{prefix}-S2",
        sizing_param=sizing_param, sizing_unit=sizing_unit,
        min_rating=min(hi_vals), max_rating=max(hi_vals),
        design_rating=round(hi_design, 2), margin_pct=margin * 100,
        condition_ids=list(hi_cids),
        coverage_pct=round(len(hi_cids) / n * 100, 1),
        vendors=ref.get("vendors", []), lead_weeks=ref.get("lead_wk", 0),
    )

    # Add shell info for HX types
    for spec in [spec_lo, spec_hi]:
        if sizing_unit == "ft²" and spec.design_rating > 0:
            spec.n_shells = _hx_shells(spec.design_rating)
            spec.shell_dia_in = round(_hx_shell_dia_in(spec.design_rating), 0)

    # ── Crossover analysis ────────────────────────────────────────────
    crossover_val = (sorted_vals[best_i - 1] + sorted_vals[best_i]) / 2
    crossover_reasoning = _crossover_sensitivity(
        sorted_vals, sorted_cids, sorted_costs,
        best_i, conditions_df, component_type,
        lo_design, hi_design, margin,
    )

    spec_lo.crossover_value = crossover_val
    spec_hi.crossover_value = crossover_val

    return spec_lo, spec_hi, crossover_reasoning


def _crossover_sensitivity(
    sorted_vals, sorted_cids, sorted_costs,
    split_i, conditions_df, component_type,
    lo_design, hi_design, margin,
):
    """Compute ±10% cost sensitivity on the crossover point.

    If component costs vary by ±10%, where does the optimal split move?
    A narrow band (±2°F) = robust crossover.
    A wide band (±15°F) = sensitive to cost assumptions, needs more data.
    """
    n = len(sorted_vals)

    # Identify which condition parameter drives the crossover
    # Look at the conditions near the split point
    if conditions_df is not None and len(conditions_df) > 0:
        # Find the driving parameter: which condition variable changes most across the split
        lo_ids = set(sorted_cids[:split_i])
        hi_ids = set(sorted_cids[split_i:])

        lo_rows = conditions_df[conditions_df["condition_id"].isin(lo_ids)]
        hi_rows = conditions_df[conditions_df["condition_id"].isin(hi_ids)]

        crossover_param = "brine_inlet_F"
        crossover_value = 0
        if len(lo_rows) > 0 and len(hi_rows) > 0:
            # Check which parameter has the cleanest separation
            for param in ["brine_inlet_F", "ambient_F", "target_MW"]:
                if param in lo_rows.columns and param in hi_rows.columns:
                    lo_mean = lo_rows[param].mean()
                    hi_mean = hi_rows[param].mean()
                    if abs(hi_mean - lo_mean) > abs(crossover_value):
                        crossover_value = (lo_mean + hi_mean) / 2
                        crossover_param = param
    else:
        crossover_param = "unknown"
        crossover_value = (sorted_vals[split_i - 1] + sorted_vals[split_i]) / 2

    # ── ±10% cost perturbation ────────────────────────────────────
    # Perturb component costs by ±10% and re-find optimal split
    # The idea: if the low-spec costs 10% more (making two specs more expensive
    # relative to oversize), does the split point move?
    sensitivity_band = 0.0
    split_moves = []

    for cost_mult in [0.90, 1.10]:
        # Perturbed costs change the "waste cost" calculation
        # More expensive components → less tolerance for oversize → split moves toward smaller spec
        perturbed_best_i = split_i
        perturbed_best_waste = float("inf")

        for i in range(2, n - 1):
            lo_d = sorted_vals[i - 1] * (1 + margin)
            hi_d = sorted_vals[-1] * (1 + margin)
            # Cost of oversize = (oversize fraction) × component cost × cost_mult
            waste_lo = sum((lo_d - v) / lo_d * cost_mult for v in sorted_vals[:i]) / i if lo_d > 0 else 0
            waste_hi = sum((hi_d - v) / hi_d * cost_mult for v in sorted_vals[i:]) / (n - i) if hi_d > 0 else 0
            # Two-spec inventory penalty: higher cost mult → more expensive to carry two specs
            inventory_penalty = 0.02 * cost_mult  # 2% base penalty for dual inventory
            total = (waste_lo * i + waste_hi * (n - i)) / n + inventory_penalty
            if total < perturbed_best_waste:
                perturbed_best_waste = total
                perturbed_best_i = i

        split_moves.append(perturbed_best_i)

    # Convert split index movement to condition parameter movement
    if conditions_df is not None and len(conditions_df) > 0 and crossover_param in conditions_df.columns:
        # Map split indices back to condition values
        base_crossover = crossover_value
        move_values = []
        for move_i in split_moves:
            if 0 < move_i < n:
                move_ids = set(sorted_cids[:move_i])
                move_rows = conditions_df[conditions_df["condition_id"].isin(move_ids)]
                other_ids = set(sorted_cids[move_i:])
                other_rows = conditions_df[conditions_df["condition_id"].isin(other_ids)]
                if len(move_rows) > 0 and len(other_rows) > 0:
                    move_val = (move_rows[crossover_param].max() + other_rows[crossover_param].min()) / 2
                    move_values.append(move_val)
        if move_values:
            sensitivity_band = max(abs(v - base_crossover) for v in move_values)
    else:
        # Fall back to index-based estimate
        idx_spread = max(abs(m - split_i) for m in split_moves) if split_moves else 0
        sensitivity_band = idx_spread * 5  # rough: 5°F per index step

    # ── Build reasoning string ─────────────────────────────────────
    param_label = crossover_param.replace("_", " ").replace(" F", " °F").replace("MW", " MW")
    band_label = "°F" if "F" in crossover_param else ("MW" if "MW" in crossover_param else "units")

    if sensitivity_band <= 5:
        robustness = "ROBUST — crossover is well-defined"
    elif sensitivity_band <= 15:
        robustness = "MODERATE — crossover is reasonably stable"
    else:
        robustness = "SENSITIVE — crossover depends heavily on cost assumptions, needs more data"

    reasoning = (
        f"Two specs recommended. Crossover at ~{crossover_value:.0f} {param_label}. "
        f"Spec 1 ({lo_design:.1f} {sorted_vals[0]:.0f}–{sorted_vals[split_i-1]:.0f} range) "
        f"covers {split_i}/{n} conditions ({split_i/n*100:.0f}%). "
        f"Spec 2 ({hi_design:.1f} {sorted_vals[split_i]:.0f}–{sorted_vals[-1]:.0f} range) "
        f"covers {n-split_i}/{n} conditions ({(n-split_i)/n*100:.0f}%). "
        f"±10% cost sensitivity: crossover moves ±{sensitivity_band:.0f} {band_label}. "
        f"{robustness}."
    )

    return reasoning


def _total_waste(specs, all_vals):
    """Average oversize percentage across all values assigned to their covering spec."""
    total = 0
    n = 0
    for v in all_vals:
        if v <= 0:
            continue
        # Find smallest spec that covers this value
        covering = None
        for s in specs:
            if v <= s.design_rating and (covering is None or s.design_rating < covering.design_rating):
                covering = s
        if covering and covering.design_rating > 0:
            total += (covering.design_rating - v) / covering.design_rating
            n += 1
    return (total / n * 100) if n > 0 else 0


def analyze_standardization(
    df: pd.DataFrame,
    max_oversize_pct: float = DEFAULT_MAX_OVERSIZE_PCT,
    max_specs: int = DEFAULT_MAX_SPECS,
) -> list[StandardizationResult]:
    """Run standardization analysis for all component types."""
    converged = df[df["converged"] == True].copy()
    if len(converged) == 0:
        return []

    results = []
    for sizing_col, unit, comp_type, cost_col in COMPONENT_DEFS:
        if sizing_col not in converged.columns:
            continue
        values = converged[sizing_col].fillna(0).tolist()
        cids = converged["condition_id"].tolist()
        costs = converged[cost_col].fillna(0).tolist() if cost_col in converged.columns else []

        result = find_standard_specs(
            values=values,
            condition_ids=cids,
            component_type=comp_type,
            sizing_param=sizing_col,
            sizing_unit=unit,
            costs=costs,
            conditions_df=converged,
            max_oversize_pct=max_oversize_pct,
            max_specs=max_specs,
        )
        results.append(result)

    return results


# ── Assembly grammar ──────────────────────────────────────────────────────────

def build_assembly_grammar(
    df: pd.DataFrame,
    std_results: list[StandardizationResult],
) -> list[AssemblyRule]:
    """Build lookup table mapping condition bands to component spec sets."""
    converged = df[df["converged"] == True].copy()
    if len(converged) == 0:
        return []

    # Build spec assignment: for each condition, which spec covers each component
    spec_map = {}  # condition_id -> {component_type: spec_id}
    for sr in std_results:
        if sr.n_specs == 0:
            continue
        for spec in sr.specs:
            for cid in spec.condition_ids:
                if cid not in spec_map:
                    spec_map[cid] = {}
                spec_map[cid][sr.component_type] = spec.spec_id

    # Group by unique spec combinations
    combo_groups = {}  # frozenset of (comp, spec_id) -> list of rows
    for _, row in converged.iterrows():
        cid = row["condition_id"]
        spec_set = spec_map.get(cid, {})
        key = frozenset(spec_set.items())
        if key not in combo_groups:
            combo_groups[key] = []
        combo_groups[key].append(row)

    # Build rules
    rules = []
    for combo_key, rows in combo_groups.items():
        rows_df = pd.DataFrame(rows)
        brine_lo = rows_df["brine_inlet_F"].min()
        brine_hi = rows_df["brine_inlet_F"].max()
        amb_lo = rows_df["ambient_F"].min()
        amb_hi = rows_df["ambient_F"].max()
        mw_lo = rows_df["target_MW"].min()
        mw_hi = rows_df["target_MW"].max()

        specs_dict = dict(combo_key)
        avg_capex = rows_df["capex_per_kW"].mean()

        rules.append(AssemblyRule(
            brine_band=f"{brine_lo:.0f}–{brine_hi:.0f}°F" if brine_lo != brine_hi else f"{brine_lo:.0f}°F",
            ambient_band=f"{amb_lo:.0f}–{amb_hi:.0f}°F" if amb_lo != amb_hi else f"{amb_lo:.0f}°F",
            mw_band=f"{mw_lo:.0f}–{mw_hi:.0f} MW" if mw_lo != mw_hi else f"{mw_lo:.0f} MW",
            component_specs=specs_dict,
            estimated_capex_per_kw=round(avg_capex, 0),
            condition_count=len(rows),
        ))

    rules.sort(key=lambda r: r.estimated_capex_per_kw)
    return rules


# ── Commercial summary ────────────────────────────────────────────────────────

def generate_commercial_summary(
    std_results: list[StandardizationResult],
    program_mw: float,
    delivery_years: float,
    unit_size_mw: float = 50.0,
) -> list[dict]:
    """Generate per-component frame contract summary."""
    n_units = max(1, round(program_mw / unit_size_mw))
    annual_units = n_units / delivery_years if delivery_years > 0 else n_units

    summaries = []
    for sr in std_results:
        if sr.n_specs == 0:
            continue
        for spec in sr.specs:
            # Estimate quantity per unit from typical counts
            qty_per_unit = _estimate_qty_per_unit(spec)
            total_qty = qty_per_unit * n_units
            annual_qty = qty_per_unit * annual_units

            summaries.append({
                "component": spec.component_type,
                "spec_id": spec.spec_id,
                "design_rating": f"{spec.design_rating:.1f} {spec.sizing_unit}",
                "qty_per_unit": qty_per_unit,
                "total_program_qty": round(total_qty),
                "annual_qty": round(annual_qty, 1),
                "vendors": ", ".join(spec.vendors),
                "lead_weeks": spec.lead_weeks,
                "coverage_pct": spec.coverage_pct,
            })

    return summaries


def _estimate_qty_per_unit(spec: ComponentSpec) -> int:
    """Estimate how many of this component per ORC unit."""
    ct = spec.component_type
    if ct == "turbine":
        return 2  # default 2 trains
    elif ct in ("vaporizer", "preheater"):
        return max(1, spec.n_shells) if spec.n_shells > 0 else 4
    elif ct == "recuperator":
        return max(1, spec.n_shells) if spec.n_shells > 0 else 2
    elif ct == "acc":
        return 1  # 1 ACC system (multiple bays counted separately)
    elif ct == "pump":
        return 3  # 2 running + 1 spare typical
    return 1
