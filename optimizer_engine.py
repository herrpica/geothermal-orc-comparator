"""
Autonomous ORC Optimizer Engine.

Systematically searches working fluids, cycle topologies, heat rejection
strategies, and pinch points to find configurations meeting the $2,000/kW
installed cost target on a 10-12 month construction schedule.

Uses the existing CoolProp engine via analysis_bridge.run_orc_analysis().
"""

import hashlib
import json
import logging
import os
import random
import time
from dataclasses import dataclass, field, asdict
from typing import Optional

logger = logging.getLogger(__name__)

DATA_DIR = os.path.join(os.path.dirname(__file__), "knowledge", "data")
RESULTS_PATH = os.path.join(DATA_DIR, "optimizer_results.json")

# ── Search Space Constants ─────────────────────────────────────────────────

WORKING_FLUIDS = ["isopentane", "isobutane", "propane", "r245fa", "cyclopentane"]
TOPOLOGIES = ["basic", "recuperated", "dual_pressure"]
HEAT_REJECTIONS = ["direct_acc", "propane_intermediate", "hybrid_wet_dry"]
VAPORIZER_PINCHES = [5, 8, 10, 12, 15, 20]
ACC_APPROACHES = [10, 15, 20, 25, 30]
PREHEATER_PINCHES = [5, 10, 15]
RECUPERATOR_PINCHES = [10, 15, 20, 25]

PROCUREMENT_STRATEGIES = ["oem_lump_sum", "direct_lump_sum", "oem_self_perform", "direct_self_perform"]

TARGET_CAPEX_PER_KW = 2000.0
TARGET_SCHEDULE_WEEKS = 70
TARGET_MIN_NET_MW = 53.0
TARGET_MAX_NET_MW = 57.0

# ── Complexity Penalties (NPV of lifecycle costs beyond minimum viable design) ──
# Per-component operational complexity, expressed as $/kW over 20yr at 8% discount rate.
# These capture O&M, training, spare parts, and failure mode costs not in capital $/kW.

COMPLEXITY_PENALTIES = {
    "recuperator": 18,          # thermal cycling, fluid inventory, pinch degradation
    "propane_intermediate": 50, # leak risk, regulatory, fluid mgmt, isolation valves
    "dual_pressure": 100,       # 2x rotating equipment, 2x HX sets, 2x controls
    "hybrid_wet_dry": 30,       # water treatment, seasonal switchover, freeze protection
}


def calculate_complexity_penalty(cfg: "OptConfig") -> float:
    """Return total complexity $/kW penalty based on topology and heat rejection.

    basic topology + direct_acc = 0 (minimum viable design).
    Penalties are additive.
    """
    penalty = 0.0
    # Topology penalties
    if cfg.topology == "recuperated":
        penalty += COMPLEXITY_PENALTIES["recuperator"]
    elif cfg.topology == "dual_pressure":
        penalty += COMPLEXITY_PENALTIES["dual_pressure"]

    # Heat rejection penalties
    if cfg.heat_rejection == "propane_intermediate":
        penalty += COMPLEXITY_PENALTIES["propane_intermediate"]
    elif cfg.heat_rejection == "hybrid_wet_dry":
        penalty += COMPLEXITY_PENALTIES["hybrid_wet_dry"]

    return penalty


# ── Data Classes ──────────────────────────────────────────────────────────

@dataclass
class OptConfig:
    """One point in the search space."""
    working_fluid: str
    topology: str            # basic / recuperated
    heat_rejection: str      # direct_acc / propane_intermediate
    vaporizer_pinch_F: float
    acc_approach_F: float
    preheater_pinch_F: float
    recuperator_pinch_F: float  # 999 for basic topology
    procurement_strategy: str = "oem_lump_sum"

    def config_hash(self) -> str:
        """SHA256[:16] for deduplication."""
        key = (
            f"{self.working_fluid}|{self.topology}|{self.heat_rejection}|"
            f"{self.vaporizer_pinch_F}|{self.acc_approach_F}|"
            f"{self.preheater_pinch_F}|{self.recuperator_pinch_F}|"
            f"{self.procurement_strategy}"
        )
        return hashlib.sha256(key.encode()).hexdigest()[:16]

    def to_tool_input(self) -> dict:
        """Convert to run_orc_analysis tool_input format."""
        # hybrid_wet_dry uses Config A thermo model; propane_intermediate uses Config B
        if self.heat_rejection in ("direct_acc", "hybrid_wet_dry"):
            config = "A"
        else:
            config = "B"
        ti = {
            "config": config,
            "working_fluid": self.working_fluid,
            "evaporator_approach_delta_F": self.vaporizer_pinch_F,
            "acc_approach_delta_F": self.acc_approach_F,
            "preheater_approach_delta_F": self.preheater_pinch_F,
            "procurement_strategy": self.procurement_strategy,
        }
        # dual_pressure uses same thermo as recuperated
        if self.topology in ("recuperated", "dual_pressure"):
            ti["recuperator_approach_delta_F"] = self.recuperator_pinch_F
        else:
            # No recuperation: set very large pinch to effectively disable
            ti["recuperator_approach_delta_F"] = 999.0
        return ti

    def is_valid(self) -> bool:
        """Filter invalid combinations."""
        # Can't use propane as working fluid with propane intermediate loop
        if self.working_fluid == "propane" and self.heat_rejection == "propane_intermediate":
            return False
        # Validate procurement strategy
        if self.procurement_strategy not in PROCUREMENT_STRATEGIES:
            return False
        return True

    def label(self) -> str:
        """Human-readable label."""
        topo_map = {"recuperated": "recup", "basic": "basic", "dual_pressure": "dual"}
        hr_map = {"direct_acc": "ACC", "propane_intermediate": "IHX", "hybrid_wet_dry": "HYB"}
        strat_map = {
            "oem_lump_sum": "OEM+LS", "direct_lump_sum": "DIR+LS",
            "oem_self_perform": "OEM+SP", "direct_self_perform": "DIR+SP",
        }
        topo = topo_map.get(self.topology, self.topology)
        hr = hr_map.get(self.heat_rejection, self.heat_rejection)
        strat = strat_map.get(self.procurement_strategy, self.procurement_strategy)
        return f"{self.working_fluid}/{topo}/{hr}/{strat}/vap{self.vaporizer_pinch_F}/acc{self.acc_approach_F}"


@dataclass
class OptResult:
    """Output from one optimizer run."""
    run_id: int
    config_hash: str
    config: dict
    converged: bool
    net_power_MW: float = 0.0
    gross_power_MW: float = 0.0
    cycle_efficiency: float = 0.0
    capex_total_USD: float = 0.0
    capex_per_kW: float = 0.0
    equipment_per_kW: float = 0.0   # equipment package only (vendor bid comparable)
    construction_weeks: int = 0
    schedule_fit: bool = False
    target_fit: bool = False
    pareto_optimal: bool = False
    npv_USD: float = 0.0
    lcoe_per_MWh: float = 0.0
    complexity_per_kW: float = 0.0
    total_adjusted_per_kW: float = 0.0  # capex_per_kW + complexity_per_kW
    procurement_strategy: str = "oem_lump_sum"
    bom_per_kw: dict = field(default_factory=dict)
    parasitic_breakdown: dict = field(default_factory=dict)
    thermal_detail: dict = field(default_factory=dict)
    schedule_phases: list = field(default_factory=list)
    acc_n_bays: int = 0
    warnings: list = field(default_factory=list)
    error: str = ""
    timestamp: float = 0.0
    duration_seconds: float = 0.0


# ── Result Store (JSON persistence) ──────────────────────────────────────

class ResultStore:
    """Persistent storage for optimizer results."""

    def __init__(self, path: str = RESULTS_PATH):
        self.path = path
        self.results: list[OptResult] = []
        self._hashes: set[str] = set()
        self._load()

    @staticmethod
    def _coerce_bool(val) -> bool:
        """Safely coerce a value to bool (handles string 'False'/'True' from JSON)."""
        if isinstance(val, str):
            return val.lower() == "true"
        return bool(val)

    def _load(self):
        if os.path.exists(self.path):
            try:
                with open(self.path, "r", encoding="utf-8") as f:
                    data = json.load(f)
                self.results = []
                _bool_fields = {"converged", "schedule_fit", "target_fit", "pareto_optimal"}
                for d in data:
                    clean = {k: v for k, v in d.items()
                             if k in OptResult.__dataclass_fields__}
                    # Coerce boolean fields that may have been serialized as strings
                    for bf in _bool_fields:
                        if bf in clean:
                            clean[bf] = self._coerce_bool(clean[bf])
                    r = OptResult(**clean)
                    self.results.append(r)
                    self._hashes.add(r.config_hash)
            except (json.JSONDecodeError, TypeError):
                self.results = []
                self._hashes = set()

    def _save(self):
        os.makedirs(os.path.dirname(self.path), exist_ok=True)
        rows = []
        for r in self.results:
            d = asdict(r)
            # Ensure booleans are actual bools, not numpy.bool_ or other types
            for bf in ("converged", "schedule_fit", "target_fit", "pareto_optimal"):
                if bf in d:
                    d[bf] = bool(d[bf])
            rows.append(d)
        with open(self.path, "w", encoding="utf-8") as f:
            json.dump(rows, f, indent=2, default=str)

    def has_config(self, config_hash: str) -> bool:
        return config_hash in self._hashes

    def add_result(self, result: OptResult):
        self.results.append(result)
        self._hashes.add(result.config_hash)
        self._save()

    def next_run_id(self) -> int:
        return len(self.results) + 1

    def get_converged(self) -> list[OptResult]:
        return [r for r in self.results if r.converged]

    def get_pareto(self) -> list[OptResult]:
        return [r for r in self.results if r.pareto_optimal]

    def get_best_per_kw(self) -> Optional[OptResult]:
        converged = self.get_converged()
        if not converged:
            return None
        return min(converged, key=lambda r: r.capex_per_kW)

    def get_best_adjusted(self) -> Optional[OptResult]:
        """Best config by total adjusted $/kW (installed + complexity NPV)."""
        converged = self.get_converged()
        if not converged:
            return None
        return min(converged, key=lambda r: r.total_adjusted_per_kW)

    def get_best_npv(self) -> Optional[OptResult]:
        converged = self.get_converged()
        if not converged:
            return None
        return max(converged, key=lambda r: r.npv_USD)

    def get_target_hits(self) -> list[OptResult]:
        return [r for r in self.results if r.target_fit]

    def reset(self):
        self.results = []
        self._hashes = set()
        if os.path.exists(self.path):
            os.remove(self.path)

    def stats(self) -> dict:
        total = len(self.results)
        converged = len(self.get_converged())
        failed = total - converged
        target_hits = len(self.get_target_hits())
        best = self.get_best_per_kw()
        best_adj = self.get_best_adjusted()
        return {
            "total_runs": total,
            "converged": converged,
            "failed": failed,
            "target_hits": target_hits,
            "best_capex_per_kw": best.capex_per_kW if best else None,
            "best_adjusted_per_kw": best_adj.total_adjusted_per_kW if best_adj else None,
            "best_config": best.config if best else None,
            "pareto_count": len(self.get_pareto()),
        }


# ── Pareto Frontier ───────────────────────────────────────────────────────

def update_pareto_frontier(store: ResultStore):
    """Recompute Pareto-optimal flags: maximize efficiency, minimize adjusted $/kW."""
    converged = store.get_converged()
    if not converged:
        return

    # Reset all
    for r in store.results:
        r.pareto_optimal = False

    # A point is Pareto-optimal if no other point dominates it
    # (higher efficiency AND lower total_adjusted_per_kW)
    for r in converged:
        dominated = False
        for other in converged:
            if other is r:
                continue
            if (other.cycle_efficiency >= r.cycle_efficiency and
                other.total_adjusted_per_kW <= r.total_adjusted_per_kW and
                (other.cycle_efficiency > r.cycle_efficiency or
                 other.total_adjusted_per_kW < r.total_adjusted_per_kW)):
                dominated = True
                break
        if not dominated:
            r.pareto_optimal = True

    store._save()


def reevaluate_targets(store: ResultStore):
    """Recompute schedule_fit, target_fit, pareto for all results against current thresholds.

    Also recomputes complexity_per_kW and total_adjusted_per_kW from config data.
    """
    for r in store.results:
        # Recompute complexity from stored config
        cfg_data = r.config or {}
        topo = cfg_data.get("topology", "basic")
        hr = cfg_data.get("heat_rejection", "direct_acc")
        complexity = 0.0
        if topo == "recuperated":
            complexity += COMPLEXITY_PENALTIES["recuperator"]
        elif topo == "dual_pressure":
            complexity += COMPLEXITY_PENALTIES["dual_pressure"]
        if hr == "propane_intermediate":
            complexity += COMPLEXITY_PENALTIES["propane_intermediate"]
        elif hr == "hybrid_wet_dry":
            complexity += COMPLEXITY_PENALTIES["hybrid_wet_dry"]
        r.complexity_per_kW = complexity
        r.total_adjusted_per_kW = r.capex_per_kW + complexity

        r.schedule_fit = bool(r.construction_weeks <= TARGET_SCHEDULE_WEEKS)
        r.target_fit = bool(
            r.total_adjusted_per_kW <= TARGET_CAPEX_PER_KW
            and r.schedule_fit
            and bool(r.net_power_MW >= TARGET_MIN_NET_MW)
            and bool(r.net_power_MW <= TARGET_MAX_NET_MW)
            and r.converged
        )
    update_pareto_frontier(store)


# ── Search Space Generation ───────────────────────────────────────────────

def generate_search_space(store: Optional[ResultStore] = None,
                          strategies: Optional[list[str]] = None,
                          heat_rejections: Optional[list[str]] = None) -> list[OptConfig]:
    """Generate full search space, filtered by validity and dedup.

    Smart ordering: promising fluids and moderate pinch points first.
    Strategies parameter allows limiting which procurement strategies to search.
    heat_rejections parameter allows limiting which heat rejection types to search.
    """
    if strategies is None:
        strategies = list(PROCUREMENT_STRATEGIES)
    if heat_rejections is None:
        heat_rejections = list(HEAT_REJECTIONS)

    configs = []

    # Order fluids by typical performance (best first for geothermal)
    fluid_order = ["isopentane", "isobutane", "r245fa", "cyclopentane", "propane"]
    # Order pinches: moderate values first (likely sweet spots)
    vap_order = [10, 8, 12, 5, 15, 20]
    acc_order = [15, 20, 10, 25, 30]
    pre_order = [10, 5, 15]
    recup_order = [15, 10, 20, 25]

    # Topologies that use recuperator pinch sweeps
    _recup_topos = {"recuperated", "dual_pressure"}

    for strat in strategies:
        for fluid in fluid_order:
            for topo in ["recuperated", "dual_pressure", "basic"]:
                for hr in heat_rejections:
                    for vap in vap_order:
                        for acc in acc_order:
                            for pre in pre_order:
                                if topo in _recup_topos:
                                    for recup in recup_order:
                                        cfg = OptConfig(fluid, topo, hr, vap, acc, pre, recup, strat)
                                        if cfg.is_valid():
                                            if store is None or not store.has_config(cfg.config_hash()):
                                                configs.append(cfg)
                                else:
                                    cfg = OptConfig(fluid, topo, hr, vap, acc, pre, 999.0, strat)
                                    if cfg.is_valid():
                                        if store is None or not store.has_config(cfg.config_hash()):
                                            configs.append(cfg)
    return configs


def total_search_space_size(strategies: Optional[list[str]] = None,
                            heat_rejections: Optional[list[str]] = None) -> int:
    """Count total valid configurations (without dedup)."""
    if strategies is None:
        strategies = list(PROCUREMENT_STRATEGIES)
    if heat_rejections is None:
        heat_rejections = list(HEAT_REJECTIONS)
    count = 0
    _recup_topos = {"recuperated", "dual_pressure"}
    for strat in strategies:
        for fluid in WORKING_FLUIDS:
            for topo in TOPOLOGIES:
                for hr in heat_rejections:
                    cfg = OptConfig(fluid, topo, hr, 10, 15, 10, 15, strat)
                    if not cfg.is_valid():
                        continue
                    n_vap = len(VAPORIZER_PINCHES)
                    n_acc = len(ACC_APPROACHES)
                    n_pre = len(PREHEATER_PINCHES)
                    if topo in _recup_topos:
                        count += n_vap * n_acc * n_pre * len(RECUPERATOR_PINCHES)
                    else:
                        count += n_vap * n_acc * n_pre
    return count


# ── Batch Runner ─────────────────────────────────────────────────────────

def run_single_config(cfg: OptConfig, design_basis: dict, store: ResultStore) -> OptResult:
    """Run one configuration through the analysis pipeline."""
    from analysis_bridge import run_orc_analysis

    run_id = store.next_run_id()
    config_hash = cfg.config_hash()
    tool_input = cfg.to_tool_input()
    t_start = time.time()

    try:
        output = run_orc_analysis(tool_input, design_basis)
    except Exception as e:
        duration = time.time() - t_start
        result = OptResult(
            run_id=run_id,
            config_hash=config_hash,
            config=asdict(cfg),
            converged=False,
            error=str(e),
            timestamp=time.time(),
            duration_seconds=round(duration, 2),
        )
        store.add_result(result)
        return result

    converged = output.get("converged", False)
    net_mw = output.get("net_power_MW", 0)
    gross_mw = output.get("gross_power_MW", 0)
    efficiency = output.get("cycle_efficiency", 0)
    capex = output.get("capex_total_USD", 0)
    capex_kw = output.get("capex_per_kW", 0)
    equip_kw = output.get("equipment_per_kW", 0)
    npv = output.get("npv_USD", 0)
    lcoe = output.get("lcoe_per_MWh", 0)
    weeks = output.get("construction_weeks_critical_path", 0)
    warnings = output.get("warnings", [])

    # ── Parametric adjustments for new config types ─────────────────
    from cost_model import COST_FACTORS as CF

    if cfg.heat_rejection == "hybrid_wet_dry" and converged:
        # Reduce ACC bays by 17%, add water system cost
        bay_reduction = CF["hybrid_acc_bay_reduction"]
        water_per_kw = CF["water_system_per_kw"]
        detail = output.get("_detail", {})
        original_acc = detail.get("cost_acc", 0)
        acc_n_bays = detail.get("acc_n_bays", 0)
        if acc_n_bays > 0:
            import math as _math
            n_bays_hybrid = _math.ceil(acc_n_bays * (1 - bay_reduction))
            acc_per_bay = detail.get("cost_acc", 0) / acc_n_bays if acc_n_bays > 0 else 347200
            new_acc_cost = n_bays_hybrid * acc_per_bay
        else:
            new_acc_cost = original_acc * (1 - bay_reduction)
        water_cost = water_per_kw * gross_mw * 1000
        acc_delta = new_acc_cost + water_cost - original_acc
        capex += acc_delta
        net_kw = net_mw * 1000 if net_mw > 0 else 1
        capex_kw = capex / net_kw if net_kw > 0 else capex_kw
        equip_kw_raw = output.get("equipment_cost_USD", 0) + acc_delta
        equip_kw = equip_kw_raw / net_kw if net_kw > 0 else equip_kw
        warnings.append("hybrid_wet_dry: ACC bays reduced 17%, +$50/kW water system")

    if cfg.topology == "dual_pressure" and converged:
        # Apply +3% efficiency gain and additional HX cost
        eff_gain = CF["dual_pressure_efficiency_gain"]
        hx_per_kw = CF["dual_pressure_hx_per_kw"]
        sched_add = CF["dual_pressure_schedule_weeks"]
        # Adjust efficiency and power
        old_eff = efficiency
        efficiency = old_eff + eff_gain
        if old_eff > 0:
            power_scale = efficiency / old_eff
            gross_mw *= power_scale
            net_mw *= power_scale
        # Additional HX cost
        dual_hx_cost = hx_per_kw * gross_mw * 1000
        capex += dual_hx_cost
        net_kw = net_mw * 1000 if net_mw > 0 else 1
        capex_kw = capex / net_kw if net_kw > 0 else capex_kw
        # Schedule impact
        weeks += sched_add
        warnings.append(f"dual_pressure: +{eff_gain*100:.0f}% eff, +${hx_per_kw}/kW HX, +{sched_add}wk")

    # BOM per kW breakdown (equipment + installation items)
    detail = output.get("_detail", {})
    net_kw = net_mw * 1000 if net_mw > 0 else 1
    bom_per_kw = {}
    for key in ["turbine_generator", "iso_pump", "vaporizer", "preheater",
                "recuperator", "acc", "ductwork", "structural_steel",
                "intermediate_hx", "propane_system",
                "bop_piping", "civil_structural", "ei_installation",
                "construction_labor", "engineering", "commissioning",
                "contingency", "equipment_subtotal"]:
        cost_val = detail.get(f"cost_{key}", 0)
        bom_per_kw[key] = round(cost_val / net_kw, 1) if net_kw > 0 else 0

    # ── Parasitic & thermal breakdown for detailed display ────────────
    power_bal = detail.get("power_balance", {})
    parasitic_breakdown = {
        "P_gross_kw": power_bal.get("P_gross", 0),
        "W_iso_pump_kw": power_bal.get("W_iso_pump", 0),
        "W_prop_pump_kw": power_bal.get("W_prop_pump", 0),
        "W_fans_kw": power_bal.get("W_fans", 0),
        "W_auxiliary_kw": power_bal.get("W_auxiliary", 0),
        "W_total_parasitic_kw": power_bal.get("W_total_parasitic", 0),
        "parasitic_pct": power_bal.get("parasitic_pct", 0),
    }
    thermal_detail_dict = {
        "T_evap_F": detail.get("T_evap_F", 0),
        "T_cond_F": detail.get("T_cond_F", 0),
        "P_high_psia": detail.get("P_high_psia", 0),
        "P_low_psia": detail.get("P_low_psia", 0),
        "pressure_ratio": detail.get("pressure_ratio", 0),
        "brine_effectiveness": power_bal.get("brine_effectiveness", 0),
    }

    # Schedule phases from construction_schedule output
    sched_output = output.get("_detail", {}).get("schedule_phases", [])

    # ACC bay count
    result_acc_n_bays = detail.get("acc_n_bays", 0) or int(
        detail.get("cost_acc", 0) / 347200) if detail.get("cost_acc", 0) > 0 else 0

    # ── Complexity penalty ─────────────────────────────────────────
    complexity_kw = calculate_complexity_penalty(cfg)
    total_adjusted_kw = round(capex_kw + complexity_kw, 0)

    schedule_fit = bool(weeks <= TARGET_SCHEDULE_WEEKS)
    net_power_ok = bool(net_mw >= TARGET_MIN_NET_MW)
    net_power_max_ok = bool(net_mw <= TARGET_MAX_NET_MW)
    target_fit = bool(
        total_adjusted_kw <= TARGET_CAPEX_PER_KW
        and schedule_fit
        and net_power_ok
        and net_power_max_ok
        and converged
    )

    duration = time.time() - t_start
    result = OptResult(
        run_id=run_id,
        config_hash=config_hash,
        config=asdict(cfg),
        converged=converged,
        net_power_MW=round(net_mw, 3),
        gross_power_MW=round(gross_mw, 3),
        cycle_efficiency=round(efficiency, 4),
        capex_total_USD=round(capex, 0),
        capex_per_kW=round(capex_kw, 0),
        equipment_per_kW=round(equip_kw, 0),
        complexity_per_kW=complexity_kw,
        total_adjusted_per_kW=total_adjusted_kw,
        construction_weeks=weeks,
        schedule_fit=schedule_fit,
        target_fit=target_fit,
        npv_USD=round(npv, 0),
        lcoe_per_MWh=round(lcoe, 1),
        procurement_strategy=cfg.procurement_strategy,
        bom_per_kw=bom_per_kw,
        parasitic_breakdown=parasitic_breakdown,
        thermal_detail=thermal_detail_dict,
        schedule_phases=sched_output,
        acc_n_bays=result_acc_n_bays,
        warnings=warnings,
        timestamp=time.time(),
        duration_seconds=round(duration, 2),
    )
    store.add_result(result)
    return result


def run_batch(configs: list[OptConfig], design_basis: dict,
              store: ResultStore, batch_size: int = 10) -> list[OptResult]:
    """Run a batch of configurations. Returns results for this batch."""
    results = []
    for cfg in configs[:batch_size]:
        if store.has_config(cfg.config_hash()):
            continue
        result = run_single_config(cfg, design_basis, store)
        results.append(result)
    return results


# ── Progress Report ──────────────────────────────────────────────────────

def generate_report(store: ResultStore) -> dict:
    """Generate a progress report with insights."""
    stats = store.stats()
    converged = store.get_converged()

    # Fluid rankings
    fluid_stats = {}
    for r in converged:
        fl = r.config.get("working_fluid", "unknown")
        if fl not in fluid_stats:
            fluid_stats[fl] = {"count": 0, "best_kw": float("inf"), "best_eff": 0, "avg_kw": 0}
        fluid_stats[fl]["count"] += 1
        fluid_stats[fl]["best_kw"] = min(fluid_stats[fl]["best_kw"], r.capex_per_kW)
        fluid_stats[fl]["best_eff"] = max(fluid_stats[fl]["best_eff"], r.cycle_efficiency)
        fluid_stats[fl]["avg_kw"] += r.capex_per_kW

    for fl in fluid_stats:
        n = fluid_stats[fl]["count"]
        fluid_stats[fl]["avg_kw"] = round(fluid_stats[fl]["avg_kw"] / n, 0) if n > 0 else 0

    # Topology comparison
    topo_stats = {}
    for r in converged:
        topo = r.config.get("topology", "unknown")
        if topo not in topo_stats:
            topo_stats[topo] = {"count": 0, "best_kw": float("inf")}
        topo_stats[topo]["count"] += 1
        topo_stats[topo]["best_kw"] = min(topo_stats[topo]["best_kw"], r.capex_per_kW)

    # Failure patterns
    failed = [r for r in store.results if not r.converged]
    failure_fluids = {}
    for r in failed:
        fl = r.config.get("working_fluid", "unknown")
        failure_fluids[fl] = failure_fluids.get(fl, 0) + 1

    # Insights
    insights = []
    best = store.get_best_per_kw()
    best_adj = store.get_best_adjusted()
    if best_adj:
        if best_adj.total_adjusted_per_kW <= TARGET_CAPEX_PER_KW:
            insights.append(
                f"TARGET HIT (adjusted): {best_adj.config.get('working_fluid')} at "
                f"${best_adj.total_adjusted_per_kW:,.0f}/kW "
                f"(${best_adj.capex_per_kW:,.0f} installed + ${best_adj.complexity_per_kW:,.0f} complexity)"
            )
        else:
            gap = best_adj.total_adjusted_per_kW - TARGET_CAPEX_PER_KW
            insights.append(
                f"Gap to target (adjusted): ${gap:,.0f}/kW "
                f"(best: ${best_adj.total_adjusted_per_kW:,.0f}/kW = "
                f"${best_adj.capex_per_kW:,.0f} installed + ${best_adj.complexity_per_kW:,.0f} complexity)"
            )
    if best and best_adj and best is not best_adj:
        insights.append(
            f"Best raw $/kW: {best.config.get('working_fluid')} at ${best.capex_per_kW:,.0f}/kW "
            f"(adjusted: ${best.total_adjusted_per_kW:,.0f}/kW)"
        )

    if fluid_stats:
        ranked = sorted(fluid_stats.items(), key=lambda x: x[1]["best_kw"])
        insights.append(f"Best fluid: {ranked[0][0]} (${ranked[0][1]['best_kw']:,.0f}/kW)")

    if failure_fluids:
        worst = max(failure_fluids.items(), key=lambda x: x[1])
        insights.append(f"Most failures: {worst[0]} ({worst[1]} failed)")

    # Strategy rankings
    strat_stats = {}
    for r in converged:
        strat = r.procurement_strategy
        if strat not in strat_stats:
            strat_stats[strat] = {"count": 0, "best_kw": float("inf"), "avg_kw": 0}
        strat_stats[strat]["count"] += 1
        strat_stats[strat]["best_kw"] = min(strat_stats[strat]["best_kw"], r.capex_per_kW)
        strat_stats[strat]["avg_kw"] += r.capex_per_kW
    for strat in strat_stats:
        n = strat_stats[strat]["count"]
        strat_stats[strat]["avg_kw"] = round(strat_stats[strat]["avg_kw"] / n, 0) if n > 0 else 0

    return {
        **stats,
        "fluid_rankings": fluid_stats,
        "topology_comparison": topo_stats,
        "strategy_rankings": strat_stats,
        "failure_patterns": failure_fluids,
        "insights": insights,
    }


# ── AI-Guided Optimizer ──────────────────────────────────────────────────

AI_GUIDED_MAX_ROUNDS = 30
AI_GUIDED_BATCH_SIZE = 10


def generate_seed_batch(store: ResultStore,
                        strategies: Optional[list[str]] = None,
                        heat_rejections: Optional[list[str]] = None) -> list[OptConfig]:
    """Return 10 diverse seed configs covering the design space.

    Covers strategies x key fluids with moderate pinches.
    Prioritizes isopentane/recuperated across all strategies to show cost impact.
    Deduplicates against already-run configs in store.
    """
    if strategies is None:
        strategies = list(PROCUREMENT_STRATEGIES)
    if heat_rejections is None:
        heat_rejections = list(HEAT_REJECTIONS)

    default_hr = heat_rejections[0]
    seeds = []

    # Priority seeds: isopentane/recuperated across all strategies (show procurement impact)
    for strat in strategies:
        cfg = OptConfig(
            working_fluid="isopentane",
            topology="recuperated",
            heat_rejection=default_hr,
            vaporizer_pinch_F=10,
            acc_approach_F=15,
            preheater_pinch_F=10,
            recuperator_pinch_F=15,
            procurement_strategy=strat,
        )
        if cfg.is_valid() and not store.has_config(cfg.config_hash()):
            seeds.append(cfg)

    # Fill remaining with fluid diversity on direct_self_perform (lowest cost strategy)
    for fluid in WORKING_FLUIDS:
        if len(seeds) >= AI_GUIDED_BATCH_SIZE:
            break
        if fluid == "isopentane":
            continue  # already covered above
        recup = 15.0
        cfg = OptConfig(fluid, "recuperated", default_hr, 10, 15, 10, recup,
                        "direct_self_perform")
        if cfg.is_valid() and not store.has_config(cfg.config_hash()):
            seeds.append(cfg)

    # If some seeds already run, fill with alternative pinch combos
    if len(seeds) < AI_GUIDED_BATCH_SIZE:
        alternates = [
            ("isopentane", "basic", 8, 20, 5, 999.0, "direct_self_perform"),
            ("isobutane", "recuperated", 12, 15, 10, 20, "direct_lump_sum"),
            ("r245fa", "basic", 10, 20, 5, 999.0, "oem_self_perform"),
            ("cyclopentane", "recuperated", 8, 15, 10, 15, "direct_self_perform"),
            ("propane", "basic", 12, 20, 10, 999.0, "direct_self_perform"),
        ]
        for fluid, topo, vap, acc, pre, recup, strat in alternates:
            if len(seeds) >= AI_GUIDED_BATCH_SIZE:
                break
            cfg = OptConfig(fluid, topo, default_hr, vap, acc, pre, recup, strat)
            if cfg.is_valid() and not store.has_config(cfg.config_hash()):
                seeds.append(cfg)

    return seeds[:AI_GUIDED_BATCH_SIZE]


def build_ai_system_prompt(design_basis: dict,
                            heat_rejections: Optional[list[str]] = None) -> str:
    """System prompt for the AI optimizer agent.

    Includes DBD design rules and complexity penalty framework.
    """
    # Load DBD for design rules (best-effort, fallback to empty)
    try:
        from design_basis_document import load_dbd
        dbd = load_dbd()
    except Exception:
        dbd = {}

    # Extract design philosophy summary
    philosophy_rules = ""
    sec2 = dbd.get("section_2_philosophy", {})
    values = sec2.get("values", [])
    if values:
        philosophy_rules = "\n".join(
            f"  - {v.get('name', '')}: {v.get('description', '')}"
            for v in values
        )

    # Extract equipment standards for ACC fan control
    sec3 = dbd.get("section_3_equipment", {})
    acc_standards = sec3.get("acc", {})
    fan_control = acc_standards.get("fan_control", "Staging only")
    fan_rationale = acc_standards.get("fan_control_rationale", "")

    return f"""You are an ORC power plant optimization engineer. Your job is to intelligently
explore the design space and find the lowest TOTAL ADJUSTED cost configuration that meets performance targets.

DESIGN BASIS (fixed site conditions):
- Brine inlet temperature: {design_basis.get('T_geo_in_F', 'N/A')} °F
- Minimum brine outlet: {design_basis.get('T_geo_out_min_F', 'N/A')} °F
- Brine flow rate: {design_basis.get('m_dot_geo_lb_s', 'N/A')} lb/s
- Ambient temperature: {design_basis.get('T_ambient_F', 'N/A')} °F
- Turbine isentropic efficiency: {design_basis.get('eta_turbine', 'N/A')}
- Pump isentropic efficiency: {design_basis.get('eta_pump', 'N/A')}

DESIGN RULES (from Design Basis Document):
{philosophy_rules}

  SPECIFIC RULES:
  1. Complexity burden of proof: Every component beyond basic/direct_acc must justify its lifecycle cost.
  2. Fan control: {fan_control}. {fan_rationale}
  3. Single turbine preferred unless dual-unit saves >$100/kW total lifecycle.
  4. Propane loop burden of proof: $50/kW complexity NPV must be offset by capital savings.
  5. 40ft ACC deck height baseline.
  6. Pipe schedule floors: ISO vapor Sch 10S, ISO liquid Sch 40, propane Sch 80.
  7. Information before assumption: flag data gaps, don't guess.

COMPLEXITY PENALTIES (NPV of lifecycle costs beyond minimum viable design):
  - recuperator:          ${COMPLEXITY_PENALTIES['recuperator']}/kW  (thermal cycling, fluid inventory, pinch degradation)
  - propane_intermediate: ${COMPLEXITY_PENALTIES['propane_intermediate']}/kW  (leak risk, regulatory, fluid mgmt, isolation valves)
  - dual_pressure:        ${COMPLEXITY_PENALTIES['dual_pressure']}/kW  (2x rotating equipment, 2x HX sets, 2x controls)
  - hybrid_wet_dry:       ${COMPLEXITY_PENALTIES['hybrid_wet_dry']}/kW  (water treatment, seasonal switchover, freeze protection)
  - basic + direct_acc:   $0/kW   (minimum viable — reference design)
  Penalties are additive: recuperated + propane = ${COMPLEXITY_PENALTIES['recuperator'] + COMPLEXITY_PENALTIES['propane_intermediate']}/kW.

IMPORTANT: Minimize total_adjusted_per_kW = installed_$/kW + complexity_$/kW, NOT just installed_$/kW.
A basic/ACC config at $1,600/kW beats a recuperated/propane config at $1,550/kW
because $1,550 + $68 complexity = $1,618 > $1,600.

SEARCH SPACE:
- Working fluids: {', '.join(WORKING_FLUIDS)}
- Topologies: basic, recuperated, dual_pressure
- Heat rejection: {', '.join(heat_rejections or HEAT_REJECTIONS)}
- Procurement strategies: {', '.join(PROCUREMENT_STRATEGIES)}
- Vaporizer pinch: 5-20 °F (integers)
- ACC approach: 10-30 °F (integers)
- Preheater pinch: 5-15 °F (integers)
- Recuperator pinch: 10-25 °F (integers, only for recuperated/dual_pressure; use 999 for basic)
- INVALID: propane fluid + propane_intermediate heat rejection

PROCUREMENT STRATEGIES (first-class optimizer variable):
- oem_lump_sum: OEM package + EPC contractor. Baseline ~$2,500/kW. Reflects OEM integration margin + contractor markup.
- direct_lump_sum: Direct vendor purchase + EPC contractor. Strips OEM integration margin (30-40% on equipment).
- oem_self_perform: OEM package + owner T&M self-perform. Eliminates contractor markup on construction.
- direct_self_perform: Direct vendor + owner self-perform. Both equipment and construction savings. Target ~$1,600-1,900/kW.
Procurement strategy changes cost factors but NOT thermodynamic performance. Same fluid/topology/pinch produces same power output but different $/kW.

TARGETS:
- Total adjusted cost: <= ${TARGET_CAPEX_PER_KW:,.0f}/kW (installed + complexity NPV)
- Construction schedule: <= {TARGET_SCHEDULE_WEEKS} weeks
- Net power range: {TARGET_MIN_NET_MW:.0f} – {TARGET_MAX_NET_MW:.0f} MW

INSTRUCTIONS:
- RIGHT-SIZING: Prefer configs closer to the minimum net power target over oversized designs. Configs above {TARGET_MAX_NET_MW:.0f} MW are NOT target hits.
- Analyze the results from previous rounds to identify patterns and promising regions
- Propose exactly {AI_GUIDED_BATCH_SIZE} new configurations to test next
- Focus on regions near the best-performing configs (exploit) while also testing new combinations (explore)
- Favor simpler topologies (basic > recuperated > dual_pressure) unless complexity is justified
- Avoid re-testing configurations that have already been run
- Set "converged" to true ONLY when further exploration is unlikely to improve on the best result
  (e.g., you've thoroughly tested variations around the optimum)

Return ONLY a JSON object with this structure:
{{
  "configs": [
    {{
      "working_fluid": "isopentane",
      "topology": "recuperated",
      "heat_rejection": "direct_acc",
      "procurement_strategy": "direct_self_perform",
      "vaporizer_pinch_F": 8,
      "acc_approach_F": 15,
      "preheater_pinch_F": 5,
      "recuperator_pinch_F": 15
    }}
  ],
  "round_summary": "Brief summary of strategy and what you learned this round",
  "insights": ["Key insight 1", "Key insight 2"],
  "converged": false
}}"""


def build_ai_context(store: ResultStore, round_number: int) -> str:
    """Build user message with all results so far for the AI optimizer."""
    stats = store.stats()
    converged = store.get_converged()
    all_results = store.results

    lines = [f"## Optimizer Status — Round {round_number}"]
    lines.append(f"Total configs tested: {stats['total_runs']}")
    lines.append(f"Converged: {stats['converged']} | Failed: {stats['failed']} | Target hits: {stats['target_hits']}")

    # Best config (by adjusted $/kW)
    best = store.get_best_adjusted()
    if best:
        bcfg = best.config
        lines.append(f"\n### Best Config So Far (by total adjusted $/kW)")
        lines.append(
            f"- {bcfg.get('working_fluid')}/{bcfg.get('topology')}/{bcfg.get('heat_rejection')} "
            f"strategy={bcfg.get('procurement_strategy', 'oem_lump_sum')} "
            f"vap={bcfg.get('vaporizer_pinch_F')} acc={bcfg.get('acc_approach_F')} "
            f"pre={bcfg.get('preheater_pinch_F')} recup={bcfg.get('recuperator_pinch_F')}"
        )
        lines.append(
            f"- **Adjusted: ${best.total_adjusted_per_kW:,.0f}/kW** "
            f"(Installed: ${best.capex_per_kW:,.0f} + Complexity: ${best.complexity_per_kW:,.0f}) | "
            f"Equipment: ${best.equipment_per_kW:,.0f}/kW | "
            f"Efficiency: {best.cycle_efficiency*100:.1f}% | Net: {best.net_power_MW:.1f} MW | "
            f"Schedule: {best.construction_weeks} wk | NPV: ${best.npv_USD:,.0f}"
        )

    # Pareto frontier
    pareto = store.get_pareto()
    if pareto:
        lines.append(f"\n### Pareto Frontier ({len(pareto)} points, by adjusted $/kW)")
        for r in sorted(pareto, key=lambda x: x.total_adjusted_per_kW)[:5]:
            rc = r.config
            lines.append(
                f"- {rc.get('working_fluid')}/{rc.get('topology')}: "
                f"${r.total_adjusted_per_kW:,.0f}/kW adj (${r.capex_per_kW:,.0f} inst + "
                f"${r.complexity_per_kW:,.0f} cmplx), "
                f"{r.cycle_efficiency*100:.1f}% eff, {r.net_power_MW:.1f} MW"
            )

    # Fluid rankings
    fluid_stats = {}
    for r in converged:
        fl = r.config.get("working_fluid", "unknown")
        if fl not in fluid_stats:
            fluid_stats[fl] = {"count": 0, "best_kw": float("inf"), "best_eff": 0, "avg_kw": 0}
        fluid_stats[fl]["count"] += 1
        fluid_stats[fl]["best_kw"] = min(fluid_stats[fl]["best_kw"], r.capex_per_kW)
        fluid_stats[fl]["best_eff"] = max(fluid_stats[fl]["best_eff"], r.cycle_efficiency)
        fluid_stats[fl]["avg_kw"] += r.capex_per_kW
    if fluid_stats:
        lines.append("\n### Fluid Rankings (by best $/kW)")
        for fl, data in sorted(fluid_stats.items(), key=lambda x: x[1]["best_kw"]):
            avg = data["avg_kw"] / data["count"] if data["count"] > 0 else 0
            lines.append(
                f"- {fl}: best ${data['best_kw']:,.0f}/kW, avg ${avg:,.0f}/kW, "
                f"best eff {data['best_eff']*100:.1f}%, {data['count']} runs"
            )

    # Topology rankings
    topo_stats = {}
    for r in converged:
        topo = r.config.get("topology", "unknown")
        if topo not in topo_stats:
            topo_stats[topo] = {"count": 0, "best_kw": float("inf")}
        topo_stats[topo]["count"] += 1
        topo_stats[topo]["best_kw"] = min(topo_stats[topo]["best_kw"], r.capex_per_kW)
    if topo_stats:
        lines.append("\n### Topology Rankings")
        for topo, data in sorted(topo_stats.items(), key=lambda x: x[1]["best_kw"]):
            lines.append(f"- {topo}: best ${data['best_kw']:,.0f}/kW, {data['count']} runs")

    # Strategy rankings (best $/kW per procurement strategy)
    strat_stats = {}
    for r in converged:
        strat = r.procurement_strategy
        if strat not in strat_stats:
            strat_stats[strat] = {"count": 0, "best_kw": float("inf")}
        strat_stats[strat]["count"] += 1
        strat_stats[strat]["best_kw"] = min(strat_stats[strat]["best_kw"], r.capex_per_kW)
    if strat_stats:
        lines.append("\n### Strategy Rankings (by best $/kW)")
        strat_labels = {
            "oem_lump_sum": "OEM+LS", "direct_lump_sum": "DIR+LS",
            "oem_self_perform": "OEM+SP", "direct_self_perform": "DIR+SP",
        }
        for strat, data in sorted(strat_stats.items(), key=lambda x: x[1]["best_kw"]):
            label = strat_labels.get(strat, strat)
            lines.append(f"- {label}: best ${data['best_kw']:,.0f}/kW, {data['count']} runs")

    # Last batch details (most recent AI_GUIDED_BATCH_SIZE results)
    batch_start = max(0, len(all_results) - AI_GUIDED_BATCH_SIZE)
    last_batch = all_results[batch_start:]
    if last_batch:
        lines.append(f"\n### Last Batch Results ({len(last_batch)} configs)")
        for r in last_batch:
            rc = r.config
            strat_short = {"oem_lump_sum": "OEM+LS", "direct_lump_sum": "DIR+LS",
                           "oem_self_perform": "OEM+SP", "direct_self_perform": "DIR+SP"
                           }.get(r.procurement_strategy, r.procurement_strategy)
            if r.converged:
                lines.append(
                    f"- {rc.get('working_fluid')}/{rc.get('topology')}/{rc.get('heat_rejection')}/{strat_short} "
                    f"vap={rc.get('vaporizer_pinch_F')} acc={rc.get('acc_approach_F')} "
                    f"pre={rc.get('preheater_pinch_F')} recup={rc.get('recuperator_pinch_F')} → "
                    f"${r.capex_per_kW:,.0f}/kW, {r.cycle_efficiency*100:.1f}% eff, "
                    f"{r.net_power_MW:.1f} MW, {r.construction_weeks} wk"
                )
            else:
                err_short = r.error[:60] if r.error else "unknown"
                lines.append(
                    f"- {rc.get('working_fluid')}/{rc.get('topology')}/{rc.get('heat_rejection')}/{strat_short} "
                    f"vap={rc.get('vaporizer_pinch_F')} acc={rc.get('acc_approach_F')} → FAILED: {err_short}"
                )

    # Failure patterns
    failed = [r for r in all_results if not r.converged]
    if failed:
        failure_combos = {}
        for r in failed:
            key = f"{r.config.get('working_fluid')}/{r.config.get('topology')}"
            failure_combos[key] = failure_combos.get(key, 0) + 1
        lines.append("\n### Failure Patterns")
        for combo, count in sorted(failure_combos.items(), key=lambda x: -x[1])[:5]:
            lines.append(f"- {combo}: {count} failures")

    # Already-run config hashes (so Claude can avoid them)
    lines.append(f"\n### Already Tested: {len(store._hashes)} unique configurations")

    lines.append("\nPropose the next batch of configurations to test.")
    return "\n".join(lines)


def call_ai_optimizer(store: ResultStore, round_number: int,
                      design_basis: dict, api_key: str = None,
                      heat_rejections: Optional[list[str]] = None) -> dict:
    """Call Claude to analyze results and propose next batch of configs.

    Returns dict with keys: configs, round_summary, insights, converged.
    On error, returns dict with error info and empty configs.
    """
    try:
        import anthropic
    except ImportError:
        return {
            "configs": [], "round_summary": "anthropic package not installed",
            "insights": [], "converged": False, "error": "anthropic not installed",
        }

    # Get API key (same pattern as app.py)
    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return {
            "configs": [], "round_summary": "No API key available",
            "insights": [], "converged": False, "error": "no_api_key",
        }

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=build_ai_system_prompt(design_basis, heat_rejections=heat_rejections),
            messages=[{"role": "user", "content": build_ai_context(store, round_number)}],
        )

        response_text = response.content[0].text
        from synthesis import _extract_json
        parsed = _extract_json(response_text)

        return {
            "configs": parsed.get("configs", []),
            "round_summary": parsed.get("round_summary", ""),
            "insights": parsed.get("insights", []),
            "converged": parsed.get("converged", False),
        }

    except Exception as e:
        logger.warning(f"AI optimizer API call failed: {e}")
        return {
            "configs": [], "round_summary": f"API error: {str(e)[:200]}",
            "insights": [], "converged": False, "error": str(e),
        }


def _generate_random_unexplored(store: ResultStore, count: int = 5) -> list[OptConfig]:
    """Generate random unexplored configs as fallback."""
    candidates = []
    _recup_topos = {"recuperated", "dual_pressure"}
    attempts = 0
    while len(candidates) < count and attempts < 500:
        attempts += 1
        fluid = random.choice(WORKING_FLUIDS)
        topo = random.choice(TOPOLOGIES)
        hr = random.choice(HEAT_REJECTIONS)
        strat = random.choice(PROCUREMENT_STRATEGIES)
        vap = random.choice(VAPORIZER_PINCHES)
        acc = random.choice(ACC_APPROACHES)
        pre = random.choice(PREHEATER_PINCHES)
        recup = random.choice(RECUPERATOR_PINCHES) if topo in _recup_topos else 999.0
        cfg = OptConfig(fluid, topo, hr, vap, acc, pre, recup, strat)
        if cfg.is_valid() and not store.has_config(cfg.config_hash()):
            candidates.append(cfg)
    return candidates


def parse_ai_configs(ai_response: dict, store: ResultStore) -> list[OptConfig]:
    """Convert AI response configs into validated OptConfig objects.

    Skips invalid or already-run configs. Falls back to random unexplored
    configs if none are valid.
    """
    configs = []
    raw_configs = ai_response.get("configs", [])

    _recup_topos = {"recuperated", "dual_pressure"}

    for raw in raw_configs:
        try:
            fluid = raw.get("working_fluid", "")
            topo = raw.get("topology", "")
            hr = raw.get("heat_rejection", "direct_acc")
            strat = raw.get("procurement_strategy", "direct_self_perform")
            vap = float(raw.get("vaporizer_pinch_F", 10))
            acc = float(raw.get("acc_approach_F", 15))
            pre = float(raw.get("preheater_pinch_F", 10))
            recup_raw = raw.get("recuperator_pinch_F", 999)
            recup = float(recup_raw) if topo in _recup_topos else 999.0

            # Validate ranges
            if fluid not in WORKING_FLUIDS:
                logger.warning(f"AI proposed invalid fluid: {fluid}")
                continue
            if topo not in TOPOLOGIES:
                logger.warning(f"AI proposed invalid topology: {topo}")
                continue
            if hr not in HEAT_REJECTIONS:
                logger.warning(f"AI proposed invalid heat_rejection: {hr}")
                continue
            if strat not in PROCUREMENT_STRATEGIES:
                logger.warning(f"AI proposed invalid strategy: {strat}, defaulting to direct_self_perform")
                strat = "direct_self_perform"
            if not (5 <= vap <= 20):
                vap = max(5, min(20, vap))
            if not (10 <= acc <= 30):
                acc = max(10, min(30, acc))
            if not (5 <= pre <= 15):
                pre = max(5, min(15, pre))
            if topo in _recup_topos and not (10 <= recup <= 25):
                recup = max(10, min(25, recup))

            cfg = OptConfig(fluid, topo, hr, vap, acc, pre, recup, strat)
            if not cfg.is_valid():
                logger.warning(f"AI proposed invalid combo: {fluid}/{topo}/{hr}")
                continue
            if store.has_config(cfg.config_hash()):
                continue
            configs.append(cfg)
        except (ValueError, TypeError, KeyError) as e:
            logger.warning(f"AI proposed unparseable config: {e}")
            continue

    if len(configs) >= AI_GUIDED_BATCH_SIZE:
        return configs[:AI_GUIDED_BATCH_SIZE]

    # Fallback: fill with random unexplored configs
    if not configs:
        logger.warning("AI returned 0 valid configs, falling back to random exploration")
    needed = AI_GUIDED_BATCH_SIZE - len(configs)
    configs.extend(_generate_random_unexplored(store, needed))
    return configs[:AI_GUIDED_BATCH_SIZE]


# ── DBD Update Proposal Generator ─────────────────────────────────────────

def generate_dbd_update_proposal(store: ResultStore, design_basis: dict,
                                  api_key: str = None) -> dict:
    """Call Claude with the current DBD + optimizer report to propose structured
    Design Basis Document updates.

    Returns a proposal container:
    {
        generated_at: str,
        optimizer_stats: dict,
        items: [
            {id, section, action, target_path, description,
             old_value, new_value, confidence, evidence,
             decision, requires_approval}
        ],
        session_summary: str,
        status: "pending"
    }

    Falls back to a deterministic history-only proposal if API is unavailable.
    """
    import uuid
    from datetime import datetime, timezone

    try:
        from design_basis_document import load_dbd
        dbd = load_dbd()
    except Exception:
        dbd = {}

    report = generate_report(store)
    context = build_ai_context(store, round_number=-1)
    stats = store.stats()

    # Build the proposal shell
    proposal = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "optimizer_stats": stats,
        "items": [],
        "session_summary": "",
        "status": "pending",
    }

    # Always include a deterministic Section 8 (opt_history) APPEND entry
    best = store.get_best_adjusted() or store.get_best_per_kw()
    best_cfg = best.config if best else {}
    history_entry = {
        "date": datetime.now(timezone.utc).strftime("%Y-%m-%d"),
        "event": f"Optimizer session: {stats['total_runs']} configs tested",
        "finding": (
            f"Best adjusted: ${best.total_adjusted_per_kW:,.0f}/kW "
            f"({best_cfg.get('working_fluid', '?')}/{best_cfg.get('topology', '?')}/"
            f"{best_cfg.get('heat_rejection', '?')}, "
            f"strategy={best.procurement_strategy}). "
            f"{stats['target_hits']} target hits out of {stats['converged']} converged."
        ) if best else f"No converged results from {stats['total_runs']} configs.",
        "action": (
            f"Best config: {best_cfg.get('working_fluid', '?')}/"
            f"{best_cfg.get('topology', '?')} at "
            f"${best.total_adjusted_per_kW:,.0f}/kW adjusted"
        ) if best else "No viable configurations found — review constraints.",
    }
    fallback_history_item = {
        "id": str(uuid.uuid4())[:8],
        "section": "section_8_opt_history",
        "action": "APPEND",
        "target_path": "entries",
        "description": f"Log optimizer session ({stats['total_runs']} configs tested)",
        "old_value": None,
        "new_value": history_entry,
        "confidence": "HIGH",
        "evidence": f"{stats['total_runs']} configs tested, {stats['target_hits']} target hits",
        "decision": "pending",
        "requires_approval": False,
    }

    # Try calling Claude for richer proposals
    try:
        import anthropic
    except ImportError:
        proposal["items"].append(fallback_history_item)
        proposal["session_summary"] = generate_session_summary(store, dbd, proposal["items"])
        return proposal

    if not api_key:
        try:
            import streamlit as st
            api_key = st.secrets.get("ANTHROPIC_API_KEY")
        except Exception:
            pass
    if not api_key:
        api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        proposal["items"].append(fallback_history_item)
        proposal["session_summary"] = generate_session_summary(store, dbd, proposal["items"])
        return proposal

    dbd_json = json.dumps(dbd, indent=2, default=str)
    report_json = json.dumps(report, indent=2, default=str)

    system_prompt = f"""You are a Design Basis Document (DBD) maintenance engineer for an ORC geothermal power project.

Given the current DBD and optimizer results, propose specific updates to the DBD.

RULES:
- You may ONLY propose changes to these sections:
  * section_3_equipment (UPDATE only, HIGH confidence + strong evidence required)
  * section_6_info_requests (NEW items for data gaps, CLOSE for resolved items)
  * section_7_kb_inventory (UPDATE coverage/gaps based on new data)
  * section_8_opt_history (APPEND new entries — always include at least one)
- Sections 1, 2, 4, 5 are user-approved only — NEVER propose changes to these.
- Each proposal item must have: section, action (UPDATE/NEW/APPEND/CLOSE),
  target_path (dot-separated path within the section), description,
  old_value (current value or null), new_value (proposed value),
  confidence (HIGH/MEDIUM/LOW), evidence (what data supports this change).
- For APPEND actions on list fields: new_value is the item to append.
- For CLOSE actions on info_requests: set status to "closed" and add closed_date.
- Be conservative with section_3_equipment — only propose when optimizer data strongly
  supports a change (e.g., consistent results across many configs).

Return ONLY a JSON object:
{{
  "items": [
    {{
      "section": "section_8_opt_history",
      "action": "APPEND",
      "target_path": "entries",
      "description": "Log optimizer session results",
      "old_value": null,
      "new_value": {{"date": "...", "event": "...", "finding": "...", "action": "..."}},
      "confidence": "HIGH",
      "evidence": "Direct optimizer data"
    }}
  ]
}}"""

    user_message = f"""## Current Design Basis Document
{dbd_json}

## Optimizer Results Summary
{report_json}

## Detailed Optimizer Context
{context}

Based on these optimizer results, propose DBD updates."""

    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=4096,
            system=system_prompt,
            messages=[{"role": "user", "content": user_message}],
        )
        response_text = response.content[0].text

        from synthesis import _extract_json
        parsed = _extract_json(response_text)
        raw_items = parsed.get("items", [])

        # Process each item: add id, decision, requires_approval
        for item in raw_items:
            item["id"] = str(uuid.uuid4())[:8]
            item["decision"] = "pending"
            section = item.get("section", "")
            # Sections 1-5 require user approval; 6-8 are Claude-maintained
            item["requires_approval"] = section in (
                "section_1_identity", "section_2_philosophy",
                "section_3_equipment", "section_4_constraints",
                "section_5_cost_anchors",
            )
            proposal["items"].append(item)

        # Ensure we always have the history entry
        has_history = any(
            i.get("section") == "section_8_opt_history" for i in proposal["items"]
        )
        if not has_history:
            proposal["items"].append(fallback_history_item)

    except Exception as e:
        logger.warning(f"DBD update proposal API call failed: {e}")
        proposal["items"].append(fallback_history_item)

    proposal["session_summary"] = generate_session_summary(store, dbd, proposal["items"])
    return proposal


def generate_session_summary(store: ResultStore, design_basis: dict,
                              proposals: list[dict]) -> str:
    """Deterministic formatter: produce a plain-text session summary (no API call).

    Includes session stats, best config, accepted proposals, and next-run
    recommendations.
    """
    from datetime import datetime, timezone

    stats = store.stats()
    best = store.get_best_adjusted() or store.get_best_per_kw()
    pareto = store.get_pareto()

    lines = []
    lines.append("=" * 60)
    lines.append("ORC OPTIMIZER SESSION SUMMARY")
    lines.append("=" * 60)
    lines.append(f"Date: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}")

    # Determine mode from session state (best-effort)
    try:
        import streamlit as st
        mode = st.session_state.get("opt_mode", "unknown")
        ai_rounds = st.session_state.get("opt_ai_round", 0)
        config_times = st.session_state.get("opt_config_times", [])
    except Exception:
        mode = "unknown"
        ai_rounds = 0
        config_times = []

    lines.append(f"Mode: {mode}")
    if mode == "ai_guided":
        lines.append(f"AI Rounds: {ai_rounds}")
    if config_times:
        total_time = sum(config_times)
        lines.append(f"Duration: {total_time:.0f}s ({total_time/60:.1f} min)")

    lines.append("")
    lines.append("--- RESULTS ---")
    lines.append(f"Total configs tested:  {stats['total_runs']}")
    lines.append(f"Converged:             {stats['converged']}")
    lines.append(f"Failed:                {stats['failed']}")
    lines.append(f"Target hits:           {stats['target_hits']}")
    lines.append(f"Pareto-optimal points: {stats['pareto_count']}")

    if best:
        cfg = best.config or {}
        lines.append("")
        lines.append("--- BEST CONFIGURATION (by adjusted $/kW) ---")
        lines.append(f"Fluid:              {cfg.get('working_fluid', '?')}")
        lines.append(f"Topology:           {cfg.get('topology', '?')}")
        lines.append(f"Heat rejection:     {cfg.get('heat_rejection', '?')}")
        lines.append(f"Strategy:           {best.procurement_strategy}")
        lines.append(f"Installed $/kW:     ${best.capex_per_kW:,.0f}")
        lines.append(f"Complexity $/kW:    ${best.complexity_per_kW:,.0f}")
        lines.append(f"Adjusted $/kW:      ${best.total_adjusted_per_kW:,.0f}")
        lines.append(f"Equipment $/kW:     ${best.equipment_per_kW:,.0f}")
        lines.append(f"Cycle efficiency:   {best.cycle_efficiency*100:.1f}%")
        lines.append(f"Net power:          {best.net_power_MW:.1f} MW")
        lines.append(f"Gross power:        {best.gross_power_MW:.1f} MW")
        lines.append(f"Schedule:           {best.construction_weeks} weeks")
        lines.append(f"NPV:                ${best.npv_USD:,.0f}")
        lines.append(f"LCOE:               ${best.lcoe_per_MWh:.1f}/MWh")
        lines.append(f"Target met:         {'YES' if best.target_fit else 'NO'}")

        vap = cfg.get("vaporizer_pinch_F", "?")
        acc = cfg.get("acc_approach_F", "?")
        pre = cfg.get("preheater_pinch_F", "?")
        recup = cfg.get("recuperator_pinch_F", "?")
        lines.append(f"Pinch points:       vap={vap}F, acc={acc}F, pre={pre}F, recup={recup}F")

    # DBD updates summary
    accepted = [p for p in proposals if p.get("decision") in ("accepted", "modified")]
    rejected = [p for p in proposals if p.get("decision") == "rejected"]
    pending = [p for p in proposals if p.get("decision") == "pending"]
    if proposals:
        lines.append("")
        lines.append("--- DBD UPDATES ---")
        lines.append(f"Total proposed:  {len(proposals)}")
        lines.append(f"Accepted:        {len(accepted)}")
        lines.append(f"Rejected:        {len(rejected)}")
        lines.append(f"Pending:         {len(pending)}")
        for p in accepted:
            lines.append(f"  [{p.get('action', '?')}] {p.get('section', '?')}: {p.get('description', '?')}")

    # Recommendations for next run
    lines.append("")
    lines.append("--- RECOMMENDED NEXT RUN ---")
    if best and best.target_fit:
        lines.append("Target already met. Consider:")
        lines.append("  - Fine-tune pinch points around best config (±2F)")
        lines.append("  - Verify results with vendor quotes")
        lines.append("  - Run sensitivity analysis on key assumptions")
    elif best:
        gap = best.total_adjusted_per_kW - TARGET_CAPEX_PER_KW
        lines.append(f"Gap to target: ${gap:,.0f}/kW")
        if gap < 200:
            lines.append("  - Close to target — try finer pinch point sweeps around best config")
        else:
            lines.append("  - Significant gap — explore different procurement strategies or simpler topologies")
            lines.append("  - Consider if any open info requests could change assumptions")
    else:
        lines.append("No converged results — review input constraints and brine conditions")

    lines.append("")
    lines.append("=" * 60)
    return "\n".join(lines)
