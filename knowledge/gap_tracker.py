"""
Knowledge gap tracking — CRUD, severity classification, JSON persistence,
and uncertainty estimation.
"""

import json
import os
import uuid
from datetime import datetime, timezone

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")
GAPS_PATH = os.path.join(DATA_DIR, "gaps.json")


def _load_gaps() -> list[dict]:
    if os.path.exists(GAPS_PATH):
        with open(GAPS_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    return []


def _save_gaps(gaps: list[dict]):
    os.makedirs(DATA_DIR, exist_ok=True)
    with open(GAPS_PATH, "w", encoding="utf-8") as f:
        json.dump(gaps, f, indent=2)


def add_gap(domain: str, title: str, description: str,
            severity: str = "soft", impact_on_cost_per_kW: float = 0.0,
            detected_by: str = "system",
            related_technologies: list[str] | None = None) -> dict:
    """Create a new knowledge gap. severity: 'hard' or 'soft'."""
    gap = {
        "gap_id": str(uuid.uuid4()),
        "domain": domain,
        "title": title,
        "description": description,
        "severity": severity if severity in ("hard", "soft") else "soft",
        "impact_on_cost_per_kW": impact_on_cost_per_kW,
        "detected_by": detected_by,
        "detected_at": datetime.now(timezone.utc).isoformat(),
        "resolved": False,
        "resolved_at": None,
        "resolution_document_id": None,
        "related_technologies": related_technologies or [],
    }
    gaps = _load_gaps()
    gaps.append(gap)
    _save_gaps(gaps)
    return gap


def resolve_gap(gap_id: str, resolution_document_id: str | None = None) -> dict | None:
    """Mark a gap as resolved. Returns updated gap or None if not found."""
    gaps = _load_gaps()
    for g in gaps:
        if g["gap_id"] == gap_id:
            g["resolved"] = True
            g["resolved_at"] = datetime.now(timezone.utc).isoformat()
            g["resolution_document_id"] = resolution_document_id
            _save_gaps(gaps)
            return g
    return None


def get_active_gaps(domain: str | None = None,
                    severity: str | None = None) -> list[dict]:
    """Return unresolved gaps, optionally filtered."""
    gaps = _load_gaps()
    active = [g for g in gaps if not g["resolved"]]
    if domain:
        active = [g for g in active if g["domain"] == domain]
    if severity:
        active = [g for g in active if g["severity"] == severity]
    return active


def get_hard_gaps() -> list[dict]:
    return get_active_gaps(severity="hard")


def get_soft_gaps() -> list[dict]:
    return get_active_gaps(severity="soft")


def estimate_total_uncertainty() -> dict:
    """Estimate total cost uncertainty from active gaps."""
    hard = get_hard_gaps()
    soft = get_soft_gaps()
    hard_impact = sum(g["impact_on_cost_per_kW"] for g in hard)
    soft_impact = sum(g["impact_on_cost_per_kW"] for g in soft)
    return {
        "hard_gap_count": len(hard),
        "soft_gap_count": len(soft),
        "total_hard_impact": hard_impact,
        "total_soft_impact": soft_impact,
        "total_impact": hard_impact + soft_impact,
        "hard_domains": sorted(set(g["domain"] for g in hard)),
        "soft_domains": sorted(set(g["domain"] for g in soft)),
    }
