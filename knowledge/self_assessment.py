"""
Self-assessment against $2,000/kW geothermal ORC target.

Decomposes the goal across 8 knowledge domains derived from COST_FACTORS
in cost_model.py, evaluates coverage, and identifies critical path gaps.

Coverage uses keyword-based content matching so that chunks tagged "general"
still count toward the correct technical domain.
"""

import re

from knowledge.vector_store import table_stats
from knowledge.ingestion import get_all_documents
from knowledge.gap_tracker import (
    get_active_gaps, get_hard_gaps, get_soft_gaps,
    estimate_total_uncertainty, add_gap,
)

TARGET_COST_PER_KW = 2000  # $/kW installed target

# 8 knowledge domains with cost share weights and related cost_model keys
KNOWLEDGE_DOMAINS = {
    "turbine_technology": {
        "label": "Turbine & Generator",
        "cost_share_pct": 25,
        "cost_factor_keys": ["turbine_per_kw"],
        "description": "Turbine design, efficiency, materials, OEM options, "
                       "multi-stage vs single-stage, generator coupling.",
        "target_contribution": 500,  # $500/kW
    },
    "heat_exchangers": {
        "label": "Heat Exchangers",
        "cost_share_pct": 20,
        "cost_factor_keys": [
            "vaporizer_per_ft2", "preheater_per_ft2",
            "recup_per_ft2", "hx_per_ft2",
        ],
        "description": "Shell-and-tube, plate, PCHE designs; materials for "
                       "brine service; fouling factors; U-values.",
        "target_contribution": 400,
    },
    "air_cooled_condenser": {
        "label": "Air-Cooled Condenser",
        "cost_share_pct": 15,
        "cost_factor_keys": ["acc_per_ft2"],
        "description": "ACC sizing, fan selection, fin-tube bundles, "
                       "approach temperature, ambient derating.",
        "target_contribution": 300,
    },
    "working_fluid": {
        "label": "Working Fluid & Piping",
        "cost_share_pct": 10,
        "cost_factor_keys": ["iso_duct_per_ft2", "prop_pipe_per_ft2", "prop_piping_pct"],
        "description": "Isobutane/isopentane/propane properties, piping design, "
                       "duct sizing, insulation, safety codes.",
        "target_contribution": 200,
    },
    "structural_civil": {
        "label": "Structural & Civil",
        "cost_share_pct": 10,
        "cost_factor_keys": ["steel_per_lb", "foundation_pct"],
        "description": "Steel structures, pipe racks, foundations, "
                       "seismic design, site preparation.",
        "target_contribution": 200,
    },
    "construction_schedule": {
        "label": "Construction & Indirects",
        "cost_share_pct": 15,
        "cost_factor_keys": ["engineering_pct", "construction_mgmt_pct", "contingency_pct"],
        "description": "EPC strategy, modularization, schedule risk, "
                       "engineering hours, construction management.",
        "target_contribution": 300,
    },
    "resource_characterization": {
        "label": "Resource Characterization",
        "cost_share_pct": 0,
        "cost_factor_keys": [],
        "description": "Brine chemistry, NCG content, scaling/corrosion potential, "
                       "temperature decline, well productivity.",
        "target_contribution": 0,
    },
    "economics_market": {
        "label": "Economics & Market",
        "cost_share_pct": 5,
        "cost_factor_keys": [],
        "description": "PPA pricing, discount rates, O&M costs, capacity factor, "
                       "tax credits (ITC/PTC), market outlook.",
        "target_contribution": 100,
    },
}


# Keyword patterns for content-based domain matching.
# A chunk matches a domain if its text contains any of these patterns.
# This lets chunks tagged "general" still contribute to the right domain.
DOMAIN_KEYWORDS = {
    "turbine_technology": [
        r"\bturbine\b", r"\bgenerator\b", r"\bturbo.generator\b",
        r"\bexpander\b", r"\brotor\b", r"\bnozzle\b", r"\bblade\b",
        r"\btg\b", r"\bturbomachinery\b", r"\bimpeller\b",
    ],
    "heat_exchangers": [
        r"\bheat exchanger\b", r"\bvaporizer\b", r"\bpreheater\b",
        r"\bevaporator\b", r"\brecuperator\b", r"\bshell.and.tube\b",
        r"\bplate heat\b", r"\bpche\b", r"\bfouling\b", r"\bu.value\b",
        r"\bhx\b", r"\bhtri\b", r"\btema\b",
    ],
    "air_cooled_condenser": [
        r"\bair.cooled\b", r"\bcondenser\b", r"\bacc\b", r"\bfin.tube\b",
        r"\bcooling tower\b", r"\bfan\b", r"\bambient derat\b",
        r"\bheat rejection\b", r"\bdry cooling\b",
    ],
    "working_fluid": [
        r"\bisopentane\b", r"\bisobutane\b", r"\bpropane\b", r"\br245fa\b",
        r"\bworking fluid\b", r"\brefrigerant\b", r"\borganic fluid\b",
        r"\bcyclopentane\b", r"\bn.pentane\b", r"\bflammab\b",
    ],
    "structural_civil": [
        r"\bstructural\b", r"\bfoundation\b", r"\bsteel\b",
        r"\bpipe rack\b", r"\bseismic\b", r"\bcivil\b",
        r"\bconcrete\b", r"\bsite prep\b",
    ],
    "construction_schedule": [
        r"\bepc\b", r"\bschedule\b", r"\bmodular\b", r"\bindirect\b",
        r"\bconstruction\b", r"\bengineering hours\b", r"\bstick.build\b",
        r"\bcommissioning\b", r"\bmobilization\b", r"\blabor\b",
        r"\bcontingency\b",
    ],
    "resource_characterization": [
        r"\bbrine\b", r"\bgeothermal\b", r"\bresource\b", r"\bncg\b",
        r"\bnon.condensable\b", r"\bscaling\b", r"\bwell productiv\b",
        r"\bsilica\b", r"\bcorrosion\b", r"\breservoir\b",
        r"\bgeochemistry\b", r"\bdownhole\b",
    ],
    "economics_market": [
        r"\bppa\b", r"\bdiscount rate\b", r"\blcoe\b", r"\bnpv\b",
        r"\bcapacity factor\b", r"\btax credit\b", r"\bo&m\b",
        r"\blevelized\b", r"\birr\b", r"\btariff\b",
    ],
}

# Compiled patterns (case-insensitive) for performance
_COMPILED_PATTERNS: dict[str, list[re.Pattern]] = {}
for _dom, _pats in DOMAIN_KEYWORDS.items():
    _COMPILED_PATTERNS[_dom] = [re.compile(p, re.IGNORECASE) for p in _pats]


def _count_content_matches(domain_id: str, df) -> int:
    """Count chunks whose text matches domain keywords, regardless of tag."""
    patterns = _COMPILED_PATTERNS.get(domain_id, [])
    if not patterns:
        return 0
    count = 0
    for text in df["text"]:
        for pat in patterns:
            if pat.search(str(text)):
                count += 1
                break
    return count


def _assess_domain_coverage(domain_id: str, domain_info: dict,
                            docs: list[dict], chunks_by_domain: dict,
                            gaps: list[dict]) -> dict:
    """Evaluate knowledge coverage for a single domain.

    Uses the HIGHER of:
      - chunk_count_by_tag: chunks explicitly tagged with this domain
      - chunk_count_by_content: chunks whose text matches domain keywords
    This ensures documents tagged "general" still contribute.
    """
    domain_docs = [d for d in docs if d.get("domain") == domain_id]
    chunk_count_by_tag = chunks_by_domain.get(domain_id, 0)
    chunk_count_by_content = chunks_by_domain.get(f"_content_{domain_id}", 0)
    chunk_count = max(chunk_count_by_tag, chunk_count_by_content)
    domain_gaps = [g for g in gaps if g["domain"] == domain_id]
    hard_gaps = [g for g in domain_gaps if g["severity"] == "hard"]

    # Coverage scoring
    if chunk_count == 0:
        coverage = "none"
        score = 0.0
    elif chunk_count < 3:
        coverage = "weak"
        score = 0.2
    elif chunk_count < 10:
        coverage = "partial"
        score = 0.5
    elif chunk_count < 25:
        coverage = "moderate"
        score = 0.7
    else:
        coverage = "strong"
        score = 0.9

    # Penalize for hard gaps
    if hard_gaps:
        score = max(0.0, score - 0.3 * len(hard_gaps))

    return {
        "domain_id": domain_id,
        "label": domain_info["label"],
        "cost_share_pct": domain_info["cost_share_pct"],
        "target_contribution": domain_info["target_contribution"],
        "document_count": len(domain_docs),
        "chunk_count": chunk_count,
        "chunk_count_by_tag": chunk_count_by_tag,
        "chunk_count_by_content": chunk_count_by_content,
        "coverage": coverage,
        "score": round(score, 2),
        "active_gaps": len(domain_gaps),
        "hard_gaps": len(hard_gaps),
    }


def run_self_assessment() -> dict:
    """
    Run full self-assessment against $2,000/kW target.

    Returns domain assessments, readiness score, critical path gaps,
    uncertainty band, and next steps.
    """
    docs = get_all_documents()
    stats = table_stats()
    all_gaps = get_active_gaps()
    uncertainty = estimate_total_uncertainty()

    # Count chunks per domain from vector store — two methods:
    #   1. By explicit domain tag (chunks_by_domain[domain_id])
    #   2. By keyword content matching (chunks_by_domain["_content_" + domain_id])
    chunks_by_domain = {}
    if stats["total_chunks"] > 0:
        from knowledge.vector_store import ensure_table
        df = ensure_table().to_pandas()
        # Tag-based counts
        for domain in df["domain"].unique():
            chunks_by_domain[domain] = int((df["domain"] == domain).sum())
        # Content-based counts (keyword matching across ALL chunks)
        for domain_id in KNOWLEDGE_DOMAINS:
            content_count = _count_content_matches(domain_id, df)
            chunks_by_domain[f"_content_{domain_id}"] = content_count

    # Assess each domain
    domain_assessments = {}
    for domain_id, info in KNOWLEDGE_DOMAINS.items():
        domain_assessments[domain_id] = _assess_domain_coverage(
            domain_id, info, docs, chunks_by_domain, all_gaps
        )

    # Weighted readiness score
    total_weight = sum(d["cost_share_pct"] for d in KNOWLEDGE_DOMAINS.values())
    if total_weight > 0:
        weighted_score = sum(
            domain_assessments[d]["score"] * KNOWLEDGE_DOMAINS[d]["cost_share_pct"]
            for d in KNOWLEDGE_DOMAINS
        ) / total_weight
    else:
        weighted_score = 0.0

    # Baseline $/kW from cost_model defaults
    baseline_cost_per_kw = _estimate_baseline_cost_per_kw()

    # Critical path: domains with hard gaps or zero coverage and >0% cost share
    critical_path = [
        domain_assessments[d] for d in KNOWLEDGE_DOMAINS
        if (domain_assessments[d]["coverage"] in ("none", "weak")
            and KNOWLEDGE_DOMAINS[d]["cost_share_pct"] > 0)
        or domain_assessments[d]["hard_gaps"] > 0
    ]
    critical_path.sort(key=lambda x: x["cost_share_pct"], reverse=True)

    # Seed default hard gaps for key domains if none exist
    _seed_default_gaps(all_gaps)

    # Generate next steps
    next_steps = _generate_next_steps(domain_assessments, critical_path)

    return {
        "target_cost_per_kw": TARGET_COST_PER_KW,
        "baseline_cost_per_kw": baseline_cost_per_kw,
        "gap_to_target": baseline_cost_per_kw - TARGET_COST_PER_KW,
        "readiness_score": round(weighted_score, 2),
        "domain_assessments": domain_assessments,
        "critical_path": critical_path,
        "uncertainty": uncertainty,
        "total_documents": len(docs),
        "total_chunks": stats["total_chunks"],
        "next_steps": next_steps,
    }


def _estimate_baseline_cost_per_kw() -> float:
    """Estimate $/kW from cost_model defaults at reference conditions."""
    # Calibrated to vendor benchmarks: Exergy $683/kW, Turboden $760/kW
    # at 53 MW net, 220°C brine, isopentane working fluid
    reference_net_kw = 53_000
    reference_total_cost = 38_000_000  # ~$717/kW at calibrated defaults
    return reference_total_cost / reference_net_kw


def _seed_default_gaps(existing_gaps: list[dict]):
    """Seed hard gaps for domains that critically lack knowledge."""
    existing_domains = {g["domain"] for g in existing_gaps}

    defaults = [
        {
            "domain": "resource_characterization",
            "title": "No site-specific brine chemistry data",
            "description": "Brine chemistry (TDS, silica, NCG, pH) determines "
                           "materials selection and fouling allowances. Without "
                           "this data, heat exchanger costs are uncertain.",
            "severity": "hard",
            "impact_on_cost_per_kW": 150.0,
        },
        {
            "domain": "construction_schedule",
            "title": "No EPC strategy or schedule baseline",
            "description": "Modularization vs stick-build, labor availability, "
                           "and schedule duration drive indirect costs (engineering, "
                           "CM, contingency). No reference data available.",
            "severity": "hard",
            "impact_on_cost_per_kW": 200.0,
        },
    ]

    for d in defaults:
        if d["domain"] not in existing_domains:
            add_gap(
                domain=d["domain"],
                title=d["title"],
                description=d["description"],
                severity=d["severity"],
                impact_on_cost_per_kW=d["impact_on_cost_per_kW"],
                detected_by="self_assessment",
            )


def _generate_next_steps(assessments: dict, critical_path: list[dict]) -> list[str]:
    """Generate prioritized next steps."""
    steps = []

    # Priority 1: Hard gaps
    hard_domains = [a for a in critical_path if a["hard_gaps"] > 0]
    if hard_domains:
        for d in hard_domains[:3]:
            steps.append(
                f"CRITICAL: Resolve hard gaps in {d['label']} "
                f"({d['hard_gaps']} gap(s), {d['cost_share_pct']}% cost share)"
            )

    # Priority 2: No-coverage domains with cost impact
    empty = [a for a in critical_path if a["coverage"] == "none" and a["cost_share_pct"] > 0]
    for d in empty[:3]:
        steps.append(
            f"Upload reference documents for {d['label']} "
            f"({d['cost_share_pct']}% of installed cost)"
        )

    # Priority 3: Weak-coverage domains
    weak = [a for a in critical_path if a["coverage"] == "weak"]
    for d in weak[:2]:
        steps.append(
            f"Strengthen {d['label']} knowledge "
            f"(currently {d['chunk_count']} chunks, need 10+)"
        )

    if not steps:
        steps.append("Knowledge base has baseline coverage across all domains. "
                      "Continue adding vendor quotes and project-specific data.")

    return steps
