"""
Living Design Basis Document (DBD) — Single source of truth for project design
philosophy, equipment standards, physical constraints, and validated cost anchors.

The optimizer reads the DBD at startup and enforces its constraints.
Claude proposes updates to Sections 6-8 after each optimization round.

Persistence: JSON at knowledge/data/design_basis.json
"""

import json
import os
from datetime import datetime, timezone

import streamlit as st

DBD_PATH = os.path.join(os.path.dirname(__file__), "knowledge", "data", "design_basis.json")

# ── Default DBD ────────────────────────────────────────────────────────────

DEFAULT_DBD = {
    "version": "1.0",
    "last_updated": datetime.now(timezone.utc).isoformat(),
    "last_updated_by": "system",

    # ── Section 1: Project Identity & Strategic Objective ──────────────
    "section_1_identity": {
        "title": "Project Identity & Strategic Objective",
        "project_name": "Cape Station Unit 2 — Surface Power Conversion Facility",
        "developer": "Fervo Energy",
        "location": "Beaver County, Utah",
        "resource_type": "Enhanced Geothermal System (EGS) — closed-loop horizontal wells",
        "strategic_objective": (
            "Demonstrate that EGS surface facilities can achieve $2,000/kW installed cost "
            "with a 70-week construction schedule at 53+ MW net, proving commercial viability "
            "of EGS power at utility-scale."
        ),
        "design_life_years": 30,
        "capacity_factor_pct": 95,
        "ppa_target_per_MWh": 35,
    },

    # ── Section 2: Design Philosophy ──────────────────────────────────
    "section_2_philosophy": {
        "title": "Design Philosophy",
        "values": [
            {
                "name": "Simplicity over marginal efficiency",
                "description": (
                    "Every additional component must justify its lifecycle cost including O&M, "
                    "training, spare parts, and failure modes — not just its capital cost."
                ),
            },
            {
                "name": "Proven equipment, aggressive procurement",
                "description": (
                    "Use commercially proven components (API 610 pumps, TEMA HX, standard ACC) "
                    "but strip OEM integration margins through direct vendor procurement and "
                    "owner self-perform construction."
                ),
            },
            {
                "name": "Single turbine preferred",
                "description": (
                    "One turbine-generator set unless dual-unit configuration provides >$100/kW "
                    "total lifecycle benefit. Single TG simplifies controls, reduces spare parts "
                    "inventory, and shortens commissioning."
                ),
            },
            {
                "name": "Information before assumption",
                "description": (
                    "When a design decision hinges on data we don't have, get the data. "
                    "Vendor quotes, site surveys, and lab tests are cheaper than discovering "
                    "wrong assumptions during construction."
                ),
            },
            {
                "name": "Design for construction speed",
                "description": (
                    "Favor modular, shop-fabricated assemblies over field-erected structures. "
                    "70-week schedule requires parallel construction paths and minimal "
                    "field welding."
                ),
            },
        ],
    },

    # ── Section 3: Equipment Design Standards ─────────────────────────
    "section_3_equipment": {
        "title": "Equipment Design Standards",
        "turbine_generator": {
            "type": "Radial inflow or axial, single-stage preferred",
            "design_code": "API 612 or manufacturer standard",
            "generator": "Synchronous, air-cooled, 13.8 kV",
            "gearbox": "Epicyclic or parallel-shaft per OEM",
            "lube_system": "Integral forced-oil with redundant pumps",
            "control": "Integrated TG control panel with DCS interface",
        },
        "heat_exchangers": {
            "vaporizer": "Shell-and-tube, TEMA BEM/AES, ASME VIII Div 1, brine in tubes",
            "preheater": "Shell-and-tube, TEMA BEM, liquid-liquid, corrosion-resistant",
            "recuperator": "Welded plate or shell-and-tube, organic vapor-liquid",
            "design_pressure": "150% of MAWP or 50 psi above operating, whichever greater",
            "tube_material": "CS or duplex SS per brine chemistry",
            "U_values_Btu_hr_ft2_F": {
                "preheater": 250,
                "recuperator": 60,
                "vaporizer": 150,
                "acc": 80,
            },
        },
        "acc": {
            "type": "A-frame air-cooled condenser",
            "fan_control": "Staging only — NO VFDs",
            "fan_control_rationale": (
                "45 fans = 2.2% resolution per step. VFD capital + O&M NPV exceeds "
                "energy saving by $12-22/kW over plant life."
            ),
            "deck_height_ft": 40,
            "fan_diameter_ft": 36,
            "design_ambient_F": 57,
            "face_velocity_fpm": 400,
        },
        "pumps": {
            "type": "API 610 centrifugal, horizontal or vertical",
            "redundancy": "1x100% operating + 1x100% installed spare",
            "seal": "Dual mechanical seal with barrier fluid",
            "motor": "TEFC, premium efficiency",
        },
        "piping": {
            "iso_vapor_duct": "Field-erected, insulated, min Schedule 10S for <150 psi",
            "iso_liquid": "Schedule 40 minimum",
            "propane": "Schedule 80 minimum (flammable service)",
            "brine": "Per ASME B31.1, schedule per brine chemistry",
        },
    },

    # ── Section 4: Physical Constraints (Non-Negotiable) ──────────────
    "section_4_constraints": {
        "title": "Physical Constraints — Non-Negotiable",
        "constraints": [
            {
                "name": "Brine outlet temperature floor",
                "value": "180°F (82°C)",
                "basis": "Silica saturation limit — scaling risk below this temperature",
                "override_condition": "Only with anti-scaling treatment AND brine chemistry verification",
            },
            {
                "name": "Single-phase brine (no flash)",
                "value": "Brine pressure maintained above saturation at all points",
                "basis": "Scaling, erosion, and well deliverability protection",
                "override_condition": "None — absolute constraint",
            },
            {
                "name": "Working fluid containment",
                "value": "Zero fugitive emissions target; double containment on hydrocarbon service",
                "basis": "Environmental permit and safety",
                "override_condition": "None",
            },
            {
                "name": "Seismic design",
                "value": "IBC Seismic Design Category per site-specific study",
                "basis": "Utah seismic zone + induced seismicity monitoring",
                "override_condition": "None",
            },
            {
                "name": "Ambient temperature range",
                "value": "-20°F to +105°F design envelope",
                "basis": "Beaver County, Utah historical extremes with margin",
                "override_condition": "None",
            },
            {
                "name": "iC4 transcritical cycle — evaluation pending",
                "value": "Isobutane transcritical ORC requires validation before adoption",
                "basis": (
                    "Potentially higher efficiency at 420°F resource, but requires: "
                    "(1) supercritical-rated HX design (P > 529 psia), "
                    "(2) turbine expander validated for transcritical expansion, "
                    "(3) updated safety analysis for high-pressure iC4. "
                    "Cannot be adopted without vendor data and safety review."
                ),
                "override_condition": "Vendor quotes + safety review + engineering analysis",
            },
        ],
    },

    # ── Section 5: Validated Cost Anchors ─────────────────────────────
    "section_5_cost_anchors": {
        "title": "Validated Cost Anchors",
        "anchors": [
            {
                "component": "ACC — A-frame air-cooled condenser",
                "vendor": "Worldwide Cooling / SPX",
                "value": "$347,200/bay",
                "basis": "Competitive bid, 2024 pricing, delivered to Utah site",
                "confidence": "High — actual bid",
                "date": "2024-Q3",
            },
            {
                "component": "Heat exchangers — shell & tube",
                "vendor": "Precision Heat Exchanger (Cleburne, TX)",
                "value": "$50/ft2 (vaporizer), $43/ft2 (preheater), $37/ft2 (recuperator)",
                "basis": "Indicative budget pricing, direct purchase",
                "confidence": "Medium — budget quote, not firm bid",
                "date": "2024-Q4",
            },
            {
                "component": "Turbine-generator package",
                "vendor": "Multiple (Atlas Copco, GE Nuovo Pignone, Hanbell)",
                "value": "$150-200/kW gross (direct purchase)",
                "basis": "Budget quotes for 50+ MW single-unit ORC TG",
                "confidence": "Medium — range, not firm",
                "date": "2024-Q4",
            },
            {
                "component": "Project baseline — real project reference",
                "vendor": "Turboden (reference: 93 MW lump-sum EPC)",
                "value": "$2,566/kW installed (OEM lump-sum EPC baseline)",
                "basis": "Actual contracted project cost, normalized to $/kW",
                "confidence": "High — contracted price",
                "date": "2024",
            },
            {
                "component": "Vendor equipment-only benchmarks",
                "vendor": "Multiple",
                "value": "Exergy $683/kW, Turboden $760/kW, Baker Hughes $837/kW",
                "basis": "Equipment package only (vendor-comparable basis)",
                "confidence": "High — vendor proposals",
                "date": "2024",
            },
        ],
    },

    # ── Section 6: Open Information Requests (Claude-Maintained) ──────
    "section_6_info_requests": {
        "title": "Open Information Requests",
        "requests": [
            {
                "priority": "HIGH",
                "data_needed": "iC4 transcritical turbine expander — vendor capability & cost",
                "current_assumption": "Subcritical isobutane only; transcritical excluded from optimization",
                "impact_if_wrong": (
                    "Could miss 5-8% efficiency gain at 420°F resource temperature. "
                    "Transcritical iC4 may enable simpler cycle topology with fewer HX."
                ),
                "source": "Atlas Copco, GE, or Hanbell — request for transcritical expander data",
                "status": "open",
            },
            {
                "priority": "HIGH",
                "data_needed": "Site-specific brine chemistry analysis (silica, TDS, pH, gases)",
                "current_assumption": "180°F brine outlet floor based on generic silica saturation",
                "impact_if_wrong": (
                    "If brine allows lower outlet temperature, could gain 5-10 MW additional power. "
                    "If more aggressive scaling, may need higher floor."
                ),
                "source": "Fervo well test data / geochemistry lab",
                "status": "open",
            },
            {
                "priority": "MEDIUM",
                "data_needed": "ACC performance at altitude (site elevation & air density)",
                "current_assumption": "Sea-level air density in fan power calculations",
                "impact_if_wrong": (
                    "Utah site at ~5,500 ft elevation — 15-18% lower air density. "
                    "Fan power may be higher, ACC area larger than modeled."
                ),
                "source": "Site survey / ACC vendor performance curves at altitude",
                "status": "open",
            },
            {
                "priority": "MEDIUM",
                "data_needed": "Firm turbine-generator pricing for 50+ MW single unit",
                "current_assumption": "$150-200/kW gross range from budget quotes",
                "impact_if_wrong": (
                    "TG is ~15% of equipment cost. ±$50/kW swing = ±$2.5M on 50 MW project."
                ),
                "source": "RFQ to Atlas Copco, Hanbell, GE Nuovo Pignone",
                "status": "open",
            },
            {
                "priority": "LOW",
                "data_needed": "Owner self-perform construction labor rates (Utah market)",
                "current_assumption": "$185/kW gross based on industry benchmarks",
                "impact_if_wrong": "±20% on labor = ±$37/kW installed",
                "source": "Fervo construction team / local contractor survey",
                "status": "open",
            },
        ],
    },

    # ── Section 7: Knowledge Base Inventory (Claude-Maintained) ────────
    "section_7_kb_inventory": {
        "title": "Knowledge Base Inventory",
        "entries": [
            {
                "domain": "turbine_technology",
                "coverage": "Budget quotes from 3 vendors, general radial inflow ORC literature",
                "gaps": "No firm pricing; no transcritical expander data",
            },
            {
                "domain": "heat_exchangers",
                "coverage": "Precision HX budget quote, TEMA standards, U-value correlations",
                "gaps": "No brine-side fouling data specific to EGS",
            },
            {
                "domain": "air_cooled_condenser",
                "coverage": "Worldwide Cooling validated bid ($347.2k/bay), fan power model",
                "gaps": "No altitude-corrected performance curves",
            },
            {
                "domain": "working_fluid",
                "coverage": "CoolProp properties for 6 fluids, isopentane optimization data",
                "gaps": "No transcritical iC4 cycle modeling; no fluid degradation data",
            },
            {
                "domain": "construction_schedule",
                "coverage": "70-week critical path estimate, phase breakdown",
                "gaps": "No site-specific construction productivity data",
            },
            {
                "domain": "economics_market",
                "coverage": "PPA target $35/MWh, 8% discount rate, 30-year life",
                "gaps": "No ITC/PTC analysis; no capacity market revenue modeling",
            },
        ],
    },

    # ── Section 8: Optimization History Log (Claude-Maintained) ────────
    "section_8_opt_history": {
        "title": "Optimization History",
        "entries": [
            {
                "date": "2026-03-08",
                "event": "Initial design basis established",
                "finding": (
                    "Baseline OEM lump-sum EPC at ~$2,566/kW. Direct self-perform strategy "
                    "reduces installed cost to $1,600-1,900/kW range. Isopentane subcritical "
                    "recuperated cycle with direct ACC is current optimum topology."
                ),
                "action": "Set $2,000/kW target with direct self-perform as primary strategy",
            },
            {
                "date": "2026-03-08",
                "event": "Complexity penalty framework introduced",
                "finding": (
                    "Recuperator adds $18/kW lifecycle complexity cost. Propane intermediate loop "
                    "adds $50/kW. Dual-pressure adds $100/kW. These NPV-adjusted costs are not "
                    "captured in capital $/kW alone."
                ),
                "action": "Optimizer now minimizes total_adjusted_per_kW (installed + complexity NPV)",
            },
        ],
    },
}


# ── Load / Save ────────────────────────────────────────────────────────────

def load_dbd() -> dict:
    """Load DBD from JSON, creating from defaults if missing."""
    if os.path.exists(DBD_PATH):
        try:
            with open(DBD_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
    # Create from defaults
    save_dbd(DEFAULT_DBD)
    return DEFAULT_DBD.copy()


def _navigate_path(obj: dict, dotted_path: str):
    """Navigate a dotted path like 'entries' or 'heat_exchangers.U_values_Btu_hr_ft2_F'
    into a nested dict/list structure.

    Returns (parent, final_key) so the caller can read or write the target.
    For a simple key like 'entries', returns (obj, 'entries').
    For a path like 'a.b.c', returns (obj['a']['b'], 'c').
    """
    parts = dotted_path.split(".")
    current = obj
    for part in parts[:-1]:
        if isinstance(current, dict):
            current = current.get(part, {})
        else:
            return None, None
    return current, parts[-1]


def apply_dbd_updates(proposals: list[dict]) -> tuple[dict, str]:
    """Apply accepted/modified proposals to the Design Basis Document.

    Only processes items where decision is 'accepted' or 'modified'.

    Actions:
      APPEND  — append new_value to a list at section→target_path
      UPDATE  — replace the value at section→target_path
      NEW     — same as APPEND for list fields
      CLOSE   — set status='closed' + closed_date on matching info request

    Returns (updated_dbd, new_version).
    """
    dbd = load_dbd()

    for item in proposals:
        decision = item.get("decision", "pending")
        if decision not in ("accepted", "modified"):
            continue

        section_key = item.get("section", "")
        action = item.get("action", "").upper()
        target_path = item.get("target_path", "")
        new_value = item.get("new_value")

        if section_key not in dbd:
            continue

        section = dbd[section_key]

        if action in ("APPEND", "NEW"):
            parent, key = _navigate_path(section, target_path)
            if parent is None or key is None:
                continue
            target = parent.get(key) if isinstance(parent, dict) else None
            if isinstance(target, list):
                target.append(new_value)
            elif isinstance(parent, dict):
                # If target doesn't exist yet, create it as a list
                parent[key] = [new_value]

        elif action == "UPDATE":
            parent, key = _navigate_path(section, target_path)
            if parent is None or key is None:
                continue
            if isinstance(parent, dict):
                parent[key] = new_value

        elif action == "CLOSE":
            # For info_requests: find the matching request and close it
            parent, key = _navigate_path(section, target_path)
            if parent is None or key is None:
                continue
            target = parent.get(key) if isinstance(parent, dict) else None
            if isinstance(target, list):
                # Match by data_needed field if provided in new_value
                match_field = (new_value or {}).get("data_needed", "") if isinstance(new_value, dict) else ""
                for entry in target:
                    if isinstance(entry, dict) and entry.get("data_needed", "") == match_field:
                        entry["status"] = "closed"
                        entry["closed_date"] = datetime.now(timezone.utc).strftime("%Y-%m-%d")
                        break

    dbd["last_updated_by"] = "optimizer"
    save_dbd(dbd)
    return dbd, dbd.get("version", "?")


def save_dbd(dbd: dict) -> None:
    """Write DBD to JSON, auto-incrementing version and timestamping."""
    # Auto-increment version
    old_version = dbd.get("version", "1.0")
    try:
        major, minor = old_version.split(".")
        dbd["version"] = f"{major}.{int(minor) + 1}"
    except (ValueError, AttributeError):
        dbd["version"] = "1.1"

    dbd["last_updated"] = datetime.now(timezone.utc).isoformat()

    os.makedirs(os.path.dirname(DBD_PATH), exist_ok=True)
    with open(DBD_PATH, "w", encoding="utf-8") as f:
        json.dump(dbd, f, indent=2, ensure_ascii=False)


# ── Streamlit Tab Renderer ─────────────────────────────────────────────────

def render_dbd_tab(design_basis: dict):
    """Render the Design Basis Document tab."""
    dbd = load_dbd()

    st.header("Living Design Basis Document")
    st.caption(
        f"Version **{dbd.get('version', '?')}** | "
        f"Last updated: {dbd.get('last_updated', 'N/A')} | "
        f"By: {dbd.get('last_updated_by', 'N/A')}"
    )

    # ── Section 1: Project Identity ────────────────────────────────────
    _render_section_1(dbd)

    # ── Section 2: Design Philosophy ───────────────────────────────────
    _render_section_2(dbd)

    # ── Section 3: Equipment Design Standards ──────────────────────────
    _render_section_3(dbd)

    # ── Section 4: Physical Constraints ────────────────────────────────
    _render_section_4(dbd)

    # ── Section 5: Validated Cost Anchors ──────────────────────────────
    _render_section_5(dbd)

    # ── Section 6: Open Information Requests ───────────────────────────
    _render_section_6(dbd)

    # ── Section 7: Knowledge Base Inventory ────────────────────────────
    _render_section_7(dbd)

    # ── Section 8: Optimization History ────────────────────────────────
    _render_section_8(dbd)


def _render_section_1(dbd: dict):
    sec = dbd.get("section_1_identity", {})
    with st.expander("Section 1: Project Identity & Strategic Objective", expanded=True):
        st.markdown(f"**Project:** {sec.get('project_name', 'N/A')}")
        st.markdown(f"**Developer:** {sec.get('developer', 'N/A')}")
        st.markdown(f"**Location:** {sec.get('location', 'N/A')}")
        st.markdown(f"**Resource:** {sec.get('resource_type', 'N/A')}")
        st.markdown(f"**Strategic Objective:** {sec.get('strategic_objective', 'N/A')}")
        st.markdown(
            f"**Design Life:** {sec.get('design_life_years', 'N/A')} years | "
            f"**Capacity Factor:** {sec.get('capacity_factor_pct', 'N/A')}% | "
            f"**PPA Target:** ${sec.get('ppa_target_per_MWh', 'N/A')}/MWh"
        )
        _edit_button("section_1_identity", dbd)


def _render_section_2(dbd: dict):
    sec = dbd.get("section_2_philosophy", {})
    with st.expander("Section 2: Design Philosophy"):
        values = sec.get("values", [])
        for i, val in enumerate(values, 1):
            st.markdown(f"**{i}. {val.get('name', '')}**")
            st.markdown(f"> {val.get('description', '')}")
        _edit_button("section_2_philosophy", dbd)


def _render_section_3(dbd: dict):
    sec = dbd.get("section_3_equipment", {})
    with st.expander("Section 3: Equipment Design Standards"):
        for equip_key in ["turbine_generator", "heat_exchangers", "acc", "pumps", "piping"]:
            equip = sec.get(equip_key, {})
            if not equip:
                continue
            st.markdown(f"**{equip_key.replace('_', ' ').title()}**")
            for k, v in equip.items():
                if isinstance(v, dict):
                    st.markdown(f"- *{k.replace('_', ' ').title()}:*")
                    for dk, dv in v.items():
                        st.markdown(f"  - {dk}: {dv}")
                else:
                    st.markdown(f"- *{k.replace('_', ' ').title()}:* {v}")
            st.markdown("---")
        _edit_button("section_3_equipment", dbd)


def _render_section_4(dbd: dict):
    sec = dbd.get("section_4_constraints", {})
    with st.expander("Section 4: Physical Constraints — Non-Negotiable"):
        constraints = sec.get("constraints", [])
        for c in constraints:
            st.markdown(f"**{c.get('name', '')}**")
            st.markdown(f"- Value: {c.get('value', '')}")
            st.markdown(f"- Basis: {c.get('basis', '')}")
            st.markdown(f"- Override: {c.get('override_condition', 'None')}")
            st.markdown("---")
        _edit_button("section_4_constraints", dbd)


def _render_section_5(dbd: dict):
    sec = dbd.get("section_5_cost_anchors", {})
    with st.expander("Section 5: Validated Cost Anchors"):
        anchors = sec.get("anchors", [])
        if anchors:
            import pandas as pd
            rows = []
            for a in anchors:
                rows.append({
                    "Component": a.get("component", ""),
                    "Vendor": a.get("vendor", ""),
                    "Value": a.get("value", ""),
                    "Basis": a.get("basis", ""),
                    "Confidence": a.get("confidence", ""),
                    "Date": a.get("date", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        _edit_button("section_5_cost_anchors", dbd)


def _render_section_6(dbd: dict):
    sec = dbd.get("section_6_info_requests", {})
    with st.expander("Section 6: Open Information Requests"):
        requests = sec.get("requests", [])
        if requests:
            import pandas as pd
            rows = []
            for r in requests:
                rows.append({
                    "Priority": r.get("priority", ""),
                    "Data Needed": r.get("data_needed", ""),
                    "Current Assumption": r.get("current_assumption", ""),
                    "Impact if Wrong": r.get("impact_if_wrong", ""),
                    "Source": r.get("source", ""),
                    "Status": r.get("status", "open"),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No open information requests.")
        st.caption("Claude-maintained section — updated after optimizer rounds")


def _render_section_7(dbd: dict):
    sec = dbd.get("section_7_kb_inventory", {})
    with st.expander("Section 7: Knowledge Base Inventory"):
        entries = sec.get("entries", [])
        if entries:
            import pandas as pd
            rows = []
            for e in entries:
                rows.append({
                    "Domain": e.get("domain", ""),
                    "Coverage": e.get("coverage", ""),
                    "Gaps": e.get("gaps", ""),
                })
            st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)
        else:
            st.info("No knowledge base inventory entries.")
        st.caption("Claude-maintained section — updated after optimizer rounds")


def _render_section_8(dbd: dict):
    sec = dbd.get("section_8_opt_history", {})
    with st.expander("Section 8: Optimization History"):
        entries = sec.get("entries", [])
        if entries:
            # Reverse chronological
            for entry in reversed(entries):
                st.markdown(f"**{entry.get('date', '')}** — {entry.get('event', '')}")
                st.markdown(f"> {entry.get('finding', '')}")
                st.markdown(f"Action: _{entry.get('action', '')}_")
                st.markdown("---")
        else:
            st.info("No optimization history entries yet.")
        st.caption("Claude-maintained section — updated after optimizer rounds")


def _edit_button(section_key: str, dbd: dict):
    """Add an Edit Section button for user-editable sections (1-5)."""
    edit_key = f"dbd_editing_{section_key}"

    if st.session_state.get(edit_key, False):
        # Show text area with current JSON
        current = json.dumps(dbd.get(section_key, {}), indent=2)
        new_text = st.text_area(
            f"Edit {section_key}",
            value=current,
            height=300,
            key=f"dbd_editor_{section_key}",
        )
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Save", key=f"dbd_save_{section_key}", type="primary"):
                try:
                    parsed = json.loads(new_text)
                    dbd[section_key] = parsed
                    dbd["last_updated_by"] = "user"
                    save_dbd(dbd)
                    st.session_state[edit_key] = False
                    st.success("Saved.")
                    st.rerun()
                except json.JSONDecodeError as e:
                    st.error(f"Invalid JSON: {e}")
        with col2:
            if st.button("Cancel", key=f"dbd_cancel_{section_key}"):
                st.session_state[edit_key] = False
                st.rerun()
    else:
        if st.button("Edit Section", key=f"dbd_edit_btn_{section_key}"):
            st.session_state[edit_key] = True
            st.rerun()
