"""
Fluid property abstraction layer.
Tries REFPROP bridge at localhost:8000 first, falls back to CoolProp.
All public methods use imperial units (°F, psia, BTU/lb, BTU/lb·R, lb/ft³).
CoolProp uses SI internally — conversions happen at the boundary.
"""

import warnings
import requests
import numpy as np

try:
    import CoolProp.CoolProp as CP
    HAS_COOLPROP = True
except ImportError:
    HAS_COOLPROP = False


# ── Unit conversion helpers ──────────────────────────────────────────────────

def f_to_k(t_f):
    """°F → K"""
    return (t_f - 32) * 5 / 9 + 273.15

def k_to_f(t_k):
    """K → °F"""
    return (t_k - 273.15) * 9 / 5 + 32

def psia_to_pa(p_psia):
    """psia → Pa"""
    return p_psia * 6894.757

def pa_to_psia(p_pa):
    """Pa → psia"""
    return p_pa / 6894.757

def j_to_btu(j):
    """J/kg → BTU/lb"""
    return j / 2326.0

def btu_to_j(btu):
    """BTU/lb → J/kg"""
    return btu * 2326.0

def j_per_kgK_to_btu_per_lbR(s):
    """J/(kg·K) → BTU/(lb·R)"""
    return s / (2326.0 * 5 / 9)

def btu_per_lbR_to_j_per_kgK(s):
    """BTU/(lb·R) → J/(kg·K)"""
    return s * (2326.0 * 5 / 9)

def kgm3_to_lbft3(rho):
    """kg/m³ → lb/ft³"""
    return rho * 0.062428

def lbft3_to_kgm3(rho):
    """lb/ft³ → kg/m³"""
    return rho / 0.062428

def w_to_btu_per_hr(w):
    """W → BTU/hr"""
    return w * 3.41214

def lbhr_to_kgs(m):
    """lb/hr → kg/s"""
    return m * 0.000125998


# ── Fluid name mapping ──────────────────────────────────────────────────────

FLUID_MAP = {
    "isopentane": "Isopentane",
    "propane": "Propane",
    "isobutane": "Isobutane",
    "n-pentane": "n-Pentane",
    "r245fa": "R245fa",
    "cyclopentane": "CycloPentane",
}


def _coolprop_name(fluid):
    """Normalise user-facing fluid name to CoolProp identifier."""
    return FLUID_MAP.get(fluid.lower(), fluid)


class FluidProperties:
    """Unified interface for thermodynamic fluid properties in imperial units."""

    def __init__(self, refprop_url="http://localhost:8000"):
        self.refprop_url = refprop_url
        self.use_refprop = self._check_refprop()
        if not self.use_refprop and not HAS_COOLPROP:
            raise RuntimeError("Neither REFPROP bridge nor CoolProp is available.")

    # ── REFPROP availability ─────────────────────────────────────────────

    def _check_refprop(self):
        try:
            r = requests.get(f"{self.refprop_url}/health", timeout=1)
            return r.status_code == 200
        except Exception:
            return False

    # ── Core CoolProp wrapper (SI) ───────────────────────────────────────

    def _cp_props(self, fluid, output, input1, val1, input2, val2):
        """Call CoolProp.PropsSI and return the result."""
        name = _coolprop_name(fluid)
        try:
            result = CP.PropsSI(output, input1, val1, input2, val2, name)
            if np.isnan(result) or np.isinf(result):
                raise ValueError(f"CoolProp returned {result}")
            return result
        except Exception as e:
            raise ValueError(
                f"CoolProp error for {name}: {output}({input1}={val1}, "
                f"{input2}={val2}): {e}"
            )

    # ── REFPROP bridge call ──────────────────────────────────────────────

    def _refprop_call(self, endpoint, payload):
        r = requests.post(f"{self.refprop_url}{endpoint}", json=payload, timeout=5)
        r.raise_for_status()
        return r.json()

    # ── Public methods (all imperial) ────────────────────────────────────

    def saturation_props(self, fluid, T=None, P=None):
        """
        Return saturation properties at given T (°F) or P (psia).

        Returns dict with keys:
            T_sat (°F), P_sat (psia),
            h_f, h_g (BTU/lb), s_f, s_g (BTU/lb·R),
            rho_f, rho_g (lb/ft³)
        """
        if self.use_refprop:
            payload = {"fluid": fluid}
            if T is not None:
                payload["T"] = T
            if P is not None:
                payload["P"] = P
            return self._refprop_call("/saturation", payload)

        name = _coolprop_name(fluid)
        if T is not None:
            T_k = f_to_k(T)
            P_pa = self._cp_props(name, "P", "T", T_k, "Q", 0)
        elif P is not None:
            P_pa = psia_to_pa(P)
            T_k = self._cp_props(name, "T", "P", P_pa, "Q", 0)
        else:
            raise ValueError("Must provide T or P")

        h_f = self._cp_props(name, "H", "T", T_k, "Q", 0)
        h_g = self._cp_props(name, "H", "T", T_k, "Q", 1)
        s_f = self._cp_props(name, "S", "T", T_k, "Q", 0)
        s_g = self._cp_props(name, "S", "T", T_k, "Q", 1)
        rho_f = self._cp_props(name, "D", "T", T_k, "Q", 0)
        rho_g = self._cp_props(name, "D", "T", T_k, "Q", 1)

        return {
            "T_sat": k_to_f(T_k),
            "P_sat": pa_to_psia(P_pa),
            "h_f": j_to_btu(h_f),
            "h_g": j_to_btu(h_g),
            "s_f": j_per_kgK_to_btu_per_lbR(s_f),
            "s_g": j_per_kgK_to_btu_per_lbR(s_g),
            "rho_f": kgm3_to_lbft3(rho_f),
            "rho_g": kgm3_to_lbft3(rho_g),
        }

    def state_point(self, fluid, prop1, val1, prop2, val2):
        """
        General state point calculation.

        prop1/prop2: one of 'T' (°F), 'P' (psia), 'H' (BTU/lb),
                     'S' (BTU/lb·R), 'Q' (0-1)
        Returns dict: T (°F), P (psia), h (BTU/lb), s (BTU/lb·R),
                      rho (lb/ft³), phase
        """
        if self.use_refprop:
            payload = {
                "fluid": fluid,
                "prop1": prop1, "val1": val1,
                "prop2": prop2, "val2": val2,
            }
            return self._refprop_call("/state_point", payload)

        name = _coolprop_name(fluid)
        # Convert inputs to SI
        p1_si, v1_si = self._to_si(prop1, val1)
        p2_si, v2_si = self._to_si(prop2, val2)

        T_k = self._cp_props(name, "T", p1_si, v1_si, p2_si, v2_si)
        P_pa = self._cp_props(name, "P", p1_si, v1_si, p2_si, v2_si)
        h_j = self._cp_props(name, "H", p1_si, v1_si, p2_si, v2_si)
        s_j = self._cp_props(name, "S", p1_si, v1_si, p2_si, v2_si)
        rho = self._cp_props(name, "D", p1_si, v1_si, p2_si, v2_si)

        # Determine phase
        try:
            phase_idx = self._cp_props(name, "Phase", p1_si, v1_si, p2_si, v2_si)
            phase = self._phase_name(phase_idx)
        except Exception:
            phase = "unknown"

        return {
            "T": k_to_f(T_k),
            "P": pa_to_psia(P_pa),
            "h": j_to_btu(h_j),
            "s": j_per_kgK_to_btu_per_lbR(s_j),
            "rho": kgm3_to_lbft3(rho),
            "phase": phase,
        }

    def vapor_density(self, fluid, T, P):
        """Vapor density at T (°F) and P (psia) → lb/ft³."""
        sp = self.state_point(fluid, "T", T, "P", P)
        return sp["rho"]

    def enthalpy(self, fluid, T, P):
        """Enthalpy at T (°F) and P (psia) → BTU/lb."""
        sp = self.state_point(fluid, "T", T, "P", P)
        return sp["h"]

    def entropy(self, fluid, T, P):
        """Entropy at T (°F) and P (psia) → BTU/(lb·R)."""
        sp = self.state_point(fluid, "T", T, "P", P)
        return sp["s"]

    def critical_point(self, fluid):
        """Return critical temperature (°F) and pressure (psia) for a fluid."""
        name = _coolprop_name(fluid)
        T_crit_K = CP.PropsSI("Tcrit", name)
        P_crit_Pa = CP.PropsSI("pcrit", name)
        return {
            "T_crit": k_to_f(T_crit_K),
            "P_crit": pa_to_psia(P_crit_Pa),
        }

    def latent_heat(self, fluid, T):
        """Latent heat of vaporisation at T (°F) → BTU/lb."""
        sat = self.saturation_props(fluid, T=T)
        return sat["h_g"] - sat["h_f"]

    def density_at_state(self, fluid, prop1, val1, prop2, val2):
        """Density via general state point → lb/ft³."""
        sp = self.state_point(fluid, prop1, val1, prop2, val2)
        return sp["rho"]

    # ── Internal helpers ─────────────────────────────────────────────────

    def _to_si(self, prop, val):
        """Convert an imperial property/value pair to SI for CoolProp."""
        prop_upper = prop.upper()
        if prop_upper == "T":
            return "T", f_to_k(val)
        elif prop_upper == "P":
            return "P", psia_to_pa(val)
        elif prop_upper == "H":
            return "H", btu_to_j(val)
        elif prop_upper == "S":
            return "S", btu_per_lbR_to_j_per_kgK(val)
        elif prop_upper == "Q":
            return "Q", val
        elif prop_upper == "D":
            return "D", lbft3_to_kgm3(val)
        else:
            return prop, val

    @staticmethod
    def _phase_name(idx):
        """Map CoolProp phase index to human-readable name."""
        phases = {
            0: "liquid",
            1: "supercritical",
            2: "supercritical_gas",
            3: "supercritical_liquid",
            5: "gas",
            6: "twophase",
            8: "not_imposed",
        }
        return phases.get(int(idx), "unknown")
