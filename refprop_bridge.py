"""
Optional FastAPI REFPROP Bridge Service

This service wraps NIST REFPROP via the ctREFPROP Python interface and
exposes fluid property calculations as a REST API. The main application
(fluid_properties.py) calls this service when available, otherwise falls
back to CoolProp.

Setup Instructions:
  1. Install REFPROP (v10+) from NIST. Note the installation directory.
  2. Install the Python wrapper:
       pip install ctREFPROP
  3. Set the RPPREFIX environment variable to your REFPROP install path:
       export RPPREFIX="C:/Program Files (x86)/REFPROP"  (Windows)
       export RPPREFIX="/opt/refprop"                      (Linux)
  4. Run this service:
       uvicorn refprop_bridge:app --host 0.0.0.0 --port 8000
  5. The main app will auto-detect the bridge at localhost:8000.

Dependencies: fastapi, uvicorn, ctREFPROP, numpy
"""

import os
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import numpy as np

app = FastAPI(title="REFPROP Bridge", version="1.0")

# ── REFPROP initialization ───────────────────────────────────────────────────

RP = None

def _init_refprop():
    global RP
    if RP is not None:
        return
    try:
        from ctREFPROP.ctREFPROP import REFPROPFunctionLibrary
        rp_path = os.environ.get("RPPREFIX", r"C:\Program Files (x86)\REFPROP")
        RP = REFPROPFunctionLibrary(rp_path)
        RP.SETPATHdll(rp_path)
    except Exception as e:
        raise RuntimeError(f"Failed to initialize REFPROP: {e}")


# ── Unit conversions (imperial ↔ SI for REFPROP) ────────────────────────────

def f_to_k(t_f):
    return (t_f - 32) * 5 / 9 + 273.15

def k_to_f(t_k):
    return (t_k - 273.15) * 9 / 5 + 32

def psia_to_kpa(p):
    return p * 6.89476

def kpa_to_psia(p):
    return p / 6.89476

def j_to_btu(j):
    return j / 2326.0

def jk_to_btuR(s):
    return s / (2326.0 * 5 / 9)

def kgm3_to_lbft3(rho):
    return rho * 0.062428


# ── REFPROP fluid name mapping ───────────────────────────────────────────────

FLUID_MAP = {
    "isopentane": "IPENTANE",
    "propane": "PROPANE",
    "isobutane": "ISOBUTAN",
    "n-pentane": "PENTANE",
    "r245fa": "R245FA",
}

def _rp_name(fluid):
    return FLUID_MAP.get(fluid.lower(), fluid.upper()) + ".FLD"


# ── Request / Response models ────────────────────────────────────────────────

class SaturationRequest(BaseModel):
    fluid: str
    T: Optional[float] = None  # °F
    P: Optional[float] = None  # psia

class StatePointRequest(BaseModel):
    fluid: str
    prop1: str
    val1: float
    prop2: str
    val2: float


# ── Endpoints ────────────────────────────────────────────────────────────────

@app.get("/health")
def health():
    try:
        _init_refprop()
        return {"status": "ok", "backend": "REFPROP"}
    except Exception as e:
        raise HTTPException(status_code=503, detail=str(e))


@app.post("/saturation")
def saturation(req: SaturationRequest):
    _init_refprop()
    fluid_file = _rp_name(req.fluid)

    try:
        if req.T is not None:
            T_k = f_to_k(req.T)
            # REFPROP SATP with temperature input
            result = RP.REFPROPdll(
                fluid_file, "TQ", "P;H;S;D",
                RP.MOLAR_BASE_SI, 0, 0, T_k, 0, [1.0]
            )
            P_kpa = result.Output[0] / 1000  # Pa → kPa
            h_f_j = result.Output[1]
            s_f_j = result.Output[2]
            rho_f = result.Output[3]

            result_g = RP.REFPROPdll(
                fluid_file, "TQ", "H;S;D",
                RP.MOLAR_BASE_SI, 0, 0, T_k, 1, [1.0]
            )
            h_g_j = result_g.Output[0]
            s_g_j = result_g.Output[1]
            rho_g = result_g.Output[2]

            T_sat = req.T
            P_sat = kpa_to_psia(P_kpa)

        elif req.P is not None:
            P_pa = psia_to_kpa(req.P) * 1000
            result = RP.REFPROPdll(
                fluid_file, "PQ", "T;H;S;D",
                RP.MOLAR_BASE_SI, 0, 0, P_pa, 0, [1.0]
            )
            T_k = result.Output[0]
            h_f_j = result.Output[1]
            s_f_j = result.Output[2]
            rho_f = result.Output[3]

            result_g = RP.REFPROPdll(
                fluid_file, "PQ", "H;S;D",
                RP.MOLAR_BASE_SI, 0, 0, P_pa, 1, [1.0]
            )
            h_g_j = result_g.Output[0]
            s_g_j = result_g.Output[1]
            rho_g = result_g.Output[2]

            T_sat = k_to_f(T_k)
            P_sat = req.P
        else:
            raise HTTPException(status_code=400, detail="Provide T or P")

        return {
            "T_sat": T_sat,
            "P_sat": P_sat,
            "h_f": j_to_btu(h_f_j),
            "h_g": j_to_btu(h_g_j),
            "s_f": jk_to_btuR(s_f_j),
            "s_g": jk_to_btuR(s_g_j),
            "rho_f": kgm3_to_lbft3(rho_f),
            "rho_g": kgm3_to_lbft3(rho_g),
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"REFPROP error: {e}")


@app.post("/state_point")
def state_point(req: StatePointRequest):
    _init_refprop()
    fluid_file = _rp_name(req.fluid)

    prop_map = {"T": "T", "P": "P", "H": "H", "S": "S", "Q": "Q", "D": "D"}
    p1 = prop_map.get(req.prop1.upper(), req.prop1.upper())
    p2 = prop_map.get(req.prop2.upper(), req.prop2.upper())

    # Convert input values to SI
    v1 = _to_si_val(req.prop1, req.val1)
    v2 = _to_si_val(req.prop2, req.val2)

    try:
        result = RP.REFPROPdll(
            fluid_file, f"{p1}{p2}", "T;P;H;S;D",
            RP.MOLAR_BASE_SI, 0, 0, v1, v2, [1.0]
        )
        T_k = result.Output[0]
        P_pa = result.Output[1]
        h_j = result.Output[2]
        s_j = result.Output[3]
        rho = result.Output[4]

        return {
            "T": k_to_f(T_k),
            "P": kpa_to_psia(P_pa / 1000),
            "h": j_to_btu(h_j),
            "s": jk_to_btuR(s_j),
            "rho": kgm3_to_lbft3(rho),
            "phase": "unknown",
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"REFPROP error: {e}")


def _to_si_val(prop, val):
    """Convert an imperial value to SI for REFPROP."""
    p = prop.upper()
    if p == "T":
        return f_to_k(val)
    elif p == "P":
        return psia_to_kpa(val) * 1000  # Pa
    elif p == "H":
        return val * 2326.0  # BTU/lb → J/kg
    elif p == "S":
        return val * (2326.0 * 5 / 9)  # BTU/(lb·R) → J/(kg·K)
    elif p == "Q":
        return val
    elif p == "D":
        return val / 0.062428  # lb/ft³ → kg/m³
    return val


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
