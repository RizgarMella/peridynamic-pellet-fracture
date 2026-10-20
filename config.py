"""config.py - shared defaults: UO2 properties, calibration mode, standard histories, cost model.

CALIBRATION (read DESIGN.md section 3 before changing)
  A single critical stretch cannot match both strength and fracture energy unless
  the horizon delta is the material length delta_mat = G_c / (C (sigma_f/E)^2)
  (~0.1 mm for UO2: R/delta ~ 40, too expensive for a 600-run campaign).
  Two modes are therefore supported everywhere:
    "strength": s_c = sigma_f / E with a mesh-independent Weibull GRAIN field
                (wcorr fixed in mm).  delta is a fixed physical length chosen by
                the mesh-convergence study; dissipated energy G_eff = C s_c^2 delta
                is recorded.  Default for the campaign (initiation-controlled problem).
    "energy":   s_c from G_c at the chosen delta (strength then scales ~delta^-1/2).
  The mesh-convergence study runs BOTH so the choice is evidence-based.
"""
from __future__ import annotations

import json
import math
import os

# ---- UO2 reference properties (assumed nominal values; replace with your data) ----
UO2 = dict(E=200e9, nu=0.32, alpha=1.0e-5, Gc=10.0, kth=3.0, Tref=293.15, gsurf=0.0192)
SIGMA_F = 130e6          # assumed tensile/fracture strength [Pa] (porosity dependent)
GEOM = dict(R=4.1e-3, H=9.8e-3)
CREEP = dict(tauRef=3600.0, TtauRef=1700.0, Qact=380e3, lamv=1.0)   # assumed relaxation law

# ---- default calibration (overridden by meshconv/derived/convergence.json when it exists) ----
CALIB = dict(mode="strength", sigmaf=SIGMA_F, delta=0.40e-3, mfac=3.0, weibull=10.0, wcorr=0.25e-3)

# ---- where to run: cheapest Spot capacity found (Oct 2026, gcloud-compute.com tracker; Spot prices
# change up to daily - re-check before a large submission). Keep the BUCKET in the SAME region:
# compute and storage co-located means no inter-region transfer charges.
# Prices are in pounds sterling, converted from Google's USD list prices at GBP_PER_USD below
# (a UK billing account is invoiced in GBP; update the rate and prices together).
#   c3d-highcpu-30 Spot: australia-southeast1 ~£0.0032/vCPU-h, us-central1 ~£0.0071/vCPU-h
#   c4d-highcpu-32 Spot: us-west4 ~£0.0059/vCPU-h;  t2d-standard-32: europe-southwest1 ~£0.0055/vCPU-h
GBP_PER_USD = 0.75             # exchange rate used for the prices below (20 Oct 2026)
REGION = "australia-southeast1"
MACHINE = "c3d-highcpu-30"
IMAGE = f"{REGION}-docker.pkg.dev/PROJECT/pellet/pellet-pd:2"

# ---- cost model (refine with bench.py results) ----
COST = dict(
    gbes_per_vcpu=0.20,          # G bond-evaluations per second per vCPU (2-vCPU sandbox measured ~0.23)
    iters_per_step_per_nr=110.0, # ADR iterations per load step per particle-across-radius (pilot runs)
    gbp_per_vcpu_h=0.00315,      # £ Spot c3d-highcpu-30, australia-southeast1, Oct 2026 ($0.0042 x 0.75; update from bench)
)


def load_calibration(path: str = "studies/meshconv/derived/convergence.json") -> dict:
    c = dict(CALIB)
    if os.path.exists(path):
        with open(path) as f:
            rec = json.load(f)
        for k in ("mode", "delta", "mfac", "sigmaf", "weibull", "wcorr"):
            if rec.get(k) is not None:
                c[k] = rec[k]
        c["source"] = path
    else:
        c["source"] = "config.CALIB defaults (mesh-convergence study not yet analysed)"
    return c


def calib_params(c: dict) -> dict:
    """Solver options implementing a calibration record."""
    p = dict(delta=c["delta"], mfac=c["mfac"], weibull=c["weibull"], wcorr=c["wcorr"])
    if c["mode"] == "strength":
        p["sigmaf"] = c["sigmaf"]
    return p


# ---- histories (rows: time_h, q_Wpm[, p_Pa]) ----
def ramp(q_peak: float, t_ramp_h: float = 1.0):
    return [(0.0, 0.0), (t_ramp_h, q_peak)]


def ramp_hold_cool(q_peak: float, t_ramp_h: float = 2.0, t_hold_h: float = 24.0, t_cool_h: float = 2.0, p_Pa: float = 0.0):
    return [(0.0, 0.0, p_Pa), (t_ramp_h, q_peak, p_Pa), (t_ramp_h + t_hold_h, q_peak, p_Pa),
            (t_ramp_h + t_hold_h + t_cool_h, 0.0, p_Pa)]


# ---- size / cost estimates ----
def estimate_run(R: float, H: float, delta: float, mfac: float, n_steps: int, n_bodies_h: float = 1.0,
                 creep: bool = False) -> dict:
    nr = max(4, round(R * mfac / delta))
    dx = R / nr
    N = math.pi * R * R * H * n_bodies_h / dx ** 3
    fam = 4.0 / 3.0 * math.pi * ((delta + 0.5 * dx) / dx) ** 3 * 0.85          # surface truncation ~15 %
    NB = N * fam
    iters = COST["iters_per_step_per_nr"] * nr * n_steps
    work = 2.0 * NB * iters
    vcpu_h = work / (COST["gbes_per_vcpu"] * 1e9) / 3600.0
    mem_gb = (NB * (23 if creep else 17) + N * 330) / 2 ** 30
    return dict(nr=nr, dx=dx, N=int(N), NB=int(NB), iters=int(iters), vcpu_h=vcpu_h, gbp=vcpu_h * COST["gbp_per_vcpu_h"],
                mem_gb=mem_gb)


def steps_in_history(rows, dqmax: float, dtmax_h: float | None) -> int:
    n = 0
    for a, b in zip(rows[:-1], rows[1:]):
        k = max(1, math.ceil(abs(b[1] - a[1]) / dqmax - 1e-9))
        if dtmax_h:
            k = max(k, math.ceil((b[0] - a[0]) / dtmax_h - 1e-9))
        n += k
    return n
