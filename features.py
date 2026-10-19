"""features.py - dimensionless inputs and targets for the fragment-size model.

ONE implementation used by campaign.py (dataset), fit_pysr.py (training) and
predict_fragments.py (end users), so a prediction is always computed from
exactly the features the model was trained on.

Inputs a user supplies: power history (time_h, q_Wpm), geometry (R, H, dish),
material (E, nu, alpha, k, strength sigma_f or G_c, Weibull m, creep law).

Dimensionless groups
  Pi_T      = alpha E dT0 / ((1-nu) sigma_c)   peak elastic thermal stress / strength,
              dT0 = q_max / (4 pi k) (centre-surface difference at peak power)
  Pi_R      = R / l_ch, l_ch = E G_c / sigma_c^2   size / Irwin length (brittleness)
  Pi_H      = H / (2R)                          aspect ratio
  Pi_dish   = dish volume / pellet volume
  nu, weibull_m                                 (already dimensionless)
  Pi_creep  = log10(t_hot / tau(T_c,max))       relaxation achieved at power
  n_cycles  = power cycles above 50 % of q_max
  Pi_drop   = largest single power decrease / q_max
  n_shutdowns = returns below 5 % of q_max after operation
Targets (log10, base-10 so equations read naturally)
  log10_nfrag, log10_deq_vw_over_R, log10_Leq_over_R, log10_specific_area (A R / V)
"""
from __future__ import annotations

import math

import numpy as np

RG = 8.314462618
C_LPS = lambda E, nu: (3 * E / (2 * (1 + nu)) + 0.75 ** 4 * (E / (3 * (1 - 2 * nu)) - 5 * E / (6 * (1 + nu))))

FEATURES = ["Pi_T", "Pi_R", "Pi_H", "Pi_dish", "nu", "weibull_m", "Pi_creep", "n_cycles", "Pi_drop", "n_shutdowns"]
TARGETS = ["log10_nfrag", "log10_deq_vw_over_R", "log10_Leq_over_R", "log10_specific_area"]


def history_features(rows) -> dict:
    h = np.asarray(rows, float)
    t, q = h[:, 0], h[:, 1]
    qmax = float(q.max())
    if qmax <= 0:
        raise ValueError("history has no power")
    above = q >= 0.5 * qmax
    n_cycles = int(above[0]) + int(np.sum(~above[:-1] & above[1:]))
    tt = np.linspace(t[0], t[-1], 20001)
    qq = np.interp(tt, t, q)
    t_hot = float(np.mean(qq >= 0.8 * qmax) * (t[-1] - t[0]))
    drops, k = [], 0
    while k < len(q) - 1:
        if q[k + 1] < q[k]:
            j = k
            while j < len(q) - 1 and q[j + 1] <= q[j]:
                j += 1
            drops.append(q[k] - q[j])
            k = j
        else:
            k += 1
    rates = [abs(q[i + 1] - q[i]) / max(t[i + 1] - t[i], 1e-6) for i in range(len(q) - 1) if q[i + 1] != q[i]]
    n_shut, op = 0, False
    for v in qq:
        if v > 0.5 * qmax:
            op = True
        elif op and v <= 0.05 * qmax:
            n_shut += 1
            op = False
    return dict(q_max_Wpm=qmax, n_cycles=n_cycles, t_hot_h=t_hot, Pi_drop=(max(drops) if drops else 0.0) / qmax,
                ramp_rate_max_Wpm_per_h=max(rates) if rates else 0.0, q_final_frac=float(q[-1] / qmax),
                n_shutdowns=n_shut, duration_h=float(t[-1] - t[0]))


def strength(mat: dict, calib: dict) -> tuple[float, float]:
    """(sigma_c, G_c) consistent with the calibration mode used to generate the data."""
    E, nu = mat["E"], mat["nu"]
    if calib.get("mode", "strength") == "strength":
        # the solver uses s_c = sigma_f/E; the energy it dissipates per crack area is
        # G_eff = C (s_c)^2 delta, independent of any G_c supplied -> use G_eff
        sig = mat.get("sigmaf") or calib["sigmaf"]
        gc = C_LPS(E, nu) * (sig / E) ** 2 * calib["delta"]
    else:
        gc = mat["Gc"]
        sig = E * math.sqrt(gc / (C_LPS(E, nu) * calib["delta"]))
    return sig, gc


def groups(geom: dict, mat: dict, hist_rows, calib: dict) -> dict:
    hf = history_features(hist_rows)
    R, H = geom["R"], geom["H"]
    E, nu, alpha, k = mat["E"], mat["nu"], mat["alpha"], mat["kth"]
    sig, gc = strength(mat, calib)
    dT0 = hf["q_max_Wpm"] / (4 * math.pi * k)
    lch = E * gc / sig ** 2
    a, d = geom.get("dishR", 0.0), geom.get("dishD", 0.0)
    vdish = 2 * math.pi * d * (3 * a * a + d * d) / 6 if a > 0 and d > 0 else 0.0
    Tc = mat.get("Tref", 293.15) + mat.get("gsurf", 0.0192) * hf["q_max_Wpm"] + dT0
    tau = mat.get("tauRef", 3600.0) * math.exp(mat.get("Qact", 380e3) / RG * (1 / Tc - 1 / mat.get("TtauRef", 1700.0)))
    pc = math.log10(max(hf["t_hot_h"] * 3600.0, 1.0) / tau) if mat.get("creep", True) else -8.0
    return dict(hf, Pi_T=alpha * E * dT0 / ((1 - nu) * sig), Pi_R=R / lch, Pi_H=H / (2 * R),
                Pi_dish=vdish / (math.pi * R * R * H), nu=nu, weibull_m=mat.get("weibull", 10.0),
                Pi_creep=float(np.clip(pc, -8, 8)), sigma_c_Pa=sig, Gc_eff=gc, l_ch_m=lch, dT0_K=dT0, T_centre_max_K=Tc,
                tau_at_Tc_s=tau)


def targets(summary: dict, geom: dict, body: int = 0) -> dict:
    fr = (summary.get("fragments_final") or [{}])[body]
    R, H = geom["R"], geom["H"]
    V = math.pi * R * R * H
    A = summary["crack_area_final_mm2"][body] * 1e-6
    n = fr.get("n") or 0
    d = fr.get("deq_volweighted_mm")
    L = fr.get("Leq_mm")
    f = lambda v: math.log10(v) if v and v > 0 and math.isfinite(v) else float("nan")
    return dict(log10_nfrag=f(n), log10_deq_vw_over_R=f(d * 1e-3 / R) if d else float("nan"),
                log10_Leq_over_R=f(L * 1e-3 / R) if L else float("nan"), log10_specific_area=f(A * R / V),
                nfrag=n, deq_vw_mm=d, Leq_mm=L, crack_area_mm2=A * 1e6)
