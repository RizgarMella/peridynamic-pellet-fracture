#!/usr/bin/env python3
"""campaign.py - parametric dataset for the fragment-size model (PySR).

Space-filling design (scrambled Sobol) over geometry, material properties
(ASSUMED distributions below - replace with measured ones), power histories
with heating/cooling cycles, and microstructure seeds.  Creep is on, so cool-
downs produce centre/circumferential cracking.  Mesh/calibration come from the
mesh-convergence study (derived/convergence.json).

  python campaign.py plan    [--n 512 --nval 128 --name campaign-A]
  python study_common.py run studies/campaign-A ...   or Cloud Batch (see DESIGN.md)
  python campaign.py collect studies/campaign-A       -> derived/dataset.csv (+ .meta.json)

dataset.csv columns: case_id, split, every sampled input (units in names),
history features, dimensionless groups (features.FEATURES), targets
(features.TARGETS, plus raw values), run status and cost.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import config
import features as ft
import pdio
import study_common as sc

# name, distribution, parameters -- every sampled variable; units in the name.
# U(lo,hi) uniform, LU log-uniform, N(mu,sd,lo,hi) truncated normal, LN(median, sigma_ln), I(lo,hi) integer, B(p)
DISTS = [
    ("R_mm", "U", (3.5, 5.0)),
    ("H_over_D", "U", (0.8, 1.6)),
    ("dish_frac", "U", (0.0, 0.75)),          # dish radius / R; < 0.15 -> flat ends
    ("dish_depth_mm", "U", (0.1, 0.4)),
    ("E_GPa", "N", (200.0, 8.0, 176.0, 224.0)),
    ("nu", "U", (0.30, 0.33)),
    ("alpha_per_MK", "N", (10.0, 0.5, 8.5, 11.5)),
    ("k_WpmK", "U", (2.5, 4.5)),
    ("sigmaf_MPa", "LN", (130.0, 0.20)),
    ("Gc_Jpm2", "LN", (10.0, 0.30)),
    ("weibull_m", "U", (6.0, 16.0)),
    ("Qact_kJpmol", "U", (350.0, 550.0)),
    ("tauRef_h", "LU", (0.2, 10.0)),
    ("gsurf_Km_per_W", "U", (0.012, 0.025)),
    ("q_peak_kWpm", "U", (10.0, 45.0)),
    ("n_cycles", "I", (1, 4)),
    ("ramp_kWpm_per_h", "LU", (1.0, 100.0)),
    ("hold_h", "LU", (0.1, 200.0)),
    ("cool_frac", "U", (0.0, 0.6)),
    ("scram", "B", (0.3,)),
    ("seed", "I", (1, 999999)),
]


def _inv(u, kind, par):
    from scipy import stats
    if kind == "U":
        return par[0] + u * (par[1] - par[0])
    if kind == "LU":
        return math.exp(math.log(par[0]) + u * (math.log(par[1]) - math.log(par[0])))
    if kind == "N":
        mu, sd, lo, hi = par
        a, b = (lo - mu) / sd, (hi - mu) / sd
        return float(stats.truncnorm.ppf(u, a, b, loc=mu, scale=sd))
    if kind == "LN":
        return float(par[0] * math.exp(par[1] * stats.norm.ppf(min(max(u, 1e-9), 1 - 1e-9))))
    if kind == "I":
        return int(min(par[1], par[0] + math.floor(u * (par[1] - par[0] + 1))))
    if kind == "B":
        return int(u < par[0])
    raise ValueError(kind)


def make_history(s: dict):
    qp = s["q_peak_kWpm"] * 1e3
    rate = s["ramp_kWpm_per_h"] * 1e3
    rows, t, q = [(0.0, 0.0)], 0.0, 0.0
    for c in range(s["n_cycles"]):
        t += (qp - q) / rate; q = qp; rows.append((t, q))
        t += s["hold_h"]; rows.append((t, q))
        if c < s["n_cycles"] - 1:
            ql = s["cool_frac"] * qp
            t += (q - ql) / rate; q = ql; rows.append((t, q))
            t += max(0.25 * s["hold_h"], 0.1); rows.append((t, q))
    t += 0.01 if s["scram"] else q / rate
    rows.append((t, 0.0))
    return rows


def sample(n: int, seed: int, skip: int = 0):
    """n Sobol points after skipping the first `skip` (extensions continue the same sequence)."""
    from scipy.stats import qmc
    eng = qmc.Sobol(d=len(DISTS), scramble=True, seed=seed)
    if skip:
        eng.fast_forward(skip)
    u = eng.random(n)
    return [{name: _inv(float(u[i, j]), kind, par) for j, (name, kind, par) in enumerate(DISTS)} for i in range(n)]


NUMERICS = {  # 'lean' halves load steps and quarters hold sub-steps; check against 'fine' on a few cases
    "fine": dict(dqmax=1e3, dt_div=8, dt_min=1.0, dt_max=25.0),
    "lean": dict(dqmax=2e3, dt_div=4, dt_min=2.0, dt_max=50.0),
}


def to_solver(s: dict, cal: dict, numerics: str = "lean"):
    nm = NUMERICS[numerics]
    R = s["R_mm"] * 1e-3
    H = s["H_over_D"] * 2 * R
    p = dict(R=R, H=H, E=s["E_GPa"] * 1e9, nu=s["nu"], alpha=s["alpha_per_MK"] * 1e-6, kth=s["k_WpmK"],
             Gc=s["Gc_Jpm2"], Tref=config.UO2["Tref"], gsurf=s["gsurf_Km_per_W"], weibull=s["weibull_m"],
             wcorr=cal["wcorr"], seed=s["seed"], delta=cal["delta"], mfac=cal["mfac"], creep=True,
             tauRef=s["tauRef_h"] * 3600.0, TtauRef=1700.0, Qact=s["Qact_kJpmol"] * 1e3, lamv=1.0,
             dqmax=nm["dqmax"], dtmax=float(np.clip(s["hold_h"] / nm["dt_div"], nm["dt_min"], nm["dt_max"])), every=1000)
    if s["dish_frac"] >= 0.15:
        p.update(dishR=s["dish_frac"] * R, dishD=s["dish_depth_mm"] * 1e-3)
    if cal["mode"] == "strength":
        p["sigmaf"] = s["sigmaf_MPa"] * 1e6
    return p


def plan(a):
    cal = config.load_calibration(a.calib)
    sdir = os.path.join(a.root, a.name)
    old = []
    skip_tr = skip_va = 0
    if a.extend:
        import pandas as pd
        prev = pd.read_csv(os.path.join(sdir, "cases.csv"))
        sdef = pdio.read_json(os.path.join(sdir, "study.json"))
        cal, a.seed, a.numerics = sdef["calibration"], sdef["sobol_seed"], sdef["numerics"]
        skip_tr, skip_va = int((prev.split == "train").sum()), int((prev.split == "val").sum())
        for _, r in prev.iterrows():
            s = {k: r[k] for k, _, _ in DISTS}
            for k, kind, _ in DISTS:
                if kind in ("I", "B"):
                    s[k] = int(s[k])
            old.append(sc.Case(r.case_id, to_solver(s, cal, a.numerics), history=make_history(s), split=r.split, factors=s))
    tr = sample(a.n, a.seed, skip_tr)
    va = sample(a.nval, a.seed + 7919, skip_va)
    cases, new_ids, tot, mem = list(old), [], 0.0, 0.0
    for split, smp in (("train", tr), ("val", va)):
        for s in smp:
            p = to_solver(s, cal, a.numerics)
            hist = make_history(s)
            cid = sc.case_id("cp", len(cases), p)
            new_ids.append(cid)
            cases.append(sc.Case(cid, p, history=hist, split=split, factors=dict(s)))
            e = config.estimate_run(p["R"], p["H"], cal["delta"], cal["mfac"],
                                    config.steps_in_history(hist, p["dqmax"], p["dtmax"]), creep=True)
            tot += e["vcpu_h"]; mem = max(mem, e["mem_gb"])
    sc.plan_study(sdir, a.name, cases,
                  {"purpose": "parametric dataset for symbolic regression of fragment size",
                   "design": "scrambled Sobol (staged: --extend continues the sequence)",
                   "n_train": skip_tr + a.n, "n_val": skip_va + a.nval, "sobol_seed": a.seed, "numerics": a.numerics,
                   "distributions": [dict(name=n, dist=k, params=p) for n, k, p in DISTS],
                   "distributions_note": "ASSUMED modelling distributions, not measured data",
                   "calibration": cal, "features": ft.FEATURES, "targets": ft.TARGETS},
                  bucket=a.bucket, image=a.image, machine=a.machine, keep_state=a.keep_state,
                  card=__doc__.split("dataset.csv")[0])
    if a.extend:
        sc.restrict_jobs(sdir, new_ids, f"stage{skip_tr + a.n}")
    print(f"{len(new_ids)} new runs: estimated {tot:,.0f} vCPU-h (~£{tot * config.COST['gbp_per_vcpu_h']:,.0f} Spot); "
          f"max {mem:.2f} GB/run")


def collect(a):
    import pandas as pd
    sdir = a.study_dir
    sdef = pdio.read_json(os.path.join(sdir, "study.json"))
    cal = sdef["calibration"]
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        s = c.to_dict()
        p = to_solver({k: s[k] for k, _, _ in DISTS}, cal, sdef.get("numerics", "lean"))
        geom = dict(R=p["R"], H=p["H"], dishR=p.get("dishR", 0.0), dishD=p.get("dishD", 0.0))
        mat = dict(E=p["E"], nu=p["nu"], alpha=p["alpha"], kth=p["kth"], Gc=p["Gc"], sigmaf=p.get("sigmaf"),
                   weibull=p["weibull"], tauRef=p["tauRef"], TtauRef=p["TtauRef"], Qact=p["Qact"], gsurf=p["gsurf"],
                   Tref=p["Tref"], creep=True)
        hist = pdio.read_history_file(os.path.join(sdir, c.history_file))
        row = dict(case_id=c.case_id, split=c.split, **{k: s[k] for k, _, _ in DISTS}, **ft.groups(geom, mat, hist, cal))
        if pdio.run_complete(d):
            summ = pdio.read_json(os.path.join(d, "summary.json"))
            row.update(ft.targets(summ, geom), status="complete", onset_q_Wpm=summ["onset_q_Wpm"],
                       solver_time_s=summ["solver_time_s"], N=summ["N"])
        else:
            row["status"] = "incomplete" if os.path.exists(os.path.join(d, "manifest.json")) else "missing"
        rows.append(row)
    df = pd.DataFrame(rows)
    units = {"Pi_T": "-", "Pi_R": "-", "Pi_H": "-", "Pi_dish": "-", "Pi_creep": "log10(-)", "Pi_drop": "-",
             "sigma_c_Pa": "Pa", "Gc_eff": "J/m2", "l_ch_m": "m", "dT0_K": "K", "tau_at_Tc_s": "s",
             "ramp_rate_max_Wpm_per_h": "W/m/h", "R_mm": "mm", "E_GPa": "GPa", "alpha_per_MK": "1e-6/K",
             "k_WpmK": "W/m/K", "sigmaf_MPa": "MPa", "Gc_Jpm2": "J/m2", "Qact_kJpmol": "kJ/mol", "tauRef_h": "h",
             "gsurf_Km_per_W": "K m/W", "q_peak_kWpm": "kW/m", "ramp_kWpm_per_h": "kW/m/h", "hold_h": "h"}
    sc.save_table(df, os.path.join(sdir, "derived", "dataset.csv"),
                  "Inputs, history features, dimensionless groups and fragment targets for every case "
                  f"(features: {ft.FEATURES}; targets: {ft.TARGETS})", units=units, sources=list(df.case_id))
    print(df["status"].value_counts().to_string())


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--root", default="studies"); p.add_argument("--name", default="campaign-A")
    p.add_argument("--n", type=int, default=128, help="training runs in this stage (power of 2)")
    p.add_argument("--nval", type=int, default=32, help="validation runs in this stage (power of 2)")
    p.add_argument("--extend", action="store_true", help="add a stage to an existing campaign (same Sobol sequence)")
    p.add_argument("--numerics", choices=["lean", "fine"], default="lean")
    p.add_argument("--seed", type=int, default=2026); p.add_argument("--calib", default="studies/meshconv/derived/convergence.json")
    p.add_argument("--keep-state", action="store_true", help="keep state dumps at knots (default: summary data only)")
    p.add_argument("--bucket", default="BUCKET"); p.add_argument("--machine", default=config.MACHINE)
    p.add_argument("--image", default=config.IMAGE)
    q = sp.add_parser("collect"); q.add_argument("study_dir")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else collect(a)


if __name__ == "__main__":
    main()
