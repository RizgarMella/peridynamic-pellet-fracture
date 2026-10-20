#!/usr/bin/env python3
"""mesh_convergence.py - discretisation study based on CRACK AREA.

Two refinement families x two calibration modes x seeds:
  family "m"     : horizon delta FIXED (physical length), m = delta/dx in M_LEVELS  -> m-convergence
  family "delta" : m = 3 fixed, nr in NR_LEVELS (delta shrinks with dx)            -> delta-convergence
  mode "strength": s_c = sigma_f/E, Weibull grain field of fixed physical size (wcorr)
  mode "energy"  : s_c from G_c at the current delta
Expected: m-convergence in both modes; delta-convergence only for "strength"
(in "energy" mode the onset power grows ~ delta^-1/2: no length scale -> no limit).

Metrics (from history.csv / summary.json, main body):
  crack area A(q*) on the first heating ramp at q* = 5..30 kW/m, onset power,
  radial cracks at q*, final fragment count, L_eq = 3V/A.
Per (mode, family, level): mean, sd, sem over seeds.  Per (mode, family, metric):
Celik et al. (2008) GCI on the three finest levels (iterative order p for
non-uniform refinement), Richardson extrapolation, convergence type, and a
seed-noise test (difference of finest two levels vs 2 x combined sem).
The recommended discretisation (coarsest level within --tol of the
extrapolated value for A at --q-ref) is written to derived/convergence.json and
picked up by the Weibull, campaign, high-res and stack studies.

  python mesh_convergence.py plan    [--root studies] [--seeds 4] [--bucket B]
  python study_common.py run studies/meshconv --parallel 2 --threads 8     (or Cloud Batch)
  python mesh_convergence.py analyze studies/meshconv [--tol 0.1 --q-ref 20e3]
  python mesh_convergence.py plot    studies/meshconv
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np

import config
import pdio
import study_common as sc

M_LEVELS = [2.0, 2.5, 3.0, 4.0, 5.0]
NR_LEVELS = [8, 12, 16, 24, 32, 48]
# lean profile: everything needed for the decision, nothing more.
#  strength mode (the campaign calibration): full m-family + 4 delta levels;
#  energy mode: 3 m levels and 3 delta levels - enough to show (non-)convergence.
PROFILES = {
    "full": {"strength": {"m": M_LEVELS, "delta": NR_LEVELS}, "energy": {"m": M_LEVELS, "delta": NR_LEVELS}},
    "lean": {"strength": {"m": [2.5, 3.0, 4.0, 5.0], "delta": [8, 12, 16, 24]},
             "energy": {"m": [2.5, 3.0, 4.0], "delta": [8, 12, 16]}},
}
Q_STAR = [5e3, 10e3, 15e3, 20e3, 25e3, 30e3]
Q_PEAK = 30e3
DQ = 1e3


def plan(a):
    sdir = os.path.join(a.root, a.name)
    cal = dict(config.CALIB)
    cases, k = [], 0
    est = []
    prof = PROFILES[a.profile]
    for mode in ("strength", "energy"):
        for fam in ("m", "delta"):
            for lev in prof[mode][fam]:
                for seed in range(a.seeds):
                    p = dict(config.UO2, **config.GEOM, weibull=cal["weibull"], wcorr=cal["wcorr"], seed=seed + 1,
                             qmax=Q_PEAK, dqmax=DQ, every=5)
                    if fam == "m":
                        p.update(delta=cal["delta"], mfac=lev)
                        delta, mfac = cal["delta"], lev
                    else:
                        p.update(nr=lev, mfac=3.0)
                        delta, mfac = 3.0 * config.GEOM["R"] / lev, 3.0
                    if mode == "strength":
                        p["sigmaf"] = cal["sigmaf"]
                    nr = round(config.GEOM["R"] * mfac / delta)
                    dx = config.GEOM["R"] / nr
                    cid = sc.case_id("mc", k, p)
                    cases.append(sc.Case(cid, p, history=config.ramp(Q_PEAK, 1.0),
                                         factors={"mode": mode, "family": fam, "level": lev, "seed": seed + 1,
                                                  "nr": nr, "dx_mm": dx * 1e3, "delta_mm": delta * 1e3, "m_ratio": delta / dx}))
                    if seed == 0:
                        e = config.estimate_run(config.GEOM["R"], config.GEOM["H"], delta, mfac, int(Q_PEAK / DQ))
                        est.append((mode, fam, lev, e))
                    k += 1
    sc.plan_study(sdir, a.name, cases,
                  {"purpose": "mesh convergence on crack area", "profile": a.profile, "levels": prof,
                   "q_star_Wpm": Q_STAR, "calibration": cal, "seeds": a.seeds, "material": config.UO2,
                   "geometry": config.GEOM},
                  bucket=a.bucket, image=a.image, machine=a.machine, region=a.region,
                  card=__doc__.split("Metrics")[0])
    tot = sum(e["vcpu_h"] for *_, e in est) * a.seeds
    print(f"estimated compute: {tot:,.0f} vCPU-h  (~£{tot * config.COST['gbp_per_vcpu_h']:,.0f} Spot); largest run "
          f"{max(e['mem_gb'] for *_, e in est):.1f} GB")


# ----------------------------------------------------------------------------- analysis
def celik_gci(h, phi, fs=1.25):
    """Celik et al. (2008) procedure. h = (h1<h2<h3) fine->coarse, phi likewise."""
    h1, h2, h3 = h
    f1, f2, f3 = phi
    out = dict(phi1=f1, phi2=f2, phi3=f3, r21=h2 / h1, r32=h3 / h2, p=np.nan, phi_ext=np.nan, ea21=np.nan,
               gci21=np.nan, kind="undetermined")
    e21, e32 = f2 - f1, f3 - f2
    if not np.all(np.isfinite([f1, f2, f3])) or e21 == 0 or e32 == 0 or f1 == 0:
        return out
    r21, r32 = h2 / h1, h3 / h2
    s = np.sign(e32 / e21)
    p = abs(np.log(abs(e32 / e21))) / np.log(r21)
    for _ in range(200):
        num, den = r21 ** p - s, r32 ** p - s
        if num <= 0 or den <= 0:
            return {**out, "kind": "oscillatory-diverging"}
        pn = abs(np.log(abs(e32 / e21)) + np.log(num / den)) / np.log(r21)
        if abs(pn - p) < 1e-10:
            break
        p = pn
    fext = (r21 ** p * f1 - f2) / (r21 ** p - 1)
    ea = abs((f1 - f2) / f1)
    out.update(p=p, phi_ext=fext, ea21=ea, gci21=fs * ea / (r21 ** p - 1),
               kind="monotone" if s > 0 else "oscillatory")
    return out


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    per = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pdio.run_complete(d):
            continue
        r = pdio.read_run(d)
        h, s = r["history"], r["summary"]
        row = dict(case_id=c.case_id, mode=c["mode"], family=c.family, level=c.level, seed=c.seed, nr=c.nr,
                   dx_mm=c.dx_mm, delta_mm=c.delta_mm, m_ratio=c.m_ratio, onset_q_Wpm=s["onset_q_Wpm"],
                   N=s["N"], solver_time_s=s["solver_time_s"])
        A = pdio.value_at_q(h, "b0_crack_area_mm2", Q_STAR)
        rc = pdio.value_at_q(h, "b0_radial_cracks", Q_STAR)
        for q, av, rv in zip(Q_STAR, A, rc):
            row[f"A_q{int(q / 1e3)}_mm2"] = av
            row[f"cracks_q{int(q / 1e3)}"] = rv
        fr = (s.get("fragments_final") or [{}])[0]
        row.update(frag_n_final=fr.get("n"), Leq_final_mm=fr.get("Leq_mm"), deq_vw_final_mm=fr.get("deq_volweighted_mm"))
        per.append(row)
    per = pd.DataFrame(per)
    dd = os.path.join(sdir, "derived")
    sc.save_table(per, os.path.join(dd, "meshconv_runs.csv"), "One row per completed run: metrics at q* and final",
                  sources=list(per.case_id) if len(per) else [])
    if per.empty:
        print("no completed runs yet"); return
    metrics = [c for c in per.columns if c.startswith(("A_q", "cracks_q"))] + ["onset_q_Wpm", "frag_n_final", "Leq_final_mm"]
    g = per.groupby(["mode", "family", "level"])
    agg = []
    for key, grp in g:
        row = dict(zip(["mode", "family", "level"], key), dx_mm=grp.dx_mm.iloc[0], delta_mm=grp.delta_mm.iloc[0],
                   m_ratio=grp.m_ratio.iloc[0], nr=grp.nr.iloc[0], n_seeds=len(grp))
        for m in metrics:
            v = grp[m].astype(float)
            row[f"{m}_mean"] = v.mean(); row[f"{m}_sd"] = v.std(ddof=1); row[f"{m}_sem"] = v.std(ddof=1) / np.sqrt(v.count())
        agg.append(row)
    agg = pd.DataFrame(agg)
    sc.save_table(agg, os.path.join(dd, "meshconv_levels.csv"), "Seed statistics per (mode, family, level)")

    gci_rows = []
    for (mode, fam), grp in agg.groupby(["mode", "family"]):
        grp = grp.sort_values("dx_mm")
        if len(grp) < 3:
            continue
        fine3 = grp.iloc[:3]
        for m in metrics:
            res = celik_gci(tuple(fine3.dx_mm), tuple(fine3[f"{m}_mean"]))
            sem12 = float(np.hypot(fine3[f"{m}_sem"].iloc[0], fine3[f"{m}_sem"].iloc[1]))
            res.update(mode=mode, family=fam, metric=m, dx1_mm=fine3.dx_mm.iloc[0],
                       within_seed_noise=bool(abs(res["phi1"] - res["phi2"]) <= 2 * sem12) if np.isfinite(sem12) else None)
            gci_rows.append(res)
    gci = pd.DataFrame(gci_rows)
    sc.save_table(gci, os.path.join(dd, "meshconv_gci.csv"),
                  "Celik GCI on the three finest levels per (mode, family, metric); phi_ext = Richardson extrapolation")

    # recommendation
    key = f"A_q{int(a.q_ref / 1e3)}_mm2"
    rec = {"tol": a.tol, "metric": key, "candidates": []}
    for mode in ("strength", "energy"):
        sub = agg[(agg["mode"] == mode) & (agg.family == "m")].sort_values("dx_mm")
        gr = gci[(gci["mode"] == mode) & (gci.family == "m") & (gci.metric == key)]
        if sub.empty:
            continue
        ref = gr.phi_ext.iloc[0] if len(gr) and np.isfinite(gr.phi_ext.iloc[0]) else sub[f"{key}_mean"].iloc[0]
        sub = sub.assign(rel_err=(sub[f"{key}_mean"] - ref).abs() / abs(ref))
        ok = sub[sub.rel_err <= a.tol]
        pick = ok.sort_values("dx_mm", ascending=False).iloc[0] if len(ok) else sub.iloc[0]
        rec["candidates"].append(dict(mode=mode, mfac=float(pick.m_ratio), delta=float(pick.delta_mm) * 1e-3,
                                      nr=int(pick.nr), rel_err=float(pick.rel_err), reference=float(ref),
                                      gci_kind=(gr.kind.iloc[0] if len(gr) else None)))
    dsub = gci[(gci.family == "delta") & (gci.metric == key)]
    rec["delta_convergence"] = dsub[["mode", "kind", "p", "gci21", "within_seed_noise"]].to_dict("records")
    best = next((c for c in rec["candidates"] if c["mode"] == config.CALIB["mode"]), rec["candidates"][0] if rec["candidates"] else None)
    if best:
        rec.update(mode=best["mode"], delta=best["delta"], mfac=best["mfac"], sigmaf=config.CALIB["sigmaf"],
                   weibull=config.CALIB["weibull"], wcorr=config.CALIB["wcorr"])
    with open(os.path.join(dd, "convergence.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(json.dumps(rec, indent=1))


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd = os.path.join(a.study_dir, "derived")
    fd = os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    agg = pd.read_csv(os.path.join(dd, "meshconv_levels.csv"))
    qs = [int(q / 1e3) for q in Q_STAR]
    for (mode, fam), grp in agg.groupby(["mode", "family"]):
        fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        cmap = plt.get_cmap("Blues")
        grp = grp.sort_values("dx_mm", ascending=False)
        for i, (_, r) in enumerate(grp.iterrows()):
            mean = [r[f"A_q{q}_mm2_mean"] for q in qs]
            sd = [r[f"A_q{q}_mm2_sd"] for q in qs]
            col = cmap(0.35 + 0.6 * i / max(1, len(grp) - 1))
            ax.plot(qs, mean, color=col, lw=2, label=f"dx={r.dx_mm:.3f} mm (m={r.m_ratio:.1f})")
            ax.fill_between(qs, np.subtract(mean, sd), np.add(mean, sd), color=col, alpha=0.15, lw=0)
        ax.set_xlabel("linear heat rate q' [kW/m]"); ax.set_ylabel("crack area [mm²]")
        ax.set_title(f"{mode} calibration, {fam}-refinement")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fd, f"meshconv_{mode}_{fam}_area_vs_q.png"), dpi=150)
        plt.close(fig)
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    key = f"A_q{int(a.q_ref / 1e3)}_mm2_mean"
    for (mode, fam), grp in agg.groupby(["mode", "family"]):
        grp = grp.sort_values("dx_mm")
        ax.errorbar(grp.dx_mm, grp[key], yerr=grp[key.replace("_mean", "_sem")], marker="o", ms=5, lw=1.5,
                    capsize=2, label=f"{mode} / {fam}")
    ax.set_xscale("log"); ax.set_xlabel("particle spacing dx [mm]"); ax.set_ylabel(f"crack area at {a.q_ref/1e3:.0f} kW/m [mm²]")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "meshconv_area_vs_dx.png"), dpi=150)
    plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--root", default="studies"); p.add_argument("--name", default="meshconv")
    p.add_argument("--seeds", type=int, default=3); p.add_argument("--bucket", default="BUCKET")
    p.add_argument("--profile", choices=["lean", "full"], default="lean")
    p.add_argument("--region", default=config.REGION)
    p.add_argument("--image", default=config.IMAGE)
    p.add_argument("--machine", default=config.MACHINE)
    for name in ("analyze", "plot"):
        q = sp.add_parser(name); q.add_argument("study_dir"); q.add_argument("--tol", type=float, default=0.10)
        q.add_argument("--q-ref", type=float, default=20e3)
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
