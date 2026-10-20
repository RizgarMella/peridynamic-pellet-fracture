#!/usr/bin/env python3
"""weibull_study.py - how the strength DISTRIBUTION controls cracking.

Factors (full factorial x seeds; geometry, material, mesh from the calibration):
  weibull modulus m      : 0 (= no scatter), 3, 5, 8, 12, 20, 40
  correlation length     : per-bond i.i.d. (0), grain field 0.15 mm, 0.40 mm
  seed                   : 1..SEEDS (independent realisations)
Optional second scenario (--with-cycle): heat-hold-cool with creep, to see the
distribution's effect on cool-down (centre / circumferential) cracking.

Outputs (derived/, all with .meta.json):
  weibull_runs.csv        one row per run: onset power, crack area and radial
                          cracks at q*, fragments (count, d_eq, L_eq), timings
  weibull_levels.csv      per (scenario, m, wcorr): mean, sd, CV, p5/p50/p95,
                          bootstrap 95% CI of the mean, Weibull fit of onset power
  weibull_curves.csv      crack area and crack count vs q (mean, p5, p95) per level
  weibull_fragments.csv   all fragment volumes at the final step (for distributions)
Figures (figures/): onset CV vs m, area-vs-q bands, crack count vs m, fragment CDFs.

  python weibull_study.py plan [--seeds 10] [--with-cycle]
  python study_common.py run studies/weibull --parallel 2 --threads 8   (or Cloud Batch)
  python weibull_study.py analyze studies/weibull
  python weibull_study.py plot    studies/weibull
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np

import config
import pdio
import study_common as sc

M_LEVELS = [0, 3, 5, 8, 12, 20, 40]
WCORR_LEVELS = [0.0, 0.15e-3, 0.40e-3]
# lean: brackets the realistic range (m 5-20) plus the no-scatter reference, per-bond vs one grain size
PROFILES = {"full": dict(m=M_LEVELS, wcorr=WCORR_LEVELS, seeds=10),
            "lean": dict(m=[0, 5, 10, 20], wcorr=[0.0, 0.25e-3], seeds=5)}
Q_GRID = np.arange(1e3, 40.01e3, 1e3)
Q_STAR = [5e3, 10e3, 20e3, 30e3, 40e3]
Q_PEAK = 40e3


def plan(a):
    cal = config.load_calibration(a.calib)
    sdir = os.path.join(a.root, a.name)
    scen = [("ramp", config.ramp(Q_PEAK, 2.0), False)]
    if a.with_cycle:
        scen.append(("cycle", config.ramp_hold_cool(Q_PEAK, 2.0, 24.0, 1.0), True))
    prof = PROFILES[a.profile]
    seeds = a.seeds or prof["seeds"]
    cases, k = [], 0
    for sname, hist, creep in scen:
        for m in prof["m"]:
            for wc in prof["wcorr"]:
                for seed in range(1, seeds + 1):
                    p = dict(config.UO2, **config.GEOM, **config.calib_params(cal), dqmax=1e3, every=10)
                    p.update(weibull=m, wcorr=wc, seed=seed)
                    if creep:
                        p.update(creep=True, **config.CREEP, dtmax=2.0)
                    cid = sc.case_id("wb", k, {**p, "scen": sname})
                    cases.append(sc.Case(cid, p, history=hist,
                                         factors={"scenario": sname, "weibull_m": m, "wcorr_mm": wc * 1e3, "seed": seed,
                                                  "calib_mode": cal["mode"]}))
                    k += 1
    sc.plan_study(sdir, a.name, cases,
                  {"purpose": "Weibull strength-distribution study", "profile": a.profile, "m_levels": prof["m"],
                   "wcorr_levels_m": prof["wcorr"], "seeds": seeds, "scenarios": [s[0] for s in scen], "calibration": cal, "material": config.UO2,
                   "geometry": config.GEOM},
                  bucket=a.bucket, image=a.image, machine=a.machine, card=__doc__.split("Outputs")[0])
    e = config.estimate_run(config.GEOM["R"], config.GEOM["H"], cal["delta"], cal["mfac"], 40)
    tot = e["vcpu_h"] * len(cases)
    print(f"estimated compute: {tot:,.0f} vCPU-h (~£{tot * config.COST['gbp_per_vcpu_h']:,.0f} Spot), {e['mem_gb']:.2f} GB/run")


def _boot_ci(v, n=2000, rng=np.random.default_rng(0)):
    v = np.asarray(v, float)
    v = v[np.isfinite(v)]
    if len(v) < 3:
        return np.nan, np.nan
    bs = rng.choice(v, size=(n, len(v)), replace=True).mean(1)
    return np.percentile(bs, 2.5), np.percentile(bs, 97.5)


def analyze(a):
    import pandas as pd
    from scipy import stats
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    runs, curves, frags = [], [], []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pdio.run_complete(d):
            continue
        r = pdio.read_run(d)
        h, s = r["history"], r["summary"]
        row = dict(case_id=c.case_id, scenario=c.scenario, weibull_m=c.weibull_m, wcorr_mm=c.wcorr_mm, seed=c.seed,
                   onset_q_Wpm=s["onset_q_Wpm"], first_bond_q_Wpm=s["first_bond_failure_q_Wpm"],
                   crack_area_final_mm2=s["crack_area_final_mm2"][0], solver_time_s=s["solver_time_s"])
        A = pdio.value_at_q(h, "b0_crack_area_mm2", Q_GRID)
        C = pdio.value_at_q(h, "b0_radial_cracks", Q_GRID)
        for q in Q_STAR:
            i = int(round(q / 1e3)) - 1
            row[f"A_q{int(q/1e3)}_mm2"] = A[i]
            row[f"cracks_q{int(q/1e3)}"] = C[i]
        for key, fr in (("final", (s.get("fragments_final") or [{}])[0]), ("peak", (s.get("fragments_at_peak") or [{}])[0])):
            row.update({f"frag_n_{key}": fr.get("n"), f"deq_vw_{key}_mm": fr.get("deq_volweighted_mm"),
                        f"Leq_{key}_mm": fr.get("Leq_mm"), f"largest_frac_{key}": fr.get("largest_frac")})
        runs.append(row)
        curves.append(pd.DataFrame(dict(case_id=c.case_id, scenario=c.scenario, weibull_m=c.weibull_m, wcorr_mm=c.wcorr_mm,
                                        seed=c.seed, q_Wpm=Q_GRID, crack_area_mm2=A, radial_cracks=C)))
        fr = r["fragments"]
        if fr is not None and len(fr):
            last = fr[fr.step == fr.step.max()]
            frags.append(last.assign(case_id=c.case_id, scenario=c.scenario, weibull_m=c.weibull_m, wcorr_mm=c.wcorr_mm,
                                     seed=c.seed))
    if not runs:
        print("no completed runs yet"); return
    runs = pd.DataFrame(runs)
    dd = os.path.join(sdir, "derived")
    sc.save_table(runs, os.path.join(dd, "weibull_runs.csv"), "One row per completed run", sources=list(runs.case_id))
    cur = pd.concat(curves, ignore_index=True)
    sc.save_table(cur, os.path.join(dd, "weibull_curves_per_run.csv"), "Crack area / radial cracks vs q, first heating ramp")
    if frags:
        sc.save_table(pd.concat(frags, ignore_index=True), os.path.join(dd, "weibull_fragments.csv"),
                      "Every fragment at the final output step of every run")

    metrics = [c for c in runs.columns if c.startswith(("onset_", "first_bond", "A_q", "cracks_q", "frag_n", "deq_vw", "Leq", "crack_area"))]
    lev = []
    for key, g in runs.groupby(["scenario", "weibull_m", "wcorr_mm"]):
        row = dict(zip(["scenario", "weibull_m", "wcorr_mm"], key), n_runs=len(g))
        for m in metrics:
            v = g[m].astype(float)
            lo, hi = _boot_ci(v)
            row.update({f"{m}_mean": v.mean(), f"{m}_sd": v.std(ddof=1), f"{m}_cv": v.std(ddof=1) / v.mean() if v.mean() else np.nan,
                        f"{m}_p5": v.quantile(.05), f"{m}_p50": v.median(), f"{m}_p95": v.quantile(.95),
                        f"{m}_mean_ci_lo": lo, f"{m}_mean_ci_hi": hi})
        on = g.onset_q_Wpm.astype(float)
        on = on[on > 0]
        if len(on) >= 5:
            shape, _, scale = stats.weibull_min.fit(on, floc=0)
            row.update(onset_weibull_shape=shape, onset_weibull_scale_Wpm=scale)
        if key[1] > 0:   # CV of a Weibull(m) variate: sqrt(G(1+2/m)/G(1+1/m)^2 - 1)
            mm = float(key[1])
            row["input_strength_cv"] = math.sqrt(math.gamma(1 + 2 / mm) / math.gamma(1 + 1 / mm) ** 2 - 1)
        lev.append(row)
    lev = pd.DataFrame(lev)
    sc.save_table(lev, os.path.join(dd, "weibull_levels.csv"),
                  "Statistics per (scenario, Weibull modulus, correlation length); input_strength_cv is the CV of the "
                  "input Weibull distribution for comparison with onset CV")
    band = cur.groupby(["scenario", "weibull_m", "wcorr_mm", "q_Wpm"]).agg(
        area_mean_mm2=("crack_area_mm2", "mean"), area_p5_mm2=("crack_area_mm2", lambda x: x.quantile(.05)),
        area_p95_mm2=("crack_area_mm2", lambda x: x.quantile(.95)), cracks_mean=("radial_cracks", "mean"),
        cracks_p5=("radial_cracks", lambda x: x.quantile(.05)), cracks_p95=("radial_cracks", lambda x: x.quantile(.95)),
        n=("crack_area_mm2", "count")).reset_index()
    sc.save_table(band, os.path.join(dd, "weibull_curves.csv"), "Seed bands of crack area / radial cracks vs q per level")
    print(f"analysed {len(runs)} runs -> {dd}")


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    lev = pd.read_csv(os.path.join(dd, "weibull_levels.csv"))
    band = pd.read_csv(os.path.join(dd, "weibull_curves.csv"))
    ramp = lev[lev.scenario == "ramp"]
    cols = {0.0: "#2a6fdb", 0.15: "#d9822b", 0.25: "#d9822b", 0.4: "#3a9a5b"}
    fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
    for wc, g in ramp[ramp.weibull_m > 0].groupby("wcorr_mm"):
        g = g.sort_values("weibull_m")
        ax.plot(g.weibull_m, g.onset_q_Wpm_cv, "o-", color=cols.get(round(wc, 2), "#777"), lw=2, ms=5,
                label=f"onset CV, wcorr={'per bond' if wc == 0 else f'{wc:.2f} mm'}")
    g = ramp[ramp.weibull_m > 0].drop_duplicates("weibull_m").sort_values("weibull_m")
    ax.plot(g.weibull_m, g.input_strength_cv, "--", color="#555", lw=1.5, label="input strength CV")
    ax.set_xscale("log"); ax.set_xlabel("Weibull modulus m"); ax.set_ylabel("coefficient of variation")
    ax.legend(frameon=False, fontsize=8)
    fig.savefig(os.path.join(fd, "weibull_onset_cv.png"), dpi=150); plt.close(fig)

    for wc, g in band[band.scenario == "ramp"].groupby("wcorr_mm"):
        fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        ms = sorted(g.weibull_m.unique())
        cmap = plt.get_cmap("Blues")
        for i, m in enumerate(ms):
            s = g[g.weibull_m == m].sort_values("q_Wpm")
            col = cmap(0.3 + 0.65 * i / max(1, len(ms) - 1))
            ax.plot(s.q_Wpm / 1e3, s.area_mean_mm2, color=col, lw=2, label="no scatter" if m == 0 else f"m={m:g}")
            ax.fill_between(s.q_Wpm / 1e3, s.area_p5_mm2, s.area_p95_mm2, color=col, alpha=0.12, lw=0)
        ax.set_xlabel("linear heat rate q' [kW/m]"); ax.set_ylabel("crack area [mm²]")
        ax.set_title(f"wcorr = {'per bond' if wc == 0 else f'{wc:.2f} mm'} (mean, 5-95 % band)")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fd, f"weibull_area_vs_q_wcorr{wc:.2f}.png"), dpi=150); plt.close(fig)

    fp = os.path.join(dd, "weibull_fragments.csv")
    if os.path.exists(fp):
        fr = pd.read_csv(fp)
        fig, ax = plt.subplots(figsize=(6, 4.2), constrained_layout=True)
        for i, (m, g) in enumerate(fr[(fr.scenario == "ramp") & (fr.wcorr_mm == fr.wcorr_mm[fr.wcorr_mm > 0].min())].groupby("weibull_m")):
            v = np.sort(g.volume_mm3.to_numpy())
            ax.plot(v, np.linspace(0, 1, len(v)), lw=2, color=plt.get_cmap("Blues")(0.3 + 0.1 * i),
                    label="no scatter" if m == 0 else f"m={m:g}")
        ax.set_xscale("log"); ax.set_xlabel("fragment volume [mm³]"); ax.set_ylabel("cumulative fraction")
        ax.legend(frameon=False, fontsize=8)
        fig.savefig(os.path.join(fd, "weibull_fragment_cdf.png"), dpi=150); plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--root", default="studies"); p.add_argument("--name", default="weibull")
    p.add_argument("--profile", choices=["lean", "full"], default="lean")
    p.add_argument("--seeds", type=int, default=0, help="0 = profile default"); p.add_argument("--with-cycle", action="store_true")
    p.add_argument("--calib", default="studies/meshconv/derived/convergence.json")
    p.add_argument("--bucket", default="BUCKET"); p.add_argument("--machine", default=config.MACHINE)
    p.add_argument("--image", default=config.IMAGE)
    for n in ("analyze", "plot"):
        q = sp.add_parser(n); q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
