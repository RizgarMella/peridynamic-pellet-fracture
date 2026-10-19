#!/usr/bin/env python3
"""stack_study.py - sympathetic cracking between pellets.

Stack (bottom to top): half pellet (bottom face pinned in z) | the CRACKED pellet
from highres.py (restored from its restart file: cracks, displacements, creep
state) | half pellet with the axial pressure applied on its top face.  The stack
is pressed (preload step), heated to Q_PEAK, held (creep), and cooled.

Factors
  pressure_MPa      nominal axial pressure            : 0, 2, 10, 30
  mu                interface friction coefficient    : 0 (frictionless), 0.2, 0.5, 1.0, -1 (= stick)
  contact_frac      nominal contact-area fraction     : 1.0, 0.75, 0.51, 0.28
                    (contact only for r >= R sqrt(1 - contact_frac); models
                    dishes / chamfers / end-face waviness)
  middle            cracked (restart) | intact (CONTROL, same loads)
  seed              Weibull realisation of the half pellets
Design 'full' = full factorial (cracked x 2 seeds + intact control x 1 seed);
design 'oat'  = one-factor-at-a-time around (10 MPa, 0.5, 0.75) + the four
corners of (pressure, mu) at each contact level - for a cheaper first pass.

Metrics (derived/stack_runs.csv, one row per run)
  crack area of each body at peak power (end of hold) and at the end;
  new crack area in the cracked pellet (end - after preload);
  half-pellet crack area minus the matched INTACT control = sympathetic excess;
  crack ALIGNMENT across each interface from angular_XXXX.csv: Pearson r of the
  face damage profiles, and enrichment = P(half-pellet face crack within +-w
  of a middle-pellet crack mouth) / (fraction of the circumference within +-w
  of a mouth)  [1 = no sympathy, >1 = cracks seeded across the interface];
  interface state at peak: max/mean pressure, fraction sliding, mean slip.
derived/stack_effects.csv: seed means per factor combination; figures/: heat
maps of enrichment and sympathetic excess vs (pressure, mu) per contact level.

  python stack_study.py plan --restart studies/highres/runs/<stackref-case>/restart.bin [--design oat]
  python stack_study.py analyze studies/stack
  python stack_study.py plot    studies/stack
"""
from __future__ import annotations

import argparse
import itertools
import math
import os
import shutil

import numpy as np

import config
import pdio
import study_common as sc

PRESSURE_MPA = [0.0, 2.0, 10.0, 30.0]
MU = [0.0, 0.2, 0.5, 1.0, -1.0]
CONTACT_FRAC = [1.0, 0.75, 0.51, 0.28]
SEEDS = [11, 12]
Q_PEAK = 30e3


def design(kind: str):
    combos = []
    if kind == "full":
        for p, mu, cf in itertools.product(PRESSURE_MPA, MU, CONTACT_FRAC):
            for s in SEEDS:
                combos.append((p, mu, cf, "cracked", s))
            combos.append((p, mu, cf, "intact", SEEDS[0]))
    else:
        base = (10.0, 0.5, 0.75)
        pts = {base}
        pts |= {(p, base[1], base[2]) for p in PRESSURE_MPA} | {(base[0], m, base[2]) for m in MU}
        pts |= {(base[0], base[1], c) for c in CONTACT_FRAC}
        pts |= {(p, m, c) for c in CONTACT_FRAC for p in (PRESSURE_MPA[0], PRESSURE_MPA[-1]) for m in (MU[0], MU[3])}
        for p, mu, cf in sorted(pts):
            combos += [(p, mu, cf, "cracked", SEEDS[0]), (p, mu, cf, "intact", SEEDS[0])]
    return combos


def plan(a):
    hdr = pdio.read_restart_header(a.restart)
    sdir = os.path.join(a.root, a.name)
    os.makedirs(os.path.join(sdir, "inputs"), exist_ok=True)
    rst = "inputs/cracked_pellet.bin"
    shutil.copy(a.restart, os.path.join(sdir, rst))
    cal = config.load_calibration(a.calib)
    R, H, dx, delta = hdr["R"], hdr["H_eff"], hdr["dx"], hdr["delta"]
    base = dict(config.UO2, R=R, H=H, delta=delta, mfac=delta / dx, jitter=hdr["jitter"], weibull=cal["weibull"],
                wcorr=cal["wcorr"], stack="0.5,1,0.5", pin="z", creep=hdr["creep"], dqmax=1e3, dtmax=4.0, every=10)
    if hdr["creep"]:
        base.update(config.CREEP)
    if hdr["dishR"] > 0:
        base.update(dishR=hdr["dishR"], dishD=hdr["dishD"])
    if hdr["chamfer"] > 0:
        base["chamfer"] = hdr["chamfer"]
    if cal["mode"] == "strength":
        base["sigmaf"] = cal["sigmaf"]
    cases = []
    for p_mpa, mu, cf, middle, seed in design(a.design):
        p = dict(base, mu=mu, contact_rmin=R * math.sqrt(max(0.0, 1.0 - cf)), seed=seed)
        if middle == "cracked":
            p.update(restart=rst, restart_body=1)
        hist = config.ramp_hold_cool(Q_PEAK, 3.0, 24.0, 3.0, p_Pa=p_mpa * 1e6)
        cid = sc.case_id("st", len(cases), {**p, "p": p_mpa})
        cases.append(sc.Case(cid, p, history=hist, factors={"pressure_MPa": p_mpa, "mu": mu, "contact_frac": cf,
                                                            "middle": middle, "seed": seed}))
    sc.plan_study(sdir, a.name, cases,
                  {"purpose": "sympathetic cracking in a pellet stack", "design": a.design, "restart": a.restart,
                   "restart_header": hdr, "levels": {"pressure_MPa": PRESSURE_MPA, "mu": MU, "contact_frac": CONTACT_FRAC,
                                                     "seeds": SEEDS}, "q_peak_Wpm": Q_PEAK},
                  bucket=a.bucket, image=a.image, machine=a.machine, card=__doc__.split("  python")[0])
    nsteps = config.steps_in_history(config.ramp_hold_cool(Q_PEAK, 3.0, 24.0, 3.0), 1e3, 4.0)
    e = config.estimate_run(R, H, delta, delta / dx, nsteps, n_bodies_h=2.0, creep=hdr["creep"])
    print(f"{len(cases)} runs; each ~{e['N']:,} particles, {e['mem_gb']:.1f} GB, ~{e['vcpu_h']:.0f} vCPU-h -> "
          f"total ~{e['vcpu_h'] * len(cases):,.0f} vCPU-h (~${e['usd'] * len(cases):,.0f} Spot)")


# ----------------------------------------------------------------------------- analysis
def alignment(mid_face, half_face, w_bins=3, thr=0.4):
    """Pearson r of two angular damage profiles + enrichment of half-face cracks near middle-face crack mouths."""
    a, b = np.asarray(mid_face, float), np.asarray(half_face, float)
    r = float(np.corrcoef(a, b)[0, 1]) if a.std() > 0 and b.std() > 0 else np.nan
    ma, mb = a > thr, b > thr
    if ma.sum() == 0 or mb.sum() == 0:
        return r, np.nan, int(mb.sum())
    near = np.zeros_like(ma)
    for s in range(-w_bins, w_bins + 1):
        near |= np.roll(ma, s)
    expected = near.mean()
    observed = (mb & near).sum() / mb.sum()
    return r, float(observed / expected) if expected > 0 else np.nan, int(mb.sum())


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pdio.run_complete(d):
            continue
        r = pdio.read_run(d)
        h = r["history"]
        qmax = h.q_Wpm.max()
        peak = int(h.loc[np.isclose(h.q_Wpm, qmax), "step"].max())       # end of hold
        hp, hf, h0 = h[h.step == peak].iloc[0], h.iloc[-1], h.iloc[0]
        row = dict(case_id=c.case_id, pressure_MPa=c.pressure_MPa, mu=c.mu, contact_frac=c.contact_frac, middle=c.middle,
                   seed=c.seed, peak_step=peak)
        for b in range(3):
            row[f"b{b}_A_peak_mm2"] = hp[f"b{b}_crack_area_mm2"]
            row[f"b{b}_A_final_mm2"] = hf[f"b{b}_crack_area_mm2"]
        row["mid_new_A_mm2"] = hf["b1_crack_area_mm2"] - h0["b1_crack_area_mm2"]
        row["halves_A_peak_mm2"] = 0.5 * (hp["b0_crack_area_mm2"] + hp["b2_crack_area_mm2"])
        row["halves_A_final_mm2"] = 0.5 * (hf["b0_crack_area_mm2"] + hf["b2_crack_area_mm2"])
        for col in ("max_contact_p_MPa", "mean_contact_p_MPa", "contact_frac_closed", "contact_frac_slip", "mean_slip_um"):
            row[f"{col}_peak"] = hp[col]
        for tag, step in (("peak", peak), ("final", int(hf.step))):
            p = os.path.join(d, f"angular_{step:04d}.csv")
            if not os.path.exists(p):
                continue
            ang = pd.read_csv(p)
            for iface, (mid, half) in enumerate((("b1_bottom", "b0_top"), ("b1_top", "b2_bottom"))):
                rr, enr, n = alignment(ang[mid], ang[half])
                row[f"if{iface}_pearson_{tag}"] = rr
                row[f"if{iface}_enrichment_{tag}"] = enr
                row[f"if{iface}_half_crack_bins_{tag}"] = n
            row[f"pearson_{tag}"] = np.nanmean([row.get(f"if0_pearson_{tag}"), row.get(f"if1_pearson_{tag}")])
            row[f"enrichment_{tag}"] = np.nanmean([row.get(f"if0_enrichment_{tag}"), row.get(f"if1_enrichment_{tag}")])
        rows.append(row)
    if not rows:
        print("no completed runs yet"); return
    df = pd.DataFrame(rows)
    keys = ["pressure_MPa", "mu", "contact_frac"]
    ctrl = df[df.middle == "intact"].groupby(keys)[["halves_A_peak_mm2", "halves_A_final_mm2"]].mean().add_suffix("_control")
    df = df.merge(ctrl, left_on=keys, right_index=True, how="left")
    df["sympathetic_excess_peak_mm2"] = df.halves_A_peak_mm2 - df.halves_A_peak_mm2_control
    df["sympathetic_excess_final_mm2"] = df.halves_A_final_mm2 - df.halves_A_final_mm2_control
    dd = os.path.join(sdir, "derived")
    sc.save_table(df, os.path.join(dd, "stack_runs.csv"), "One row per stack run: crack areas per body, sympathetic "
                  "excess vs matched intact control, cross-interface alignment, interface state", sources=list(df.case_id))
    cr = df[df.middle == "cracked"]
    eff = cr.groupby(keys).agg(**{f"{m}_mean": (m, "mean") for m in
                                  ("enrichment_peak", "enrichment_final", "pearson_peak", "pearson_final",
                                   "sympathetic_excess_peak_mm2", "sympathetic_excess_final_mm2", "mid_new_A_mm2",
                                   "contact_frac_slip_peak", "mean_slip_um_peak", "max_contact_p_MPa_peak")},
                               n_seeds=("seed", "count")).reset_index()
    sc.save_table(eff, os.path.join(dd, "stack_effects.csv"), "Seed means per (pressure, friction, contact fraction), cracked middle")
    print(f"analysed {len(df)} runs -> {dd}")


def plot(a):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    from matplotlib.colors import TwoSlopeNorm
    dd, fd = os.path.join(a.study_dir, "derived"), os.path.join(a.study_dir, "figures")
    os.makedirs(fd, exist_ok=True)
    eff = pd.read_csv(os.path.join(dd, "stack_effects.csv"))
    for metric, cmap, center, label in (("enrichment_final_mean", "RdBu_r", 1.0, "crack alignment enrichment (1 = none)"),
                                        ("sympathetic_excess_final_mm2_mean", "RdBu_r", 0.0, "half-pellet crack area vs control [mm²]")):
        cfs = sorted(eff.contact_frac.unique(), reverse=True)
        fig, axs = plt.subplots(1, len(cfs), figsize=(3.6 * len(cfs), 3.4), constrained_layout=True, squeeze=False)
        vals = eff[metric].to_numpy(float)
        span = np.nanmax(np.abs(vals - center)) if np.isfinite(vals).any() else 1.0
        norm = TwoSlopeNorm(center, center - span - 1e-12, center + span + 1e-12)
        im = None
        for ax, cf in zip(axs[0], cfs):
            g = eff[eff.contact_frac == cf].pivot_table(index="mu", columns="pressure_MPa", values=metric)
            im = ax.imshow(g.to_numpy(), cmap=cmap, norm=norm, aspect="auto", origin="lower")
            ax.set_xticks(range(len(g.columns)), [f"{v:g}" for v in g.columns])
            ax.set_yticks(range(len(g.index)), ["stick" if v < 0 else f"{v:g}" for v in g.index])
            ax.set_xlabel("pressure [MPa]"); ax.set_title(f"contact {cf:.0%}")
        axs[0][0].set_ylabel("friction coefficient")
        fig.colorbar(im, ax=axs[0].tolist(), label=label, shrink=0.85)
        fig.savefig(os.path.join(fd, f"stack_{metric}.png"), dpi=150); plt.close(fig)
    print("figures ->", fd)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--restart", required=True); p.add_argument("--root", default="studies")
    p.add_argument("--name", default="stack"); p.add_argument("--design", choices=["full", "oat"], default="oat", help="oat = lean first pass (54 runs)")
    p.add_argument("--calib", default="studies/meshconv/derived/convergence.json")
    p.add_argument("--bucket", default="BUCKET"); p.add_argument("--machine", default="c3d-highcpu-60")
    p.add_argument("--image", default=config.IMAGE)
    for n in ("analyze", "plot"):
        q = sp.add_parser(n); q.add_argument("study_dir")
    a = ap.parse_args()
    {"plan": plan, "analyze": analyze, "plot": plot}[a.cmd](a)


if __name__ == "__main__":
    main()
