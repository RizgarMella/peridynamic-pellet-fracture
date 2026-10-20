#!/usr/bin/env python3
"""highres.py - production cracked pellets: one 'hero' high-resolution run (nr 48 lean / 64 full) and the
reference cracked pellet that the stack study restarts from.

Protocol (both runs): creep on, heat to Q_PEAK over 3 h, hold 48 h, cool to zero
over 3 h -> radial cracks at power, centre / circumferential cracks on cool-down.
  stackref : the campaign calibration (delta, m) at --nr-stack; writes
             restart.bin -> copied to studies/stack/inputs/ by stack_study.py plan
  hero     : --nr-hero particles across the radius, m = 3 (delta = 3 dx): sharp
             cracks for visualisation and as the finest single data point.
             State dumps every --every-hero steps for animation.

  python highres.py plan [--nr-hero 64 --nr-stack 32]
     -> jobs/dryrun.sh (exact sizes/memory without running), Batch configs:
        jobs/batch_job_hero.json (one large Spot VM, checkpointed) and
        jobs/batch_job_stackref.json
  python highres.py post studies/highres       # crackmesh + figures from all state dumps
"""
from __future__ import annotations

import argparse
import glob
import json
import os

import config
import pdio
import study_common as sc

Q_PEAK = 30e3


def plan(a):
    cal = config.load_calibration(a.calib)
    sdir = os.path.join(a.root, a.name)
    hist = config.ramp_hold_cool(Q_PEAK, 3.0, 48.0, 3.0)
    common = dict(config.UO2, **config.GEOM, weibull=cal["weibull"], wcorr=cal["wcorr"], seed=1, creep=True,
                  **config.CREEP, dqmax=0.5e3, dtmax=4.0)
    if cal["mode"] == "strength":
        common["sigmaf"] = cal["sigmaf"]
    hero = dict(common, nr=a.nr_hero, mfac=3.0, every=a.every_hero, restart_out="restart.bin")
    stref = dict(common, mfac=cal["mfac"], every=10, restart_out="restart.bin")
    if a.nr_stack > 0:
        stref["nr"] = a.nr_stack                  # explicit resolution, delta = m dx
    else:
        stref["delta"] = cal["delta"]             # campaign calibration: fixed physical horizon
    cases = [sc.Case(sc.case_id("hr", 0, hero), hero, history=hist, factors={"role": "hero", "nr": a.nr_hero}),
             sc.Case(sc.case_id("hr", 1, stref), stref, history=hist, factors={"role": "stackref", "nr": a.nr_stack})]
    for c in cases:   # restart file lands inside the run directory
        c.params["restart_out"] = f"runs/{c.case_id}/restart.bin"
    sc.plan_study(sdir, a.name, cases, {"purpose": "high-resolution cracked pellets", "history": hist, "calibration": cal},
                  bucket=a.bucket, image=a.image, ckpt_every=2, card=__doc__.split("  python")[0])
    lines = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(sdir, "jobs", "jobs.tsv")) if l.strip())
    for c, machine in ((cases[0], a.machine_hero), (cases[1], a.machine_stack)):
        role = c.factors["role"]
        jf = f"jobs/jobs_{role}.tsv"
        open(os.path.join(sdir, jf), "w").write(f"{c.case_id}\t{lines[c.case_id]}\n")
        sc.write_batch_job(sdir, a.name, 1, machine=machine, parallelism=1, region=a.region, image=a.image,
                           bucket=a.bucket, max_run_h=a.max_hours)
        job = json.load(open(os.path.join(sdir, "jobs", "batch_job.json")))
        tsk = job["taskGroups"][0]
        tsk["taskSpec"]["environment"]["variables"]["JOBS_FILE"] = jf
        tsk["taskSpec"]["computeResource"]["memoryMib"] = int(a.mem_gb * 1024)
        if a.ondemand and role == "hero":
            job["allocationPolicy"]["instances"][0]["policy"]["provisioningModel"] = "STANDARD"
        json.dump(job, open(os.path.join(sdir, "jobs", f"batch_job_{role}.json"), "w"), indent=1)
    with open(os.path.join(sdir, "jobs", "dryrun.sh"), "w") as f:
        f.write("#!/bin/bash\n# exact particle/bond counts and memory, no simulation\ncd \"$(dirname \"$0\")/..\"\n")
        for c in cases:
            args = " ".join(sc.to_args({k: v for k, v in c.params.items() if k not in ("restart_out",)}))
            f.write(f'"${{PELLET_PD:-pellet_pd}}" {args} --out runs/{c.case_id} --dryrun --history histories/{c.case_id}.txt\n')
    nsteps = config.steps_in_history(hist, 0.5e3, 4.0)
    for c in cases:
        if c.params.get("nr"):
            delta = c.params["mfac"] * config.GEOM["R"] / c.params["nr"]
            e = config.estimate_run(config.GEOM["R"], config.GEOM["H"], delta, c.params["mfac"], nsteps, creep=True)
        else:
            e = config.estimate_run(config.GEOM["R"], config.GEOM["H"], cal["delta"], cal["mfac"], nsteps, creep=True)
        print(f"{c.factors['role']:8s} nr={e['nr']:3d}: N~{e['N']:,} bonds~{e['NB']:,} mem~{e['mem_gb']:.1f} GB "
              f"~{e['vcpu_h']:,.0f} vCPU-h (~£{e['gbp']:,.0f} Spot)")


def post(a):
    import crackmesh
    for run in sorted(glob.glob(os.path.join(a.study_dir, "runs", "*"))):
        states = sorted(glob.glob(os.path.join(run, "state_*.bin")))
        if not states:
            continue
        od = os.path.join(a.study_dir, "derived", "mesh", os.path.basename(run))
        series = []
        for s in states:
            series.append(crackmesh.process(s, a.warp, None, a.render, od))
        import pandas as pd
        df = pd.DataFrame(series).drop(columns=["file"])
        sc.save_table(df, os.path.join(a.study_dir, "derived", f"crackmesh_series_{os.path.basename(run)}.csv"),
                      "crackmesh statistics for every state dump of the run", sources=[os.path.basename(run)])
        midplane_montage(run, states, os.path.join(a.study_dir, "figures"))


def midplane_montage(run, states, fd):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    os.makedirs(fd, exist_ok=True)
    pick = states if len(states) <= 6 else [states[int(i)] for i in np.linspace(0, len(states) - 1, 6)]
    fig, axs = plt.subplots(1, len(pick), figsize=(3.2 * len(pick), 3.4), constrained_layout=True)
    axs = np.atleast_1d(axs)
    for ax, s in zip(axs, pick):
        st = pdio.read_state(s)
        h = st["header"]
        kmid = int(np.median(st["gk"]))
        m = st["gk"] == kmid
        ax.scatter(st["x0"][m, 0] * 1e3, st["x0"][m, 1] * 1e3, c=st["damage"][m], cmap="Oranges", vmin=0, vmax=0.6,
                   s=max(0.5, 2500 / m.sum()), marker="s", linewidths=0)
        ax.set_aspect("equal"); ax.set_xticks([]); ax.set_yticks([])
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(f"t={h['time_h']:.1f} h, q'={h['q_Wpm']/1e3:.1f} kW/m", fontsize=9)
    fig.savefig(os.path.join(fd, f"midplane_damage_{os.path.basename(run)}.png"), dpi=160)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--root", default="studies"); p.add_argument("--name", default="highres")
    p.add_argument("--nr-hero", type=int, default=48, help="48 lean (~1/3 the cost of 64); 64 = full"); p.add_argument("--every-hero", type=int, default=5)
    p.add_argument("--nr-stack", type=int, default=0, help="0 = use the calibrated delta/m")
    p.add_argument("--calib", default="studies/meshconv/derived/convergence.json")
    p.add_argument("--machine-hero", default="c3d-highcpu-180"); p.add_argument("--machine-stack", default="c3d-highcpu-60")
    p.add_argument("--mem-gb", type=float, default=32.0); p.add_argument("--max-hours", type=float, default=72.0)
    p.add_argument("--ondemand", action="store_true", help="hero on a standard (non-Spot) VM")
    p.add_argument("--bucket", default="BUCKET"); p.add_argument("--region", default=config.REGION)
    p.add_argument("--image", default=config.IMAGE)
    q = sp.add_parser("post"); q.add_argument("study_dir"); q.add_argument("--warp", type=float, default=20.0)
    q.add_argument("--render", action="store_true")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else post(a)


if __name__ == "__main__":
    main()
