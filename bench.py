#!/usr/bin/env python3
"""bench.py - pick the cheapest Google Cloud machine for pellet_pd by measurement.

Runs one fixed, representative case (fracture on, ~15 load steps) on several Spot
machine types and thread counts, then ranks them by USD per 10^12 bond-evaluations
(the solver's unit of work) and by wall time.  Run this BEFORE the large studies;
its result updates config.COST via derived/bench_recommendation.json.

  python bench.py plan --bucket B --project P           # study dir studies/bench-YYYYMMDD
  (push, then submit one Batch job per machine: jobs/batch_job_<machine>.json)
  python bench.py analyze studies/bench-YYYYMMDD --prices cloud/prices.json
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import os

import config
import pdio
import study_common as sc

MACHINES = ["c3d-highcpu-30", "c4d-highcpu-32", "t2d-standard-32", "n2d-highcpu-32"]

# Spot USD per vCPU-hour in each family's cheapest region (whole-VM Spot price / vCPUs),
# from the gcloud-compute.com tracker, Oct 2026.  Throughput per vCPU is region-independent, so the
# benchmark can run anywhere; the analysis prices each family at its cheapest region and
# recommends a (machine, region) pair.  Spot prices change up to daily: refresh before analysing.
DEFAULT_PRICES = {
    "_note": "family -> [USD per vCPU-hour, region]. Refresh from cloud.google.com/spot-vms/pricing.",
    "c3d": [0.0042, "australia-southeast1"],
    "c4d": [0.0079, "us-west4"],
    "t2d": [0.0073, "europe-southwest1"],
    "n2d": [0.0086, "asia-southeast1"],
}


def plan(a):
    name = a.name or f"bench-{dt.date.today():%Y%m%d}"
    sdir = os.path.join(a.root, name)
    base = dict(config.UO2, **config.GEOM, nr=24, mfac=3.0, sigmaf=config.SIGMA_F, weibull=10.0, wcorr=0.25e-3,
                qmax=15e3, steps=15, every=1000)
    cases, per_machine = [], {}
    k = 0
    for m in MACHINES:
        vcpu = int(m.rsplit("-", 1)[1])
        for thr in sorted({vcpu, max(1, vcpu // 2)}):      # all vCPUs (SMT) vs physical cores
            p = dict(base)
            cid = sc.case_id("bench", k, {**p, "machine": m, "threads": thr})
            cases.append(sc.Case(cid, p, factors={"machine": m, "vcpu": vcpu, "threads": thr}))
            per_machine.setdefault(m, []).append((cid, thr))
            k += 1
    sc.plan_study(sdir, name, cases, {"purpose": "machine-type benchmark", "machines": MACHINES, "case": base},
                  ckpt_every=0, bucket=a.bucket, image=a.image, keep_state=False,
                  card="Fixed fracture case on several Spot machine types / thread counts to measure bond-evaluation "
                       "throughput and price-performance.")
    # one jobs file + Batch job per machine type (a Batch job has a single machine type)
    lines = dict(l.rstrip("\n").split("\t", 1) for l in open(os.path.join(sdir, "jobs", "jobs.tsv")) if l.strip())
    for m, items in per_machine.items():
        jf = f"jobs/jobs_{m}.tsv"
        with open(os.path.join(sdir, jf), "w") as f:
            for cid, thr in items:
                f.write(f"{cid}\texport OMP_NUM_THREADS={thr}; {lines[cid]}\n")
        reg = DEFAULT_PRICES[m.split("-")[0]][1] if a.region == "cheapest" else a.region
        sc.write_batch_job(sdir, name, len(items), machine=m, parallelism=len(items), region=reg, image=a.image,
                           bucket=a.bucket, max_run_h=4)
        jb = os.path.join(sdir, "jobs", "batch_job.json")
        job = json.load(open(jb))
        job["taskGroups"][0]["taskSpec"]["environment"]["variables"]["JOBS_FILE"] = jf
        job["taskGroups"][0]["parallelism"] = 1                 # one at a time: clean timing
        json.dump(job, open(os.path.join(sdir, "jobs", f"batch_job_{m}.json"), "w"), indent=1)
    json.dump(DEFAULT_PRICES, open(os.path.join(sdir, "prices.json"), "w"), indent=1)
    for m in MACHINES:
        reg = DEFAULT_PRICES[m.split("-")[0]][1] if a.region == "cheapest" else a.region
        print(f"gcloud batch jobs submit bench-{m.replace('highcpu', 'hc').replace('standard', 'st')} --location {reg} "
              f"--config {sdir}/jobs/batch_job_{m}.json")
    print("(the container image must be pullable from each region: use a multi-region Artifact Registry repo, e.g. 'us' "
          "or 'asia', or push the image to each region)")


def analyze(a):
    import pandas as pd
    sdir = a.study_dir
    prices = json.load(open(a.prices or os.path.join(sdir, "prices.json")))
    cases = pd.read_csv(os.path.join(sdir, "cases.csv"))
    rows = []
    for _, c in cases.iterrows():
        d = os.path.join(sdir, "runs", c.case_id)
        if not pdio.run_complete(d):
            continue
        s = pdio.read_json(os.path.join(d, "summary.json"))
        fam = c.machine.split("-")[0]
        price, region = prices.get(fam, [float("nan"), ""])
        usd_h = price * c.vcpu
        gbes = s["throughput_Mbond_per_s"] / 1e3
        rows.append(dict(case_id=c.case_id, machine=c.machine, region=region, vcpu=c.vcpu, threads=c.threads,
                         solver_time_s=s["solver_time_s"], throughput_Gbond_per_s=gbes,
                         Gbond_per_s_per_vcpu=gbes / c.vcpu, vm_usd_per_h=usd_h,
                         usd_per_Tbond=usd_h / (gbes * 3600 / 1e3) if gbes > 0 else float("nan")))
    df = pd.DataFrame(rows).sort_values("usd_per_Tbond")
    sc.save_table(df, os.path.join(sdir, "derived", "bench_results.csv"),
                  "Measured throughput and price-performance per machine type / thread count",
                  units={"usd_per_Tbond": "USD per 1e12 bond evaluations", "vm_usd_per_h": "USD/h",
                         "throughput_Gbond_per_s": "1e9 bond evaluations / s", "Gbond_per_s_per_vcpu": "1e9/s/vCPU"},
                  sources=list(df.case_id))
    if len(df):
        best = df.iloc[0]
        rec = {"machine": best.machine, "region": best.region, "threads": int(best.threads), "gbes_per_vcpu": float(best.Gbond_per_s_per_vcpu),
               "usd_per_vcpu_h": float(best.vm_usd_per_h / best.vcpu), "usd_per_Tbond": float(best.usd_per_Tbond)}
        json.dump(rec, open(os.path.join(sdir, "derived", "bench_recommendation.json"), "w"), indent=1)
        print(df.to_string(index=False))
        print("recommendation:", rec, "\n-> set config.REGION / MACHINE and copy gbes_per_vcpu / usd_per_vcpu_h into config.COST")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sp = ap.add_subparsers(dest="cmd", required=True)
    p = sp.add_parser("plan"); p.add_argument("--root", default="studies"); p.add_argument("--name")
    p.add_argument("--bucket", default="BUCKET"); p.add_argument("--region", default="cheapest", help="'cheapest' = each family in its cheapest region")
    p.add_argument("--image", default=config.IMAGE)
    q = sp.add_parser("analyze"); q.add_argument("study_dir"); q.add_argument("--prices")
    a = ap.parse_args()
    plan(a) if a.cmd == "plan" else analyze(a)


if __name__ == "__main__":
    main()
