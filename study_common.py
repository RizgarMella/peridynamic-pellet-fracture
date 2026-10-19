"""study_common.py - cases, job files, catalogues, labelled derived tables and the
Google Cloud Storage layout shared by every study script.

BUCKET LAYOUT (one prefix per solver major version; nothing is ever overwritten
in place by a different solver build - the build id is in every manifest):

  gs://<bucket>/pellet-pd/v2/
    _meta/schema.json                 data dictionary (schema.py)
    _meta/solver/<build>/             exact pellet_pd.cpp + build flags used
    _meta/env/                        Dockerfile, requirements.txt
    benchmarks/<YYYYMMDD>/...         machine-type benchmark runs
    studies/<study>/                  e.g. weibull, meshconv, campaign-A, highres, stack
      study.json                      study definition (config, levels, distributions, seeds)
      DATASET_CARD.md                 human description of the dataset
      cases.csv                       design matrix (one row per case, units in column names)
      histories/<case_id>.txt         power/pressure histories
      inputs/                         shared inputs (e.g. cracked-pellet restart files)
      jobs/jobs.tsv                   case_id <TAB> command  (one line per Batch task)
      jobs/batch_job.json             Google Cloud Batch job (Spot, retries, task array)
      jobs/slurm_array.sh             same tasks for a Slurm cluster
      runs/<case_id>/                 raw solver outputs (see schema.json)
      derived/*.csv + *.meta.json     tidy analysis tables (+ sidecar: columns, units, sources, code)
      figures/*.png                   figures made only from derived/
      catalog.csv                     run index: status, checksums, key metrics, paths

Case ids are <prefix>-<6-digit index>-<6-hex hash of the solver arguments>,
so a case id identifies its inputs and re-planning the same design is idempotent.
"""
from __future__ import annotations

import csv
import datetime as _dt
import hashlib
import json
import os
import shlex
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field

import pdio
from schema import SCHEMA_VERSION, schema_dict

HERE = os.path.dirname(os.path.abspath(__file__))
BUCKET_PREFIX = "pellet-pd/v2"


# ----------------------------------------------------------------------------- cases
@dataclass
class Case:
    case_id: str
    params: dict                       # solver options (python names -> CLI via to_args)
    factors: dict = field(default_factory=dict)   # labelled design variables (units in names) -> cases.csv
    history: list | None = None        # rows (time_h, q_Wpm[, p_Pa]) or None for built-in ramp
    split: str = "-"                   # train / val / - (campaign)


def to_args(params: dict) -> list[str]:
    """{'nr': 16, 'creep': True, 'contact_rmin': 1e-3} -> ['--nr','16','--creep','--contact-rmin','0.001']"""
    out = []
    for k, v in params.items():
        if v is None or v is False:
            continue
        flag = "--" + k.replace("_", "-") if k in ("contact_rmin", "restart_out", "restart_body", "calib_delta",
                                                     "ckpt_every") else "--" + k
        if v is True:
            out.append(flag)
        else:
            out += [flag, f"{v:.10g}" if isinstance(v, float) else str(v)]
    return out


def case_id(prefix: str, index: int, params: dict) -> str:
    h = hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()[:6]
    return f"{prefix}-{index:06d}-{h}"


# ----------------------------------------------------------------------------- planning
def plan_study(study_dir: str, study_name: str, cases: list[Case], study_def: dict, *, threads: int = 0,
               ckpt_every: int = 2, card: str = "", machine: str = "c3d-highcpu-30", parallelism: int = 50,
               region: str = "us-central1", image: str = "REGION-docker.pkg.dev/PROJECT/pellet/pellet-pd:2",
               bucket: str = "BUCKET", max_run_h: float = 24.0, keep_state: bool = True, keep_vtk: bool = False) -> None:
    """Write study.json, cases.csv, histories, jobs (tsv, local, Slurm, Cloud Batch), DATASET_CARD.md, schema.json."""
    os.makedirs(study_dir, exist_ok=True)
    for d in ("histories", "jobs", "runs", "derived", "figures", "inputs"):
        os.makedirs(os.path.join(study_dir, d), exist_ok=True)
    stamp = _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
    sdef = {"schema": SCHEMA_VERSION, "study": study_name, "created_utc": stamp, "n_cases": len(cases),
            "gcs_uri": gcs_study_uri(bucket, study_name), **study_def}
    with open(os.path.join(study_dir, "study.json"), "w") as f:
        json.dump(sdef, f, indent=1, default=str)
    with open(os.path.join(study_dir, "schema.json"), "w") as f:
        json.dump(schema_dict(), f, indent=1)

    rows, lines = [], []
    for c in cases:
        p = dict(c.params)
        if c.history is not None:
            hp = f"histories/{c.case_id}.txt"
            pdio.write_history(os.path.join(study_dir, hp), c.history, comment=f"study {study_name} case {c.case_id}")
            p["history"] = hp
        p["out"] = f"runs/{c.case_id}"
        p["ckpt_every"] = ckpt_every
        p["resume"] = True
        if not keep_state:
            p["nostate"] = True
        if not keep_vtk:
            p["novtk"] = True
        args = to_args(p)
        labels = {"study": study_name, "case_id": c.case_id, "split": c.split, **{k: v for k, v in c.factors.items()}}
        for k, v in labels.items():
            args += ["--label", f"{k}={v:.8g}" if isinstance(v, float) else f"{k}={v}"]
        cmd = '"${PELLET_PD:-pellet_pd}" ' + " ".join(shlex.quote(a) for a in args)
        if threads:
            cmd = f"OMP_NUM_THREADS={threads} " + cmd
        cmd = f"mkdir -p runs/{c.case_id} && {cmd} > runs/{c.case_id}/stdout.log 2> runs/{c.case_id}/stderr.log"
        lines.append(f"{c.case_id}\t{cmd}")
        rows.append({"case_id": c.case_id, "split": c.split, **c.factors,
                     "history_file": p.get("history", ""), "solver_args": " ".join(to_args(c.params))})
    keys = list(dict.fromkeys(k for r in rows for k in r))
    with open(os.path.join(study_dir, "cases.csv"), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        w.writerows(rows)
    with open(os.path.join(study_dir, "jobs", "jobs.tsv"), "w") as f:
        f.write("\n".join(lines) + "\n")
    _write_slurm(study_dir, len(lines), threads or 16, max_run_h)
    write_batch_job(study_dir, study_name, len(lines), machine=machine, parallelism=parallelism, region=region,
                    image=image, bucket=bucket, max_run_h=max_run_h)
    write_card(study_dir, study_name, sdef, card)
    print(f"planned {len(cases)} cases in {study_dir}")
    print(f"  local : python study_common.py run {study_dir} --parallel 2 --threads 8")
    print(f"  cloud : python study_common.py push {study_dir} --bucket {bucket}  then  "
          f"gcloud batch jobs submit {study_name}-$(date +%s) --location {region} --config {study_dir}/jobs/batch_job.json")


def _write_slurm(study_dir: str, n: int, threads: int, max_run_h: float) -> None:
    hh = int(max_run_h)
    with open(os.path.join(study_dir, "jobs", "slurm_array.sh"), "w") as f:
        f.write(f"""#!/bin/bash
#SBATCH --job-name=pellet-pd
#SBATCH --array=1-{n}
#SBATCH --cpus-per-task={threads}
#SBATCH --time={hh:02d}:00:00
#SBATCH --requeue
# run from the study directory:  sbatch jobs/slurm_array.sh
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK OMP_PROC_BIND=close OMP_PLACES=cores
LINE=$(sed -n "${{SLURM_ARRAY_TASK_ID}}p" jobs/jobs.tsv)
CMD=$(printf '%s' "$LINE" | cut -f2-)
bash -c "$CMD"
""")


def gcs_study_uri(bucket: str, study: str) -> str:
    return f"gs://{bucket}/{BUCKET_PREFIX}/studies/{study}"


def write_batch_job(study_dir, study, ntasks, *, machine, parallelism, region, image, bucket, max_run_h) -> None:
    """Google Cloud Batch task array on Spot VMs. Each task runs cloud/task.sh for one line of jobs.tsv:
    it stages inputs from the bucket, resumes from any checkpoint already there, syncs outputs every
    10 minutes and on SIGTERM, compiles pellet_pd natively for the VM's CPU on first use."""
    vcpu = int(machine.rsplit("-", 1)[-1]) if machine.rsplit("-", 1)[-1].isdigit() else 16
    job = {
        "taskGroups": [{
            "taskSpec": {
                "runnables": [{"container": {
                    "imageUri": image,
                    "entrypoint": "/bin/bash",
                    "commands": ["/opt/pellet/cloud/task.sh"],
                }}],
                "environment": {"variables": {"STUDY_URI": gcs_study_uri(bucket, study), "OMP_PROC_BIND": "close",
                                              "OMP_PLACES": "cores"}},
                "computeResource": {"cpuMilli": vcpu * 1000 - 500, "memoryMib": 1024},
                "maxRetryCount": 5,
                "maxRunDuration": f"{int(max_run_h * 3600)}s",
                # 75 = solver received SIGTERM (Spot preemption); 50001 = Batch: VM preempted
                "lifecyclePolicies": [{"action": "RETRY_TASK", "actionCondition": {"exitCodes": [75, 50001]}}],
            },
            "taskCount": ntasks,
            "parallelism": min(parallelism, ntasks),
        }],
        "allocationPolicy": {
            "instances": [{"policy": {"machineType": machine, "provisioningModel": "SPOT"}}],
            "location": {"allowedLocations": [f"regions/{region}"]},
        },
        "labels": {"app": "pellet-pd", "study": study.lower().replace("_", "-")[:60]},
        "logsPolicy": {"destination": "CLOUD_LOGGING"},
    }
    with open(os.path.join(study_dir, "jobs", "batch_job.json"), "w") as f:
        json.dump(job, f, indent=1)


def write_card(study_dir: str, study: str, sdef: dict, text: str) -> None:
    with open(os.path.join(study_dir, "DATASET_CARD.md"), "w") as f:
        f.write(f"# Dataset: {study}\n\n")
        f.write(f"- Schema: `{SCHEMA_VERSION}` (see `schema.json`; units are in column-name suffixes)\n")
        f.write(f"- Created (UTC): {sdef['created_utc']}\n- Cases: {sdef['n_cases']}\n")
        f.write(f"- Location: `{sdef['gcs_uri']}`\n- Solver: pellet_pd v2 (build id in every `runs/*/manifest.json`)\n\n")
        f.write("## Purpose and design\n\n" + (text.strip() or "(see study.json)") + "\n\n")
        f.write("## Layout\n\n`cases.csv` design matrix -> `runs/<case_id>/` raw outputs -> `derived/` tidy tables "
                "(each with `.meta.json`) -> `figures/`. `catalog.csv` indexes runs with checksums.\n\n")
        f.write("## Provenance and caveats\n\nSimulated data (3D state-based peridynamics, prescribed temperature "
                "field). Material-property distributions are modelling assumptions recorded in `study.json`, not "
                "measurements. Licence / citation / contact: _fill in before sharing_.\n")


# ----------------------------------------------------------------------------- local execution
def run_local(study_dir: str, parallel: int = 1, threads: int = 0, only: list[str] | None = None,
              solver: str | None = None) -> None:
    """Run (or resume) every incomplete case. Safe to re-run: complete runs are skipped."""
    env = dict(os.environ)
    env["PELLET_PD"] = solver or env.get("PELLET_PD", os.path.join(HERE, "pellet_pd"))
    if threads:
        env["OMP_NUM_THREADS"] = str(threads)
    env.setdefault("OMP_PROC_BIND", "close")
    todo = []
    with open(os.path.join(study_dir, "jobs", "jobs.tsv")) as f:
        for line in f:
            if not line.strip():
                continue
            cid, cmd = line.rstrip("\n").split("\t", 1)
            if only and cid not in only:
                continue
            if pdio.run_complete(os.path.join(study_dir, "runs", cid)):
                continue
            todo.append((cid, cmd))
    print(f"{len(todo)} case(s) to run in {study_dir}")

    def _one(item):
        cid, cmd = item
        r = subprocess.run(["bash", "-c", cmd], cwd=study_dir, env=env)
        ok = r.returncode == 0
        if ok:
            ck = os.path.join(study_dir, "runs", cid, "checkpoint.bin")
            if os.path.exists(ck):
                os.remove(ck)
        print(f"  {cid}: {'ok' if ok else f'exit {r.returncode}'}", flush=True)
        return ok

    with ThreadPoolExecutor(max_workers=max(1, parallel)) as ex:
        res = list(ex.map(_one, todo))
    print(f"done: {sum(res)}/{len(res)} succeeded")


# ----------------------------------------------------------------------------- catalogue
def _sha(path: str) -> str:
    if not os.path.exists(path):
        return ""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def catalog(study_dir: str, bucket: str | None = None) -> str:
    """Index all runs: status, checksums, sizes and headline metrics -> catalog.csv."""
    import pandas as pd
    cases = pd.read_csv(os.path.join(study_dir, "cases.csv"))
    sdef = pdio.read_json(os.path.join(study_dir, "study.json"))
    rows = []
    for cid in cases["case_id"]:
        d = os.path.join(study_dir, "runs", cid)
        man = os.path.join(d, "manifest.json")
        summ = os.path.join(d, "summary.json")
        status = "complete" if pdio.run_complete(d) else ("incomplete" if os.path.exists(man) else "missing")
        row = {"case_id": cid, "status": status}
        if os.path.exists(man):
            m = pdio.read_json(man)
            row.update(build=m.get("build"), N=m["counts"]["N"], NB_directed=m["counts"]["NB_directed"],
                       dx_mm=m["derived"]["dx"] * 1e3, delta_mm=m["derived"]["delta"] * 1e3,
                       n_steps=len(m.get("schedule", [])))
        if status == "complete":
            s = pdio.read_json(summ)
            fr = s.get("fragments_final") or []
            main = max(range(len(s["crack_area_final_mm2"])), key=lambda b: s["crack_area_final_mm2"][b]) if fr else 0
            row.update(onset_q_Wpm=s["onset_q_Wpm"], crack_area_final_mm2=sum(s["crack_area_final_mm2"]),
                       fragments_final_main=fr[main]["n"] if fr else None, solver_time_s=s["solver_time_s"],
                       resumed=s.get("resumed"))
        nbytes = sum(os.path.getsize(os.path.join(dp, fn)) for dp, _, fns in os.walk(d) for fn in fns) if os.path.isdir(d) else 0
        row.update(bytes=nbytes, sha256_manifest=_sha(man), sha256_summary=_sha(summ),
                   sha256_history=_sha(os.path.join(d, "history.csv")), path_local=os.path.relpath(d, study_dir),
                   path_gcs=f"{gcs_study_uri(bucket, sdef['study'])}/runs/{cid}" if bucket else "")
        rows.append(row)
    out = os.path.join(study_dir, "catalog.csv")
    pd.DataFrame(rows).to_csv(out, index=False)
    st = pd.DataFrame(rows)["status"].value_counts().to_dict()
    print(f"catalog -> {out}  {st}")
    return out


# ----------------------------------------------------------------------------- labelled derived tables
def save_table(df, path: str, description: str, units: dict | None = None, sources: list[str] | None = None,
               producer: str | None = None) -> None:
    """CSV (+ parquet if pyarrow is available) with a .meta.json sidecar: columns, units, description, provenance."""
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    df.to_csv(path, index=False)
    try:
        df.to_parquet(os.path.splitext(path)[0] + ".parquet", index=False)
    except Exception:
        pass
    units = units or {}

    def unit_of(c):
        if c in units:
            return units[c]
        for suf, u in (("_mm3", "mm3"), ("_mm2", "mm2"), ("_mm", "mm"), ("_um", "um"), ("_MPa", "MPa"), ("_Pa", "Pa"),
                       ("_Wpm", "W/m"), ("_K", "K"), ("_h", "h"), ("_s", "s"), ("_deg", "deg")):
            if c.endswith(suf):
                return u
        return "-"

    meta = {"schema": SCHEMA_VERSION, "file": os.path.basename(path), "description": description,
            "rows": int(len(df)), "columns": [{"name": c, "unit": unit_of(c), "dtype": str(df[c].dtype)} for c in df.columns],
            "producer": producer or os.path.basename(sys.argv[0]), "created_utc":
                _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"), "sources": sources or []}
    with open(path + ".meta.json", "w") as f:
        json.dump(meta, f, indent=1)


# ----------------------------------------------------------------------------- GCS sync (gcloud storage CLI)
def _gcloud(args: list[str], execute: bool) -> None:
    cmd = ["gcloud", "storage"] + args
    print(" ".join(shlex.quote(c) for c in cmd))
    if execute:
        subprocess.run(cmd, check=True)


def push(study_dir: str, bucket: str, execute: bool = False) -> None:
    """Upload the study definition (and anything produced locally) to its canonical location."""
    sdef = pdio.read_json(os.path.join(study_dir, "study.json"))
    uri = gcs_study_uri(bucket, sdef["study"])
    _gcloud(["rsync", "-r", "--exclude", r".*checkpoint\.bin(\.tmp)?$", study_dir, uri], execute)
    _gcloud(["cp", os.path.join(study_dir, "schema.json"), f"gs://{bucket}/{BUCKET_PREFIX}/_meta/schema.json"], execute)


def pull(study_dir: str, bucket: str, study: str, what: str = "small", execute: bool = False) -> None:
    """Download results. what='small' skips state/VTK/checkpoint binaries (enough for all derived tables)."""
    uri = gcs_study_uri(bucket, study)
    excl = [] if what == "all" else ["--exclude", r".*(state_\d+\.bin|\.vtk|\.vtu|\.vtp|checkpoint\.bin)$"]
    _gcloud(["rsync", "-r", *excl, uri, study_dir], execute)


def publish_meta(bucket: str, execute: bool = False) -> None:
    """Archive the exact solver source / build recipe under _meta/solver/<build>/."""
    src = os.path.join(HERE, "pellet_pd.cpp")
    build = _sha(src)[:16]
    base = f"gs://{bucket}/{BUCKET_PREFIX}/_meta"
    for fn in ("pellet_pd.cpp", "build.sh"):
        _gcloud(["cp", os.path.join(HERE, fn), f"{base}/solver/{build}/{fn}"], execute)
    for fn in ("Dockerfile", "requirements.txt"):
        if os.path.exists(os.path.join(HERE, "cloud", fn)):
            _gcloud(["cp", os.path.join(HERE, "cloud", fn), f"{base}/env/{fn}"], execute)
    print(f"build id (sha256[:16] of pellet_pd.cpp): {build}")


# ----------------------------------------------------------------------------- CLI
def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(description="run / catalogue / sync a planned study")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("study_dir"); r.add_argument("--parallel", type=int, default=1)
    r.add_argument("--threads", type=int, default=0); r.add_argument("--only", nargs="*"); r.add_argument("--solver")
    c = sp.add_parser("catalog"); c.add_argument("study_dir"); c.add_argument("--bucket")
    p = sp.add_parser("push"); p.add_argument("study_dir"); p.add_argument("--bucket", required=True)
    p.add_argument("--execute", action="store_true")
    q = sp.add_parser("pull"); q.add_argument("study_dir"); q.add_argument("--bucket", required=True)
    q.add_argument("--study", required=True); q.add_argument("--what", choices=["small", "all"], default="small")
    q.add_argument("--execute", action="store_true")
    m = sp.add_parser("publish-meta"); m.add_argument("--bucket", required=True); m.add_argument("--execute", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "run":
        run_local(a.study_dir, a.parallel, a.threads, a.only, a.solver)
    elif a.cmd == "catalog":
        catalog(a.study_dir, a.bucket)
    elif a.cmd == "push":
        push(a.study_dir, a.bucket, a.execute)
    elif a.cmd == "pull":
        pull(a.study_dir, a.bucket, a.study, a.what, a.execute)
    elif a.cmd == "publish-meta":
        publish_meta(a.bucket, a.execute)


if __name__ == "__main__":
    main()
