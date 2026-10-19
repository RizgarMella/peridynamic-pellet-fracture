"""pdio.py - readers/writers for pellet_pd (v2) outputs.

Every study script goes through these functions, so the on-disk formats are
documented in exactly one place:

  manifest.json     inputs, derived quantities, counts, load schedule
  summary.json      final/peak metrics (written only when a run completes)
  history.csv       one row per load step
  cycles.csv        one row per break/relax cycle
  fragments.csv     one row per fragment per output step
  profile_XXXX.csv  mid-plane radial profiles (per body)
  angular_XXXX.csv  damage vs angle (mid-plane ring + end faces, per body)
  interface_XXXX.csv contact pairs (stacks)
  state_XXXX.bin    full particle state (see read_state)
  restart.bin       single-pellet restart (see read_restart_header)
"""
from __future__ import annotations

import glob
import json
import os
import re
import struct

import numpy as np

try:
    import pandas as pd
except ImportError:  # pandas is required by the study scripts, not by crackmesh.py
    pd = None


# ----------------------------------------------------------------------------- state
_DT = {"i4": "<i4", "u1": "u1", "f4": "<f4", "u4": "<u4"}


def read_state(path: str) -> dict:
    """Read a state_XXXX.bin dump. Returns dict of numpy arrays + 'header'.

    Arrays: gi, gj, gk (lattice indices), body, x0 (N,3), u (N,3), T, damage,
    sigma (N,6: xx yy zz yz xz xy, Pa, NaN where not reported), eps (N,6),
    fragment (solver fragment id, -1 = crack band), nbmask (broken-bond bits to
    the 26 lattice neighbours, bit = (di+1)*9 + (dj+1)*3 + (dk+1)).
    """
    with open(path, "rb") as f:
        if f.readline().strip() != b"PDSTATE2":
            raise ValueError(f"{path}: not a PDSTATE2 file")
        hdr = json.loads(f.readline())
        n = hdr["N"]
        out = {"header": hdr}
        for spec in hdr["arrays"]:
            name, typ, nc = spec.split(":")
            nc = int(nc)
            dt = np.dtype(_DT[typ])
            a = np.frombuffer(f.read(n * nc * dt.itemsize), dtype=dt)
            out[name] = a.reshape(n, nc) if nc > 1 else a
    return out


def cyl_components(x0: np.ndarray, t6: np.ndarray):
    """(rr, tt, zz) of symmetric tensors stored as xx yy zz yz xz xy."""
    r = np.hypot(x0[:, 0], x0[:, 1])
    c = np.where(r > 0, x0[:, 0] / np.maximum(r, 1e-300), 1.0)
    s = np.where(r > 0, x0[:, 1] / np.maximum(r, 1e-300), 0.0)
    rr = c * c * t6[:, 0] + s * s * t6[:, 1] + 2 * c * s * t6[:, 5]
    tt = s * s * t6[:, 0] + c * c * t6[:, 1] - 2 * c * s * t6[:, 5]
    return rr, tt, t6[:, 2]


def von_mises(t6: np.ndarray) -> np.ndarray:
    a1, a2, a3 = t6[:, 0] - t6[:, 1], t6[:, 1] - t6[:, 2], t6[:, 2] - t6[:, 0]
    return np.sqrt(0.5 * (a1 ** 2 + a2 ** 2 + a3 ** 2) + 3 * (t6[:, 3] ** 2 + t6[:, 4] ** 2 + t6[:, 5] ** 2))


# ----------------------------------------------------------------------------- restart
def read_restart_header(path: str) -> dict:
    """Header of a restart.bin (enough to build a geometry-matched stack run)."""
    with open(path, "rb") as f:
        magic = f.read(8)
        if not magic.startswith(b"PDRST02"):
            raise ValueError(f"{path}: not a PDRST02 restart file")
        n, nr, creep = struct.unpack("<3i", f.read(12))
        (nb,) = struct.unpack("<q", f.read(8))
        h = struct.unpack("<10d", f.read(80))
    keys = ["R", "H_eff", "dx", "delta", "dishR", "dishD", "chamfer", "jitter", "time_h", "seed"]
    d = dict(zip(keys, h))
    d.update(N=n, nr=nr, creep=bool(creep), NB=nb)
    return d


# ----------------------------------------------------------------------------- run directory
def read_json(path: str):
    with open(path) as f:
        return json.load(f)


def run_complete(out_dir: str) -> bool:
    p = os.path.join(out_dir, "summary.json")
    if not os.path.exists(p):
        return False
    try:
        return bool(read_json(p).get("complete", False))
    except Exception:
        return False


def read_run(out_dir: str) -> dict:
    """Everything small about a run: manifest, summary, history (DataFrame), fragments (DataFrame)."""
    r = {"dir": out_dir}
    for name in ("manifest", "summary"):
        p = os.path.join(out_dir, f"{name}.json")
        r[name] = read_json(p) if os.path.exists(p) else None
    if pd is not None:
        for name in ("history", "fragments", "cycles"):
            p = os.path.join(out_dir, f"{name}.csv")
            r[name] = pd.read_csv(p) if os.path.exists(p) and os.path.getsize(p) > 0 else None
    return r


def output_steps(out_dir: str, kind: str = "state") -> list[int]:
    ext = "bin" if kind == "state" else "csv"
    steps = []
    for p in glob.glob(os.path.join(out_dir, f"{kind}_*.{ext}")):
        m = re.search(rf"{kind}_(\d+)\.{ext}$", p)
        if m:
            steps.append(int(m.group(1)))
    return sorted(steps)


def read_step_csv(out_dir: str, kind: str, step: int):
    return pd.read_csv(os.path.join(out_dir, f"{kind}_{step:04d}.csv"))


def knot_steps(history) -> list[int]:
    """Load steps that end a history segment (where full outputs are guaranteed)."""
    return [int(s) for s in history.loc[history["is_knot"] == 1, "step"]]


def value_at_q(history, column: str, q_values, branch: str = "first_rise"):
    """Interpolate a history column at given powers q (W/m) on the first rising branch."""
    h = history
    if branch == "first_rise":
        q = h["q_Wpm"].to_numpy()
        stop = len(q)
        for k in range(1, len(q)):
            if q[k] < q[k - 1] - 1e-9:
                stop = k
                break
        h = h.iloc[:stop]
    q = h["q_Wpm"].to_numpy()
    y = h[column].to_numpy()
    # collapse repeated q (holds) to the last value at that power
    uq, idx = np.unique(q[::-1], return_index=True)
    yy = y[::-1][idx]
    return np.interp(np.asarray(q_values, float), uq, yy, left=np.nan, right=np.nan)


# ----------------------------------------------------------------------------- histories
def write_history(path: str, rows, comment: str = "") -> None:
    """rows: iterable of (time_h, q_Wpm) or (time_h, q_Wpm, p_Pa)."""
    with open(path, "w") as f:
        f.write("# time_h q_Wpm [p_Pa]\n")
        if comment:
            for line in comment.splitlines():
                f.write(f"# {line}\n")
        for r in rows:
            f.write(" ".join(f"{v:.10g}" for v in r) + "\n")


def read_history_file(path: str) -> np.ndarray:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.split("#", 1)[0].replace(",", " ").replace(";", " ").strip()
            if not line:
                continue
            v = [float(x) for x in line.split()]
            if len(v) >= 2:
                rows.append(v[:3] if len(v) >= 3 else v[:2] + [0.0])
    return np.array(rows, float)
