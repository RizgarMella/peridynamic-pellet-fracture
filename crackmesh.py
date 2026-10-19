#!/usr/bin/env python3
"""crackmesh.py - crack-aware mesh visualiser for pellet_pd state dumps.

Particles sit on a cubic lattice, so the dual mesh is a hexahedral grid whose
nodes ARE the particles: each cell joins the 8 particles (i..i+1, j..j+1,
k..k+1).  A cell is DELETED when any of the 28 bonds between its corners is
broken (optionally also when a corner exceeds a damage threshold).  What is
left is a solid mesh in which every crack appears as a physical gap, so
fragments separate visually and can be counted as connected cell sets.

Fields are projected exactly (nodal values = particle values; no smoothing):
damage, temperature, displacement, stresses (Cartesian + cylindrical, von
Mises), strains, solver fragment id.  Cell fields: mesh fragment id, fragment
volume, body.

Outputs per state file (all re-plottable):
  <stem>.vtu              hex mesh with cracked cells removed (ParaView)
  <stem>_cracks.vtp       crack-face surface (quads on the kept side of every crack)
  <stem>_fragments.csv    mesh fragments: id, body, cells, volume, d_eq, centroid, bbox
  <stem>_meshstats.json   counts, geometric crack area, fragment statistics
  <stem>_3d.png / _slice.png   (with --render, needs pyvista)
and, for several input files, crackmesh_series.csv (one row per file).

Usage:
  python crackmesh.py run/state_0040.bin                     # one dump
  python crackmesh.py run/state_*.bin --warp 30 --render     # series + pictures
  python crackmesh.py run/state_0040.bin --dcut 0.6          # also drop very damaged cells

Requires numpy and scipy; pyvista only for --render.
"""
from __future__ import annotations

import argparse
import base64
import json
import os
import sys

import numpy as np

from pdio import cyl_components, read_state, von_mises

CORNERS = np.array([(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0), (0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)])  # VTK_HEXAHEDRON order
PAIRS = [(a, b) for a in range(8) for b in range(a + 1, 8)]  # 12 edges + 12 face diagonals + 4 body diagonals
VTK_HEX, VTK_QUAD = 12, 9


# ----------------------------------------------------------------------------- mesh construction
def build_cells(st: dict, dcut: float | None = None):
    h = st["header"]
    nx, ny, nz, n = h["nx"], h["ny"], h["nz"], h["N"]
    idx = -np.ones((nx, ny, nz), np.int32)
    idx[st["gi"], st["gj"], st["gk"]] = np.arange(n, dtype=np.int32)
    C = np.stack([idx[a:nx - 1 + a, b:ny - 1 + b, c:nz - 1 + c] for a, b, c in CORNERS], axis=-1)
    valid = (C >= 0).all(-1)
    Cv = np.where(C >= 0, C, 0)
    body = st["body"]
    bc = body[Cv]
    valid &= (bc == bc[..., :1]).all(-1)          # no cell spans two bodies (interfaces stay open)
    mask = st["nbmask"]
    broken = np.zeros(valid.shape, bool)
    for a, b in PAIRS:
        d = CORNERS[b] - CORNERS[a]
        bit = np.uint32((d[0] + 1) * 9 + (d[1] + 1) * 3 + (d[2] + 1))
        broken |= ((mask[Cv[..., a]] >> bit) & np.uint32(1)).astype(bool)
    if dcut is not None:
        broken |= (st["damage"][Cv] > dcut).any(-1)
    kept = valid & ~broken
    cracked = valid & broken
    return idx, C, kept, cracked, bc[..., 0]


def label_fragments(kept: np.ndarray):
    from scipy import ndimage
    structure = ndimage.generate_binary_structure(3, 1)   # face connectivity
    lab, nlab = ndimage.label(kept, structure=structure)
    return lab, nlab


def crack_faces(idx, kept, cracked):
    """Quads on the kept side of every kept|cracked cell interface. Returns (quads (M,4), n_faces)."""
    quads = []
    face_corners = {
        0: lambda s: [(s, 0, 0), (s, 1, 0), (s, 1, 1), (s, 0, 1)],
        1: lambda s: [(0, s, 0), (0, s, 1), (1, s, 1), (1, s, 0)],
        2: lambda s: [(0, 0, s), (1, 0, s), (1, 1, s), (0, 1, s)],
    }
    nx1, ny1, nz1 = kept.shape
    for ax in range(3):
        sl_lo = [slice(None)] * 3
        sl_hi = [slice(None)] * 3
        sl_lo[ax] = slice(0, kept.shape[ax] - 1)
        sl_hi[ax] = slice(1, kept.shape[ax])
        for kept_side, crack_side, s in ((tuple(sl_lo), tuple(sl_hi), 1), (tuple(sl_hi), tuple(sl_lo), 0)):
            hit = kept[kept_side] & cracked[crack_side]
            ii, jj, kk = np.nonzero(hit)
            if kept_side[ax].start == 1:   # kept cell is the upper one
                if ax == 0: ii = ii + 1
                elif ax == 1: jj = jj + 1
                else: kk = kk + 1
            pts = [idx[ii + a, jj + b, kk + c] for a, b, c in face_corners[ax](s)]
            if len(ii):
                quads.append(np.stack(pts, axis=-1))
    q = np.concatenate(quads) if quads else np.zeros((0, 4), np.int32)
    return q, len(q)


# ----------------------------------------------------------------------------- VTK XML writers (binary, base64)
_VT = {np.dtype("float32"): "Float32", np.dtype("float64"): "Float64", np.dtype("int32"): "Int32",
       np.dtype("int64"): "Int64", np.dtype("uint8"): "UInt8"}


def _da(name: str, arr: np.ndarray, ncomp: int = 1) -> str:
    arr = np.ascontiguousarray(arr)
    raw = arr.tobytes()
    blob = base64.b64encode(np.array([len(raw)], "<u8").tobytes() + raw).decode()
    return (f'<DataArray type="{_VT[arr.dtype]}" Name="{name}" NumberOfComponents="{ncomp}" '
            f'format="binary">{blob}</DataArray>\n')


def write_vtu(path, points, cells, ctype, npc, point_data: dict, cell_data: dict):
    ncell = len(cells)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="UnstructuredGrid" version="1.0" '
                'byte_order="LittleEndian" header_type="UInt64">\n<UnstructuredGrid>\n')
        f.write(f'<Piece NumberOfPoints="{len(points)}" NumberOfCells="{ncell}">\n<Points>\n')
        f.write(_da("Points", points.astype(np.float32), 3))
        f.write("</Points>\n<Cells>\n")
        f.write(_da("connectivity", cells.astype(np.int64).ravel()))
        f.write(_da("offsets", (np.arange(1, ncell + 1, dtype=np.int64) * npc)))
        f.write(_da("types", np.full(ncell, ctype, np.uint8)))
        f.write("</Cells>\n<PointData>\n")
        for k, (v, nc) in point_data.items():
            f.write(_da(k, v, nc))
        f.write("</PointData>\n<CellData>\n")
        for k, (v, nc) in cell_data.items():
            f.write(_da(k, v, nc))
        f.write("</CellData>\n</Piece>\n</UnstructuredGrid>\n</VTKFile>\n")


def write_vtp_quads(path, points, quads, cell_data: dict):
    n = len(quads)
    with open(path, "w") as f:
        f.write('<?xml version="1.0"?>\n<VTKFile type="PolyData" version="1.0" byte_order="LittleEndian" '
                'header_type="UInt64">\n<PolyData>\n')
        f.write(f'<Piece NumberOfPoints="{len(points)}" NumberOfPolys="{n}">\n<Points>\n')
        f.write(_da("Points", points.astype(np.float32), 3))
        f.write("</Points>\n<Polys>\n")
        f.write(_da("connectivity", quads.astype(np.int64).ravel()))
        f.write(_da("offsets", np.arange(1, n + 1, dtype=np.int64) * 4))
        f.write("</Polys>\n<CellData>\n")
        for k, (v, nc) in cell_data.items():
            f.write(_da(k, v, nc))
        f.write("</CellData>\n</Piece>\n</PolyData>\n</VTKFile>\n")


# ----------------------------------------------------------------------------- main processing
def process(path: str, warp: float, dcut: float | None, render: bool, outdir: str | None) -> dict:
    st = read_state(path)
    h = st["header"]
    dx = h["dx"]
    stem = os.path.splitext(os.path.basename(path))[0]
    od = outdir or os.path.dirname(os.path.abspath(path))
    os.makedirs(od, exist_ok=True)
    base = os.path.join(od, stem)

    idx, C, kept, cracked, cbody = build_cells(st, dcut)
    lab, nlab = label_fragments(kept)
    kc = np.nonzero(kept)
    cells = C[kc]                                    # (M, 8) particle indices
    frag = lab[kc].astype(np.int32)
    counts = np.bincount(lab.ravel(), minlength=nlab + 1)
    vol_cell = dx ** 3
    fvol = counts[frag].astype(np.float32) * vol_cell * 1e9   # mm^3

    x0 = st["x0"].astype(np.float64)
    u = st["u"].astype(np.float64)
    pts = x0 + warp * u
    sig = st["sigma"].astype(np.float64)
    eps = st["eps"].astype(np.float64)
    srr, stt, szz = cyl_components(x0, sig)
    err, ett, ezz = cyl_components(x0, eps)
    pd_ = {
        "damage": (st["damage"], 1), "temperature_K": (st["T"], 1), "displacement_um": ((u * 1e6).astype(np.float32), 3),
        "sigma_rr_MPa": ((srr * 1e-6).astype(np.float32), 1), "sigma_tt_MPa": ((stt * 1e-6).astype(np.float32), 1),
        "sigma_zz_MPa": ((szz * 1e-6).astype(np.float32), 1), "von_mises_MPa": ((von_mises(sig) * 1e-6).astype(np.float32), 1),
        "eps_rr": (err.astype(np.float32), 1), "eps_tt": (ett.astype(np.float32), 1), "eps_zz": (ezz.astype(np.float32), 1),
        "stress_tensor_MPa": ((sig[:, [0, 5, 4, 5, 1, 3, 4, 3, 2]] * 1e-6).astype(np.float32), 9),
        "solver_fragment": (st["fragment"].astype(np.int32), 1), "body": (st["body"].astype(np.uint8), 1),
    }
    cd = {"mesh_fragment": (frag, 1), "fragment_volume_mm3": (fvol, 1), "body": (cbody[kc].astype(np.uint8), 1)}
    write_vtu(base + ".vtu", pts, cells, VTK_HEX, 8, pd_, cd)

    quads, nfaces = crack_faces(idx, kept, cracked)
    if len(quads):
        qd = st["damage"][quads].mean(1).astype(np.float32)
        write_vtp_quads(base + "_cracks.vtp", pts, quads, {"mean_damage": (qd, 1)})

    # fragment table (mesh-based, vectorised: O(cells) regardless of fragment count)
    from scipy import ndimage
    rows = []
    cent = (np.stack(kc, axis=1) + 0.5) * dx            # cell-centre lattice coords [m]
    ncell = np.bincount(frag, minlength=nlab + 1)
    sums = [np.bincount(frag, weights=cent[:, d], minlength=nlab + 1) for d in range(3)]
    fbody = np.zeros(nlab + 1, np.int32)
    fbody[frag] = cbody[kc]
    for fid, sl in enumerate(ndimage.find_objects(lab), start=1):
        if sl is None or ncell[fid] == 0:
            continue
        nc = int(ncell[fid])
        v = nc * vol_cell
        lo = [sl[d].start * dx for d in range(3)]
        hi = [sl[d].stop * dx for d in range(3)]
        rows.append((fid, int(fbody[fid]), nc, v * 1e9, (6 * v / np.pi) ** (1 / 3) * 1e3,
                     *(np.array([sums[d][fid] for d in range(3)]) / nc * 1e3), *(np.array(lo) * 1e3), *(np.array(hi) * 1e3)))
    # lattice-frame coordinates (origin at the lattice corner: x+R, y+R, z), mm
    hdr = ("frag_id,body,cells,volume_mm3,d_eq_mm,centroid_lx_mm,centroid_ly_mm,centroid_lz_mm,"
           "bbox_lx_min_mm,bbox_ly_min_mm,bbox_lz_min_mm,bbox_lx_max_mm,bbox_ly_max_mm,bbox_lz_max_mm")
    np.savetxt(base + "_fragments.csv", np.array(rows) if rows else np.zeros((0, 14)), delimiter=",", header=hdr,
               comments="", fmt=["%d", "%d", "%d"] + ["%.6f"] * 11)

    vols = np.array([r[3] for r in rows]) if rows else np.zeros(0)
    stats = {
        "file": os.path.abspath(path), "step": h["step"], "time_h": h["time_h"], "q_Wpm": h["q_Wpm"], "p_Pa": h["p_Pa"],
        "cells_valid": int(kept.sum() + cracked.sum()), "cells_kept": int(kept.sum()), "cells_cracked": int(cracked.sum()),
        "crack_faces": int(nfaces), "crack_area_geo_mm2": float(nfaces * dx * dx * 1e6 / 2.0),
        "fragments": int(len(vols)),
        "fragments_ge_1pct": int((vols >= 0.01 * vols.sum()).sum()) if len(vols) else 0,
        "vol_mean_mm3": float(vols.mean()) if len(vols) else None,
        "vol_median_mm3": float(np.median(vols)) if len(vols) else None,
        "deq_volweighted_mm": float((6 * (vols ** 2).sum() / vols.sum() / np.pi) ** (1 / 3)) if len(vols) else None,
        "largest_frac": float(vols.max() / vols.sum()) if len(vols) else None,
        "warp": warp, "dcut": dcut,
    }
    with open(base + "_meshstats.json", "w") as f:
        json.dump(stats, f, indent=1)
    if render:
        render_pngs(base, h)
    return stats


def render_pngs(base: str, h: dict) -> None:
    try:
        import pyvista as pv
    except ImportError:
        print("  (pyvista not installed: skipping --render)", file=sys.stderr)
        return
    pv.OFF_SCREEN = True
    m = pv.read(base + ".vtu")
    # 3D: cut away a quarter, colour by fragment
    clip = m.clip_box(bounds=(0, 1, 0, 1, -1, 1), invert=True)
    rng = np.random.default_rng(1)
    nf = int(m.cell_data["mesh_fragment"].max()) + 1
    lut = rng.uniform(0.25, 0.95, size=(nf, 3))        # distinct shades for fragment identity
    p = pv.Plotter(window_size=(1400, 1100))
    p.add_mesh(clip, scalars=lut[clip.cell_data["mesh_fragment"]], rgb=True, show_edges=False)
    if os.path.exists(base + "_cracks.vtp"):
        p.add_mesh(pv.read(base + "_cracks.vtp").clip_box(bounds=(0, 1, 0, 1, -1, 1), invert=True), color="#c0392b", opacity=0.6)
    p.add_text(f"step {h['step']}  q'={h['q_Wpm']/1e3:.1f} kW/m  t={h['time_h']:.2f} h", font_size=12)
    p.view_isometric()
    p.screenshot(base + "_3d.png")
    p.close()
    # mid-plane slice: hoop stress
    zmid = 0.5 * (m.bounds[4] + m.bounds[5])
    sl = m.slice(normal="z", origin=(0, 0, zmid))
    lim = float(np.nanpercentile(np.abs(sl.point_data["sigma_tt_MPa"]), 98)) if sl.n_points else 1.0
    p = pv.Plotter(window_size=(1100, 1000))
    p.add_mesh(sl, scalars="sigma_tt_MPa", cmap="RdBu_r", clim=(-lim, lim), nan_color="#d9d9d9")
    p.view_xy()
    p.screenshot(base + "_slice.png")
    p.close()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("states", nargs="+", help="state_XXXX.bin files")
    ap.add_argument("--warp", type=float, default=1.0, help="displacement magnification for point positions")
    ap.add_argument("--dcut", type=float, default=None, help="also delete cells with a corner damage above this")
    ap.add_argument("--render", action="store_true", help="write PNGs with pyvista")
    ap.add_argument("--outdir", default=None, help="output directory (default: next to each state file)")
    a = ap.parse_args(argv)
    series = []
    for s in sorted(a.states):
        st = process(s, a.warp, a.dcut, a.render, a.outdir)
        print(f"{os.path.basename(s)}: kept {st['cells_kept']} cells, cracked {st['cells_cracked']}, "
              f"crack area {st['crack_area_geo_mm2']:.2f} mm2, fragments {st['fragments']}")
        series.append(st)
    if len(series) > 1:
        keys = [k for k in series[0] if k != "file"]
        out = os.path.join(a.outdir or os.path.dirname(os.path.abspath(a.states[0])), "crackmesh_series.csv")
        with open(out, "w") as f:
            f.write("file," + ",".join(keys) + "\n")
            for s in series:
                f.write(os.path.basename(s["file"]) + "," + ",".join("" if s[k] is None else str(s[k]) for k in keys) + "\n")
        print("series ->", out)


if __name__ == "__main__":
    main()
