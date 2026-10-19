"""schema.py - data dictionary for every file produced by pellet_pd v2 and the study tools.

`python schema.py > schema.json` (or study_common.publish_meta) writes the
machine-readable version that is stored at  gs://<bucket>/pellet-pd/v2/_meta/schema.json.
Units follow the column-name suffix convention used everywhere:
  _mm _um _mm2 _mm3 (lengths/areas/volumes), _MPa _Pa, _K, _Wpm (W/m), _h, _s, _deg.
Columns without a suffix are dimensionless or counts.
"""
import json

SCHEMA_VERSION = "pellet_pd.dataset/2"

FILES = {
    "manifest.json": {
        "level": "run", "format": "json",
        "description": "All solver inputs (params), derived quantities (dx, delta, s_c0, K, mu, Irwin length...), "
                       "particle/bond counts per body, the full load schedule [time_h, dt_h, q_Wpm, p_Pa, is_knot], "
                       "labels (study, case_id, factors), solver version and build id (sha256 of source).",
    },
    "summary.json": {
        "level": "run", "format": "json",
        "description": "Written only when a run completes ('complete': true). Onset power (first step where the main "
                       "body's crack area exceeds onsetFrac*R*H), crack area per body, radial crack counts, fragment "
                       "statistics at the final and peak-power output steps, timings, throughput, labels.",
    },
    "history.csv": {
        "level": "run", "format": "csv", "row": "one load step",
        "columns": {
            "step": ("-", "load step index (1-based)"),
            "time_h": ("h", "time at end of step"), "dt_h": ("h", "step duration"),
            "q_Wpm": ("W/m", "linear heat rate"), "p_Pa": ("Pa", "nominal axial pressure on stack top"),
            "is_knot": ("-", "1 if the step ends a history segment (full outputs written)"),
            "T_centre_K": ("K", "pellet centre temperature"), "T_surface_K": ("K", "pellet surface temperature"),
            "adr_iters": ("-", "dynamic-relaxation iterations in the step (all cycles)"),
            "break_cycles": ("-", "break/re-relax cycles in the step"),
            "residual": ("-", "final max-norm force residual / load scale"),
            "scale": ("N/m3", "load scale (max |thermal force density| at zero displacement, or pressure load)"),
            "new_broken_pairs": ("-", "bond pairs broken in this step"),
            "broken_pairs": ("-", "cumulative broken bond pairs (all bodies)"),
            "crack_area_mm2": ("mm2", "crack area from cut bonds, all bodies: 5/(pi delta^5) * sum_cut x V V"),
            "mean_damage": ("-", "mean particle damage (fraction of family volume lost)"),
            "max_damage": ("-", "max particle damage"),
            "stt_p50_MPa": ("MPa", "median hoop stress over stress-valid particles"),
            "stt_p99_MPa": ("MPa", "99th percentile hoop stress"), "stt_max_MPa": ("MPa", "max hoop stress"),
            "szz_p99_MPa": ("MPa", "99th percentile axial stress"), "svm_p99_MPa": ("MPa", "99th percentile von Mises"),
            "n_stress_valid": ("-", "particles with a reported stress (crack debris excluded)"),
            "max_contact_p_MPa": ("MPa", "max inter-pellet contact pressure"),
            "mean_contact_p_MPa": ("MPa", "mean contact pressure over all contact pairs"),
            "contact_frac_closed": ("-", "fraction of contact pairs closed"),
            "contact_frac_slip": ("-", "fraction of closed pairs sliding (Coulomb limit)"),
            "mean_slip_um": ("um", "mean accumulated interface slip"),
            "wall_s": ("s", "wall time of the step"),
            "b{k}_crack_area_mm2": ("mm2", "crack area of body k"), "b{k}_broken_pairs": ("-", "broken pairs of body k"),
            "b{k}_radial_cracks": ("-", "radial cracks on body k mid-plane outer ring (-1 = ring fully damaged)"),
            "b{k}_ur_mid_um": ("um", "mean radial displacement of body k outer surface at mid-plane"),
            "b{k}_ur_end_um": ("um", "mean radial displacement of body k outer surface at end layers (hourglassing = end - mid)"),
            "b{k}_dH_um": ("um", "axial elongation of body k (mean top - mean bottom displacement)"),
        },
    },
    "cycles.csv": {
        "level": "run", "format": "csv", "row": "one break/relax cycle",
        "columns": {"step": ("-", "load step"), "cycle": ("-", "cycle index within step"),
                    "max_ratio": ("-", "max stretch/critical stretch before breaking"),
                    "bonds_broken": ("-", "pairs broken in the cycle"), "relax_iters": ("-", "ADR iterations"),
                    "residual": ("-", "final residual")},
    },
    "fragments.csv": {
        "level": "run", "format": "csv", "row": "one fragment at one output step",
        "description": "Connected components of intact bonds between particles with damage < dcut, >= minfrag particles.",
        "columns": {"step": ("-", ""), "time_h": ("h", ""), "q_Wpm": ("W/m", ""), "body": ("-", "body index"),
                    "frag_id": ("-", "fragment id (per output step)"), "n_particles": ("-", ""),
                    "volume_mm3": ("mm3", ""), "d_eq_mm": ("mm", "equivalent-sphere diameter"),
                    "cx_mm": ("mm", "centroid x"), "cy_mm": ("mm", "centroid y"), "cz_mm": ("mm", "centroid z"),
                    "rmin_mm": ("mm", "min radius"), "rmax_mm": ("mm", "max radius"),
                    "zmin_mm": ("mm", "min z"), "zmax_mm": ("mm", "max z")},
    },
    "profile_XXXX.csv": {
        "level": "run/output-step", "format": "csv", "row": "one radial bin of one body's mid-plane layer",
        "columns": {"body": ("-", ""), "r_mm": ("mm", "bin centre radius"), "n": ("-", "particles in bin"),
                    "T_K": ("K", ""), "u_r_um": ("um", "radial displacement"), "damage": ("-", ""),
                    "sigma_rr_MPa": ("MPa", ""), "sigma_tt_MPa": ("MPa", ""), "sigma_zz_MPa": ("MPa", ""),
                    "sigma_vm_MPa": ("MPa", ""), "eps_rr": ("-", "total strain"), "eps_tt": ("-", ""), "eps_zz": ("-", ""),
                    "n_stress": ("-", "particles with valid stress"),
                    "an_sigma_rr_MPa": ("MPa", "analytic elastic long-cylinder solution (uncracked reference)"),
                    "an_sigma_tt_MPa": ("MPa", ""), "an_sigma_zz_MPa": ("MPa", "")},
    },
    "angular_XXXX.csv": {
        "level": "run/output-step", "format": "csv", "row": "one 2-degree angular bin",
        "columns": {"theta_deg": ("deg", "bin centre"),
                    "b{k}_mid": ("-", "max damage, body k mid-plane, r > 0.75R"),
                    "b{k}_bottom": ("-", "max damage, body k lowest two layers, r > 0.4R"),
                    "b{k}_top": ("-", "max damage, body k highest two layers, r > 0.4R")},
    },
    "interface_XXXX.csv": {
        "level": "run/output-step", "format": "csv", "row": "one contact pair (stacks only)",
        "columns": {"iface": ("-", "interface index (between body k and k+1)"), "r_mm": ("mm", ""), "theta_deg": ("deg", ""),
                    "gap_um": ("um", "normal gap (<0 = closed)"), "p_MPa": ("MPa", "contact pressure"),
                    "tau_MPa": ("MPa", "shear traction magnitude"), "slip_um": ("um", "accumulated slip"),
                    "state": ("-", "0 open, 1 stick, 2 slip"), "dmg_lower": ("-", "damage of lower particle"),
                    "dmg_upper": ("-", "damage of upper particle")},
    },
    "state_XXXX.bin": {
        "level": "run/output-step", "format": "binary",
        "description": "Line 1 'PDSTATE2'; line 2 JSON header (N, lattice dims, dx, delta, step, time_h, q_Wpm, p_Pa, "
                       "labels, array list); then little-endian arrays in header order: gi,gj,gk int32; body uint8; "
                       "x0 float32[N,3] m; u float32[N,3] m; T float32 K; damage float32; sigma float32[N,6] Pa "
                       "(xx,yy,zz,yz,xz,xy; NaN = not reported); eps float32[N,6]; fragment int32; nbmask uint32 "
                       "(broken bond to lattice neighbour (di,dj,dk): bit (di+1)*9+(dj+1)*3+(dk+1)). Reader: pdio.read_state.",
    },
    "restart.bin": {
        "level": "run", "format": "binary",
        "description": "Single-pellet restart (PDRST02): geometry header, x0, u, family sizes, broken flags, critical "
                       "stretches, creep back-extensions. Reader: pdio.read_restart_header; consumed by --restart.",
    },
    "checkpoint.bin": {
        "level": "run", "format": "binary",
        "description": "Whole-model checkpoint for Spot preemption (PDCKP02). Deleted after successful completion.",
    },
    "*_meshstats.json / *_fragments.csv / *.vtu / *_cracks.vtp": {
        "level": "run/output-step", "format": "json/csv/vtk",
        "description": "crackmesh.py products: hex mesh with cracked cells removed, crack faces, mesh-based "
                       "fragments (face-connected kept cells) and geometric crack area (crack faces * dx^2 / 2).",
    },
    "cases.csv": {
        "level": "study", "format": "csv", "row": "one case (design matrix)",
        "description": "case_id, split (train/val/-), every factor/sampled variable with units in the name, the exact "
                       "solver argument string, and the history file name.",
    },
    "catalog.csv": {
        "level": "study", "format": "csv", "row": "one run",
        "description": "status (complete/incomplete/missing), solver build, counts, key metrics, sha256 of manifest/"
                       "summary/history, bytes, local and gs:// paths.",
    },
    "derived/*.csv (+ .meta.json)": {
        "level": "study", "format": "csv + json sidecar",
        "description": "Tidy analysis tables; every CSV has a .meta.json sidecar listing columns, units, description, "
                       "the code that produced it and the source cases. Figures in figures/ are made only from these.",
    },
}


def schema_dict() -> dict:
    return {"schema": SCHEMA_VERSION, "units_convention": __doc__.split("Units follow")[1].strip(), "files": FILES}


if __name__ == "__main__":
    print(json.dumps(schema_dict(), indent=1))
