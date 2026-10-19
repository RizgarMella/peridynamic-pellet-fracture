#!/usr/bin/env python3
"""predict_fragments.py - fragment-size estimate from a heating/cooling history,
geometry and material properties, using the equations found by fit_pysr.py.

  python predict_fragments.py --history my_cycles.txt --R 4.1e-3 --H 9.8e-3 \
      --model studies/campaign-A/derived/pysr/model_log10_nfrag.json \
      --model studies/campaign-A/derived/pysr/model_log10_deq_vw_over_R.json [--json out.json]

History file: rows "time_h q_Wpm" (heating and cooling cycles; '#' comments).
Material defaults are the nominal UO2 values in config.py; override any of them.
Prints each estimate with a +-1 sigma band (validation residual sd, log10 space)
and WARNS when an input lies outside the training domain (extrapolation).
"""
from __future__ import annotations

import argparse
import json

import config
import features as ft
import pdio


def evaluate(card: dict, feats: dict) -> float:
    import sympy as sp
    syms = {f: sp.Symbol(f) for f in card["features"]}
    expr = sp.sympify(card["expression"], locals=syms)
    fn = sp.lambdify([syms[f] for f in card["features"]], expr, "math")
    return float(fn(*[feats[f] for f in card["features"]]))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--history", required=True)
    ap.add_argument("--model", action="append", required=True)
    ap.add_argument("--R", type=float, default=config.GEOM["R"]); ap.add_argument("--H", type=float, default=config.GEOM["H"])
    ap.add_argument("--dishR", type=float, default=0.0); ap.add_argument("--dishD", type=float, default=0.0)
    for k, v in dict(config.UO2, sigmaf=config.SIGMA_F, weibull=10.0, **config.CREEP).items():
        ap.add_argument(f"--{k}", type=float, default=v)
    ap.add_argument("--no-creep", action="store_true")
    ap.add_argument("--json", help="also write results here")
    a = ap.parse_args()
    hist = pdio.read_history_file(a.history)
    geom = dict(R=a.R, H=a.H, dishR=a.dishR, dishD=a.dishD)
    mat = {k: getattr(a, k) for k in list(config.UO2) + ["sigmaf", "weibull"] + list(config.CREEP)}
    mat["creep"] = not a.no_creep
    out = {"inputs": {"geometry": geom, "material": mat, "history": a.history}, "predictions": []}
    for path in a.model:
        card = json.load(open(path))
        f = ft.groups(geom, mat, hist, card["calibration"] or config.CALIB)
        y = evaluate(card, f)
        sd = card["residual_sd"]
        warn = [k for k in card["features"] if not (card["domain"][k][0] <= f[k] <= card["domain"][k][1])]
        name = card["target"].replace("log10_", "")
        scale = a.R if name.endswith("_over_R") else 1.0
        val, lo, hi = (10 ** y) * scale, (10 ** (y - sd)) * scale, (10 ** (y + sd)) * scale
        unit = " m" if scale != 1.0 else ""
        print(f"{name.replace('_over_R', ''):>18s}: {val:.4g}{unit}  (+-1 sigma: {lo:.4g} - {hi:.4g}{unit})"
              + (f"   WARNING extrapolating in {warn}" if warn else ""))
        out["predictions"].append(dict(target=card["target"], value=val, lo_1sigma=lo, hi_1sigma=hi, log10=y,
                                       extrapolated_features=warn, model=path, features={k: f[k] for k in card["features"]}))
    if a.json:
        json.dump(out, open(a.json, "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
