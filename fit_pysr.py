#!/usr/bin/env python3
"""fit_pysr.py - symbolic regression of fragment size on the campaign dataset.

For each target in features.TARGETS:
  * fit PySRRegressor (PySR 2.x; options filtered against the installed
    version's signature, so minor API changes don't break the script) on the
    'train' split using the dimensionless features in features.FEATURES;
  * evaluate EVERY equation of the Pareto front on the held-out 'val' split;
  * select the simplest equation whose validation RMSE is within --slack of the
    best one (Occam choice), and report it next to a gradient-boosting baseline
    (how much accuracy the closed form gives up);
  * save a self-contained model card JSON that predict_fragments.py can use:
    sympy expression, features, target transform, val metrics, residual sd,
    training-domain bounds, calibration used to generate the data.

Outputs in <study>/derived/pysr/:
  pareto_<target>.csv        every equation: complexity, train loss, val RMSE / R2
  model_<target>.json        selected equation (+ LaTeX) and metadata
  parity_<target>.csv        y_true / y_pred for train and val (re-plottable)
  summary.csv                one row per target: chosen complexity, RMSE, baseline RMSE
Figures in <study>/figures/: parity and Pareto plots.

  python fit_pysr.py studies/campaign-A [--niter 400 --maxsize 30 --targets log10_nfrag]
"""
from __future__ import annotations

import argparse
import inspect
import json
import os

import numpy as np

import features as ft
import pdio
import study_common as sc


def make_model(a, outdir, target):
    from pysr import PySRRegressor
    want = dict(
        niterations=a.niter, populations=a.populations, maxsize=a.maxsize,
        binary_operators=["+", "-", "*", "/"],
        unary_operators=["log", "exp", "sqrt", "square"],
        constraints={"/": (-1, 9), "exp": 7, "log": 7, "sqrt": 7, "square": 7},
        nested_constraints={"exp": {"exp": 0, "log": 0}, "log": {"exp": 0, "log": 0}},
        elementwise_loss="L2DistLoss()", model_selection="best", random_state=a.seed,
        output_directory=outdir, run_id=f"pysr_{target}", progress=False, verbosity=1,
    )
    sig = inspect.signature(PySRRegressor.__init__).parameters
    use = {k: v for k, v in want.items() if k in sig}
    dropped = sorted(set(want) - set(use))
    if dropped:
        print(f"  (installed PySR does not accept: {dropped})")
    return PySRRegressor(**use)


def fit(a):
    import pandas as pd
    import sympy as sp
    from sklearn.ensemble import HistGradientBoostingRegressor
    from sklearn.metrics import mean_squared_error, r2_score

    sdir = a.study_dir
    df = pd.read_csv(os.path.join(sdir, "derived", "dataset.csv"))
    sdef = pdio.read_json(os.path.join(sdir, "study.json"))
    df = df[df.status == "complete"]
    od = os.path.join(sdir, "derived", "pysr")
    os.makedirs(od, exist_ok=True)
    feats = list(ft.FEATURES)
    summary = []
    for target in (a.targets or ft.TARGETS):
        d = df[np.isfinite(df[target])]
        tr, va = d[d.split == "train"], d[d.split == "val"]
        if len(tr) < 30 or len(va) < 10:
            print(f"{target}: not enough data (train {len(tr)}, val {len(va)})"); continue
        Xtr, ytr, Xva, yva = tr[feats].to_numpy(float), tr[target].to_numpy(float), va[feats].to_numpy(float), va[target].to_numpy(float)
        print(f"== {target}: train {len(tr)} val {len(va)}")
        model = make_model(a, od, target)
        model.fit(Xtr, ytr, variable_names=feats)
        eqs = model.equations_.copy()
        val_rmse, val_r2, tr_rmse = [], [], []
        for i in range(len(eqs)):
            pv = model.predict(Xva, index=i)
            pt = model.predict(Xtr, index=i)
            val_rmse.append(float(np.sqrt(mean_squared_error(yva, pv))))
            val_r2.append(float(r2_score(yva, pv)))
            tr_rmse.append(float(np.sqrt(mean_squared_error(ytr, pt))))
        eqs["train_rmse"], eqs["val_rmse"], eqs["val_r2"] = tr_rmse, val_rmse, val_r2
        keep = [c for c in ("complexity", "loss", "score", "equation", "train_rmse", "val_rmse", "val_r2") if c in eqs.columns]
        sc.save_table(eqs[keep], os.path.join(od, f"pareto_{target}.csv"),
                      f"PySR Pareto front for {target}; val metrics on the held-out split", sources=list(d.case_id))
        best = min(val_rmse)
        ok = [i for i, v in enumerate(val_rmse) if v <= best * (1 + a.slack)]
        idx = min(ok, key=lambda i: eqs["complexity"].iloc[i])
        expr = model.sympy(idx)
        base = HistGradientBoostingRegressor(random_state=a.seed).fit(Xtr, ytr)
        base_rmse = float(np.sqrt(mean_squared_error(yva, base.predict(Xva))))
        resid = yva - model.predict(Xva, index=idx)
        card = {
            "schema": "pellet_pd.fragment_model/1", "target": target, "target_transform": "log10",
            "features": feats, "expression": str(expr), "latex": sp.latex(expr),
            "complexity": int(eqs["complexity"].iloc[idx]), "val_rmse": val_rmse[idx], "val_r2": val_r2[idx],
            "train_rmse": tr_rmse[idx], "residual_sd": float(np.std(resid, ddof=1)),
            "baseline_gbr_val_rmse": base_rmse, "n_train": int(len(tr)), "n_val": int(len(va)),
            "domain": {f: [float(d[f].min()), float(d[f].max())] for f in feats},
            "calibration": sdef.get("calibration"), "dataset": os.path.abspath(os.path.join(sdir, "derived", "dataset.csv")),
            "selection": f"simplest equation with val RMSE <= (1+{a.slack}) x best",
        }
        with open(os.path.join(od, f"model_{target}.json"), "w") as f:
            json.dump(card, f, indent=1)
        par = pd.concat([pd.DataFrame(dict(case_id=tr.case_id, split="train", y_true=ytr, y_pred=model.predict(Xtr, index=idx))),
                         pd.DataFrame(dict(case_id=va.case_id, split="val", y_true=yva, y_pred=model.predict(Xva, index=idx)))])
        sc.save_table(par, os.path.join(od, f"parity_{target}.csv"), f"Selected equation predictions for {target}")
        summary.append(dict(target=target, complexity=card["complexity"], val_rmse=card["val_rmse"], val_r2=card["val_r2"],
                            baseline_gbr_val_rmse=base_rmse, expression=card["expression"]))
        print(f"   chosen (complexity {card['complexity']}): {expr}\n   val RMSE {card['val_rmse']:.4f} "
              f"(R2 {card['val_r2']:.3f}); GBR baseline {base_rmse:.4f}")
    if summary:
        sc.save_table(pd.DataFrame(summary), os.path.join(od, "summary.csv"), "Selected equation per target")
        plot(sdir)


def plot(sdir):
    import glob
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import pandas as pd
    od, fd = os.path.join(sdir, "derived", "pysr"), os.path.join(sdir, "figures")
    os.makedirs(fd, exist_ok=True)
    for p in glob.glob(os.path.join(od, "parity_*.csv")):
        t = os.path.basename(p)[7:-4]
        d = pd.read_csv(p)
        fig, ax = plt.subplots(figsize=(4.8, 4.6), constrained_layout=True)
        for split, col in (("train", "#9bb7e0"), ("val", "#d9822b")):
            s = d[d.split == split]
            ax.plot(s.y_true, s.y_pred, "o", ms=4, color=col, mec="white", mew=0.5, label=split)
        lo, hi = d[["y_true", "y_pred"]].min().min(), d[["y_true", "y_pred"]].max().max()
        ax.plot([lo, hi], [lo, hi], color="#888", lw=1)
        ax.set_xlabel(f"simulated {t}"); ax.set_ylabel("equation"); ax.legend(frameon=False)
        fig.savefig(os.path.join(fd, f"pysr_parity_{t}.png"), dpi=150); plt.close(fig)
        pf = pd.read_csv(os.path.join(od, f"pareto_{t}.csv"))
        fig, ax = plt.subplots(figsize=(5, 3.8), constrained_layout=True)
        ax.plot(pf.complexity, pf.train_rmse, "o-", color="#9bb7e0", lw=2, ms=4, label="train")
        ax.plot(pf.complexity, pf.val_rmse, "o-", color="#d9822b", lw=2, ms=4, label="val")
        ax.set_xlabel("equation complexity"); ax.set_ylabel(f"RMSE of {t}"); ax.legend(frameon=False)
        fig.savefig(os.path.join(fd, f"pysr_pareto_{t}.png"), dpi=150); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("study_dir")
    ap.add_argument("--targets", nargs="*")
    ap.add_argument("--niter", type=int, default=400)
    ap.add_argument("--populations", type=int, default=31)
    ap.add_argument("--maxsize", type=int, default=30)
    ap.add_argument("--slack", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--plot-only", action="store_true")
    a = ap.parse_args()
    plot(a.study_dir) if a.plot_only else fit(a)


if __name__ == "__main__":
    main()
