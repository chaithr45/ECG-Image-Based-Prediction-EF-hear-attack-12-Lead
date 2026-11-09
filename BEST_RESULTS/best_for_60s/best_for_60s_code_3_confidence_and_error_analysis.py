# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/train100/train100_scores.py
# >>> Run from the repository root: PYTHONPATH=. python research_experiments/train100/train100_scores.py (see the docstring below for the options)
# >>> Part of BEST_RESULTS/best_for_60s; see the README there.
"""TRAIN100 scores that need the held-out probabilities: heart-failure screen AUC, accuracy against coverage
(selective prediction) and error rate against the distance of the echo EF from the nearest class limit.

Reruns the cross-validation cells of train100_features.py unchanged (same seeds, folds and models) and keeps, per cell,
the held-out class probabilities in a PRIVATE folder outside the repository (--private; per-ECG, never committed).
The reproduction is checked cell by cell against the committed confusion matrices. Only aggregates go to --out.
    PYTHONPATH=<repo with data> python3 train100_scores.py --private <scratch dir> --out <dir> [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

import train100_features as T

HERE = Path(__file__).resolve().parent
CAP = {}


def _members(rows, pred_sets):
    P, info = ORIG_MEMBERS(rows, pred_sets)
    if len(pred_sets) == 2:                      # the outer fit: [held-out fold, training rows]
        CAP["P"] = {m: v[0] for m, v in P.items()}
    return P, info


def _ef_models(rows, sets, *a):
    e = ORIG_EF(rows, sets, *a)
    if len(sets) == 2:
        CAP["ef"] = e[0]
    return e


ORIG_MEMBERS, ORIG_EF = T.members, T.ef_models
T.members, T.ef_models = _members, _ef_models


def run(task, private):
    CAP.clear()
    out = T.cell(task)
    r, k = task
    fv = T.B.fold_vec(r); te = np.flatnonzero(fv == k)
    np.savez(Path(private) / f"cell_r{r}_k{k}.npz", te=te, ef_hat=CAP["ef"], **{f"P_{m}": v for m, v in CAP["P"].items()})
    return out


def aggregate(private, out):
    """Aggregates only, from the private per-cell probabilities. Pooled within each repeat, averaged over the 3 repeats."""
    from research_experiments.phase_5.p5common import decide
    Y = T.Y; tab = T.B.table(); ef = tab.ef.values.astype(float); bnd = tab.sex_boundary.values
    F = json.load(open(HERE / "results" / "train100_features.json"))
    alph = {(c["repeat"], c["fold"]): c["alphas"] for c in F["cells"]}
    cells = {}
    for r in (0, 1, 2):
        for k in range(5):
            z = np.load(Path(private) / f"cell_r{r}_k{k}.npz")
            fv = T.B.fold_vec(r); tr = np.flatnonzero(fv != k)
            prior = np.bincount(Y[tr], minlength=4) / len(tr)
            d = {m[2:]: z[m] for m in z.files if m.startswith("P_")}
            d["C3"] = decide(d["C1"], "prior", alph[(r, k)]["C3"], prior); d["IMG-C3"] = decide(d["IMG-vote"], "prior", alph[(r, k)]["IMG-C3"], prior)
            cells[(r, k)] = dict(te=z["te"], ef_hat=z["ef_hat"], **d)
    prob_models = [m for m in cells[(0, 0)] if m not in ("te", "ef_hat", "C3", "IMG-C3")]

    def per_repeat(fn):
        return float(np.mean([fn(np.concatenate([cells[(r, k)]["te"] for k in range(5)]), r) for r in (0, 1, 2)]))

    def cat(m, r):
        return np.concatenate([cells[(r, k)][m] for k in range(5)])

    res = dict(note="aggregates of TRAIN100 held-out predictions (rerun of train100_features.py, identical held-out confusion matrices)")
    res["screen_auc"] = {m: per_repeat(lambda te, r, m=m: roc_auc_score(Y[te] != 0, 1 - cat(m, r)[:, 0])) for m in prob_models}
    res["screen_auc"]["C3"] = res["screen_auc"]["C1"]; res["screen_auc"]["IMG-C3"] = res["screen_auc"]["IMG-vote"]
    pairs = {"Normal vs Severe": (0, 3), "Normal vs Moderate": (0, 2), "Normal vs Mild": (0, 1), "Mild vs Moderate": (1, 2), "Moderate vs Severe": (2, 3)}
    res["pairwise_auc"] = {}
    for m in ("C1", "FUSION-MLP", "IMG-vote", "LGBM-FEAT"):
        res["pairwise_auc"][m] = {}
        for nm, (a_, b_) in pairs.items():
            def f(te, r, m=m, a_=a_, b_=b_):
                P = cat(m, r); s = P @ np.arange(4); keep = np.isin(Y[te], (a_, b_))
                return roc_auc_score(Y[te][keep] == b_, s[keep])
            res["pairwise_auc"][m][nm] = per_repeat(f)
    covs = np.round(np.arange(1.0, 0.09, -0.05), 2)
    res["coverage"] = {}
    for m in ("C1", "FUSION-MLP", "IMG-vote", "LGBM-FEAT"):
        pts = []
        for cv in covs:
            def f(te, r, m=m, cv=cv):
                P = cat(m, r); conf = P.max(1); keep = conf >= np.quantile(conf, 1 - cv)
                return np.mean(P.argmax(1)[keep] == Y[te][keep])
            pts.append(dict(coverage=float(cv), accuracy=per_repeat(f)))
        res["coverage"][m] = pts
    res["mild_offset_imgvote"] = []
    for w in (1, 1.25, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 12):
        def acc(te, r, w=w):
            P = cat("IMG-vote", r) * np.array([1, w, 1, 1]); return np.mean(P.argmax(1) == Y[te])

        def mrec(te, r, w=w):
            P = cat("IMG-vote", r) * np.array([1, w, 1, 1]); return np.mean(P.argmax(1)[Y[te] == 1] == 1)
        res["mild_offset_imgvote"].append(dict(weight=w, accuracy=per_repeat(acc), mild_recall=per_repeat(mrec)))
    # error rate against the distance of the echo EF from the nearest class limit
    edges, labels = [0, 3, 6, 10, 100], ["< 3 points", "3--6 points", "6--10 points", "$\\ge$ 10 points"]
    dist = np.min(np.abs(np.column_stack([ef - bnd, ef - 41, ef - 31])), 1)
    res["distance_bins"] = dict(labels=labels, n=[int(np.sum((dist >= lo) & (dist < hi))) for lo, hi in zip(edges[:-1], edges[1:])])
    for m in ("C3", "IMG-C3"):
        err = []
        for lo, hi in zip(edges[:-1], edges[1:]):
            def f(te, r, m=m, lo=lo, hi=hi):
                p = cat(m, r); sel = (dist[te] >= lo) & (dist[te] < hi); return np.mean(p[sel] != Y[te][sel])
            err.append(per_repeat(f))
        res["distance_bins"][m] = err
    near = dist < 6
    res["share_within_6"] = float(near.mean())
    # simulation: a perfect grader whose EF estimate has an error of SD s (837 ECGs, 200 draws per SD)
    rng = np.random.default_rng(0); sim = []
    from research_experiments.phase_6b_threshold_sampling_lstm.mildfix import ef_grade
    for s in np.arange(0, 20.5, 0.5):
        acc = np.mean([np.mean(ef_grade(ef + rng.normal(0, s, len(ef)), bnd, 0.0, 41.0, 31.0) == Y) for _ in range(200)])
        acc_c = np.mean([np.mean(ef_grade(ef + rng.normal(0, s, len(ef)), bnd, 0.0, 41.0, 31.0) == ef_grade(ef, bnd, 0.0, 41.0, 31.0)) for _ in range(50)])
        sim.append(dict(sd=float(s), accuracy=float(acc), accuracy_echo_consistent=float(acc_c)))
    res["simulation"] = sim
    rm = [float(np.sqrt(np.mean((cells[(r, k)]["ef_hat"] - ef[cells[(r, k)]["te"]]) ** 2))) for r in (0, 1, 2) for k in range(5)]
    res["ef_sd_heldout"] = [float(np.min(rm)), float(np.max(rm))]
    res["label_echo_disagreement"] = float(np.mean(ef_grade(ef, bnd, 0.0, 41.0, 31.0) != Y))
    res["ef_sd_population"] = float(np.std(ef))
    json.dump(res, open(Path(out) / "train100_scores.json", "w"), indent=1)
    print(json.dumps({k: v for k, v in res.items() if k in ("screen_auc", "pairwise_auc", "distance_bins", "ef_sd_heldout", "label_echo_disagreement", "ef_sd_population",
                                                           "share_within_6")}, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--private", required=True); ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=4); ap.add_argument("--aggregate-only", action="store_true")
    a = ap.parse_args()
    if a.aggregate_only:
        return aggregate(a.private, a.out)
    priv = Path(a.private); priv.mkdir(parents=True, exist_ok=True)
    T.C.verified(); T.B.table(); T.clin()
    tasks = [(r, k) for r in (0, 1, 2) for k in range(5)]
    with Pool(a.jobs) as pool:
        cells = pool.starmap(run, [(t, str(priv)) for t in tasks])
    ref = {(c["repeat"], c["fold"]): c for c in json.load(open(HERE / "results" / "train100_features.json"))["cells"]}
    same = all(c["models"][m]["heldout_confusion"] == ref[(c["repeat"], c["fold"])]["models"][m]["heldout_confusion"]
               and c["triage"]["heldout_confusion"] == ref[(c["repeat"], c["fold"])]["triage"]["heldout_confusion"]
               for c in cells for m in c["models"])
    print("reproduces the committed held-out confusion matrices:", same, flush=True)
    json.dump(dict(reproduced=same), open(Path(a.out) / "train100_scores_check.json", "w"))
    aggregate(priv, a.out)


if __name__ == "__main__":
    main()
