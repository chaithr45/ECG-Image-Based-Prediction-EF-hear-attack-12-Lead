# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/train100/train100_features.py
# >>> Run from the repository root: PYTHONPATH=. python research_experiments/train100/train100_features.py (see the docstring below for the options)
# >>> Part of BEST_RESULTS/best_for_50s; see the README there.
"""TRAIN100 (AMENDMENT_TRAIN100.md): the project's models retrained until (nearly) 100 % training accuracy, with as many
epochs / iterations as needed and no early stopping. Feature-based models, ensembles, decision rules and the triage.
Same inputs, labels (original four-class), and Phase-2 patient-grouped 5-fold CV x 3 repeats as the original runs.
Development data only (837 ECGs); the locked test set is not used. Aggregates only; nothing per ECG is written.
    PYTHONPATH=<repo> python3 train100_features.py --out <dir> [--jobs 4]
Models: ECGF-linear, ECGF-MLP, HDR-LR, HDRCLIN-LR, CLIN-LR (image-only features), FUSION-MLP (header + ECGFounder),
LGBM-FEAT (header + engineered feature families, as Phase-5 E2 "eng"), C1 (vote), C3 (C1 + nested prior rule), IMG-vote, IMG-C3, AGT-img triage.
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from research_experiments.phase_2.p2heads import Head, class_w
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.p5common import blockstd, corr_ci, decide, ecgf_f6, impute, make_X
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1
from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin
from research_experiments.phase_7 import p7common as C

B = C.B
Y = C.Y
SEEDS = (42, 43, 44)
MLP_MAX_EPOCHS, LIN_MAX_STEPS, LR_C = 3000, 5000, 1e4
FEAT_FAMILIES = "feat:clin3+freq+hrv+leadstat+qrs+sq+st+wavelet"
CLASSES = ("Normal", "Mild", "Moderate", "Severe")


def bal_w(y):
    n = np.bincount(y, minlength=4).astype(float)
    return (len(y) / (4 * np.maximum(n, 1)))[y]


# ------------------------------------------------------------------ learners trained to (nearly) 100 % training accuracy
def _std(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6
    return lambda idx: torch.tensor((X[idx] - mu) / sd, dtype=torch.float32)


def mlp(X, rows, seed):
    """Project MLP head (dropout 0.5, 64 units, AdamW 1e-3, wd 0.1, batch 64, balanced CE) until 100 % training accuracy."""
    T = _std(X, rows); xtr = T(rows); yt = torch.tensor(Y[rows]); cw = class_w(Y[rows], "balanced")
    torch.manual_seed(seed); m = Head(X.shape[1], "ce", "mlp", 64, 0.5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); g = torch.Generator().manual_seed(seed)
    ep, acc = 0, 0.0
    while ep < MLP_MAX_EPOCHS and acc < 1.0:
        m.train(); idx = torch.randperm(len(rows), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; lo = m.loss(xtr[b], yt[b], cw, {}); opt.zero_grad(); lo.backward(); opt.step()
        ep += 1; m.eval()
        with torch.no_grad():
            acc = float((m.probs(xtr).argmax(1).numpy() == Y[rows]).mean())
    m.eval()

    def f(idx):
        with torch.no_grad():
            return m.probs(T(idx)).numpy()
    return f, dict(epochs=ep, train_acc=acc)


def mlp3(X, rows):
    fs = [mlp(X, rows, s) for s in SEEDS]
    return (lambda idx: np.mean([f(idx) for f, _ in fs], 0)), dict(epochs=[i["epochs"] for _, i in fs])


def linear(X, rows, seed=42):
    """ECGFounder linear head (distance-penalised CE), L-BFGS with L2 = 1e-6 until 100 % training accuracy."""
    T = _std(X, rows); xtr = T(rows); yt = torch.tensor(Y[rows]); cw = class_w(Y[rows], "balanced")
    torch.manual_seed(seed); m = Head(X.shape[1], "ce_dist", "linear")
    opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=100, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lo = m.loss(xtr, yt, cw, {"lam": 1.0}) + 1e-6 * sum((p ** 2).sum() for p in m.weight_params())
        lo.backward(); return lo
    steps, acc = 0, 0.0
    while steps < LIN_MAX_STEPS and acc < 1.0:
        opt.step(closure); steps += 100
        with torch.no_grad():
            acc = float((m.probs(xtr).argmax(1).numpy() == Y[rows]).mean())
    m.eval()

    def f(idx):
        with torch.no_grad():
            return m.probs(T(idx)).numpy()
    return f, dict(steps=steps, train_acc=acc)


def logreg(X, rows):
    """Logistic regression with almost no regularisation (C = 1e4), class-balanced."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(C=LR_C, class_weight="balanced", max_iter=20000)).fit(X[rows], Y[rows])
    return (lambda idx: clf.predict_proba(X[idx])), {}


def lgbm(X, rows):
    """LightGBM without early stopping or shrinkage-style limits (2,000 trees, 31 leaves, min_child_samples 1)."""
    import lightgbm as lgb
    m = lgb.LGBMClassifier(objective="multiclass", n_estimators=2000, learning_rate=0.05, num_leaves=31, min_child_samples=1,
                           subsample=1.0, colsample_bytree=1.0, reg_lambda=0.0, random_state=0, n_jobs=1, verbose=-1)
    m.fit(X[rows], Y[rows], sample_weight=bal_w(Y[rows]))
    return (lambda idx: m.predict_proba(X[idx])), {}


def ef_models(fit_rows, pred_sets, clean, ef, E, IMG):
    """Triage EF regressors without shrinkage: ridge (alpha 1e-6) + LightGBM (4,000 trees, no early stopping); mean."""
    import lightgbm as lgb
    w = np.where(clean[fit_rows], 1.0, 0.3)
    fm = np.zeros(C.nD, bool); fm[fit_rows] = True
    X1 = blockstd([E, impute(IMG, fm)], fm)
    r1 = Ridge(alpha=1e-6).fit(X1[fit_rows], ef[fit_rows], sample_weight=w)
    mu, sd = E[fm].mean(0), E[fm].std(0) + 1e-6
    X2 = np.column_stack([IMG, PCA(32, random_state=0).fit(((E - mu) / sd)[fm]).transform((E - mu) / sd)])
    r2 = lgb.LGBMRegressor(n_estimators=4000, learning_rate=0.05, num_leaves=31, min_child_samples=1, subsample=1.0,
                           colsample_bytree=1.0, reg_lambda=0.0, random_state=0, n_jobs=1, verbose=-1).fit(X2[fit_rows], ef[fit_rows], sample_weight=w)
    return [(r1.predict(X1[idx]) + r2.predict(X2[idx])) / 2 for idx in pred_sets]


# ------------------------------------------------------------------ inputs (fit-row statistics only)
def inputs(fit_rows):
    fm = np.zeros(C.nD, bool); fm[fit_rows] = True
    E = ecgf_f6().astype(float)
    return dict(ecgf=E, hdr=make_X(["hdr"], fm)[0], hdrclin=make_X(["hdr", "feat:clin3"], fm)[0],
                fusion=make_X(["hdr", "ecgf"], fm)[0], feat=make_X(["hdr", FEAT_FAMILIES], fm)[0], clin=impute(clin()[0], fm))


def members(rows, pred_sets):
    """Fit every single model on `rows`; probabilities for each index set in pred_sets; plus info."""
    X = inputs(rows)
    fit = {"ECGF-linear": linear(X["ecgf"], rows), "ECGF-MLP": mlp3(X["ecgf"], rows), "HDR-LR": logreg(X["hdr"], rows),
           "HDRCLIN-LR": logreg(X["hdrclin"], rows), "CLIN-LR": logreg(X["clin"], rows), "FUSION-MLP": mlp3(X["fusion"], rows),
           "LGBM-FEAT": lgbm(X["feat"], rows)}
    P = {name: [f(idx) for idx in pred_sets] for name, (f, _) in fit.items()}
    P["C1"] = [np.mean([P[m][i] for m in ("FUSION-MLP", "ECGF-linear", "HDRCLIN-LR")], 0) for i in range(len(pred_sets))]
    P["IMG-vote"] = [np.mean([P[m][i] for m in ("ECGF-MLP", "ECGF-linear")], 0) for i in range(len(pred_sets))]
    return P, {name: inf for name, (_, inf) in fit.items()}


# ------------------------------------------------------------------ one outer fold cell
def cell(task):
    r, k = task
    warnings.filterwarnings("ignore"); torch.set_num_threads(1); t0 = time.time()
    ef = B.table().ef.values.astype(float); bnd = B.table().sex_boundary.values; clean = C.verified()[0]
    E = ecgf_f6().astype(float); IMG = clin()[0]
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k); yTR, yte = Y[TR], Y[te]
    prior = np.bincount(yTR, minlength=4) / len(TR)
    split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, C.PAT[TR]))
    # outer fit: predictions for the held-out fold and in-sample for the training rows
    P, info = members(TR, [te, TR])
    # inner OOF (for the nested prior rules and the triage cut-points / margin)
    Pin = {m: np.zeros((len(TR), 4)) for m in ("C1", "IMG-vote")}; ein = np.zeros(len(TR))
    for ia, ib in split:
        Pi, _ = members(TR[ia], [TR[ib]])
        for m in Pin:
            Pin[m][ib] = Pi[m][0]
        ein[ib] = ef_models(TR[ia], [TR[ib]], clean, ef, E, IMG)[0]
    pred = {m: (P[m][0].argmax(1), P[m][1].argmax(1)) for m in P}          # (held-out, training) argmax predictions
    alphas = {}
    for base, name in (("C1", "C3"), ("IMG-vote", "IMG-C3")):
        a = choose_alpha(Pin[base], yTR, prior, "protected"); alphas[name] = float(a)
        pred[name] = (decide(P[base][0], "prior", a, prior), decide(P[base][1], "prior", a, prior))
        pred[name + "_inner"] = decide(Pin[base], "prior", a, prior)
    out = dict(repeat=r, fold=k, alphas=alphas, info=info, models={})
    for m, pp in pred.items():
        if m.endswith("_inner"):
            continue
        p_te, p_tr = pp
        out["models"][m] = dict(train_acc=float(np.mean(p_tr == yTR)), heldout_acc=float(np.mean(p_te == yte)),
                                heldout_recall=[float(np.mean(p_te[yte == c] == c)) for c in range(4)],
                                heldout_confusion=np.array([[np.sum((yte == i) & (p_te == j)) for j in range(4)] for i in range(4)]).tolist())
    # AGT-img triage (agreement with IMG-C3)
    e_te, e_tr = ef_models(TR, [te, TR], clean, ef, E, IMG)
    grid = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]
    p = F1.floor_best(((q, F1.ef_grade(ein, bnd[TR], *q)) for q in grid), yTR) or (0.0, 41.0, 31.0)
    dist = lambda e, b: np.abs(e[:, None] - np.column_stack([b + p[0], np.full(len(e), p[1]), np.full(len(e), p[2])])).min(1)
    g_in, g_te, g_tr = F1.ef_grade(ein, bnd[TR], *p), F1.ef_grade(e_te, bnd[te], *p), F1.ef_grade(e_tr, bnd[TR], *p)
    ag_in = pred["IMG-C3_inner"] == g_in
    t = F1.threshold(dist(ein, bnd[TR])[ag_in], (g_in == yTR)[ag_in])
    k_te = (pred["IMG-C3"][0] == g_te) & (dist(e_te, bnd[te]) >= t)
    k_tr = (pred["IMG-C3"][1] == g_tr) & (dist(e_tr, bnd[TR]) >= t)
    cm = np.array([[np.sum((yte[k_te] == i) & (g_te[k_te] == j)) for j in range(4)] for i in range(4)])
    out["triage"] = dict(cuts=[float(x) for x in p], margin=float(t),
                         train_coverage=float(k_tr.mean()), train_acc=float(np.mean(g_tr[k_tr] == yTR[k_tr])) if k_tr.any() else None,
                         grade_train_acc_all=float(np.mean(g_tr == yTR)), ef_train_rmse=float(np.sqrt(np.mean((e_tr - ef[TR]) ** 2))),
                         heldout_coverage=float(k_te.mean()), heldout_acc=float(np.mean(g_te[k_te] == yte[k_te])) if k_te.any() else None,
                         ef_heldout_rmse=float(np.sqrt(np.mean((e_te - ef[te]) ** 2))), heldout_confusion=cm.tolist(),
                         heldout_referred=[int(np.sum((yte == c) & ~k_te)) for c in range(4)])
    out["minutes"] = (time.time() - t0) / 60
    tri = out["triage"]
    print(f"[r{r} k{k}] {out['minutes']:.1f} min | IMG-C3 train {out['models']['IMG-C3']['train_acc']:.3f} held-out "
          f"{out['models']['IMG-C3']['heldout_acc']:.3f} | C3 train {out['models']['C3']['train_acc']:.3f} held-out "
          f"{out['models']['C3']['heldout_acc']:.3f} | triage train {tri['train_acc']} held-out {tri['heldout_acc']} "
          f"({tri['heldout_coverage']:.2f} answered)", flush=True)
    return out


def summarise(cells):
    def ms(v):
        v = [x for x in v if x is not None]
        return dict(mean=float(np.mean(v)), ci95_corrected=[float(x) for x in corr_ci(v)])
    names = list(cells[0]["models"])
    S = {}
    for m in names:
        c = [x["models"][m] for x in cells]
        S[m] = dict(train_accuracy=ms([x["train_acc"] for x in c]), heldout_accuracy=ms([x["heldout_acc"] for x in c]),
                    heldout_recall_N_Mi_Mo_S=np.mean([x["heldout_recall"] for x in c], 0).round(3).tolist(),
                    heldout_confusion_mean_per_repeat=(np.sum([x["heldout_confusion"] for x in c], 0) / 3).round(1).tolist())
    T = [x["triage"] for x in cells]
    S["AGT-img triage"] = dict(train_accuracy_answered=ms([x["train_acc"] for x in T]), train_coverage=ms([x["train_coverage"] for x in T]),
                               grade_train_accuracy_all=ms([x["grade_train_acc_all"] for x in T]), ef_train_rmse=ms([x["ef_train_rmse"] for x in T]),
                               heldout_accuracy_answered=ms([x["heldout_acc"] for x in T]), heldout_coverage=ms([x["heldout_coverage"] for x in T]),
                               ef_heldout_rmse=ms([x["ef_heldout_rmse"] for x in T]),
                               heldout_confusion_mean_per_repeat=(np.sum([x["heldout_confusion"] for x in T], 0) / 3).round(1).tolist(),
                               heldout_referred_mean_per_repeat=(np.sum([x["heldout_referred"] for x in T], 0) / 3).round(1).tolist())
    ep = {m: sum([x["info"][m].get("epochs", []) if isinstance(x["info"][m].get("epochs"), list) else [] for x in cells], [])
          for m in ("ECGF-MLP", "FUSION-MLP")}
    S["_epochs_to_100"] = {m: dict(median=float(np.median(v)), max=int(max(v))) for m, v in ep.items() if v}
    S["_linear_steps"] = dict(median=float(np.median([x["info"]["ECGF-linear"]["steps"] for x in cells])))
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cells", default="all")
    a = ap.parse_args()
    C.verified(); B.table(); clin()
    tasks = [(r, k) for r in (0, 1, 2) for k in range(5)] if a.cells == "all" else [tuple(map(int, c.split(":"))) for c in a.cells.split(",")]
    with Pool(a.jobs) as pool:
        cells = list(pool.imap_unordered(cell, tasks))
    cells.sort(key=lambda c: (c["repeat"], c["fold"]))
    res = dict(amendment="AMENDMENT_TRAIN100.md", n_cells=len(cells), summary=summarise(cells), cells=cells)
    with open(f"{a.out}/train100_features.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
    for m, s in res["summary"].items():
        if m.startswith("_"):
            print(m, s); continue
        if "train_accuracy" in s:
            print(f"{m:16s} train {s['train_accuracy']['mean']:.3f} | held-out {s['heldout_accuracy']['mean']:.3f} {np.round(s['heldout_accuracy']['ci95_corrected'], 3).tolist()}")
        else:
            print(f"{m:16s} train {s['train_accuracy_answered']['mean']:.3f} ({s['train_coverage']['mean']:.2f}) | held-out "
                  f"{s['heldout_accuracy_answered']['mean']:.3f} {np.round(s['heldout_accuracy_answered']['ci95_corrected'], 3).tolist()} "
                  f"({s['heldout_coverage']['mean']:.2f} answered)")


if __name__ == "__main__":
    main()
