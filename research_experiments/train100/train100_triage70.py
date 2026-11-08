"""TRAIN100-70s (AMENDMENT_TRAIN100_ECHONEXT.md, Addendum A): the 70s model, i.e. the frozen header-free triage model
AGT-img (BEST_RESULTS/best_for_60s + best_for_70s), retrained under the TRAIN100 rules and evaluated exactly as its
cross-validation estimate (phase_7/e6_cascade.py, scored as phase_7/triage_mild_curve.py): Phase-2 patient-grouped
5-fold CV x 3 repeats on the 837 development ECGs. The locked test set is NOT loaded and NOT rescored. Hospital data
only (no EchoNext). Per-ECG arrays stay in the git-ignored phase_7/data/private/train100_triage70/; aggregates only.
From the repository root, PYTHONPATH=. :
    python research_experiments/train100/train100_triage70.py --jobs 8 --threads 1 [--cells 0:0]   (cached, resumable)
    python research_experiments/train100/train100_triage70.py --report
"""
import os
import sys


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


THREADS = int(_opt("--threads", "1"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ[_v] = str(THREADS)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # noqa: E402
from sklearn.linear_model import Ridge  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train100_echonext as T1  # noqa: E402  (its TRAIN100 head learners: head100 / mlp3)
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha  # noqa: E402
from research_experiments.phase_5.p5common import blockstd, corr_ci, decide, ecgf_f6, impute  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin  # noqa: E402
from research_experiments.phase_7 import p7common as C  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(THREADS)
B, Y, nD = C.B, C.Y, C.nD
PRIV = C.PRIV7 / "train100_triage70"
RES_JSON = HERE / "results" / "train100_triage70.json"
EF_ALPHA, EF_TREES = 1e-6, 4000
WEIGHTS = {"W1": 1.0, "W.3": 0.3, "W0": 0.0}
VARIANTS = [(m, w) for m in ("H1", "H2", "H3") for w in WEIGHTS]
GRID = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]


def guard(p):
    p = Path(p).resolve()
    if not (PRIV.resolve() in p.parents or p == RES_JSON.resolve()):
        raise PermissionError(f"refusing to write outside the TRAIN100-70s folders: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def ef_fitpred(model, wv, fit_rows, pred_sets, clean, ef, E, IMG):
    """e6_cascade.fitpred with the brake removed: H1 ridge alpha 1e-6 (was RidgeCV); H2 LightGBM 4,000 trees,
    min_child_samples 1, L2 0, no early stopping, all fit rows (was <= 600 trees with early stopping on a 20 % part)."""
    if wv == 0.0:
        fit_rows = fit_rows[clean[fit_rows]]; w = np.ones(len(fit_rows))
    else:
        w = np.where(clean[fit_rows], 1.0, wv)
    fm = np.zeros(nD, bool); fm[fit_rows] = True
    if model == "H1":
        X = blockstd([E, impute(IMG, fm)], fm)
        m = Ridge(alpha=EF_ALPHA).fit(X[fit_rows], ef[fit_rows], sample_weight=w)
    else:
        mu, sd = E[fm].mean(0), E[fm].std(0) + 1e-6
        X = np.column_stack([IMG, PCA(32, random_state=0).fit(((E - mu) / sd)[fm]).transform((E - mu) / sd)])
        m = lgb.LGBMRegressor(n_estimators=EF_TREES, learning_rate=0.03, num_leaves=7, min_child_samples=1, subsample=0.8,
                              subsample_freq=1, colsample_bytree=0.5, reg_lambda=0.0, random_state=0, n_jobs=THREADS, verbose=-1)
        m.fit(X[fit_rows], ef[fit_rows], sample_weight=w)
    return [m.predict(X[p]) for p in pred_sets]


def fold(rk):
    r, k = rk
    f, fj = PRIV / f"r{r}_k{k}.npz", PRIV / f"r{r}_k{k}.json"
    if f.exists() and fj.exists():
        return r, k
    warnings.filterwarnings("ignore"); torch.set_num_threads(THREADS); t0 = time.time()
    ef = B.table().ef.values.astype(float); bnd = B.table().sex_boundary.values; clean = C.verified()[0]
    E = ecgf_f6().astype(float); IMG = clin()[0]
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k); yTR = Y[TR]
    split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, C.PAT[TR]))
    # ---- IMG-vote members (ECGF-MLP 3 seeds + ECGF-lin), TRAIN100 heads; inner OOF + outer fit on all training rows
    H, info = {}, {}
    for name, fit in (("ECGF-MLP", lambda rows: T1.mlp3(E, rows)), ("ECGF-lin", lambda rows: T1.head100(E, rows, Y, "lin", 42))):
        inn, infs = np.zeros((len(TR), 4)), []
        for ia, ib in split:
            fn, inf = fit(TR[ia]); inn[ib] = fn(TR[ib]); infs.append(inf)
        fn, inf = fit(TR)
        H[name] = dict(inner=inn, te=fn(te), tr=fn(TR)); info[name] = dict(outer=inf, inner=infs)
    Pin = (H["ECGF-MLP"]["inner"] + H["ECGF-lin"]["inner"]) / 2
    Pte = (H["ECGF-MLP"]["te"] + H["ECGF-lin"]["te"]) / 2; Ptr = (H["ECGF-MLP"]["tr"] + H["ECGF-lin"]["tr"]) / 2
    prior = np.bincount(yTR, minlength=4) / len(TR)
    a = choose_alpha(Pin, yTR, prior, "protected")                                  # IMG-C3 rule, nested (unchanged)
    c3_in, c3_te, c3_tr = (decide(P, "prior", a, prior) for P in (Pin, Pte, Ptr))
    log(f"r{r} k{k}: heads done {(time.time() - t0) / 60:.1f} min (MLP epochs {info['ECGF-MLP']['outer']['epochs']})")
    # ---- EF regressors: 9 variants, one chosen by inner-OOF RMSE on verified ECGs (unchanged rule)
    efo, efi = {}, {}
    for m in ("H1", "H2"):
        for w, wv in WEIGHTS.items():
            efo[(m, w)] = ef_fitpred(m, wv, TR, [te, TR], clean, ef, E, IMG)
            e = np.zeros(len(TR))
            for ia, ib in split:
                e[ib] = ef_fitpred(m, wv, TR[ia], [TR[ib]], clean, ef, E, IMG)[0]
            efi[(m, w)] = e
    for w in WEIGHTS:
        efo[("H3", w)] = [(x + y) / 2 for x, y in zip(efo[("H1", w)], efo[("H2", w)])]
        efi[("H3", w)] = (efi[("H1", w)] + efi[("H2", w)]) / 2
    cl = clean[TR]
    rmse = {v: float(np.sqrt(np.mean((efi[v][cl] - ef[TR][cl]) ** 2))) for v in VARIANTS}
    vs = min(rmse, key=rmse.get); (e_te, e_tr), e_in = efo[vs], efi[vs]
    # ---- grading cuts and answer threshold (unchanged rules, on the inner OOF)
    p = F1.floor_best(((q, F1.ef_grade(e_in, bnd[TR], *q)) for q in GRID), yTR) or (0.0, 41.0, 31.0)
    dist = lambda e, b: np.abs(e[:, None] - np.column_stack([b + p[0], np.full(len(e), p[1]), np.full(len(e), p[2])])).min(1)  # noqa: E731
    g_in, g_te, g_tr = F1.ef_grade(e_in, bnd[TR], *p), F1.ef_grade(e_te, bnd[te], *p), F1.ef_grade(e_tr, bnd[TR], *p)
    d_in, d_te, d_tr = dist(e_in, bnd[TR]), dist(e_te, bnd[te]), dist(e_tr, bnd[TR])
    ag_in = c3_in == g_in
    t = F1.threshold(d_in[ag_in], (g_in == yTR)[ag_in])
    k_te, k_tr = (c3_te == g_te) & (d_te >= t), (c3_tr == g_tr) & (d_tr >= t)
    np.savez(guard(f), te=te, g_te=g_te, k_te=k_te, c3_te=c3_te, g_tr=g_tr, k_tr=k_tr, c3_tr=c3_tr, e_te=e_te, e_tr=e_tr,
             mlp_te=H["ECGF-MLP"]["te"], mlp_tr=H["ECGF-MLP"]["tr"], lin_te=H["ECGF-lin"]["te"], lin_tr=H["ECGF-lin"]["tr"])
    acc = lambda p_, y_: float(np.mean(p_ == y_)) if len(y_) else None  # noqa: E731
    agg = dict(repeat=r, fold=k, alpha=float(a), ef_variant=f"{vs[0]}_{vs[1]}",
               ef_inner_rmse_verified={f"{m}_{w}": v for (m, w), v in rmse.items()}, cuts=[float(x) for x in p],
               threshold=float(t),
               train=dict(answered_share=float(k_tr.mean()), accuracy_answered=acc(g_tr[k_tr], yTR[k_tr]), grade_accuracy_all=acc(g_tr, yTR),
                          imgc3_accuracy=acc(c3_tr, yTR), imgvote_accuracy=acc(Ptr.argmax(1), yTR),
                          ecgf_mlp_accuracy=acc(H["ECGF-MLP"]["tr"].argmax(1), yTR), ecgf_lin_accuracy=acc(H["ECGF-lin"]["tr"].argmax(1), yTR),
                          ef_rmse=float(np.sqrt(np.mean((e_tr - ef[TR]) ** 2)))),
               heldout=dict(answered_share=float(k_te.mean()), accuracy_answered=acc(g_te[k_te], Y[te][k_te]), grade_accuracy_all=acc(g_te, Y[te]),
                            imgc3_accuracy=acc(c3_te, Y[te]), imgvote_accuracy=acc(Pte.argmax(1), Y[te]),
                            ecgf_mlp_accuracy=acc(H["ECGF-MLP"]["te"].argmax(1), Y[te]), ecgf_lin_accuracy=acc(H["ECGF-lin"]["te"].argmax(1), Y[te]),
                            ef_rmse=float(np.sqrt(np.mean((e_te - ef[te]) ** 2)))),
               heads=info, minutes=round((time.time() - t0) / 60, 2))
    guard(fj).write_text(json.dumps(agg, default=float))
    tr_, ho = agg["train"], agg["heldout"]
    log(f"r{r} k{k}: {agg['minutes']} min | train answered {tr_['answered_share']:.2f} acc {tr_['accuracy_answered']} | "
        f"held-out answered {ho['answered_share']:.2f} acc {ho['accuracy_answered']} | {agg['ef_variant']}")
    return r, k


def report():
    cells_rk = [(r, k) for r in (0, 1, 2) for k in range(5)]
    F = {rk: json.loads((PRIV / f"r{rk[0]}_k{rk[1]}.json").read_text()) for rk in cells_rk}
    Z = {rk: np.load(PRIV / f"r{rk[0]}_k{rk[1]}.npz") for rk in cells_rk}
    scores, cl, cm, c3m, c3c, counts_true, counts_pred, share = [], [], [], [], [], [], [], []
    for r in (0, 1, 2):
        g, keep, c3 = np.full(nD, -1), np.zeros(nD, bool), np.full(nD, -1)
        for k in range(5):
            z = Z[(r, k)]; g[z["te"]] = z["g_te"]; keep[z["te"]] = z["k_te"]; c3[z["te"]] = z["c3_te"]
        P = np.eye(4)[g]
        scores.append(B.metrics(Y[keep], P[keep], g[keep])); share.append(float(keep.mean()))
        cl += [c["accuracy"] for c in C.cells(P, g, r, mask=keep)]
        cm.append([[int(np.sum(keep & (Y == i) & (g == j))) for j in range(4)] + [int(np.sum(~keep & (Y == i)))] for i in range(4)])
        c3m.append(B.metrics(Y, np.eye(4)[c3], c3)); c3c += [c["accuracy"] for c in C.cells(np.eye(4)[c3], c3, r)]
        counts_true.append([int(np.sum(keep & (Y == i))) for i in range(4)]); counts_pred.append([int(np.sum(keep & (g == i))) for i in range(4)])
    keys = ["accuracy", "balanced_accuracy", "macro_f1"] + [f"{a}_{c}" for c in C.CLASSES for a in ("precision", "recall")]
    S = {k_: float(np.mean([m[k_] for m in scores])) for k_ in keys}
    orig = json.loads((B.ROOT / "research_experiments" / "phase_7" / "results" / "triage_scores.json").read_text())
    # original held-out numbers of the parts (read-only aggregates of the committed / cached original outputs)
    from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import ecgf_lin, ecgf_mlp
    c3_orig = [float(np.mean(B.load_run(f"P6B-imgc3-IMGF-v1__s0_r{r}")[1] == Y)) for r in (0, 1, 2)]
    vote_orig = [float(np.mean(B.load_run(f"P6B-imgvote-IMGF-v1__s0_r{r}")[1] == Y)) for r in (0, 1, 2)]
    mlp_orig = [float(np.mean(ecgf_mlp(r).argmax(1) == Y)) for r in (0, 1, 2)]
    lin_orig = [float(np.mean(ecgf_lin(r).argmax(1) == Y)) for r in (0, 1, 2)]
    efv = B.table().ef.values.astype(float)
    ef_orig = [float(np.sqrt(np.mean((np.load(C.PRIV7 / "e6" / f"r{r}.npz")["SEL"] - efv) ** 2))) for r in (0, 1, 2)]
    mean = lambda part, key: float(np.mean([F[rk][part][key] for rk in cells_rk if F[rk][part][key] is not None]))  # noqa: E731
    mn = lambda part, key: float(np.min([F[rk][part][key] for rk in cells_rk if F[rk][part][key] is not None]))  # noqa: E731
    ep = [e for rk in cells_rk for x in [F[rk]["heads"]["ECGF-MLP"]["outer"]] + F[rk]["heads"]["ECGF-MLP"]["inner"] for e in x["epochs"]]
    it = [x["iterations"] for rk in cells_rk for x in [F[rk]["heads"]["ECGF-lin"]["outer"]] + F[rk]["heads"]["ECGF-lin"]["inner"]]
    out = dict(
        amendment="research_experiments/train100/AMENDMENT_TRAIN100_ECHONEXT.md (Addendum A)",
        model="70s model = frozen header-free triage AGT-img (BEST_RESULTS/best_for_60s, best_for_70s), TRAIN100 retrain",
        evaluation="Phase-2 patient-grouped 5-fold CV x 3 repeats, 837 development ECGs; locked test NOT loaded or rescored",
        triage=dict(
            train=dict(accuracy_answered=mean("train", "accuracy_answered"), accuracy_answered_min_fold=mn("train", "accuracy_answered"),
                       answered_share=mean("train", "answered_share"), grade_accuracy_all=mean("train", "grade_accuracy_all")),
            heldout=dict(S, accuracy_ci95_corrected=corr_ci(cl), accuracy_per_repeat=[round(m["accuracy"], 4) for m in scores],
                         answered_share=float(np.mean(share)), answered_share_per_repeat=[round(s, 4) for s in share],
                         answered_true_class_counts_mean_per_repeat=dict(zip(C.CLASSES, np.mean(counts_true, 0).round(1).tolist())),
                         answered_predicted_counts_mean_per_repeat=dict(zip(C.CLASSES, np.mean(counts_pred, 0).round(1).tolist())),
                         confusion_mean_per_repeat_with_referred_column=np.mean(cm, 0).round(1).tolist(),
                         grade_accuracy_all=mean("heldout", "grade_accuracy_all")),
            original_cv=dict(accuracy_answered=orig["accuracy"], ci95_corrected=orig["accuracy_ci95_corrected"], answered_share=orig["answered_share"],
                             recall=[orig[f"recall_{c}"] for c in C.CLASSES]),
            original_locked_test_not_rescored=dict(accuracy_answered=0.747, answered_share=0.497, note="BEST_RESULTS/best_for_70s; context only"),
            change_points_vs_original_cv=round(100 * (S["accuracy"] - orig["accuracy"]), 2)),
        parts=dict(
            ECGF_MLP=dict(train_accuracy=mean("train", "ecgf_mlp_accuracy"), heldout_accuracy=mean("heldout", "ecgf_mlp_accuracy"),
                          original_heldout_accuracy=float(np.mean(mlp_orig)), epochs=dict(median=float(np.median(ep)), max=int(max(ep)), fits_at_cap=int(np.sum(np.array(ep) >= T1.MLP_CAP)), n_fits=len(ep))),
            ECGF_lin=dict(train_accuracy=mean("train", "ecgf_lin_accuracy"), heldout_accuracy=mean("heldout", "ecgf_lin_accuracy"),
                          original_heldout_accuracy=float(np.mean(lin_orig)), lbfgs_iterations=dict(median=float(np.median(it)), max=int(max(it)), fits_at_cap=int(np.sum(np.array(it) >= T1.LIN_CAP)))),
            IMG_vote=dict(train_accuracy=mean("train", "imgvote_accuracy"), heldout_accuracy=mean("heldout", "imgvote_accuracy"),
                          original_heldout_accuracy=float(np.mean(vote_orig))),
            IMG_C3=dict(train_accuracy=mean("train", "imgc3_accuracy"), heldout_accuracy=float(np.mean([m["accuracy"] for m in c3m])),
                        heldout_ci95_corrected=corr_ci(c3c), original_heldout_accuracy=float(np.mean(c3_orig))),
            EF_regressor=dict(train_rmse=mean("train", "ef_rmse"), heldout_rmse=mean("heldout", "ef_rmse"), original_heldout_rmse=float(np.mean(ef_orig)),
                              variant_per_fold=[F[rk]["ef_variant"] for rk in cells_rk])),
        decision_rules_per_fold=[dict(repeat=rk[0], fold=rk[1], alpha=F[rk]["alpha"], cuts=F[rk]["cuts"], threshold=F[rk]["threshold"])
                                 for rk in cells_rk],
        per_fold=[dict(repeat=rk[0], fold=rk[1], train=F[rk]["train"], heldout=F[rk]["heldout"]) for rk in cells_rk],
        minutes_per_fold=float(np.mean([F[rk]["minutes"] for rk in cells_rk])))
    guard(RES_JSON).write_text(json.dumps(out, indent=1, default=float))
    t, h, o = out["triage"]["train"], out["triage"]["heldout"], out["triage"]["original_cv"]
    print(json.dumps(dict(train=t, heldout={k_: h[k_] for k_ in ("accuracy", "accuracy_ci95_corrected", "answered_share", "macro_f1")},
                          original=o, change=out["triage"]["change_points_vs_original_cv"], parts=out["parts"]), indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cells", default="all"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    C.verified(); B.table(); clin()
    cells = [(r, k) for r in (0, 1, 2) for k in range(5)] if a.cells == "all" else [tuple(map(int, c.split(":"))) for c in a.cells.split(",")]
    log(f"TRAIN100-70s: {len(cells)} folds, jobs {a.jobs}, threads {THREADS}")
    if a.jobs <= 1:
        for c in cells:
            fold(c)
    else:
        with Pool(a.jobs) as p:
            for r, k in p.imap_unordered(fold, cells):
                log(f"done r{r} k{k}")


if __name__ == "__main__":
    main()
