"""TRAIN100-EchoNext (AMENDMENT_TRAIN100_ECHONEXT.md): the top 3 all-patient ensembles of report 45 (public_ef/en_super.py:
SUPER-TOP10-ORD on the base pool, STAGED-4 on pool X, SUPER with greedy selection) with every member and meta-learner
retrained under the TRAIN100 rules (no early stopping, weight decay / L2 0, C = 10,000, 2,000 trees, alpha 1e-6, k 15,
networks until >= 99.5 % training accuracy). Same inputs, labels, folds, inner splits, seeds and decision rules.

Owner's laptop only (EchoNext under the PhysioNet data-use agreement + hospital data). Per-ECG arrays, per-fold
predictions and the retrained CNN weights stay in the git-ignored public_ef/data/private/train100_echonext/; the results
JSON holds aggregates only. Nothing per record is printed. The locked hospital test set is never loaded, and the
EchoNext locked-test rows (en_split.locked) are never used or scored.

From the repository root, PYTHONPATH=. :
    python research_experiments/train100/train100_echonext.py --stage en  [--threads 8]   EchoNext tabular members (CPU)
    python research_experiments/train100/train100_echonext.py --stage img [--workers 2]   PIX-ZS retrain (GPU)
    python research_experiments/train100/train100_echonext.py --stage folds --jobs 6 --threads 2 [--cells 0:0,0:1]
        hospital members per outer fold, then the three meta-learners once the EchoNext members exist (cached, resumable)
    python research_experiments/train100/train100_echonext.py --stage report               aggregates -> results JSON + tables
Add --smoke for the tiny-budget smoke test (separate private folder, fold r0 k0 only).
"""
import os
import sys


def _opt(flag, default):
    return sys.argv[sys.argv.index(flag) + 1] if flag in sys.argv else default


THREADS = int(_opt("--threads", "2"))
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):     # before numpy / sklearn / torch load
    os.environ[_v] = str(THREADS)

import argparse  # noqa: E402
import inspect  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from multiprocessing import Pool  # noqa: E402
from pathlib import Path  # noqa: E402

import lightgbm as lgb  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from scipy.stats import pearsonr  # noqa: E402
from sklearn.ensemble import HistGradientBoostingClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression, Ridge  # noqa: E402
from sklearn.model_selection import GroupShuffleSplit  # noqa: E402
from sklearn.neighbors import KNeighborsClassifier  # noqa: E402

from research_experiments.phase_2.p2heads import Head, class_w  # noqa: E402
from research_experiments.phase_5.p5common import corr_ci, ecgf_f6  # noqa: E402
from research_experiments.phase_6a_normal_mild_lstm.data_check import rule  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import ult as U  # noqa: E402
from research_experiments.public_ef import en_featx as FXM  # noqa: E402
from research_experiments.public_ef import en_image as EI  # noqa: E402
from research_experiments.public_ef import en_prep as EP  # noqa: E402
from research_experiments.public_ef import en_super as S  # noqa: E402
from research_experiments.public_ef import en_train as T  # noqa: E402
from research_experiments.public_ef.en_split import locked  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(THREADS)

# ------------------------------------------------------------------------------------------------ settings (declared)
SMOKE = "--smoke" in sys.argv
BIG_C, ALPHA, TREES, KNN_K, STOP = 1e4, 1e-6, 2000, 15, 0.995
MLP_CAP, LIN_CAP, IMG_CAP = 3000, 5000, 100          # epochs (MLP heads), L-BFGS iterations (linear head), epochs (CNN)
LR_ITER, META_ITER = 3000, 5000                       # original solver limits (members 3,000; meta-learners 5,000)
IMG_LIMIT = None
if SMOKE:                                             # tiny budgets: checks the plumbing only, numbers are meaningless
    MLP_CAP, LIN_CAP, IMG_CAP, TREES, LR_ITER, META_ITER, IMG_LIMIT = 3, 100, 1, 10, 50, 100, 256

PRIV0 = EP.PRIV                                       # original public-EF / EchoNext caches: READ ONLY
PRIV = PRIV0 / ("train100_echonext_smoke" if SMOKE else "train100_echonext")
HERE = Path(__file__).resolve().parent
RES_JSON = HERE / "results" / "train100_echonext.json"
REPORT_MD = HERE / "TRAIN100_ECHONEXT_REPORT.md"
Y, EF, nD = B.Y, E.EF.astype(float), B.nD
MM = np.isin(Y, (1, 2))                               # Mild / Moderate rows (MiMo members)
CLS = {"Normal": 0, "Mild": 1, "Moderate": 2, "Severe": 3}

HOSP = ("ECGF-MLP", "ECGF-lin", "M5", "M6", "M8", "M9", "M10", "M12", "H-TAB-LR", "H-TAB-GBM", "H-X-GBM", "H-MiMo-GBM")
CONFIGS = {"base_top10": ("base", "top10"), "X_top10": ("X", "top10"), "base_greedy": ("base", "greedy")}
HEADLINE = {  # report name: (config, output of super_fold, original results file, original model key)
    "SUPER-TOP10-ORD": ("base_top10", "ord4", "en_super_original_top10.json", "SUPER-ORD"),
    "STAGED-4 (pool X)": ("X_top10", "staged4", "en_super_original_X_top10.json", "STAGED-4"),
    "SUPER (greedy)": ("base_greedy", "sup", "en_super_original.json", "SUPER"),
}


def guard(p):
    """Writes go only to the new private folder, the aggregate results JSON and the report."""
    p = Path(p).resolve()
    if not (PRIV.resolve() in p.parents or p in (RES_JSON.resolve(), REPORT_MD.resolve())):
        raise PermissionError(f"refusing to write outside the TRAIN100-EchoNext folders: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def need(p):
    """Original caches that are read; the original functions would rebuild (and write) them if they were missing."""
    if not Path(p).exists():
        raise FileNotFoundError(f"missing original cache (not rebuilt here): {Path(p).name}")
    return p


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


# ---------------------------------------------------------------------------------------------- TRAIN100 learners
def lr100(balanced=True):
    return LogisticRegression(C=BIG_C, class_weight="balanced" if balanced else None, max_iter=LR_ITER)


def gbm100():
    """LightGBM with the original EI.GBM tree settings; 2,000 trees, min_child_samples 1, L2 0, no early stopping."""
    return lgb.LGBMClassifier(n_estimators=TREES, **dict(EI.GBM, min_child_samples=1, reg_lambda=0.0, n_jobs=THREADS))


def hgb100():
    """ULT M10 HistGB (depth 3, lr 0.05, balanced) with 2,000 iterations, L2 0, min_samples_leaf 1, no early stopping."""
    return HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=TREES, l2_regularization=0.0,
                                          min_samples_leaf=1, early_stopping=False, class_weight="balanced", random_state=0)


def pad4(classes, P):
    """As ult.fit_pred: absent classes get 1e-3, then renormalise."""
    out = np.full((len(P), 4), 1e-3); out[:, classes] = P
    return out / out.sum(1, keepdims=True)


def skfit(mdl, X, rows, y, pad=False):
    mdl.fit(X[rows], y[rows])
    return (lambda idx: pad4(mdl.classes_, mdl.predict_proba(X[idx])) if pad else mdl.predict_proba(X[idx])), {}


def head100(X, rows, y, kind, seed, no_std=False):
    """Phase-2 head (p2heads.Head) with the brake removed, trained on all `rows`.
    'mlp': MLPF (64 GELU, dropout 0.5, balanced CE), AdamW lr 1e-3 constant, weight decay 0, batch 64, no early stopping.
    'lin': linear head, CE + distance penalty (lambda 1), L-BFGS (lr 0.5, history 20, strong Wolfe), L2 0.
    Stops when the eval-mode training accuracy on all rows is >= 99.5 % or at the cap.
    no_std: inputs already block-standardised by make_X (p2heads cfg no_std), as for multi-block inputs."""
    mu, sd = (np.zeros(X.shape[1]), np.ones(X.shape[1])) if no_std else (X[rows].mean(0), X[rows].std(0) + 1e-6)
    tf = lambda idx: torch.tensor((X[idx] - mu) / sd, dtype=torch.float32)  # noqa: E731
    xtr, yt = tf(rows), torch.tensor(y[rows]); cw = class_w(y[rows], "balanced")
    torch.manual_seed(seed)

    def acc():
        m.eval()
        with torch.no_grad():
            return float((m.probs(xtr).argmax(1).numpy() == y[rows]).mean())
    if kind == "mlp":
        m = Head(X.shape[1], "ce", "mlp", 64, 0.5)
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); g = torch.Generator().manual_seed(seed)
        ep, a = 0, 0.0
        while ep < MLP_CAP and a < STOP:
            m.train(); idx = torch.randperm(len(rows), generator=g)
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]; lo = m.loss(xtr[b], yt[b], cw, {}); opt.zero_grad(); lo.backward(); opt.step()
            ep += 1; a = acc()
        info = dict(epochs=ep, train_acc=a)
    else:
        m = Head(X.shape[1], "ce_dist", "linear")
        opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=100, history_size=20, line_search_fn="strong_wolfe")

        def closure():
            opt.zero_grad(); lo = m.loss(xtr, yt, cw, {"lam": 1.0}); lo.backward(); return lo
        it, a = 0, 0.0
        while it < LIN_CAP and a < STOP:
            opt.step(closure); it += 100; a = acc()
        info = dict(iterations=it, train_acc=a)
    m.eval()

    def f(idx):
        with torch.no_grad():
            return m.probs(tf(idx)).numpy()
    return f, info


def mlp3(X, rows):
    fs = [head100(X, rows, Y, "mlp", s) for s in B.SEEDS]
    return (lambda idx: np.mean([f(idx) for f, _ in fs], 0)), dict(epochs=[i["epochs"] for _, i in fs],
                                                                   train_acc=[i["train_acc"] for _, i in fs])


def ult100(m, X, rows):
    kind = U.SPEC[m][0]
    mdl = lr100() if kind == "lr" else KNeighborsClassifier(n_neighbors=KNN_K, weights="distance") if kind == "knn" else hgb100()
    return skfit(mdl, X, rows, Y, pad=True)


# ------------------------------------------------------------------------------------- hospital members, one fold
def hosp_fold(r, k):
    """Inner OOF (4 inner folds) + outer fit on all training rows (predictions for the held-out fold and in-sample for
    the training rows) of every hospital member. Cached per fold."""
    f = PRIV / "folds" / f"hosp_r{r}_k{k}.npz"; fj = f.with_suffix(".json")
    if f.exists() and fj.exists():
        z = np.load(f); return {x: z[x] for x in z.files}, json.loads(fj.read_text())
    t0 = time.time()
    need(EI.HOSP / "plain.npz"); need(FXM.FX / "hosp_plain.npz"); need(E.D / "build.npz")
    TR, TE, inner = Bd.splits(r, k)
    fm = np.zeros(nD, bool); fm[TR] = True
    out, info = {}, {}

    def run(name, fit, ok=None):
        ts = time.time(); sel = (lambda a: a) if ok is None else (lambda a: a[ok[a]])
        P, inf_in = None, []
        for ia, ib in inner:
            fn, inf = fit(sel(TR[ia])); p = fn(TR[ib])
            P = np.zeros((len(TR), p.shape[1])) if P is None else P
            P[ib] = p; inf_in.append(inf)
        fn, inf = fit(sel(TR))
        out[f"{name}__inner"], out[f"{name}__te"], out[f"{name}__tr"] = P, fn(TE), fn(TR)
        info[name] = dict(outer=inf, inner=inf_in, seconds=round(time.time() - ts, 1))

    E6 = ecgf_f6().astype(float)
    run("ECGF-MLP", lambda rows: mlp3(E6, rows))
    run("ECGF-lin", lambda rows: head100(E6, rows, Y, "lin", 42))
    X5 = E.features("M5", r, k, fm).astype(np.float64)
    run("M5", lambda rows: skfit(lr100(), X5, rows, Y))
    # M6 and EF-band: one ridge on the M6 inputs (identical in the original: same inputs, inner splits and criterion)
    ts = time.time(); X6 = E.features("M6", r, k, fm).astype(np.float64); mu_in = np.zeros(len(TR))
    for ia, ib in inner:
        mu_in[ib] = Ridge(alpha=ALPHA).fit(X6[TR[ia]], EF[TR[ia]]).predict(X6[TR[ib]])
    rg = Ridge(alpha=ALPHA).fit(X6[TR], EF[TR]); mu_te, mu_tr = rg.predict(X6[TE]), rg.predict(X6[TR])
    rmse = float(np.sqrt(np.mean((mu_in - EF[TR]) ** 2)))
    out.update({"M6__inner": E.ef_probs(mu_in, rmse), "M6__te": E.ef_probs(mu_te, rmse), "M6__tr": E.ef_probs(mu_tr, rmse),
                "EF-band__mu_inner": mu_in, "EF-band__mu_te": mu_te, "EF-band__mu_tr": mu_tr})
    info["M6"] = info["EF-band"] = dict(inner_rmse=rmse, train_rmse=float(np.sqrt(np.mean((mu_tr - EF[TR]) ** 2))),
                                        seconds=round(time.time() - ts, 1))
    for m in U.NEW:                                                        # M8, M9, M10, M12
        Xm = U.feats(m, fm)
        run(m, lambda rows, m=m, Xm=Xm: ult100(m, Xm, rows), ok=U.ZONE if m == "M12" else None)
    h = EI.hospital_inputs("plain")
    raw = [np.asarray(h[x], float) for x in ("la", "ls", "tab")]

    def tab_lr(rows):
        bs = EI.BlockStd().fit([b[rows] for b in raw])
        mdl = lr100().fit(bs.transform([b[rows] for b in raw]), Y[rows])
        return (lambda idx: mdl.predict_proba(bs.transform([b[idx] for b in raw]))), {}
    run("H-TAB-LR", tab_lr)
    G = np.c_[raw[0], raw[1], raw[2]]
    run("H-TAB-GBM", lambda rows: skfit(gbm100(), G, rows, Y))
    GX = np.c_[h["la"], h["ls"], h["tab"], FXM.hospital_featx("plain")[0]].astype(float)
    run("H-X-GBM", lambda rows: skfit(gbm100(), GX, rows, Y))
    yb = (Y == 2).astype(int)
    run("H-MiMo-GBM", lambda rows: skfit(gbm100(), GX, rows, yb), ok=MM)
    info["_minutes"] = round((time.time() - t0) / 60, 2)
    np.savez(guard(f), **out); guard(fj).write_text(json.dumps(info, default=float))
    return out, info


# ----------------------------------------------------------------------------------- EchoNext members (one time)
def cls_m(y, p):
    return dict(n=int(len(y)), accuracy=float(np.mean(p == y)), **{f"recall_{c}": float(np.mean(p[y == i] == i)) for c, i in CLS.items()})


def bin_m(y, p):
    return dict(n=int(len(y)), accuracy=float(np.mean(p == y)))


def grade(mu, male):
    return np.array([CLS[rule(e, "M" if m else "F")] for e, m in zip(mu, male)])


def reg_m(ef, mu, y, male):
    return dict(n=int(len(ef)), rmse=float(np.sqrt(np.mean((mu - ef) ** 2))), r=float(pearsonr(mu, ef)[0]),
                graded_accuracy=float(np.mean(grade(mu, male) == y)))


def en_stage():
    f, fj = PRIV / "en_members.npz", PRIV / "en_members.json"
    if f.exists() and fj.exists():
        log("EN members cached"); return
    t0 = time.time()
    z = np.load(need(PRIV0 / "en_cache.npz"), allow_pickle=False)
    en = dict(f6a=z["f6_aligned"], f6s=z["f6_synth"], la=z["logits_aligned"], ls=z["logits_synth"], tab=z["tab"], y=z["y"],
              ef=z["ef"], g=z["patient_key"])
    male = z["male"]
    for v in ("plain", "H"):
        need(EI.HOSP / f"{v}.npz")
    need(FXM.FX / "hosp_H.npz"); need(FXM.FX / "hosp_plain.npz")
    hosp = {v: EI.hospital_inputs(v) for v in ("H", "plain")}
    y, ef, g = en["y"], en["ef"], en["g"]
    L = locked(g); D = np.flatnonzero(~L)
    outs, agg = {}, dict(n_rows=int(len(y)), n_locked_not_used=int(L.sum()), n_training_pool=int(len(D)))

    # (1) en_train.train_en recipe: EN-cls (LR) and EN-band (ridge) on standardised aligned + synthetic f6
    X = np.c_[en["f6a"], en["f6s"]].astype(float)
    mu_, sd_ = X[D].mean(0), X[D].std(0) + 1e-6; Xs = (X - mu_) / sd_
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(Xs[D], y[D], g[D])); tr, va = D[tr], D[va]
    agg.update(n_fit_part=int(len(tr)), n_validation=int(len(va)))
    lr_t = lr100().fit(Xs[tr], y[tr]); rg_t = Ridge(alpha=ALPHA).fit(Xs[tr], ef[tr])
    lr = lr100().fit(Xs[D], y[D]); rg = Ridge(alpha=ALPHA).fit(Xs[D], ef[D])
    agg["EN-cls"] = dict(training_rows=cls_m(y[D], lr.predict(Xs[D])), fit_part=cls_m(y[tr], lr_t.predict(Xs[tr])),
                         validation=cls_m(y[va], lr_t.predict(Xs[va])), solver_iterations=int(np.max(lr.n_iter_)))
    agg["EN-band"] = dict(training_rows=reg_m(ef[D], rg.predict(Xs[D]), y[D], male[D]),
                          fit_part=reg_m(ef[tr], rg_t.predict(Xs[tr]), y[tr], male[tr]),
                          validation=reg_m(ef[va], rg_t.predict(Xs[va]), y[va], male[va]))
    Xh = (np.c_[ecgf_f6(), E.build()["f6"]].astype(float) - mu_) / sd_
    outs["EN-cls"], outs["EN-band"] = lr.predict_proba(Xh), rg.predict(Xh)
    log(f"EN-cls / EN-band done {time.time() - t0:.0f}s")

    # (2) en_image.fit_en_all recipe: EN-cls-all (LR), EN-gbm (LightGBM), EN-band-all (ridge)
    bl_all = EI._blocks(en); G_all = np.c_[bl_all[2], bl_all[3], bl_all[4]]
    bl, G, yD, efD, gD, mD = [b[D] for b in bl_all], G_all[D], y[D], ef[D], g[D], male[D]
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(G, yD, gD))
    bs = EI.BlockStd().fit([b[tr] for b in bl]); Xtr, Xva = bs.transform([b[tr] for b in bl]), bs.transform([b[va] for b in bl])
    lr_t, rg_t, gb_t = lr100().fit(Xtr, yD[tr]), Ridge(alpha=ALPHA).fit(Xtr, efD[tr]), gbm100().fit(G[tr], yD[tr])
    bs = EI.BlockStd().fit(bl); Xa = bs.transform(bl)
    lr, rg, gbf = lr100().fit(Xa, yD), Ridge(alpha=ALPHA).fit(Xa, efD), gbm100().fit(G, yD)
    agg["EN-cls-all"] = dict(training_rows=cls_m(yD, lr.predict(Xa)), fit_part=cls_m(yD[tr], lr_t.predict(Xtr)),
                             validation=cls_m(yD[va], lr_t.predict(Xva)), solver_iterations=int(np.max(lr.n_iter_)))
    agg["EN-gbm"] = dict(training_rows=cls_m(yD, gbf.predict(G)), fit_part=cls_m(yD[tr], gb_t.predict(G[tr])),
                         validation=cls_m(yD[va], gb_t.predict(G[va])), trees=TREES)
    agg["EN-band-all"] = dict(training_rows=reg_m(efD, rg.predict(Xa), yD, mD), fit_part=reg_m(efD[tr], rg_t.predict(Xtr), yD[tr], mD[tr]),
                              validation=reg_m(efD[va], rg_t.predict(Xva), yD[va], mD[va]))
    for v, d in hosp.items():
        hb = EI._blocks(d); Xh = bs.transform(hb)
        outs[f"EN-cls-all-{v}"], outs[f"EN-band-all-{v}"] = lr.predict_proba(Xh), rg.predict(Xh)
        outs[f"EN-gbm-{v}"] = gbf.predict_proba(np.c_[hb[2], hb[3], hb[4]])
    log(f"EN-cls-all / EN-gbm / EN-band-all done {time.time() - t0:.0f}s")

    # (3) en_featx.en_members recipe (pool X): EN-cls-all-X, EN-gbm-X, EN-MiMo-LR, EN-MiMo-GBM, EN-mu-mid
    c = np.load(need(FXM.FX / "echonext.npz"))
    assert np.array_equal(c["row"], z["row"]), "FEATX EchoNext cache does not match en_cache rows"
    Xe = c["X"]; hH, xH = hosp["H"], FXM.hospital_featx("H")[0]
    eb = [np.asarray(a, float) for a in (z["f6_aligned"], z["f6_synth"], z["logits_aligned"], z["logits_synth"], z["tab"])]
    eb = [b[D] for b in eb[:4]] + [np.c_[eb[4], np.asarray(Xe, float)][D]]
    hb = [np.asarray(hH[k_], float) for k_ in ("f6a", "f6s", "la", "ls")] + [np.c_[np.asarray(hH["tab"], float), np.asarray(xH, float)]]
    tabe, tabh = np.c_[eb[2], eb[3], eb[4]], np.c_[hb[2], hb[3], hb[4]]
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tabe, yD, gD))
    blockfit = lambda rows: EI.BlockStd().fit([b[rows] for b in eb])  # noqa: E731
    T_ = lambda bs_, rows: bs_.transform([b[rows] for b in eb])  # noqa: E731
    bs = blockfit(tr); m_t = lr100().fit(T_(bs, tr), yD[tr]); pv = m_t.predict(T_(bs, va)); ptr = m_t.predict(T_(bs, tr))
    bsA = blockfit(np.arange(len(yD))); XA = bsA.transform(eb); mx = lr100().fit(XA, yD)
    outs["EN-cls-all-X"] = mx.predict_proba(bsA.transform(hb))
    agg["EN-cls-all-X"] = dict(training_rows=cls_m(yD, mx.predict(XA)), fit_part=cls_m(yD[tr], ptr), validation=cls_m(yD[va], pv),
                               solver_iterations=int(np.max(mx.n_iter_)))
    g_t = gbm100().fit(tabe[tr], yD[tr]); gx = gbm100().fit(tabe, yD)
    outs["EN-gbm-X"] = gx.predict_proba(tabh)
    agg["EN-gbm-X"] = dict(training_rows=cls_m(yD, gx.predict(tabe)), fit_part=cls_m(yD[tr], g_t.predict(tabe[tr])),
                           validation=cls_m(yD[va], g_t.predict(tabe[va])), trees=TREES)
    mm = np.flatnonzero(np.isin(yD, (1, 2))); tr_m, va_m = np.intersect1d(tr, mm), np.intersect1d(va, mm); yb = (yD == 2).astype(int)
    bs = blockfit(tr_m); m_t = lr100().fit(T_(bs, tr_m), yb[tr_m]); pv, ptr = m_t.predict(T_(bs, va_m)), m_t.predict(T_(bs, tr_m))
    bsM = blockfit(mm); ml = lr100().fit(T_(bsM, mm), yb[mm])
    outs["EN-MiMo-LR"] = ml.predict_proba(bsM.transform(hb))
    agg["EN-MiMo-LR"] = dict(task="Moderate vs Mild, on Mild / Moderate rows", training_rows=bin_m(yb[mm], ml.predict(T_(bsM, mm))),
                             fit_part=bin_m(yb[tr_m], ptr), validation=bin_m(yb[va_m], pv), solver_iterations=int(np.max(ml.n_iter_)))
    g_t = gbm100().fit(tabe[tr_m], yb[tr_m]); gm = gbm100().fit(tabe[mm], yb[mm])
    outs["EN-MiMo-GBM"] = gm.predict_proba(tabh)
    agg["EN-MiMo-GBM"] = dict(task="Moderate vs Mild, on Mild / Moderate rows", training_rows=bin_m(yb[mm], gm.predict(tabe[mm])),
                              fit_part=bin_m(yb[tr_m], g_t.predict(tabe[tr_m])), validation=bin_m(yb[va_m], g_t.predict(tabe[va_m])), trees=TREES)
    mid = np.flatnonzero((efD >= 25) & (efD <= 60)); tr_e, va_e = np.intersect1d(tr, mid), np.intersect1d(va, mid)
    bs = blockfit(tr_e); r_t = Ridge(alpha=ALPHA).fit(T_(bs, tr_e), efD[tr_e])
    bsE = blockfit(mid); rm = Ridge(alpha=ALPHA).fit(T_(bsE, mid), efD[mid])
    outs["EN-mu-mid"] = rm.predict(bsE.transform(hb))
    agg["EN-mu-mid"] = dict(task="EF regression on rows with EF 25-60", training_rows=reg_m(efD[mid], rm.predict(T_(bsE, mid)), yD[mid], mD[mid]),
                            fit_part=reg_m(efD[tr_e], r_t.predict(T_(bs, tr_e)), yD[tr_e], mD[tr_e]),
                            validation=reg_m(efD[va_e], r_t.predict(T_(bs, va_e)), yD[va_e], mD[va_e]))
    agg["minutes"] = round((time.time() - t0) / 60, 1)
    np.savez(guard(f), **outs); guard(fj).write_text(json.dumps(agg, indent=1, default=float))
    log(f"EN members done {agg['minutes']} min")


def img_stage(workers):
    """PIX-ZS: the IMG_PLAN ResNet-18 (ImageNet init) pre-trained on EchoNext images, TRAIN100 recipe; then applied
    zero-shot to the 837 hospital images. Checkpoint every epoch (resumable)."""
    import torch.nn as nn
    from torch.utils.data import DataLoader
    from research_experiments.public_ef import img_train as IT
    f, fj, ck = PRIV / "pix_zs.npz", PRIV / "pix_zs.json", PRIV / "pix_zs_ckpt.pt"
    if f.exists() and fj.exists():
        log("PIX-ZS cached"); return
    t0 = time.time()
    m = np.load(need(PRIV0 / "img_echonext_meta.npz")); y, ef, g, male = m["y"], m["ef"], m["patient_key"], m["male"]
    path, hpath = need(PRIV0 / "img_echonext.npy"), need(PRIV0 / "img_hospital_ordered.npy")
    L = locked(g); idx = np.flatnonzero(~L)
    tr, va = next(GroupShuffleSplit(1, test_size=0.1, random_state=0).split(idx, y[idx], g[idx])); tr, va = idx[tr], idx[va]
    if IMG_LIMIT:
        tr = tr[:IMG_LIMIT]
    torch.manual_seed(42); np.random.seed(42)
    model = IT.Net(imagenet=True).to(IT.DEV)
    w = np.bincount(y[tr], minlength=4).astype(float)
    w = torch.tensor(w.sum() / (4 * np.maximum(w, 1)), dtype=torch.float32, device=IT.DEV)
    ce, mse = nn.CrossEntropyLoss(weight=w), nn.MSELoss()
    opt = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=0.0)          # constant lr, no schedule
    scaler = torch.amp.GradScaler(enabled=IT.DEV.type == "cuda")
    start, hist = 0, []
    if ck.exists():
        s = torch.load(ck, map_location=IT.DEV, weights_only=False)
        model.load_state_dict(s["model"]); opt.load_state_dict(s["opt"]); scaler.load_state_dict(s["scaler"])
        start, hist = s["epoch"], s["hist"]; torch.manual_seed(42 + start)
        log(f"PIX-ZS resumed after epoch {start}")
    dl = DataLoader(IT.Imgs(path, tr, y, ef, False), batch_size=32, shuffle=True, num_workers=workers,  # no augmentation
                    drop_last=len(tr) > 32, persistent_workers=workers > 0)
    acc = hist[-1]["train_acc"] if hist else 0.0
    ep = start
    while ep < IMG_CAP and acc < STOP:
        te = time.time(); model.train()
        for x, yy, ee in dl:
            x, yy, ee = x.to(IT.DEV), yy.to(IT.DEV), ee.float().to(IT.DEV)
            with torch.autocast(IT.DEV.type, enabled=IT.DEV.type == "cuda"):
                lo, rr = model(x); loss = ce(lo.float(), yy) + 0.5 * mse(rr.float(), ee)
            opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.step(opt); scaler.update()
        ep += 1
        Ptr, _ = IT.predict(model, path, tr, y, ef); acc = float(np.mean(Ptr.argmax(1) == y[tr]))
        hist.append(dict(epoch=ep, train_acc=acc, seconds=round(time.time() - te, 1)))
        torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), scaler=scaler.state_dict(), epoch=ep, hist=hist), guard(ck))
        log(f"PIX-ZS epoch {ep}: training accuracy {acc:.4f} ({hist[-1]['seconds']}s)")
    Ptr, Rtr = IT.predict(model, path, tr, y, ef); Pv, Rv = IT.predict(model, path, va, y, ef)
    Ph, Rh = IT.predict(model, hpath, np.arange(nD), Y, EF)
    agg = dict(n_train=int(len(tr)), n_validation=int(len(va)), epochs=ep, reached_stop=bool(acc >= STOP),
               training_rows=dict(cls_m(y[tr], Ptr.argmax(1)), macro_f1=IT.macro_f1(y[tr], Ptr.argmax(1))),
               validation=dict(cls_m(y[va], Pv.argmax(1)), macro_f1=IT.macro_f1(y[va], Pv.argmax(1))),
               ef_head=dict(training_rows=reg_m(ef[tr], Rtr, y[tr], male[tr]), validation=reg_m(ef[va], Rv, y[va], male[va])),
               epoch_history=[dict(epoch=h["epoch"], train_acc=round(h["train_acc"], 4)) for h in hist],
               minutes=round((time.time() - t0) / 60, 1), device=str(IT.DEV))
    torch.save(model.state_dict(), guard(PRIV / "pix_zs_weights.pt"))                 # private, never committed
    np.savez(guard(f), P=Ph, mu=Rh); guard(fj).write_text(json.dumps(agg, indent=1, default=float))
    log(f"PIX-ZS done: {ep} epochs, training {acc:.4f}, validation {agg['validation']['accuracy']:.4f}, {agg['minutes']} min")


# ----------------------------------------------------------------------------------------------- meta-learners
def _meta_namespace():
    """en_super.super_fold / staged_fold / ordinal_fold, unchanged except for the C values: every C grid literal becomes
    (BIG_C,), C_SEL = BIG_C, and max_iter is the original 5,000 (META_ITER; smaller only in the smoke test)."""
    ns = dict(vars(S)); ns.update(C_SEL=BIG_C, C_GRID=(BIG_C,), META_ITER=META_ITER)
    for fn in ("ordinal_fold", "staged_fold", "super_fold"):
        src = inspect.getsource(getattr(S, fn))
        assert src.count("(0.01, 0.03, 0.1, 0.3, 1.0)") == 1, fn
        src = src.replace("(0.01, 0.03, 0.1, 0.3, 1.0)", "C_GRID").replace("max_iter=5000", "max_iter=META_ITER")
        exec(compile(src, f"<en_super.{fn}, TRAIN100 C values>", "exec"), ns)  # noqa: S102  (our own repository source)
    return ns


NS = _meta_namespace()
FIXED_BASE = ("EN-cls", "EN-band", "EN-cls-all-H", "EN-gbm-H", "EN-band-all-H", "EN-cls-all-plain", "EN-gbm-plain",
              "EN-band-all-plain", "PIX-ZS", "PIX-ZS-band")
FIXED_X = ("EN-cls-all-X", "EN-gbm-X", "EN-MiMo-LR", "EN-MiMo-GBM", "EN-mu-mid")
BAND = {"EN-band", "EN-band-all-H", "EN-band-all-plain", "PIX-ZS-band", "EN-mu-mid"}


def fixed_outputs():
    z, p = np.load(PRIV / "en_members.npz"), np.load(PRIV / "pix_zs.npz")
    d = {n: z[n] for n in z.files}; d["PIX-ZS"], d["PIX-ZS-band"] = p["P"], p["mu"]
    return d


def candidates(r, k, pool, H, fx):
    """Same members, same order and same tuples as en_super.main; the 'outer' entries hold the held-out rows followed by
    the training rows (all row-wise operations), so one call gives held-out and in-sample predictions."""
    TR, TE, _ = Bd.splits(r, k)
    two = lambda a, b: np.concatenate([a, b], 0)  # noqa: E731
    c = {}
    for n in HOSP[:10] + (HOSP[10:] if pool == "X" else ()):
        c[n] = (H[f"{n}__inner"], two(H[f"{n}__te"], H[f"{n}__tr"]), None, None)
    mi, mt, mr = H["EF-band__mu_inner"], H["EF-band__mu_te"], H["EF-band__mu_tr"]
    sd = float(np.std(mi - EF[TR]))
    c["EF-band"] = (T.band(mi, sd, TR), two(T.band(mt, sd, TE), T.band(mr, sd, TR)), mi / 50, two(mt, mr) / 50)
    for n in FIXED_BASE + (FIXED_X if pool == "X" else ()):
        v = fx[n]
        if n not in BAND:
            c[n] = (v[TR], two(v[TE], v[TR]), None, None)
        else:
            s_ = float(np.std(v[TR] - EF[TR]))
            c[n] = (T.band(v[TR], s_, TR), two(T.band(v[TE], s_, TE), T.band(v[TR], s_, TR)), v[TR] / 50, two(v[TE], v[TR]) / 50)
    return c


def meta_fold(cfg, r, k, H, fx):
    f = PRIV / "meta" / f"{cfg}_r{r}_k{k}.npz"; fj = f.with_suffix(".json")
    if f.exists() and fj.exists():
        return json.loads(fj.read_text())
    t0 = time.time(); pool, select = CONFIGS[cfg]
    TR, TE, _ = Bd.splits(r, k)
    NS["SELECT"] = select
    te_, p4, chosen, best, stg = NS["super_fold"](r, k, candidates(r, k, pool, H, fx))
    assert np.array_equal(te_, TE)
    n = len(TE); outs = dict(sup=p4, staged4=stg["staged4"], three_class=stg["three_class"], ord4=stg["ord4"])
    np.savez(guard(f), **{f"{o}__te": v[:n] for o, v in outs.items()}, **{f"{o}__tr": v[n:] for o, v in outs.items()})
    info = dict(members=chosen, inner_meta_cv_accuracy=float(best), seconds=round(time.time() - t0, 1))
    guard(fj).write_text(json.dumps(info))
    return info


def fold_task(rk):
    r, k = rk
    warnings.filterwarnings("ignore"); torch.set_num_threads(THREADS); t0 = time.time()
    H, info = hosp_fold(r, k)
    log(f"r{r} k{k}: hospital members done ({info['_minutes']} min; ECGF-MLP epochs {info['ECGF-MLP']['outer']['epochs']})")
    msg = "EchoNext members not ready: meta skipped"
    if (PRIV / "en_members.json").exists() and (PRIV / "pix_zs.json").exists():
        fx = fixed_outputs()
        for cfg in CONFIGS:
            i = meta_fold(cfg, r, k, H, fx); log(f"r{r} k{k}: {cfg} {len(i['members'])} members, {i['seconds']}s")
        msg = "meta done"
    log(f"r{r} k{k}: {msg}, {(time.time() - t0) / 60:.1f} min")
    return r, k


# ----------------------------------------------------------------------------------------------------- report
def _metrics(y, p):
    from research_experiments.public_ef.img_train import macro_f1
    return dict(T.metrics(y, p), macro_f1=macro_f1(y, p))


def report():
    cells_rk = [(0, 0)] if SMOKE else [(r, k) for r in (0, 1, 2) for k in range(5)]
    reps = sorted({r for r, _ in cells_rk})
    out = dict(amendment="research_experiments/train100/AMENDMENT_TRAIN100_ECHONEXT.md", smoke=SMOKE,
               settings=dict(C=BIG_C, ridge_alpha=ALPHA, boosting_trees=TREES, knn_k=KNN_K, stop_training_accuracy=STOP,
                             mlp_epoch_cap=MLP_CAP, linear_lbfgs_cap=LIN_CAP, cnn_epoch_cap=IMG_CAP, lr_max_iter_members=LR_ITER,
                             lr_max_iter_meta=META_ITER),
               models={}, members=dict(hospital={}, echonext={}), selection_frequency={}, per_fold=[], timing={})
    Hs = {rk: hosp_fold(*rk) for rk in cells_rk}
    # ---- the three models
    for name, (cfg, o, orig_file, orig_key) in HEADLINE.items():
        preds, cells, tr_acc = [], [], []
        for r in reps:
            p = np.full(nD, -1)
            for k in [k for r_, k in cells_rk if r_ == r]:
                TR, TE, _ = Bd.splits(r, k); z = np.load(PRIV / "meta" / f"{cfg}_r{r}_k{k}.npz")
                p[TE] = z[f"{o}__te"]; tr_acc.append(float(np.mean(z[f"{o}__tr"] == Y[TR])))
                cells.append(float(np.mean(z[f"{o}__te"] == Y[TE])))
            preds.append(p)
        ok = preds[0] >= 0
        ms = [_metrics(Y[ok], p[ok]) for p in preds]
        s = {k_: float(np.mean([m[k_] for m in ms])) for k_ in ms[0]}
        orig = json.loads((EP.RES / orig_file).read_text())["models"][orig_key]
        out["models"][name] = dict(
            en_super_run=dict(config=cfg, output=orig_key),
            train_accuracy=dict(mean=float(np.mean(tr_acc)), min=float(np.min(tr_acc)), max=float(np.max(tr_acc))),
            heldout=dict(s, accuracy_ci95_corrected=corr_ci(cells) if len(cells) > 1 else None,
                         accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms], admissible=bool(s["min_recall"] >= 0.15),
                         confusion_mean_per_repeat=np.mean([[[int(np.sum((Y[ok] == a) & (p[ok] == b))) for b in range(4)] for a in range(4)]
                                                            for p in preds], 0).round(1).tolist()),
            original=dict(accuracy=orig["accuracy"], ci95_corrected=orig["accuracy_ci95_corrected"],
                          recall=[orig[f"recall_{c}"] for c in B.CLASSES], macro_f1=orig.get("macro_f1")),
            change_points=round(100 * (s["accuracy"] - orig["accuracy"]), 2), _cells=cells, _train=tr_acc)
    for cfg in CONFIGS:
        fr = {}
        for r, k in cells_rk:
            for n in json.loads((PRIV / "meta" / f"{cfg}_r{r}_k{k}.json").read_text())["members"]:
                fr[n] = fr.get(n, 0) + 1
        out["selection_frequency"][cfg] = dict(sorted(fr.items(), key=lambda x: -x[1]))
    for i, (r, k) in enumerate(cells_rk):
        row = dict(repeat=r, fold=k)
        for name, (cfg, _, _, _) in HEADLINE.items():
            j = json.loads((PRIV / "meta" / f"{cfg}_r{r}_k{k}.json").read_text())
            row[name] = dict(train_acc=round(out["models"][name]["_train"][i], 4), heldout_acc=round(out["models"][name]["_cells"][i], 4),
                             n_members=len(j["members"]), inner_meta_cv_accuracy=round(j["inner_meta_cv_accuracy"], 4))
        out["per_fold"].append(row)
    for name in out["models"]:
        out["models"][name].pop("_cells"); out["models"][name].pop("_train")
    # ---- hospital members
    for n in HOSP + ("EF-band",):
        tr_a, cells, preds = [], [], {r: np.full(nD, -1) for r in reps}
        for (r, k) in cells_rk:
            H, inf = Hs[(r, k)]; TR, TE, _ = Bd.splits(r, k)
            if n == "EF-band":
                sd = float(np.std(H["EF-band__mu_inner"] - EF[TR]))
                pte, ptr = T.band(H["EF-band__mu_te"], sd, TE).argmax(1), T.band(H["EF-band__mu_tr"], sd, TR).argmax(1)
            else:
                pte, ptr = H[f"{n}__te"].argmax(1), H[f"{n}__tr"].argmax(1)
            if n == "H-MiMo-GBM":                                         # Mild (0) vs Moderate (1) on Mild / Moderate rows
                mt, mr = MM[TE], MM[TR]
                tr_a.append(float(np.mean(ptr[mr] == (Y[TR][mr] == 2)))); cells.append(float(np.mean(pte[mt] == (Y[TE][mt] == 2))))
                preds[r][TE] = np.where(mt, (pte == (Y[TE] == 2)).astype(int), -1)
                continue
            own = U.ZONE[TR] if n == "M12" else np.ones(len(TR), bool)
            tr_a.append(float(np.mean(ptr[own] == Y[TR][own]))); cells.append(float(np.mean(pte == Y[TE])))
            preds[r][TE] = pte
        if n == "H-MiMo-GBM":
            ho = float(np.mean([np.mean(preds[r][preds[r] >= 0]) for r in reps]))
        else:
            ho = float(np.mean([np.mean(preds[r][preds[r] >= 0] == Y[preds[r] >= 0]) for r in reps]))
        d = dict(train_accuracy=float(np.mean(tr_a)), train_accuracy_min=float(np.min(tr_a)), heldout_accuracy=ho,
                 heldout_ci95_corrected=corr_ci(cells) if len(cells) > 1 else None)
        if n == "H-MiMo-GBM":
            d["task"] = "Mild vs Moderate, Mild / Moderate rows"
        if n == "M12":
            d["train_rows"] = "zone rows of the outer training rows"
        if n == "ECGF-MLP":
            ep = [e for rk in cells_rk for x in [Hs[rk][1][n]["outer"]] + Hs[rk][1][n]["inner"] for e in x["epochs"]]
            d["epochs_to_stop"] = dict(median=float(np.median(ep)), max=int(np.max(ep)), fits_at_cap=int(np.sum(np.array(ep) >= MLP_CAP)), n_fits=len(ep))
        if n == "ECGF-lin":
            it = [x["iterations"] for rk in cells_rk for x in [Hs[rk][1][n]["outer"]] + Hs[rk][1][n]["inner"]]
            d["lbfgs_iterations"] = dict(median=float(np.median(it)), max=int(np.max(it)), fits_at_cap=int(np.sum(np.array(it) >= LIN_CAP)))
        if n in ("M6", "EF-band"):
            d["ef_rmse"] = dict(train=float(np.mean([Hs[rk][1][n]["train_rmse"] for rk in cells_rk])),
                                inner_oof=float(np.mean([Hs[rk][1][n]["inner_rmse"] for rk in cells_rk])))
        out["members"]["hospital"][n] = d
    # ---- EchoNext members: EchoNext training / validation + accuracy on the hospital ECGs (never trained on them)
    en = json.loads((PRIV / "en_members.json").read_text()); px = json.loads((PRIV / "pix_zs.json").read_text())
    fx = fixed_outputs()
    out["data"] = dict(hospital_development_ecgs=nD, echonext_rows=en["n_rows"], echonext_locked_not_used=en["n_locked_not_used"],
                       echonext_training_pool=en["n_training_pool"], echonext_fit_part=en["n_fit_part"],
                       echonext_validation=en["n_validation"], echonext_images_train=px["n_train"], echonext_images_validation=px["n_validation"])

    def hosp_acc(n):
        v = fx[n]
        if n in ("EN-MiMo-LR", "EN-MiMo-GBM"):
            return dict(hospital_accuracy=float(np.mean(v.argmax(1)[MM] == (Y[MM] == 2))), hospital_task="Mild vs Moderate, Mild / Moderate rows")
        if n not in BAND:
            return dict(hospital_accuracy=float(np.mean(v.argmax(1) == Y)))
        acc = []
        for r in reps:
            p = np.full(nD, -1)
            for k in [k for r_, k in cells_rk if r_ == r]:
                TR, TE, _ = Bd.splits(r, k); s_ = float(np.std(v[TR] - EF[TR])); p[TE] = T.band(v[TE], s_, TE).argmax(1)
            acc.append(float(np.mean(p[p >= 0] == Y[p >= 0])))
        return dict(hospital_accuracy=float(np.mean(acc)), hospital_ef_rmse=float(np.sqrt(np.mean((v - EF) ** 2))))
    src = {"EN-cls": "EN-cls", "EN-band": "EN-band", "EN-cls-all-H": "EN-cls-all", "EN-cls-all-plain": "EN-cls-all",
           "EN-gbm-H": "EN-gbm", "EN-gbm-plain": "EN-gbm", "EN-band-all-H": "EN-band-all", "EN-band-all-plain": "EN-band-all",
           **{n: n for n in FIXED_X}}
    for n in FIXED_BASE + FIXED_X:
        if n in ("PIX-ZS", "PIX-ZS-band"):
            e = dict(training_rows=px["training_rows"] if n == "PIX-ZS" else px["ef_head"]["training_rows"],
                     validation=px["validation"] if n == "PIX-ZS" else px["ef_head"]["validation"], epochs=px["epochs"],
                     reached_stop=px["reached_stop"])
        else:
            e = {k_: v for k_, v in en[src[n]].items() if k_ != "fit_part"}
            e["fit_part_training"] = en[src[n]]["fit_part"]
        out["members"]["echonext"][n] = dict(e, **hosp_acc(n))
    out["timing"] = dict(echonext_tabular_minutes=en["minutes"], pix_zs_minutes=px["minutes"],
                         hospital_members_minutes_per_fold=float(np.mean([Hs[rk][1]["_minutes"] for rk in cells_rk])),
                         meta_minutes_per_fold={cfg: float(np.mean([json.loads((PRIV / "meta" / f"{cfg}_r{r}_k{k}.json").read_text())["seconds"]
                                                                    for r, k in cells_rk]) / 60) for cfg in CONFIGS})
    dest = PRIV / "train100_echonext_smoke.json" if SMOKE else RES_JSON
    guard(dest).write_text(json.dumps(out, indent=1, default=float))
    log(f"wrote {dest.name}")
    print(tables(out), flush=True)
    return out


def tables(out):
    f3 = lambda x: f"{x:.3f}"  # noqa: E731
    L = ["| Model | Training accuracy | Held-out accuracy [95 % CI] | Recall N / Mi / Mo / S | Macro-F1 | Original [95 % CI] | Change (points) |",
         "|---|---|---|---|---|---|---|"]
    for n, s in out["models"].items():
        h, o = s["heldout"], s["original"]; ci = h["accuracy_ci95_corrected"] or [float("nan")] * 2
        L.append(f"| {n} | {f3(s['train_accuracy']['mean'])} | {f3(h['accuracy'])} [{f3(ci[0])}, {f3(ci[1])}] | "
                 + " / ".join(f"{h[f'recall_{c}']:.2f}" for c in B.CLASSES) + f" | {f3(h['macro_f1'])} | "
                 f"{f3(o['accuracy'])} [{f3(o['ci95_corrected'][0])}, {f3(o['ci95_corrected'][1])}] | {s['change_points']:+.1f} |")
    M = ["| Member | Trained on | Training accuracy | Held-out accuracy | Held-out data | Accuracy on the 837 hospital ECGs |",
         "|---|---|---|---|---|---|"]
    for n, d in out["members"]["hospital"].items():
        ci = d["heldout_ci95_corrected"] or [float("nan")] * 2
        task = " (Mild vs Moderate)" if "task" in d else ""
        rows = "hospital zone rows" if n == "M12" else "hospital Mild/Moderate rows" if "task" in d else "hospital"
        M.append(f"| {n}{task} | {rows} | {f3(d['train_accuracy'])} | {f3(d['heldout_accuracy'])} [{f3(ci[0])}, {f3(ci[1])}] | "
                 f"hospital outer folds | (held-out column) |")
    for n, d in out["members"]["echonext"].items():
        tr, va = d["training_rows"], d["validation"]
        if "rmse" in tr:
            a, b = f"{f3(tr['graded_accuracy'])} (RMSE {tr['rmse']:.1f})", f"{f3(va['graded_accuracy'])} (RMSE {va['rmse']:.1f})"
        else:
            a, b = f3(tr["accuracy"]), f3(va["accuracy"])
        task = " (Mild vs Moderate)" if "hospital_task" in d else ""
        M.append(f"| {n}{task} | EchoNext, {tr['n']} rows | {a} | {b} | EchoNext validation, {va['n']} rows | {f3(d['hospital_accuracy'])} |")
    return "\n".join(L) + "\n\n" + "\n".join(M)


# ------------------------------------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", required=True, choices=["en", "img", "folds", "report"])
    ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--threads", type=int, default=2)
    ap.add_argument("--workers", type=int, default=2); ap.add_argument("--cells", default="all"); ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()
    log(f"stage {a.stage}{' (SMOKE)' if SMOKE else ''}, threads {THREADS}, private folder {PRIV.name}")
    if a.stage == "en":
        en_stage()
    elif a.stage == "img":
        img_stage(a.workers)
    elif a.stage == "folds":
        cells = [(0, 0)] if SMOKE else ([(r, k) for r in (0, 1, 2) for k in range(5)] if a.cells == "all"
                                        else [tuple(map(int, c.split(":"))) for c in a.cells.split(",")])
        if a.jobs <= 1:
            for c in cells:
                fold_task(c)
        else:
            with Pool(a.jobs) as p:
                list(p.imap_unordered(fold_task, cells))
    else:
        report()


if __name__ == "__main__":
    main()
