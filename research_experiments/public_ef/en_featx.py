"""PUBLIC_EF_PLAN Amendment 8: FEATX caches (hospital plain / H, EchoNext) and the Mild / Moderate-targeted members.
Per-ECG arrays stay in the git-ignored data/private/featx/. Never prints EchoNext rows.
    python -m research_experiments.public_ef.en_featx --hospital        (837 development ECGs, plain + H)
    python -m research_experiments.public_ef.en_featx --echonext        (the rows of en_cache.npz)
"""
import argparse
import os
import time
from multiprocessing import Pool

import lightgbm as lgb
import numpy as np
import pandas as pd
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import GroupShuffleSplit

from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B
from research_experiments.public_ef import en_image as EI
from research_experiments.public_ef import en_prep as P
from research_experiments.public_ef.en_split import locked
from research_experiments.public_ef.featx import featx

FX = P.PRIV / "featx"
GBMX = dict(EI.GBM, n_jobs=4)
_W = {}


def _frame(rows, cols=None):
    F = pd.DataFrame(rows).apply(pd.to_numeric, errors="coerce").replace([np.inf, -np.inf], np.nan)
    return F.reindex(columns=sorted(F.columns) if cols is None else list(cols))


def _hosp_one(a):
    lv, rv, h = a
    if h:
        lv, rv = EI.harmonise(lv, rv)
    return featx(lv, rv)


def hospital_featx(variant, jobs=1):
    f = FX / f"hosp_{variant}.npz"
    if f.exists():
        z = np.load(f); return z["X"], z["cols"]
    sig = np.load(B.ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(sig["ecg_id"])), index=sig["ecg_id"]).loc[B.IDS].values     # development rows only
    L, R = sig["leads"][pos], sig["rhythm"][pos]                         # load once (not once per ECG)
    args = [(L[j], R[j], variant == "H") for j in range(len(pos))]
    if jobs <= 1:                                                         # low-memory path: no worker processes
        rows = [_hosp_one(x) for x in args]
    else:
        with Pool(jobs) as p:
            rows = p.map(_hosp_one, args, chunksize=8)
    cols = None if variant == "plain" else hospital_featx("plain")[1]
    F = _frame(rows, cols)
    FX.mkdir(parents=True, exist_ok=True)
    np.savez(f, X=F.values.astype(np.float32), cols=np.asarray(F.columns, dtype="U64"))
    return F.values.astype(np.float32), np.asarray(F.columns)


def _en_init(params, normalised):
    _W.update(params=params, normalised=normalised)


def _en_one(w):
    return featx(*P.printout(P.to_mv(w, _W["params"], _W["normalised"])))


def echonext_featx(jobs=1):
    z = np.load(P.PRIV / "en_cache.npz"); rows = z["row"]
    f = FX / "echonext.npz"
    if f.exists():
        c = np.load(f)
        if np.array_equal(c["row"], rows):
            return c["X"]
    src = P.source_dir()
    W = np.load(src / "EchoNext_test_waveforms.npy", mmap_mode="r")
    normalised = float(np.asarray(W[:200], float).std()) < 5                  # as en_prep
    cols = hospital_featx("plain")[1]
    out = []
    if jobs <= 1:                                                         # low-memory path: no worker processes
        _en_init(P.norm_params(), normalised)
        out = [_en_one(np.asarray(W[i, 0])) for i in rows]
    else:
        with Pool(jobs, initializer=_en_init, initargs=(P.norm_params(), normalised)) as p:
            for s in range(0, len(rows), 256):
                out += p.map(_en_one, [np.asarray(W[i, 0]) for i in rows[s:s + 256]], chunksize=8)
    X = _frame(out, cols).values.astype(np.float32)
    FX.mkdir(parents=True, exist_ok=True); np.savez(f, X=X, row=rows)
    return X


# ----------------------------------------------------------------------------------------------- EchoNext members
def en_members():
    """EN-cls-all-X, EN-gbm-X, EN-MiMo-LR, EN-MiMo-GBM, EN-mu-mid (variant H) -> hospital outputs + internal checks."""
    z = np.load(P.PRIV / "en_cache.npz", allow_pickle=False)
    g, y, ef = z["patient_key"], z["y"], z["ef"]
    D = np.flatnonzero(~locked(g))
    Xe = echonext_featx()
    hH, xH = EI.hospital_inputs("H"), hospital_featx("H")[0]
    eb = [np.asarray(a, float) for a in (z["f6_aligned"], z["f6_synth"], z["logits_aligned"], z["logits_synth"], z["tab"])]
    # G5 joins the tabular block (G3 + G4), so the block standardiser imputes it with the other tabular features
    eb = [b[D] for b in eb[:4]] + [np.c_[eb[4], np.asarray(Xe, float)][D]]
    hb = [np.asarray(hH[k], float) for k in ("f6a", "f6s", "la", "ls")] + [np.c_[np.asarray(hH["tab"], float), np.asarray(xH, float)]]
    tabe = np.c_[eb[2], eb[3], eb[4]]; tabh = np.c_[hb[2], hb[3], hb[4]]
    y, ef, g = y[D], ef[D], g[D]
    tr, va = next(GroupShuffleSplit(1, test_size=0.2, random_state=0).split(tabe, y, g))

    def blockfit(rows):
        return EI.BlockStd().fit([b[rows] for b in eb])

    out, internal = {}, {}
    # EN-cls-all-X
    bs = blockfit(tr); Xtr, Xva = bs.transform([b[tr] for b in eb]), bs.transform([b[va] for b in eb])
    C = min(EI.CS_ALL, key=lambda c: EI.U.logloss(y[va], EI._lr(c).fit(Xtr, y[tr]).predict_proba(Xva)))
    bsA = blockfit(np.arange(len(y))); XA = bsA.transform(eb); XH = bsA.transform(hb)
    out["EN-cls-all-X"] = EI._lr(C).fit(XA, y).predict_proba(XH)
    # EN-gbm-X
    gb = lgb.LGBMClassifier(n_estimators=400, **GBMX).fit(tabe[tr], y[tr], eval_set=[(tabe[va], y[va])],
                                                          callbacks=[lgb.early_stopping(50, verbose=False)])
    nr = int(gb.best_iteration_ or 400)
    internal["EN-gbm-X_heldout_accuracy"] = float(np.mean(gb.predict(tabe[va]) == y[va]))
    out["EN-gbm-X"] = lgb.LGBMClassifier(n_estimators=nr, **GBMX).fit(tabe, y).predict_proba(tabh)
    # EN-MiMo-LR / EN-MiMo-GBM (binary: Moderate = 1), fitted on Mild / Moderate rows only
    mm = np.flatnonzero(np.isin(y, (1, 2))); tr_m, va_m = np.intersect1d(tr, mm), np.intersect1d(va, mm)
    yb = (y == 2).astype(int)
    bs = blockfit(tr_m)
    Cm = min(EI.CS_ALL, key=lambda c: EI.U.logloss(yb[va_m], EI._lr(c).fit(bs.transform([b[tr_m] for b in eb]), yb[tr_m])
                                                   .predict_proba(bs.transform([b[va_m] for b in eb]))))
    bsM = blockfit(mm)
    mlr = EI._lr(Cm).fit(bsM.transform([b[mm] for b in eb]), yb[mm])
    out["EN-MiMo-LR"] = mlr.predict_proba(bsM.transform(hb))
    gbm_m = lgb.LGBMClassifier(n_estimators=400, **GBMX).fit(tabe[tr_m], yb[tr_m], eval_set=[(tabe[va_m], yb[va_m])],
                                                             callbacks=[lgb.early_stopping(50, verbose=False)])
    nrm = int(gbm_m.best_iteration_ or 400)
    from sklearn.metrics import roc_auc_score
    internal["EN-MiMo-GBM_heldout_auroc"] = float(roc_auc_score(yb[va_m], gbm_m.predict_proba(tabe[va_m])[:, 1]))
    out["EN-MiMo-GBM"] = lgb.LGBMClassifier(n_estimators=nrm, **GBMX).fit(tabe[mm], yb[mm]).predict_proba(tabh)
    # EN-mu-mid (EF 25-60)
    mid = np.flatnonzero((ef >= 25) & (ef <= 60)); tr_e, va_e = np.intersect1d(tr, mid), np.intersect1d(va, mid)
    bs = blockfit(tr_e)
    a = min(EI.T.ALPHAS, key=lambda al: float(np.mean((Ridge(alpha=al).fit(bs.transform([b[tr_e] for b in eb]), ef[tr_e])
                                                       .predict(bs.transform([b[va_e] for b in eb])) - ef[va_e]) ** 2)))
    bsE = blockfit(mid)
    out["EN-mu-mid"] = Ridge(alpha=a).fit(bsE.transform([b[mid] for b in eb]), ef[mid]).predict(bsE.transform(hb))
    internal.update(C_cls=float(C), C_mimo=float(Cm), alpha_mid=float(a), gbm_rounds=nr, mimo_gbm_rounds=nrm,
                    n_g5=int(Xe.shape[1]))
    return out, internal


# ----------------------------------------------------------------------------------------------- hospital members
def hosp_job(kind, r, k, cache):
    """H-X-GBM (four classes) / H-MiMo-GBM (Mild vs Moderate, fitted on Mild / Moderate rows): inner OOF + outer."""
    f = cache / f"{kind}_r{r}_k{k}.npz"
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    h = EI.hospital_inputs("plain"); xp = hospital_featx("plain")[0]
    G = np.c_[h["la"], h["ls"], h["tab"], xp].astype(float)
    TR, TE, inner = Bd.splits(r, k)
    Y = B.Y

    def fit_pred(a, b):
        if kind == "H-X-GBM":
            return lgb.LGBMClassifier(n_estimators=300, **GBMX).fit(G[a], Y[a]).predict_proba(G[b])
        a = a[np.isin(Y[a], (1, 2))]
        return lgb.LGBMClassifier(n_estimators=300, **GBMX).fit(G[a], (Y[a] == 2).astype(int)).predict_proba(G[b])

    k_ = 4 if kind == "H-X-GBM" else 2
    Pi = np.zeros((len(TR), k_))
    for ia, ib in inner:
        Pi[ib] = fit_pred(TR[ia], TR[ib])
    out = dict(inner=Pi, outer=fit_pred(TR, TE))
    cache.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


def descriptive_auroc():
    """Descriptive only: univariate Mild-vs-Moderate AUROC of each G5 feature on the hospital ECGs (plain)."""
    from sklearn.metrics import roc_auc_score
    X, cols = hospital_featx("plain")
    m = np.isin(B.Y, (1, 2)); out = {}
    for j, c in enumerate(cols):
        v = X[m, j]; ok = np.isfinite(v)
        if ok.sum() > 50:
            out[str(c)] = float(roc_auc_score(B.Y[m][ok] == 2, v[ok]))
    return dict(sorted(out.items(), key=lambda x: -abs(x[1] - 0.5))[:15])


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--hospital", action="store_true"); ap.add_argument("--echonext", action="store_true")
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    t0 = time.time()
    if a.hospital:
        for v in ("plain", "H"):
            X, c = hospital_featx(v, a.jobs); print(f"hospital {v}: {X.shape}, missing {np.isnan(X).mean():.3f}, {time.time() - t0:.0f}s", flush=True)
    if a.echonext:
        X = echonext_featx(a.jobs); print(f"EchoNext: {X.shape}, missing {np.isnan(X).mean():.3f}, {time.time() - t0:.0f}s", flush=True)
