# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/train100/train100_variants.py
# >>> Run from the repository root: PYTHONPATH=. python research_experiments/train100/train100_variants.py (see the docstring below for the options)
# >>> Part of BEST_RESULTS/best_for_50s; see the README there.
"""TRAIN100-D (AMENDMENT_TRAIN100.md, addendum D): the remaining variants of every phase, retrained until (nearly) 100 %
training accuracy, in the same Phase-2 patient-grouped 5-fold CV x 3 repeats, development ECGs only (no locked test,
no EchoNext). Same inputs and model families as the originals; every regularisation / early-stopping / shrinkage knob
removed or set to its weakest value; nested decision rules (prior rule, class weights, thresholds, simplex weights)
kept as in the original, because they are decision rules, not regularisation. Aggregates only.
    PYTHONPATH=<repo> python3 train100_variants.py --out <dir> [--jobs 4] [--cells 0:0,...] [--parts base,labels]
Part "base":  E1 E2 E3 (wavg) Stack-LR E4 E6 E7-mixup | EF regression + multitask (Phase 4) | HIER, ENH members M2-M6,
              ENH-vote, ULT members M8 M9 M10 M12 and ULT-stack (Phase 6B)
Part "labels": C3 recipe under the label studies VER, HC30, CLEAR, MMC, NF, NFS (report Table 4.7)
"""
from __future__ import annotations

import argparse
import json
import time
import warnings
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.neighbors import KNeighborsClassifier

import train100_features as T  # noqa: E402  (same directory; its learners are reused unchanged)
from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha
from research_experiments.phase_5.p5common import blockstd, corr_ci, decide, ecgf_f6, feature_family, impute, make_X

C, B, Y = T.C, T.B, T.Y
NEW = "qrs+st+hrv+freq+wavelet+leadstat+sq"
BIG_C = 1e4                                   # "almost no regularisation" for every logistic regression
LR_ITER, HGB_ITER, REG_MAX_EP = 5000, 2000, 3000
LABEL_CONFIGS = ("VER", "HC30", "CLEAR", "MMC", "NF", "NFS")


def acc(p, y):
    return float(np.mean(p == y))


def inner_split(r, k, TR):
    return list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, Y[TR], C.PAT[TR]))


def pad4(mdl, P):
    out = np.full((len(P), 4), 1e-3); out[:, mdl.classes_] = P
    return out / out.sum(1, keepdims=True)


def lr_fit(X, rows, y, balanced=True):
    m = LogisticRegression(C=BIG_C, class_weight="balanced" if balanced else None, max_iter=LR_ITER).fit(X[rows], y[rows])
    return lambda idx: pad4(m, m.predict_proba(X[idx]))


# ------------------------------------------------------------------ MLP variants (same body as the project MLP head)
class Body(nn.Module):
    def __init__(self, d, out, hidden=64, dropout=0.5):
        super().__init__()
        self.net = nn.Sequential(nn.Dropout(dropout), nn.Linear(d, hidden), nn.GELU(), nn.Dropout(dropout), nn.Linear(hidden, out))

    def forward(self, x):
        return self.net(x)


def _std(X, rows):
    mu, sd = X[rows].mean(0), X[rows].std(0) + 1e-6
    return lambda idx: torch.tensor((X[idx] - mu) / sd, dtype=torch.float32)


def mlp_cls(X, rows, y, n_out, seed, mixup=0.0, max_epochs=None):
    """Class head trained (AdamW lr 1e-3, wd 0, batch 64, balanced CE) until 100 % training accuracy (eval mode)."""
    tf = _std(X, rows); xtr = tf(rows); yt = torch.tensor(y[rows]); cw = class_w(y[rows], "balanced", n=n_out)
    torch.manual_seed(seed); m = Body(X.shape[1], n_out); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); ep, a = 0, 0.0
    max_epochs = max_epochs or T.MLP_MAX_EPOCHS
    while ep < max_epochs and a < 1.0:
        m.train(); idx = torch.randperm(len(rows), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; xb, yb = xtr[b], yt[b]
            if mixup:
                lam = float(np.random.default_rng(seed * 100003 + ep * 997 + i).beta(mixup, mixup)); pm = torch.randperm(len(b), generator=g)
                out = m(lam * xb + (1 - lam) * xb[pm])
                lo = lam * F.cross_entropy(out, yb, weight=cw) + (1 - lam) * F.cross_entropy(out, yb[pm], weight=cw)
            else:
                lo = F.cross_entropy(m(xb), yb, weight=cw)
            opt.zero_grad(); lo.backward(); opt.step()
        ep += 1; m.eval()
        with torch.no_grad():
            a = float((m(xtr).argmax(1).numpy() == y[rows]).mean())
    m.eval()

    def f(idx):
        with torch.no_grad():
            return F.softmax(m(tf(idx)), 1).numpy()
    return f, ep


def mlp_seeds(X, rows, y, n_out, **kw):
    fs = [mlp_cls(X, rows, y, n_out, s, **kw) for s in T.SEEDS]
    return (lambda idx: np.mean([f(idx) for f, _ in fs], 0)), [e for _, e in fs]


def mlp_reg(X, rows, ef, seed, target_r2=0.99, max_epochs=None):
    """EF regression MLP (Phase-4 body, Huber on z-scored EF, AdamW wd 0, no early stopping) until training R2 >= 0.99."""
    tf = _std(X, rows); xtr = tf(rows); mu, sd = ef[rows].mean(), ef[rows].std()
    yt = torch.tensor((ef[rows] - mu) / sd, dtype=torch.float32)
    torch.manual_seed(seed); m = Body(X.shape[1], 1); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); ep, r2 = 0, -1.0
    max_epochs = max_epochs or REG_MAX_EP
    while ep < max_epochs and r2 < target_r2:
        m.train(); idx = torch.randperm(len(rows), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; lo = F.huber_loss(m(xtr[b])[:, 0], yt[b], delta=1.0); opt.zero_grad(); lo.backward(); opt.step()
        ep += 1; m.eval()
        with torch.no_grad():
            pr = m(xtr)[:, 0]; r2 = float(1 - ((pr - yt) ** 2).mean() / yt.var())
    m.eval()

    def f(idx):
        with torch.no_grad():
            return m(tf(idx))[:, 0].numpy() * sd + mu
    return f, ep


def multitask(X, rows, y, ef, seed, lam=1.0, max_epochs=None):
    """Phase-4 multitask: shared body -> class head (balanced CE) + EF head (masked Huber), lam = 1, until 100 % class acc."""
    tf = _std(X, rows); xtr = tf(rows); yt = torch.tensor(y[rows]); cw = class_w(y[rows], "balanced")
    e = ef[rows]; ok = torch.tensor(~np.isnan(e)); mu, sd = np.nanmean(e), np.nanstd(e)
    et = torch.tensor(np.nan_to_num((e - mu) / sd), dtype=torch.float32)
    torch.manual_seed(seed); m = Body(X.shape[1], 5); g = torch.Generator().manual_seed(seed)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0); ep, a = 0, 0.0
    max_epochs = max_epochs or T.MLP_MAX_EPOCHS
    while ep < max_epochs and a < 1.0:
        m.train(); idx = torch.randperm(len(rows), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; out = m(xtr[b]); lo = F.cross_entropy(out[:, :4], yt[b], weight=cw)
            if ok[b].any():
                lo = lo + lam * F.huber_loss(out[ok[b], 4], et[b][ok[b]], delta=1.0)
            opt.zero_grad(); lo.backward(); opt.step()
        ep += 1; m.eval()
        with torch.no_grad():
            a = float((m(xtr)[:, :4].argmax(1).numpy() == y[rows]).mean())
    m.eval()

    def f(idx):
        with torch.no_grad():
            o = m(tf(idx)); return F.softmax(o[:, :4], 1).numpy(), o[:, 4].numpy() * sd + mu
    return f, ep


# ------------------------------------------------------------------ combiners (nested on inner OOF, as the originals)
SIMPLEX = [np.array([a, b, 10 - a - b]) / 10 for a in range(11) for b in range(11 - a)]


def wavg_weights(Ps, y):
    return max(SIMPLEX, key=lambda w: (acc(sum(wi * P for wi, P in zip(w, Ps)).argmax(1), y), -abs(w - 1 / 3).sum()))


def lp(P):
    return np.log(np.clip(P, 1e-6, 1))


def band(mu, sd, idx):
    """EF -> class probabilities by Gaussian band integration with the per-ECG Normal limit (as public_ef en_train.band)."""
    from scipy.stats import norm
    from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
    nc = Bd.normal_cut(52.0, 2.0)
    cN, cM, cS = (norm.cdf((c - mu) / sd) for c in (nc[idx], 41.0, 31.0))
    return np.c_[1 - cN, cN - cM, cM - cS, cS]


def super_top10_ord(cands, TR, yt, r, k, sel_C, C_grid, outs=("te",)):
    """SUPER-TOP10-ORD (public_ef en_super, Amendments 7 + 9): the 10 best single members by meta-CV accuracy, then a
    Frank-Hall ordinal meta-learner, one C by meta-CV log-loss, floor rule (0.18) on the meta-CV OOF.
    cands[n] = {"inner": P, "inner_mu": mu/50 or None, o: P, o + "_mu": ...} for o in outs."""
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, C.PAT[TR]))

    def Z(names, key):
        return np.column_stack([lp(cands[n][key]) for n in names]
                               + [cands[n][key + "_mu"] for n in names if cands[n].get(key + "_mu") is not None])

    def mcv(Zi, Cc):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=Cc, max_iter=LR_ITER).fit(Zi[ia], yt[ia]).predict_proba(Zi[ib])
        return mo

    def score(names):
        mo = mcv(Z(names, "inner"), sel_C)
        return acc((mo * U.tune_floor(mo, yt, 0.18)).argmax(1), yt)
    single = {n: score([n]) for n in cands}
    chosen = sorted(single, key=lambda n: -single[n])[:10]
    Zi = Z(chosen, "inner")

    def fh(Za, ya, Zb, Cc):
        g = np.column_stack([LogisticRegression(C=Cc, max_iter=LR_ITER).fit(Za, (ya > j).astype(int)).predict_proba(Zb)[:, 1]
                             for j in range(3)])
        P = np.clip(np.c_[1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]], 1e-6, None)
        return P / P.sum(1, keepdims=True)
    best = None
    for Cc in C_grid:
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = fh(Zi[ia], yt[ia], Zi[ib], Cc)
        ll = U.logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (Cc, ll, mo)
    w = U.tune_floor(best[2], yt, 0.18)
    return {o: (fh(Zi, yt, Z(chosen, o), best[0]) * w).argmax(1) for o in outs}, chosen


def super_noecho_original(r, k):
    """Reference: the same SUPER-TOP10-ORD recipe (original C values) on the ORIGINAL cached hospital-only members of
    its pool (no EchoNext member; the H-TAB members' inputs live with the EchoNext workspace and are absent). Read-only."""
    from research_experiments.phase_6.p6common import inner_oof
    from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
    from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
    from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
    from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx
    TR, TE, _ = Bd.splits(r, k); yt = Y[TR]; EFv = E.EF.astype(float)
    outer = {"ECGF-MLP": outer_ecgfmlp(r)[0], "ECGF-lin": B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]}
    c = {}
    for nm, base in (("ECGF-MLP", "ecgfmlp"), ("ECGF-lin", "ecgflindist")):
        tr_, P_ = inner_oof(base, r, k, H.SPEC[base]); assert np.array_equal(tr_, TR)
        c[nm] = {"inner": P_, "te": outer[nm][TE]}
    for m in ("M5", "M6"):
        j = E.member_job(m, r, k); c[m] = {"inner": j["inner"], "te": j["outer"]}
    for m in U.NEW:
        j = U.member(m, r, k); c[m] = {"inner": j["inner"], "te": j["outer"]}
    mu = Bd.mu_job(r, k); sd = float(np.std(mu["inner"] - EFv[TR]))
    c["EF-band"] = {"inner": band(mu["inner"], sd, TR), "inner_mu": mu["inner"] / 50, "te": band(mu["outer"], sd, TE), "te_mu": mu["outer"] / 50}
    p, chosen = super_top10_ord(c, TR, yt, r, k, 0.1, (0.01, 0.03, 0.1, 0.3, 1.0))
    return TE, p["te"], chosen


def members4(rows, sets):
    """The four TRAIN100 members the variants combine (same learners as train100_features): C1 members + IMG-vote."""
    X = T.inputs(rows)
    fit = {"ECGF-linear": T.linear(X["ecgf"], rows)[0], "ECGF-MLP": T.mlp3(X["ecgf"], rows)[0],
           "HDRCLIN-LR": T.logreg(X["hdrclin"], rows)[0], "FUSION-MLP": T.mlp3(X["fusion"], rows)[0]}
    P = {n: [f(idx) for idx in sets] for n, f in fit.items()}
    P["IMG-vote"] = [(P["ECGF-MLP"][i] + P["ECGF-linear"][i]) / 2 for i in range(len(sets))]
    return P


# ------------------------------------------------------------------ one outer fold, part "base"
def cell_base(r, k):
    from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
    from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as U
    H.WS = np.round(np.arange(0.3, 2.01, 0.1), 2)          # HIER-ext decision grid
    ef = B.table().ef.values.astype(float); bnd = B.table().sex_boundary.values; clean = C.verified()[0]
    Eemb = ecgf_f6().astype(float); IMG = T.clin()[0]
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k); yTR, yte = Y[TR], Y[te]
    fm = np.zeros(C.nD, bool); fm[TR] = True
    split = inner_split(r, k, TR); sets = [te, TR]
    res, info = {}, {}

    def put(name, Pte, Ptr=None, pte=None, ptr=None, **kw):
        pte = Pte.argmax(1) if pte is None else pte; ptr = (Ptr.argmax(1) if Ptr is not None else None) if ptr is None else ptr
        res[name] = dict(heldout_acc=acc(pte, yte), train_acc=None if ptr is None else acc(ptr, yTR),
                         heldout_recall=[float(np.mean(pte[yte == c] == c)) for c in range(4)],
                         heldout_confusion=np.array([[np.sum((yte == i) & (pte == j)) for j in range(4)] for i in range(4)]).tolist(), **kw)

    # ---- C1 members and IMG-vote members: outer fit + inner OOF (needed by E3 / Stack / E6 / HIER / ENH / ULT)
    P = members4(TR, sets)
    Pin = {m: np.zeros((len(TR), 4)) for m in ("FUSION-MLP", "ECGF-linear", "HDRCLIN-LR", "IMG-vote", "ECGF-MLP")}
    ein = np.zeros(len(TR))
    for ia, ib in split:
        Pi = members4(TR[ia], [TR[ib]])
        for m in Pin:
            Pin[m][ib] = Pi[m][0]
        ein[ib] = T.ef_models(TR[ia], [TR[ib]], clean, ef, Eemb, IMG)[0]
    e_te, e_tr = T.ef_models(TR, sets, clean, ef, Eemb, IMG)
    c1m = ("FUSION-MLP", "ECGF-linear", "HDRCLIN-LR")

    # ---- Phase 5
    X1 = make_X(["hdr", "ecgf", "feat:" + NEW], fm)[0]
    f, eps = mlp_seeds(X1, TR, Y, 4); put("E1 MLP, all feature families", f(te), f(TR), epochs=eps)
    X2 = make_X(["hdr", "feat:clin3+" + NEW, "ecgfpca:32"], fm)[0]
    f, _ = T.lgbm(X2, TR); put("E2 LightGBM (eng + ECGF-PCA)", f(te), f(TR))
    w = wavg_weights([Pin[m] for m in c1m], yTR); info["E3_weights"] = w.tolist()
    put("E3 weighted average", sum(wi * P[m][0] for wi, m in zip(w, c1m)), sum(wi * P[m][1] for wi, m in zip(w, c1m)))
    Zi = np.concatenate([lp(Pin[m]) for m in c1m], 1)
    st = LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER).fit(Zi, yTR)
    put("E3 stacking (LR, balanced)", st.predict_proba(np.concatenate([lp(P[m][0]) for m in c1m], 1)),
        st.predict_proba(np.concatenate([lp(P[m][1]) for m in c1m], 1)))
    Xf = make_X(["hdr", "ecgf"], fm)[0]
    yb = (Y > 0).astype(int)
    f1, e1 = mlp_seeds(Xf, TR, yb, 2)
    ab = TR[Y[TR] > 0]; f2, e2 = mlp_seeds(Xf, ab, Y - 1, 3)
    hier = lambda idx: np.c_[f1(idx)[:, :1], f1(idx)[:, 1:2] * f2(idx)]
    put("E4 hierarchical MLP", hier(te), hier(TR), epochs_stage1=e1, epochs_stage2=e2)
    sq, cols = feature_family("sq"); poor = np.nan_to_num(sq[:, list(cols).index("n_poor_leads")]) >= 2
    ws = {}
    for s in (False, True):
        mm = poor[TR] == s
        ws[s] = max(np.round(np.arange(0, 1.01, 0.1), 1), key=lambda w: (acc((w * Pin["FUSION-MLP"][mm] + (1 - w) * Pin["HDRCLIN-LR"][mm]).argmax(1), yTR[mm]), w)) if mm.any() else 0.5
    gate = lambda i, idx: np.where(poor[idx][:, None], ws[True], ws[False]) * P["FUSION-MLP"][i] + (1 - np.where(poor[idx][:, None], ws[True], ws[False])) * P["HDRCLIN-LR"][i]
    put("E6 signal-quality gating", gate(0, te), gate(1, TR), weights={"good": ws[False], "poor": ws[True]})
    f, eps = mlp_seeds(Xf, TR, Y, 4, mixup=0.2); put("E7 mixup MLP", f(te), f(TR), epochs=eps)

    # ---- Phase 4: EF regression and multitask (on ECGs with an EF value)
    E4 = E.EF; okr = TR[~np.isnan(E4[TR])]; okt = te[~np.isnan(E4[te])]
    fr, ep = mlp_reg(Xf, okr, E4, 42)
    ptr, pte_ = fr(okr), fr(okt)
    res["EF regression MLP (hdr + ECGF)"] = dict(train_r=float(np.corrcoef(ptr, E4[okr])[0, 1]), heldout_r=float(np.corrcoef(pte_, E4[okt])[0, 1]),
                                                train_mae=float(np.mean(np.abs(ptr - E4[okr]))), heldout_mae=float(np.mean(np.abs(pte_ - E4[okt]))), epochs=ep)
    res["EF regression ridge + LightGBM (triage)"] = dict(train_r=float(np.corrcoef(e_tr, ef[TR])[0, 1]), heldout_r=float(np.corrcoef(e_te, ef[te])[0, 1]),
                                                         train_mae=float(np.mean(np.abs(e_tr - ef[TR]))), heldout_mae=float(np.mean(np.abs(e_te - ef[te]))))
    fmt, ep = multitask(Xf, TR, Y, E4, 42)
    (Pte, Ete), (Ptr, Etr) = fmt(te), fmt(TR)
    put("Multitask MLP (class + EF, lambda 1)", Pte, Ptr, epochs=ep,
        heldout_ef_r=float(np.corrcoef(Ete[~np.isnan(E4[te])], E4[okt])[0, 1]))

    # ---- Phase 6B: HIER (LR stages, C = 1e4) + IMG-vote parts, decision tuned on inner OOF (ext grid)
    XH = make_X(["ecgf", "feat:" + H.FAMS], fm)[0]; Y3 = H.Y3; mid = np.isin(Y, (1, 2))
    in1 = np.zeros((len(TR), 3)); in2 = np.zeros(len(TR))
    for ia, ib in split:
        a, b = TR[ia], TR[ib]; am = a[mid[a]]
        in1[ib] = LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER).fit(XH[a], Y3[a]).predict_proba(XH[b])
        in2[ib] = LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER).fit(XH[am], (Y[am] == 2).astype(int)).predict_proba(XH[b])[:, 1]
    s1 = LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER).fit(XH[TR], Y3[TR]); tm = TR[mid[TR]]
    s2 = LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER).fit(XH[tm], (Y[tm] == 2).astype(int))
    vS1i, vqi = H.vote_parts(Pin["IMG-vote"]); S1i, qi = (vS1i + in1) / 2, (vqi + in2) / 2
    wm, wsv, t = H.tune(S1i, qi, yTR); info["HIER"] = [wm, wsv, t]
    hout = {}
    for nm, idx, Pv in (("te", te, P["IMG-vote"][0]), ("tr", TR, P["IMG-vote"][1])):
        vS1, vq = H.vote_parts(Pv); S1o, qo = (vS1 + s1.predict_proba(XH[idx])) / 2, (vq + s2.predict_proba(XH[idx])[:, 1]) / 2
        hout[nm] = (H.combine(S1o, qo, wm, wsv, t)[0], H.combine(S1o, qo, 1, 1, 0.5)[1])
    put("HIER (two-stage)", hout["te"][1], hout["tr"][1], pte=hout["te"][0], ptr=hout["tr"][0])
    H_in = H.combine(S1i, qi, 1, 1, 0.5)[1]

    # ---- Phase 6B: ENH members (LR C = 1e4; M6 ridge alpha 1e-6 + Gaussian bands) and ENH-vote
    Ins, Outs = {"M1": Pin["IMG-vote"]}, {"M1": (P["IMG-vote"][0], P["IMG-vote"][1])}
    names = {"M2": "ENH M2 synthetic 10-s -> ECGFounder", "M3": "ENH M3 median-beat shape", "M4": "ENH M4 MiniRocket",
             "M5": "ENH M5 LR on HIER inputs", "M6": "ENH M6 EF ridge -> bands"}
    for m in ("M2", "M3", "M4", "M5", "M6"):
        X = E.features(m, r, k, fm).astype(np.float32 if m == "M4" else np.float64); inn = np.zeros((len(TR), 4))
        if m == "M6":
            mu_in = np.zeros(len(TR))
            for ia, ib in split:
                mu_in[ib] = Ridge(alpha=1e-6).fit(X[TR[ia]], E4[TR[ia]]).predict(X[TR[ib]])
            sd = float(np.sqrt(np.mean((mu_in - E4[TR]) ** 2))); inn = E.ef_probs(mu_in, sd)
            rg = Ridge(alpha=1e-6).fit(X[TR], E4[TR]); o = (E.ef_probs(rg.predict(X[te]), sd), E.ef_probs(rg.predict(X[TR]), sd))
            MU = dict(inner=mu_in, te=rg.predict(X[te]), tr=rg.predict(X[TR]))      # = bands.mu_job (ridge on the M6 inputs)
        else:
            for ia, ib in split:
                inn[ib] = lr_fit(X, TR[ia], Y)(TR[ib])
            fo = lr_fit(X, TR, Y); o = (fo(te), fo(TR))
        Ins[m], Outs[m] = inn, o; put(names[m], o[0], o[1])
    vi = np.mean([Ins[m] for m in ("M1", "M2", "M3", "M4", "M5", "M6")], 0)
    wv = E.tune(vi, yTR); info["ENH_vote_weights"] = wv.tolist()
    vo = [np.mean([Outs[m][i] for m in ("M1", "M2", "M3", "M4", "M5", "M6")], 0) for i in (0, 1)]
    put("ENH-vote", vo[0], vo[1], pte=(vo[0] * wv).argmax(1), ptr=(vo[1] * wv).argmax(1))

    # ---- Phase 6B: ULT members (M8 LR, M9 kNN, M10 HGB without L2 / early stopping, M12 zone LR) and ULT-stack
    zone = U.ZONE
    for m, kind in (("M8", "lr"), ("M9", "knn"), ("M10", "hgb"), ("M12", "lr")):
        X = U.feats(m, fm); ok = zone if m == "M12" else np.ones(C.nD, bool)

        def fitp(rows, idx_list):
            rows = rows[ok[rows]]
            mdl = (LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=LR_ITER) if kind == "lr" else
                   KNeighborsClassifier(n_neighbors=15, weights="distance") if kind == "knn" else
                   HistGradientBoostingClassifier(max_depth=3, learning_rate=0.05, max_iter=HGB_ITER, l2_regularization=0.0,
                                                  early_stopping=False, class_weight="balanced", random_state=0)).fit(X[rows], Y[rows])
            return [pad4(mdl, mdl.predict_proba(X[i])) for i in idx_list]
        inn = np.zeros((len(TR), 4))
        for ia, ib in split:
            inn[ib] = fitp(TR[ia], [TR[ib]])[0]
        o = tuple(fitp(TR, [te, TR])); Ins[m], Outs[m] = inn, o
        put({"M8": "ULT M8 LR multi-scale ECGF", "M9": "ULT M9 kNN (k 15, distance)", "M10": "ULT M10 gradient boosting",
             "M12": "ULT M12 zone LR"}[m], o[0], o[1])
    Ins["H"] = H_in
    Outs["H"] = tuple(hout[nm][1] for nm in ("te", "tr"))
    order = list(Ins)
    Zi = np.c_[np.concatenate([lp(Ins[n]) for n in order], 1), MU["inner"] / 50, (MU["inner"] / 50) ** 2]
    meta = LogisticRegression(C=BIG_C, max_iter=LR_ITER).fit(Zi, yTR)
    Zo = [np.c_[np.concatenate([lp(Outs[n][i]) for n in order], 1), e / 50, (e / 50) ** 2] for i, e in ((0, MU["te"]), (1, MU["tr"]))]
    mo = np.zeros((len(TR), 4))
    for ia, ib in StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yTR, C.PAT[TR]):
        mo[ib] = LogisticRegression(C=BIG_C, max_iter=LR_ITER).fit(Zi[ia], yTR[ia]).predict_proba(Zi[ib])
    wt = U.tune_floor(mo, yTR, 0.18); info["ULT_weights"] = wt.tolist()
    so = [meta.predict_proba(z) for z in Zo]
    put("ULT-stack (no EchoNext)", so[0], so[1], pte=(so[0] * wt).argmax(1), ptr=(so[1] * wt).argmax(1))

    # ---- best-for-50s recipe without EchoNext: SUPER-TOP10-ORD on the hospital-only members of its pool
    sd = float(np.std(MU["inner"] - E4[TR]))
    cands = {"ECGF-MLP": dict(inner=Pin["ECGF-MLP"], te=P["ECGF-MLP"][0], tr=P["ECGF-MLP"][1]),
             "ECGF-lin": dict(inner=Pin["ECGF-linear"], te=P["ECGF-linear"][0], tr=P["ECGF-linear"][1]),
             **{m: dict(inner=Ins[m], te=Outs[m][0], tr=Outs[m][1]) for m in ("M5", "M6", "M8", "M9", "M10", "M12")},
             "EF-band": dict(inner=band(MU["inner"], sd, TR), inner_mu=MU["inner"] / 50, te=band(MU["te"], sd, te),
                             te_mu=MU["te"] / 50, tr=band(MU["tr"], sd, TR), tr_mu=MU["tr"] / 50)}
    p, chosen = super_top10_ord(cands, TR, yTR, r, k, BIG_C, (BIG_C,), outs=("te", "tr"))
    put("SUPER-TOP10-ORD without EchoNext", np.eye(4)[p["te"]], np.eye(4)[p["tr"]], pte=p["te"], ptr=p["tr"], members=chosen)
    TE0, p0, ch0 = super_noecho_original(r, k); assert np.array_equal(TE0, te)
    put("SUPER-TOP10-ORD without EchoNext, ORIGINAL recipe (reference)", np.eye(4)[p0], None, pte=p0, members=ch0)
    return res, info


# ------------------------------------------------------------------ one outer fold, part "labels" (C3 recipe)
def label_defs():
    from research_experiments.phase_6b_threshold_sampling_lstm import clear, hc30, mmclean, verified
    Fl = mmclean.flags()
    return dict(
        y={"VER": verified.Y_VER},
        excl={"HC30": ~hc30.HC, "CLEAR": clear.BORDER, "MMC": Fl.MMC.values, "NF": Fl.NF.values, "NFS": Fl.NFS.values},
        subsets={"all_837": np.ones(C.nD, bool), "HC_395": hc30.HC, "clear_cut": ~clear.BORDER, "border": clear.BORDER,
                 "kept_MMC": ~Fl.MMC.values, "NF_clear": ~Fl.NF.values & ~Fl.near_boundary.values},
        y_eval={"VER": verified.Y_VER})


def c1_members(rows, sets, y):
    """C1 members (FUSION-MLP 3 seeds, ECGF-linear, HDRCLIN-LR) fitted on `rows` with labels y; mean = C1 vote."""
    X = T.inputs(rows); Yb = T.Y
    try:
        T.Y = y                                               # the learners read the module-level label vector
        fus = T.mlp3(X["fusion"], rows)
        fs = [fus[0], T.linear(X["ecgf"], rows)[0], T.logreg(X["hdrclin"], rows)[0]]
    finally:
        T.Y = Yb
    return [np.mean([f(idx) for f in fs], 0) for idx in sets], fus[1]["epochs"]


def cell_labels(r, k):
    D = label_defs(); fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k)
    split = inner_split(r, k, TR); out = {}
    for cfg in LABEL_CONFIGS:
        y = D["y"].get(cfg, Y); ex = D["excl"].get(cfg, np.zeros(C.nD, bool))
        fit = TR[~ex[TR]]
        (Pte, Ptr), fus_epochs = c1_members(fit, [te, TR], y)
        Pin = np.zeros((len(TR), 4))
        for ia, ib in split:
            Pin[ib] = c1_members(TR[ia][~ex[TR[ia]]], [TR[ib]], y)[0][0]
        keep = ~ex[TR]; prior = np.bincount(y[TR][keep], minlength=4) / keep.sum()
        a = choose_alpha(Pin[keep], y[TR][keep], prior, "protected")
        pte, ptr = decide(Pte, "prior", a, prior), decide(Ptr, "prior", a, prior)
        sub = {}
        for sname, mask in D["subsets"].items():
            mt = mask[te]
            sub[sname] = dict(n=int(mt.sum()), correct_orig=int(np.sum(pte[mt] == Y[te][mt])),
                              **({"correct_own": int(np.sum(pte[mt] == y[te][mt]))} if cfg in D["y_eval"] else {}))
        out[cfg] = dict(alpha=float(a), train_acc=acc(ptr[keep], y[TR][keep]), n_train_rows=int(keep.sum()), subsets=sub,
                        fusion_mlp_epochs=[int(e) for e in fus_epochs])
    return out


def cell(task):
    (r, k), parts, outdir = task
    from pathlib import Path
    f = Path(outdir) / "cells" / f"variants_{'_'.join(parts)}_r{r}_k{k}.json"      # aggregates only; resume after restarts
    if f.exists():
        return json.load(open(f))
    warnings.filterwarnings("ignore"); torch.set_num_threads(1); t0 = time.time(); out = dict(repeat=r, fold=k)
    if "base" in parts:
        out["base"], out["info"] = cell_base(r, k)
    if "labels" in parts:
        out["labels"] = cell_labels(r, k)
    out["minutes"] = (time.time() - t0) / 60
    msg = " | ".join(f"{m} {v['train_acc']:.3f}/{v['heldout_acc']:.3f}" for m, v in out.get("base", {}).items()
                     if v.get("train_acc") is not None and m.startswith(("E1", "HIER", "ENH-vote", "ULT-stack")))
    print(f"[r{r} k{k}] {out['minutes']:.1f} min | {msg}", flush=True)
    f.parent.mkdir(parents=True, exist_ok=True); json.dump(out, open(f, "w"), default=float)
    return out


def summarise(cells):
    S = {}
    if "base" in cells[0]:
        for m in cells[0]["base"]:
            c = [x["base"][m] for x in cells]
            S[m] = {key: dict(mean=float(np.mean([v[key] for v in c])), ci95_corrected=[float(z) for z in corr_ci([v[key] for v in c])])
                    for key in c[0] if isinstance(c[0][key], float)}
            if "heldout_confusion" in c[0]:
                S[m]["heldout_confusion_mean_per_repeat"] = (np.sum([v["heldout_confusion"] for v in c], 0) / 3).round(1).tolist()
                S[m]["heldout_recall_N_Mi_Mo_S"] = np.mean([v["heldout_recall"] for v in c], 0).round(3).tolist()
    if "labels" in cells[0]:
        L = {}
        for cfg in LABEL_CONFIGS:
            d = dict(train_acc=float(np.mean([x["labels"][cfg]["train_acc"] for x in cells])))
            ep = sum([x["labels"][cfg].get("fusion_mlp_epochs", []) for x in cells], [])
            if ep:
                d["fusion_mlp_epochs_median"] = float(np.median(ep)); d["fusion_mlp_epochs_range"] = [int(min(ep)), int(max(ep))]
            for sname in cells[0]["labels"][cfg]["subsets"]:
                for key in ("correct_orig", "correct_own"):
                    if key not in cells[0]["labels"][cfg]["subsets"][sname]:
                        continue
                    per_rep = []
                    for r in (0, 1, 2):
                        cc = [x["labels"][cfg]["subsets"][sname] for x in cells if x["repeat"] == r]
                        if cc:
                            per_rep.append(sum(z[key] for z in cc) / max(1, sum(z["n"] for z in cc)))
                    d[f"accuracy_{sname}_{'orig_labels' if key == 'correct_orig' else 'own_labels'}"] = float(np.mean(per_rep))
            L[cfg] = d
        S["label_studies (C3 recipe)"] = L
    return S


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--cells", default="all"); ap.add_argument("--parts", default="base,labels")
    a = ap.parse_args(); parts = tuple(a.parts.split(","))
    C.verified(); B.table(); T.clin()
    cells_ = [(r, k) for r in (0, 1, 2) for k in range(5)] if a.cells == "all" else [tuple(map(int, c.split(":"))) for c in a.cells.split(",")]
    with Pool(a.jobs) as pool:
        cells = list(pool.imap_unordered(cell, [(c, parts, a.out) for c in cells_]))
    cells.sort(key=lambda c: (c["repeat"], c["fold"]))
    tag = "_".join(parts)
    json.dump(dict(amendment="AMENDMENT_TRAIN100.md (addendum D)", parts=parts, n_cells=len(cells), summary=summarise(cells), cells=cells),
              open(f"{a.out}/train100_variants_{tag}.json", "w"), indent=1, default=float)
    print(json.dumps(summarise(cells), indent=1, default=float)[:6000])


if __name__ == "__main__":
    main()
