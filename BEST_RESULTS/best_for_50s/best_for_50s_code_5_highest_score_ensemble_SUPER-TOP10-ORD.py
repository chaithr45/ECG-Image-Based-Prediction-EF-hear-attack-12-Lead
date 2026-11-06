# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/public_ef/en_super.py
# >>> Run from the repository root: python -m research_experiments.public_ef.en_super --labels original
# >>> Part of BEST_RESULTS/best_for_50s; see the README there.

"""PUBLIC_EF_PLAN Amendment 4: SUPER ensemble of all image-derived members with greedy pruning inside the folds (A), and
the echo-consistent-label sensitivity analysis (B). CPU. Per-ECG outputs stay private; results are aggregates.
    python -m research_experiments.public_ef.en_super --labels original
    python -m research_experiments.public_ef.en_super --labels echo
"""
import argparse
import json
import time
import warnings

import lightgbm as lgb
import numpy as np

from research_experiments.phase_6b_threshold_sampling_lstm import bands as Bd
from research_experiments.phase_6b_threshold_sampling_lstm import enh as E
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

warnings.filterwarnings("ignore")
MAX_MEMBERS, MIN_GAIN, C_SEL = 10, 0.002, 0.1
Y_ORIGINAL = None


def echo_labels():
    NC = Bd.normal_cut(52.0, 2.0); EF = E.EF.astype(float)
    return np.select([EF > NC, EF >= 41, EF >= 31], [0, 1, 2], 3).astype(B.Y.dtype)


def setup(labels):
    """'echo': replace the shared label array in place and send every hospital-label cache to new folders BEFORE any
    member is computed (the original-label caches are never read)."""
    from research_experiments.phase_6 import p6common as P6
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL
    from research_experiments.public_ef.en_prep import PRIV
    tag = "" if labels == "original" else "_echolabels"
    if labels == "echo":
        yc = echo_labels()
        changes = {f"{B.CLASSES[a]}->{B.CLASSES[b]}": int(((B.Y == a) & (yc == b)).sum()) for a in range(4) for b in range(4)
                   if a != b and ((B.Y == a) & (yc == b)).any()}
        global Y_ORIGINAL
        Y_ORIGINAL = B.Y.copy()                                            # Amendment 6: also score against these
        B.Y[:] = yc                                                        # one shared array (checked) -> every module
        UL.ZONE[:] = np.isin(B.Y, (1, 2)) | ((B.Y == 0) & (E.EF < 62))
        P6.INNER6 = P6.PRIV6 / "inner_echolabels"
        E.MEM = E.D / "members_echolabels"
        UL.D = B.PRIV / "ult_echolabels"
        Bd.D = E.D / "bands_echolabels"                                    # inner folds are stratified by label
    else:
        changes = {}
    return PRIV / f"super{tag}", changes


# ------------------------------------------------------------------------------------------------ hospital members
def outer_fit(name, r):
    """Outer OOF of ECGF-MLP / ECGF-lin retrained on the current labels (Phase-2 inner split, as the originals)."""
    from research_experiments.phase_5.experiments.nested import BASES
    from research_experiments.phase_6b_threshold_sampling_lstm.existing6b import fit_member
    P = np.zeros((B.nD, 4))
    for k in range(5):
        itr, iva, ote = B.split_indices(r, k)
        fm = np.zeros(B.nD, bool); fm[itr] = True
        if name == "ECGF-MLP":
            X, pre = B.make_X(["ecgf"], fm)
            P[ote] = np.mean([fit_member("mlpfusion", X[itr], B.Y[itr], X[iva], B.Y[iva], s, pre)(X[ote]) for s in B.SEEDS], 0)
        else:
            X, pre = B.make_X(BASES["ecgflindist"][0], fm)
            P[ote] = fit_member("ecgflindist", X[itr], B.Y[itr], X[iva], B.Y[iva], 42, pre)(X[ote])
    return P


def outer_pair(r, labels, cache):
    f = cache / f"outer_r{r}.npz"
    if f.exists():
        z = np.load(f); return z["mlp"], z["lin"]
    if labels == "original":
        from research_experiments.phase_6b_threshold_sampling_lstm.imgonly import P2, outer_ecgfmlp, sfx
        mlp = outer_ecgfmlp(r)[0]; lin = B.load_oof_csv(P2 / f"ecgf_f6_lin_cedist1{sfx(r)}" / "oof.csv")[0]
    else:
        mlp, lin = outer_fit("ECGF-MLP", r), outer_fit("ECGF-lin", r)
    cache.mkdir(parents=True, exist_ok=True); np.savez(f, mlp=mlp, lin=lin)
    return mlp, lin


def tab_job(kind, r, k, cache):
    """H-TAB-LR / H-TAB-GBM on the hospital plain G2-G4 features; inner OOF on the Bd.splits inner folds + outer."""
    f = cache / f"{kind}_r{r}_k{k}.npz"
    if f.exists():
        z = np.load(f); return {x: z[x] for x in z.files}
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL
    from research_experiments.public_ef import en_image as EI
    h = EI.hospital_inputs("plain")
    TR, TE, inner = Bd.splits(r, k)
    raw = [np.asarray(h[x], float) for x in ("la", "ls", "tab")]
    Y = B.Y

    def fit_pred(tr_idx, te_idx, c):
        if kind == "H-TAB-LR":
            bs = EI.BlockStd().fit([b[tr_idx] for b in raw])
            m = EI._lr(c).fit(bs.transform([b[tr_idx] for b in raw]), Y[tr_idx])
            return m.predict_proba(bs.transform([b[te_idx] for b in raw]))
        G = np.c_[raw[0], raw[1], raw[2]]
        return lgb.LGBMClassifier(n_estimators=300, **EI.GBM).fit(G[tr_idx], Y[tr_idx]).predict_proba(G[te_idx])

    grid = EI.CS_ALL if kind == "H-TAB-LR" else (None,)
    best = None
    for c in grid:
        P = np.zeros((len(TR), 4))
        for ia, ib in inner:
            P[ib] = fit_pred(TR[ia], TR[ib], c)
        ll = UL.logloss(Y[TR], P)
        if best is None or ll < best[1]:
            best = (c, ll, P)
    out = dict(inner=best[2], outer=fit_pred(TR, TE, best[0]))
    cache.mkdir(parents=True, exist_ok=True); np.savez(f, **out)
    return out


def pix_zs():
    """PIX-ZS: the IMG_PLAN EchoNext-pre-trained ResNet-18 on the 837 hospital images (no hospital training)."""
    from research_experiments.public_ef.en_prep import PRIV
    w, path = PRIV / "img_en_weights.pt", PRIV / "img_hospital_ordered.npy"
    if not (w.exists() and path.exists()):
        return None
    import torch
    from research_experiments.public_ef import img_train as IT
    m = IT.Net(imagenet=False).to(IT.DEV); m.load_state_dict(torch.load(w, map_location=IT.DEV, weights_only=True))
    return IT.predict(m, path, np.arange(B.nD), B.Y, E.EF.astype(float))


# ------------------------------------------------------------------------------------------------------- the stack
def _Z(P_list, mu_list):
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL
    return np.column_stack([np.concatenate([UL.lp(p) for p in P_list], 1), *mu_list]) if mu_list else \
        np.concatenate([UL.lp(p) for p in P_list], 1)


SELECT = "greedy"                                                      # Amendment 9: or "top10"


def super_fold(r, k, cands):
    """cands: name -> (inner P (len TR), outer P (len TE), inner mu/50 or None, outer mu/50 or None)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.model_selection import StratifiedGroupKFold
    from research_experiments.phase_6.p6common import PAT_GROUP
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL
    TR, TE, _ = Bd.splits(r, k)
    yt = B.Y[TR]
    sp = list(StratifiedGroupKFold(4, shuffle=True, random_state=3000 + 10 * r + k).split(TR, yt, PAT_GROUP()[TR]))

    def zin(names):
        return _Z([cands[n][0] for n in names], [cands[n][2] for n in names if cands[n][2] is not None])

    def mcv(Z, C):
        mo = np.zeros((len(TR), 4))
        for ia, ib in sp:
            mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Z[ia], yt[ia]).predict_proba(Z[ib])
        return mo

    def score(names):
        mo = mcv(zin(names), C_SEL)
        return float(np.mean((mo * UL.tune_floor(mo, yt, 0.18)).argmax(1) == yt))

    chosen, best = [], -1.0
    if SELECT == "top10":                                                 # Amendment 9: keep the 10 best single members
        single = {n: score([n]) for n in cands}
        chosen = sorted(single, key=lambda n: -single[n])[:10]
        best = score(chosen)
    while SELECT == "greedy" and len(chosen) < MAX_MEMBERS:
        trial = {n: score(chosen + [n]) for n in cands if n not in chosen}
        n_best = max(trial, key=trial.get)
        if trial[n_best] < best + MIN_GAIN:
            break
        chosen.append(n_best); best = trial[n_best]
    Zi = zin(chosen)
    Zo = _Z([cands[n][1] for n in chosen], [cands[n][3] for n in chosen if cands[n][3] is not None])
    fin = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = mcv(Zi, C); ll = UL.logloss(yt, mo)
        if fin is None or ll < fin[1]:
            fin = (C, ll, mo)
    so = LogisticRegression(C=fin[0], max_iter=5000).fit(Zi, yt).predict_proba(Zo)
    p4 = (so * UL.tune_floor(fin[2], yt, 0.18)).argmax(1)
    staged = staged_fold(Zi, Zo, yt, sp)                                  # Amendment 5 (same members, same meta inputs)
    staged["ord4"] = ordinal_fold(Zi, Zo, yt, sp)                         # Amendment 7
    return TE, p4, chosen, best, staged


def ordinal_fold(Zi, Zo, yt, sp):
    """Amendment 7: Frank-Hall ordinal meta-learner (P(y > j), j = 0, 1, 2) on the same meta inputs; one C by meta-CV
    log-loss; ULT decision rule on the meta-CV out-of-fold probabilities."""
    from sklearn.linear_model import LogisticRegression
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL

    def fh(Za, ya, Zb, C):
        g = np.column_stack([LogisticRegression(C=C, max_iter=5000).fit(Za, (ya > j).astype(int)).predict_proba(Zb)[:, 1]
                             for j in range(3)])
        P = np.clip(np.c_[1 - g[:, 0], g[:, 0] - g[:, 1], g[:, 1] - g[:, 2], g[:, 2]], 1e-6, None)
        return P / P.sum(1, keepdims=True)
    best = None
    for C in (0.01, 0.03, 0.1, 0.3, 1.0):
        mo = np.zeros((len(yt), 4))
        for ia, ib in sp:
            mo[ib] = fh(Zi[ia], yt[ia], Zi[ib], C)
        ll = UL.logloss(yt, mo)
        if best is None or ll < best[1]:
            best = (C, ll, mo)
    return (fh(Zi, yt, Zo, best[0]) * UL.tune_floor(best[2], yt, 0.18)).argmax(1)


# ------------------------------------------------------------------- Amendment 5: wider middle band and staged decision
W3 = np.array([(1.0, a, b) for a in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4) for b in (0.25, 0.5, 0.75, 1, 1.5, 2, 3, 4)])


def to3(y):
    """Normal / Mild+Moderate / Severe (Severe unchanged)."""
    return np.select([y == 0, y == 3], [0, 2], 1)


def tune_k(P, y, k, floor=0.18):
    best = None
    for fl in (floor, 0.15):
        for w in W3:
            p = (P * w).argmax(1)
            rec = [np.mean(p[y == c] == c) for c in range(k)]
            if min(rec) >= fl:
                a = float(np.mean(p == y))
                if best is None or a > best[0]:
                    best = (a, w)
        if best is not None:
            return best[1]
    return np.ones(k)


def staged_fold(Zi, Zo, yt, sp):
    from sklearn.linear_model import LogisticRegression
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL

    def fit_c(Z, y, k, rows=None):
        rows = np.arange(len(y)) if rows is None else rows
        best = None
        for C in (0.01, 0.03, 0.1, 0.3, 1.0):
            mo = np.zeros((len(y), k))
            for ia, ib in sp:
                a_ = np.intersect1d(ia, rows)
                mo[ib] = LogisticRegression(C=C, max_iter=5000).fit(Z[a_], y[a_]).predict_proba(Z[ib])
            ll = UL.logloss(y[rows], mo[rows])
            if best is None or ll < best[1]:
                best = (C, ll, mo)
        return best
    y3 = to3(yt)
    C3, _, mo3 = fit_c(Zi, y3, 3)
    w3 = tune_k(mo3, y3, 3)
    d3_in = (mo3 * w3).argmax(1)
    s3 = LogisticRegression(C=C3, max_iter=5000).fit(Zi, y3).predict_proba(Zo)
    d3_out = (s3 * w3).argmax(1)
    mid = np.flatnonzero(np.isin(yt, (1, 2)))
    yb = (yt == 2).astype(int)
    Cb, _, mob = fit_c(Zi, yb, 2, rows=mid)
    pm_in = mob[:, 1]
    best = None
    for fl in (0.18, 0.15):
        for t in np.linspace(0.05, 0.95, 37):
            p4 = np.where(d3_in == 0, 0, np.where(d3_in == 2, 3, np.where(pm_in >= t, 2, 1)))
            rec = [np.mean(p4[yt == c] == c) for c in range(4)]
            if min(rec) >= fl:
                a = float(np.mean(p4 == yt))
                if best is None or a > best[0]:
                    best = (a, t)
        if best is not None:
            break
    t = best[1] if best else 0.5
    pm_out = LogisticRegression(C=Cb, max_iter=5000).fit(Zi[mid], yb[mid]).predict_proba(Zo)[:, 1]
    p4_out = np.where(d3_out == 0, 0, np.where(d3_out == 2, 3, np.where(pm_out >= t, 2, 1)))
    return dict(three_class=d3_out, staged4=p4_out)


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--labels", choices=["original", "echo"], default="original")
    ap.add_argument("--warm", action="store_true", help="only compute the hospital-trained member caches")
    ap.add_argument("--repeat", type=int, default=None, help="warm only this repeat (parallel warm-up)")
    ap.add_argument("--pool", choices=["base", "X"], default="base", help="X: Amendment 8 members added")
    ap.add_argument("--select", choices=["greedy", "top10"], default="greedy", help="top10: Amendment 9")
    a = ap.parse_args()
    cache, changes = setup(a.labels)
    global SELECT
    SELECT = a.select
    from research_experiments.phase_5.p5common import corr_ci, corr_t, holm
    from research_experiments.phase_6.p6common import inner_oof
    from research_experiments.phase_6b_threshold_sampling_lstm import hier as H
    from research_experiments.phase_6b_threshold_sampling_lstm import ult as UL
    if a.warm:
        t0 = time.time()
        for r in ((0, 1, 2) if a.repeat is None else (a.repeat,)):
            outer_pair(r, a.labels, cache)
            for k in range(5):
                for base in ("ecgfmlp", "ecgflindist"):
                    inner_oof(base, r, k, H.SPEC[base])
                for m in ("M5", "M6"):
                    E.member_job(m, r, k)
                for m in UL.NEW:
                    UL.member(m, r, k)
                for m in ("H-TAB-LR", "H-TAB-GBM"):
                    tab_job(m, r, k, cache)
                Bd.mu_job(r, k)
                print(f"WARM {a.labels} r{r} k{k} {time.time() - t0:.0f}s", flush=True)
        return
    from research_experiments.public_ef import en_image as EI
    from research_experiments.public_ef import en_train as T
    from research_experiments.public_ef.en_prep import PRIV, RES
    t0 = time.time()
    Y, EF = B.Y, E.EF.astype(float)
    z = np.load(PRIV / "en_cache.npz", allow_pickle=False)
    en = dict(f6a=z["f6_aligned"], f6s=z["f6_synth"], la=z["logits_aligned"], ls=z["logits_synth"], tab=z["tab"], y=z["y"], ef=z["ef"], g=z["patient_key"])
    _, Pen, mu_en = T.train_en()
    _, outs = EI.fit_en_all(en, {v: EI.hospital_inputs(v) for v in ("H", "plain")})
    pz = pix_zs()
    fixed = {"EN-cls": (Pen, None), "EN-band": (None, mu_en)}
    for v in ("H", "plain"):
        fixed[f"EN-cls-all-{v}"] = (outs[v]["P_cls"], None); fixed[f"EN-gbm-{v}"] = (outs[v]["P_gbm"], None)
        fixed[f"EN-band-all-{v}"] = (None, outs[v]["mu"])
    if pz is not None:
        fixed["PIX-ZS"] = (pz[0], None); fixed["PIX-ZS-band"] = (None, pz[1])
    internal_x = None
    if a.pool == "X":                                                       # Amendment 8
        from research_experiments.public_ef import en_featx as FXM
        ox, internal_x = FXM.en_members()
        for n in ("EN-cls-all-X", "EN-gbm-X", "EN-MiMo-LR", "EN-MiMo-GBM"):
            fixed[n] = (ox[n], None)
        fixed["EN-mu-mid"] = (None, ox["EN-mu-mid"])
    out = dict(labels=a.labels, pool=a.pool, select=a.select, label_changes=changes, n_echonext=int(len(en["y"])), pixel_member=pz is not None,
               internal_x=internal_x, models={}, selection_frequency={}, comparisons=[], selected_per_fold=[])
    preds, cells = {n: [] for n in ("SUPER", "stack-BASE", "STAGED-4", "SUPER-ORD")}, {n: [] for n in ("SUPER", "stack-BASE", "STAGED-4", "SUPER-ORD")}
    p3_all, c3_all = [], []
    for r in (0, 1, 2):
        mlp, lin = outer_pair(r, a.labels, cache)
        p_sup, p_base = np.full(B.nD, -1), np.full(B.nD, -1)
        p_stg, p_3, p_ord = np.full(B.nD, -1), np.full(B.nD, -1), np.full(B.nD, -1)
        for k in range(5):
            TR, TE, _ = Bd.splits(r, k)
            c = {}
            for nm, base, outer in (("ECGF-MLP", "ecgfmlp", mlp), ("ECGF-lin", "ecgflindist", lin)):
                tr_, P_ = inner_oof(base, r, k, H.SPEC[base]); assert np.array_equal(tr_, TR)
                c[nm] = (P_, outer[TE], None, None)
            for m in ("M5", "M6"):
                j = E.member_job(m, r, k); c[m] = (j["inner"], j["outer"], None, None)
            for m in UL.NEW:
                j = UL.member(m, r, k); c[m] = (j["inner"], j["outer"], None, None)
            for m in ("H-TAB-LR", "H-TAB-GBM"):
                j = tab_job(m, r, k, cache); c[m] = (j["inner"], j["outer"], None, None)
            if a.pool == "X":
                for m in ("H-X-GBM", "H-MiMo-GBM"):
                    j = FXM.hosp_job(m, r, k, cache); c[m] = (j["inner"], j["outer"], None, None)
            mu = Bd.mu_job(r, k); sd = float(np.std(mu["inner"] - EF[TR]))
            c["EF-band"] = (T.band(mu["inner"], sd, TR), T.band(mu["outer"], sd, TE), mu["inner"] / 50, mu["outer"] / 50)
            for n, (P_, m_) in fixed.items():
                if P_ is not None:
                    c[n] = (P_[TR], P_[TE], None, None)
                else:
                    s_ = float(np.std(m_[TR] - EF[TR]))
                    c[n] = (T.band(m_[TR], s_, TR), T.band(m_[TE], s_, TE), m_[TR] / 50, m_[TE] / 50)
            te, p, chosen, sc, stg = super_fold(r, k, c)
            p_sup[te] = p; p_stg[te] = stg["staged4"]; p_3[te] = stg["three_class"]; p_ord[te] = stg["ord4"]
            out["selected_per_fold"].append(dict(repeat=r, fold=k, members=chosen, inner_meta_cv_accuracy=sc))
            for n in chosen:
                out["selection_frequency"][n] = out["selection_frequency"].get(n, 0) + 1
            te, p = T.fold(r, k, (mlp + lin) / 2, Pen, mu_en, False); p_base[te] = p
            print(f"r{r} k{k}: {len(chosen)} members {chosen} inner {sc:.3f}  {time.time() - t0:.0f}s", flush=True)
        fo = B.fold_vec(r)
        for n, p in (("SUPER", p_sup), ("stack-BASE", p_base), ("STAGED-4", p_stg), ("SUPER-ORD", p_ord)):
            preds[n].append(p); cells[n].extend(float(np.mean(p[fo == k] == Y[fo == k])) for k in range(5))
    if a.labels == "original":
        preds["C3 (with header; reference)"], cells["C3 (with header; reference)"] = [], []
        for r in (0, 1, 2):
            p = B.load_run(f"P6A-c3-B1-v1__s0_r{r}")[1]; fo = B.fold_vec(r)
            preds["C3 (with header; reference)"].append(p)
            cells["C3 (with header; reference)"].extend(float(np.mean(p[fo == k] == Y[fo == k])) for k in range(5))
    if Y_ORIGINAL is not None:                                             # Amendment 6: echo-trained, original-label scoring
        out["echo_trained_scored_on_original_labels"] = {}
        for n in ("SUPER", "STAGED-4", "SUPER-ORD", "stack-BASE"):
            ms = [T.metrics(Y_ORIGINAL, p) for p in preds[n]]
            cl = [float(np.mean(p[B.fold_vec(r) == k] == Y_ORIGINAL[B.fold_vec(r) == k])) for r, p in enumerate(preds[n]) for k in range(5)]
            s = {k_: float(np.mean([m[k_] for m in ms])) for k_ in ms[0]}
            s.update(accuracy_ci95_corrected=corr_ci(cl), admissible=bool(s["min_recall"] >= 0.15))
            out["echo_trained_scored_on_original_labels"][n] = s
            print(f"RESULT echo-trained, ORIGINAL-label scoring: {n:12s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} "
                  f"rec {[round(s[f'recall_{c}'], 2) for c in B.CLASSES]} adm {s['admissible']}", flush=True)
    for n, ps in preds.items():
        ms = [T.metrics(Y, p) for p in ps]
        s = {k_: float(np.mean([m[k_] for m in ms])) for k_ in ms[0]}
        from research_experiments.public_ef.img_train import macro_f1
        s.update(macro_f1=float(np.mean([macro_f1(Y, p) for p in ps])), accuracy_ci95_corrected=corr_ci(cells[n]),
                 accuracy_per_repeat=[round(m["accuracy"], 4) for m in ms], admissible=bool(s["min_recall"] >= 0.15))
        out["models"][n] = s
    pv = []
    for x, yv in [("SUPER", "stack-BASE"), ("STAGED-4", "SUPER"), ("SUPER-ORD", "SUPER")] + ([("SUPER", "C3 (with header; reference)")] if a.labels == "original" else []):
        d = np.array(cells[x]) - np.array(cells[yv]); md, p_ = corr_t(d); pv.append(p_)
        out["comparisons"].append(dict(model=x, vs=yv, delta_accuracy=md, p=p_,
                                       same_sign_all_repeats=bool(len({np.sign(d[5 * r:5 * r + 5].mean()) for r in range(3)}) == 1)))
    for c_, h in zip(out["comparisons"], holm(pv)):
        c_["p_holm"] = float(h)
    out["minutes"] = round((time.time() - t0) / 60, 1)
    RES.mkdir(parents=True, exist_ok=True)
    (RES / f"en_super_{a.labels}{'_X' if a.pool == 'X' else ''}{'_top10' if a.select == 'top10' else ''}.json").write_text(json.dumps(out, indent=1, default=float))
    for n, s in out["models"].items():
        print(f"RESULT {a.labels}: {n:28s} acc {s['accuracy']:.3f} {np.round(s['accuracy_ci95_corrected'], 3)} rec "
              f"{[round(s[f'recall_{c}'], 2) for c in B.CLASSES]} F1 {s['macro_f1']:.3f} adm {s['admissible']}", flush=True)
    print("RESULT selection frequency:", dict(sorted(out["selection_frequency"].items(), key=lambda x: -x[1])), flush=True)
    for c_ in out["comparisons"]:
        print(f"RESULT {c_['model']} vs {c_['vs']}: {c_['delta_accuracy']:+.3f}, Holm p {c_['p_holm']:.3f}", flush=True)


if __name__ == "__main__":
    main()
