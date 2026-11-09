# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/train100/train100_freeze.py
# >>> Run from the repository root: PYTHONPATH=. python research_experiments/train100/train100_freeze.py (see the docstring below for the options)
# >>> Part of BEST_RESULTS/best_for_70s; see the README there.
"""Freeze the image-only triage model trained to (nearly) 100 % training accuracy, for the test kit
(phase_7/testkit/score.py --model train100, the default). Same recipe as the triage of train100_features.py, the report's
"Best in the 60s" (0.696 on the 48 % of patients it answers):
  * IMG-vote = mean of ECGF-MLP (3 seeds) and ECGF-linear on the frozen ECGFounder embedding, each trained to 100 %;
    IMG-C3 = IMG-vote + prior rule (alpha);
  * EF = mean of ridge (alpha 1e-6) on [ECGFounder, image clinician features] and LightGBM (4,000 trees, no early stopping)
    on [image clinician features, 32 ECGFounder PCs]; sample weight 1 for verified and 0.3 for other ECGs;
  * answer when the EF grade agrees with IMG-C3 and the EF estimate is at least t_answer EF points from every class cut.
As in the original freeze (phase_7/testkit/freeze.py), alpha, the grade cuts and t_answer are chosen on the pooled outer
out-of-fold predictions of the 3 x 5-fold cross-validation (the TRAIN100 predictions kept privately by train100_scores.py),
then the members and EF regressors are fitted on all 837 development ECGs, trained to 100 % (no selection split, no early
stopping). The frozen file stays in the git-ignored phase_7/data/private/testkit/; only the card (aggregates) is committed.
From the repository root, PYTHONPATH=. :
    python research_experiments/train100/train100_freeze.py --check   --private <train100_scores private dir>
    python research_experiments/train100/train100_freeze.py --freeze  --private <train100_scores private dir>
    python research_experiments/train100/train100_freeze.py --selftest 25
"""
import argparse
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train100_features as T  # noqa: E402  (learner settings, data access, folds)
from research_experiments.phase_2.p2heads import Head, class_w  # noqa: E402
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha  # noqa: E402
from research_experiments.phase_5.p5common import decide, ecgf_f6  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import mildfix as F1  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm.mildmodels import clin  # noqa: E402
from research_experiments.phase_7.testkit import kit as K  # noqa: E402

C, B, Y, nD = T.C, T.B, T.Y, T.C.nD
FROZEN = K.FROZEN_T100
CARD = K.CARD_T100
GRID = [(d0, t1, t2) for d0 in np.arange(-15, 6.5, 0.5) for t1 in np.arange(30, 55.5, 0.5) for t2 in np.arange(20, t1, 0.5)]


# ------------------------------------------------------------------ model parts (same code path as train100_features)
def _std(X, rows):
    return X[rows].mean(0), X[rows].std(0) + 1e-6


def mlp_member(X, rows, seed):
    """As train100_features.mlp, returning the fitted head for the kit."""
    mu, sd = _std(X, rows); xtr = torch.tensor((X[rows] - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Y[rows]); cw = class_w(Y[rows], "balanced")
    torch.manual_seed(seed); m = Head(X.shape[1], "ce", "mlp", 64, 0.5)
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.1); g = torch.Generator().manual_seed(seed)
    ep, acc = 0, 0.0
    while ep < T.MLP_MAX_EPOCHS and acc < 1.0:
        m.train(); idx = torch.randperm(len(rows), generator=g)
        for i in range(0, len(idx), 64):
            b = idx[i:i + 64]; lo = m.loss(xtr[b], yt[b], cw, {}); opt.zero_grad(); lo.backward(); opt.step()
        ep += 1; m.eval()
        with torch.no_grad():
            acc = float((m.probs(xtr).argmax(1).numpy() == Y[rows]).mean())
    m.eval()
    return dict(model=m, mu=mu, sd=sd, seed=seed, cfg=dict(head="mlp", loss="ce", hidden=64, dropout=0.5), best_epoch=None,
                epochs=ep, train_acc=acc)


def linear_member(X, rows, seed=42):
    """As train100_features.linear, returning the fitted head for the kit."""
    mu, sd = _std(X, rows); xtr = torch.tensor((X[rows] - mu) / sd, dtype=torch.float32)
    yt = torch.tensor(Y[rows]); cw = class_w(Y[rows], "balanced")
    torch.manual_seed(seed); m = Head(X.shape[1], "ce_dist", "linear")
    opt = torch.optim.LBFGS(m.parameters(), lr=0.5, max_iter=100, history_size=20, line_search_fn="strong_wolfe")

    def closure():
        opt.zero_grad()
        lo = m.loss(xtr, yt, cw, {"lam": 1.0}) + 1e-6 * sum((p ** 2).sum() for p in m.weight_params())
        lo.backward(); return lo
    steps, acc = 0, 0.0
    while steps < T.LIN_MAX_STEPS and acc < 1.0:
        opt.step(closure); steps += 100
        with torch.no_grad():
            acc = float((m.probs(xtr).argmax(1).numpy() == Y[rows]).mean())
    m.eval()
    return dict(model=m, mu=mu, sd=sd, seed=seed, cfg=dict(head="linear", loss="ce_dist", lam=1.0), best_epoch=None,
                steps=steps, train_acc=acc)


def ef_parts(fit_rows, clean, ef, E, IMG):
    """As train100_features.ef_models, in the kit's H3 = mean(H1 ridge, H2 LightGBM) format."""
    import lightgbm as lgb
    w = np.where(clean[fit_rows], 1.0, 0.3)
    fm = np.zeros(nD, bool); fm[fit_rows] = True
    med = np.nanmedian(IMG[fm], 0); med = np.where(np.isnan(med), 0.0, med)
    Ii = np.where(np.isnan(IMG), med[None], IMG)
    h1 = dict(kind="H1", med=med, muE=E[fm].mean(0), sdE=E[fm].std(0) + 1e-6, muI=Ii[fm].mean(0), sdI=Ii[fm].std(0) + 1e-6)
    X1 = np.concatenate([(E - h1["muE"]) / h1["sdE"] / np.sqrt(E.shape[1]), (Ii - h1["muI"]) / h1["sdI"] / np.sqrt(IMG.shape[1])], 1)
    h1["ridge"] = Ridge(alpha=1e-6).fit(X1[fit_rows], ef[fit_rows], sample_weight=w)
    mu, sd = E[fm].mean(0), E[fm].std(0) + 1e-6
    pca = PCA(32, random_state=0).fit(((E - mu) / sd)[fm])
    X2 = np.column_stack([IMG, pca.transform((E - mu) / sd)])
    lg = lgb.LGBMRegressor(n_estimators=4000, learning_rate=0.05, num_leaves=31, min_child_samples=1, subsample=1.0,
                           colsample_bytree=1.0, reg_lambda=0.0, random_state=0, n_jobs=1, verbose=-1).fit(X2[fit_rows], ef[fit_rows], sample_weight=w)
    return dict(kind="H3", parts=[h1, dict(kind="H2", mu=mu, sd=sd, pca=pca, lgbm=lg)])


def data():
    tab = B.table()
    return dict(E=ecgf_f6().astype(float), IMG=clin()[0], cols=list(clin()[1]), ef=tab.ef.values.astype(float),
                bnd=tab.sex_boundary.values, male=tab.male.values, clean=C.verified()[0])


# ------------------------------------------------------------------ --check: the kit format reproduces train100_features
def check():
    warnings.filterwarnings("ignore"); torch.set_num_threads(1); D = data()
    fv = B.fold_vec(0); tr, te = np.flatnonzero(fv != 0), np.flatnonzero(fv == 0)
    ref_mlp, _ = T.mlp3(D["E"], tr); ref_lin, _ = T.linear(D["E"], tr)
    ref_vote = (ref_mlp(te) + ref_lin(te)) / 2
    mem = dict(mlp=[mlp_member(D["E"], tr, s) for s in T.SEEDS], lin=linear_member(D["E"], tr))
    vote = K.Frozen(mem).vote(D["E"][te])
    ref_ef = T.ef_models(tr, [te], D["clean"], D["ef"], D["E"], D["IMG"])[0]
    ef = K.ef_predict(ef_parts(tr, D["clean"], D["ef"], D["E"], D["IMG"]), D["E"][te], D["IMG"][te])
    out = dict(fold="repeat 0, fold 0", vote_max_abs_diff=float(np.abs(vote - ref_vote).max()), ef_max_abs_diff=float(np.abs(ef - ref_ef).max()))
    print(json.dumps(out, indent=1))
    assert out["vote_max_abs_diff"] < 1e-5 and out["ef_max_abs_diff"] < 1e-6, "kit-format model differs from train100_features"
    return out


# ------------------------------------------------------------------ --freeze
def pooled_oof(private):
    P, E_ = [], []
    for r in (0, 1, 2):
        p, e = np.zeros((nD, 4)), np.zeros(nD)
        for k in range(5):
            z = np.load(Path(private) / f"cell_r{r}_k{k}.npz"); p[z["te"]] = z["P_IMG-vote"]; e[z["te"]] = z["ef_hat"]
        P.append(p); E_.append(e)
    return np.concatenate(P), np.concatenate(E_)


def cv_expectation():
    F = json.load(open(HERE / "results" / "train100_features.json")); S = F["summary"]

    def pooled(get):
        reps = []
        for r in (0, 1, 2):
            Cm = np.sum([get(c) for c in F["cells"] if c["repeat"] == r], 0).astype(float)
            rec = np.diag(Cm) / np.maximum(Cm.sum(1), 1); prec = np.divide(np.diag(Cm), Cm.sum(0), out=np.zeros(4), where=Cm.sum(0) > 0)
            f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
            reps.append(dict(rec=rec, prec=prec, f1=f1))
        return {k: np.mean([x[k] for x in reps], 0) for k in reps[0]}
    t, i = pooled(lambda c: c["triage"]["heldout_confusion"]), pooled(lambda c: c["models"]["IMG-C3"]["heldout_confusion"])
    st, si = S["AGT-img triage"], S["IMG-C3"]
    return dict(source="TRAIN100 patient-grouped 5-fold CV x 3 repeats on the 837 development ECGs (train100/results/train100_features.json)",
                triage=dict(answered_share=st["heldout_coverage"]["mean"], answered_share_ci95=st["heldout_coverage"]["ci95_corrected"],
                            accuracy_answered=st["heldout_accuracy_answered"]["mean"], accuracy_answered_ci95=st["heldout_accuracy_answered"]["ci95_corrected"],
                            macro_f1_answered=float(t["f1"].mean()), balanced_accuracy_answered=float(t["rec"].mean()), recall_Mild_answered=float(t["rec"][1])),
                imgc3_all=dict(accuracy=si["heldout_accuracy"]["mean"], accuracy_ci95=si["heldout_accuracy"]["ci95_corrected"], macro_f1=float(i["f1"].mean()),
                               balanced_accuracy=float(i["rec"].mean()), recall_Mild=float(i["rec"][1]), precision_Mild=float(i["prec"][1])))


def freeze(private):
    warnings.filterwarnings("ignore"); torch.set_num_threads(4); t0 = time.time(); D = data()
    prior = np.bincount(Y, minlength=4) / nD
    Ppool, e_pool = pooled_oof(private); ypool = np.tile(Y, 3); b_pool = np.tile(D["bnd"], 3)
    alpha = choose_alpha(Ppool, ypool, prior, "protected")
    p = F1.floor_best(((q, F1.ef_grade(e_pool, b_pool, *q)) for q in GRID), ypool) or (0.0, 41.0, 31.0)
    g = F1.ef_grade(e_pool, b_pool, *p)
    d = np.abs(e_pool[:, None] - np.column_stack([b_pool + p[0], np.full(len(e_pool), p[1]), np.full(len(e_pool), p[2])])).min(1)
    c3 = decide(Ppool, "prior", alpha, prior); ag = c3 == g
    t_ans = float(F1.threshold(d[ag], (g == ypool)[ag])); keep = ag & (d >= t_ans)
    sel = dict(note="computed on the same pooled OOF used to choose these values: optimistic, NOT a performance estimate",
               answered_share=float(keep.mean()), accuracy_answered=float(np.mean(g[keep] == ypool[keep])) if keep.any() else None,
               imgc3_accuracy_all=float(np.mean(c3 == ypool)), grade_accuracy_all=float(np.mean(g == ypool)))
    print(f"alpha {alpha} | cuts {p} | t {t_ans:.2f} | {sel}", flush=True)
    allr = np.arange(nD)
    mem = dict(mlp=[mlp_member(D["E"], allr, s) for s in T.SEEDS], lin=linear_member(D["E"], allr))
    efm = ef_parts(allr, D["clean"], D["ef"], D["E"], D["IMG"])
    frozen = dict(mlp=mem["mlp"], lin=mem["lin"], alpha=float(alpha), prior=prior.tolist(), ef=efm, cuts=[float(x) for x in p],
                  t_answer=t_ans, clin_cols=D["cols"], mildfeat=K.mildfeat_settings(), ef_variant="TRAIN100: ridge 1e-6 + LightGBM 4,000 trees, weights 1 / 0.3",
                  training="trained to (nearly) 100 % training accuracy (TRAIN100)", created=time.strftime("%F %T"), git=B.git_head())
    C.guard7(K.KIT).mkdir(parents=True, exist_ok=True)
    torch.save(frozen, C.guard7(FROZEN))
    fr = K.Frozen(frozen); full = fr.run(D["E"], D["IMG"], D["male"]); ans = full["answered"]
    e_in = K.ef_predict(efm, D["E"], D["IMG"])
    card = dict(
        model="image-only triage (AGT-img), trained to (nearly) 100 % training accuracy: answer when the image EF grade agrees with IMG-C3 "
              "and the EF estimate is far from every class cut",
        created=frozen["created"], git_head_at_freeze=frozen["git"], frozen_file="phase_7/data/private/testkit/frozen_triage_train100.pt (git-ignored)",
        frozen_sha256=K.sha256(FROZEN), n_development_ecgs=int(nD),
        recipe=dict(members="ECGFounder-f6 MLP heads (seeds 42/43/44) + linear distance-CE head (seed 42), each trained on all 837 until 100 % "
                            "training accuracy (no selection split, no early stopping, as train100_features.py)",
                    alpha="choose_alpha(protected) on the TRAIN100 IMG-vote outer OOF pooled over 3 repeats",
                    ef="mean of ridge (alpha 1e-6) and LightGBM (4,000 trees, no early stopping), weights 1 / 0.3, fitted on all 837",
                    cuts_and_threshold="floor-objective cuts and distance threshold (answered accuracy >= 0.70, >= 20 answered) on the pooled TRAIN100 outer OOF"),
        chosen=dict(alpha=float(alpha), prior=dict(zip(K.CLASSES, prior.round(4).tolist())), cuts=dict(d0=float(p[0]), t1=float(p[1]), t2=float(p[2])),
                    t_answer_ef_points=t_ans, normal_mild_cut="sex boundary (52 men / 54 women) + d0", mildfeat=frozen["mildfeat"],
                    n_image_features=len(frozen["clin_cols"])),
        training_accuracy=dict(mlp_members=[dict(seed=m["seed"], epochs=m["epochs"], train_acc=m["train_acc"]) for m in mem["mlp"]],
                               linear_member=dict(steps=mem["lin"]["steps"], train_acc=mem["lin"]["train_acc"]),
                               ef_train_rmse=float(np.sqrt(np.mean((e_in - D["ef"]) ** 2))),
                               imgc3_all=float(np.mean(full["c3"] == Y)), triage_answered_share=float(ans.mean()),
                               triage_accuracy_answered=float(np.mean(full["triage"][ans] == Y[ans])) if ans.any() else None,
                               note="the frozen model on its own 837 training ECGs (training accuracy, not a performance estimate)"),
        selection_data=sel, check_vs_train100_features=json.loads(CARD.read_text()).get("check_vs_train100_features") if CARD.exists() else None,
        cv_expectation=cv_expectation(), minutes=round((time.time() - t0) / 60, 1))
    CARD.write_text(json.dumps(card, indent=1, default=float))
    print("frozen:", FROZEN.name, card["frozen_sha256"][:12], json.dumps(card["training_accuracy"], default=float))


# ------------------------------------------------------------------ --selftest: PDF reader vs cached inputs, same answers
def selftest(n):
    import pandas as pd
    warnings.filterwarnings("ignore"); torch.set_num_threads(4); D = data()
    fr = K.Frozen.load(FROZEN, CARD)
    man = pd.read_csv(C.ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source"]).set_index("ecg_id")
    idx = np.sort(np.random.default_rng(0).choice(nD, n, replace=False))
    cols, st = fr.d["clin_cols"], fr.d["mildfeat"]
    assert cols == D["cols"], "frozen feature columns differ from the development feature table"
    F6c, Ic = D["E"][idx], D["IMG"][idx]
    F6p, Ip, errors = np.full_like(F6c, np.nan), np.full_like(Ic, np.nan), 0
    for j, i in enumerate(idx):
        f, c, e = K.inputs_from_pdf(C.ROOT / man.loc[C.IDS[i], "source"], cols, st)          # path never printed
        if e:
            errors += 1; continue
        F6p[j], Ip[j] = f, c
    ok = ~np.isnan(F6p).any(1)
    cos = np.sum(F6p[ok] * F6c[ok], 1) / (np.linalg.norm(F6p[ok], axis=1) * np.linalg.norm(F6c[ok], axis=1))
    eq = np.isclose(Ip[ok], Ic[ok], rtol=1e-5, atol=1e-6, equal_nan=True)
    a, b = fr.run(F6p[ok], Ip[ok], D["male"][idx][ok]), fr.run(F6c[ok], Ic[ok], D["male"][idx][ok])
    out = dict(n_sampled=int(n), sample_seed=0, unreadable=int(errors), f6_min_cosine=float(cos.min()),
               f6_max_abs_diff=float(np.abs(F6p[ok] - F6c[ok]).max()), image_features_equal_fraction=float(eq.mean()),
               same_triage_decision=float(np.mean(a["triage"] == b["triage"])), same_imgc3_class=float(np.mean(a["c3"] == b["c3"])),
               ef_estimate_max_abs_diff=float(np.abs(a["ef_est"] - b["ef_est"]).max()))
    c = json.loads(CARD.read_text()); c.setdefault("checks", {})["pdf_reproduction"] = out
    CARD.write_text(json.dumps(c, indent=1, default=float))
    print(json.dumps(out, indent=1))


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--private"); ap.add_argument("--check", action="store_true")
    ap.add_argument("--freeze", action="store_true"); ap.add_argument("--selftest", type=int, default=0)
    a = ap.parse_args()
    if a.check:
        out = check()
        if CARD.exists():
            c = json.loads(CARD.read_text()); c["check_vs_train100_features"] = out; CARD.write_text(json.dumps(c, indent=1, default=float))
        else:
            CARD.write_text(json.dumps(dict(check_vs_train100_features=out), indent=1))
    if a.freeze:
        freeze(a.private)
    if a.selftest:
        selftest(a.selftest)


if __name__ == "__main__":
    main()
