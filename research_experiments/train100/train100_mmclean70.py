"""TRAIN100-MMCLEAN-70 (AMENDMENT_TRAIN100_ECHONEXT.md, Addendum B): the owner's "70s model" = MMCLEAN selective, i.e.
C3 trained on MMC (no fitting on the MMC-flagged suspect Mild / Moderate ECGs) answering only when its vote confidence
passes a nested threshold for 0.70 accuracy (phase_6b_threshold_sampling_lstm/mmclean.py `selective`, existing6b
`run_config("MMC")`), retrained under the TRAIN100 rules. Same Phase-2 patient-grouped 5-fold CV x 3 repeats on the 837
development ECGs, original labels for scoring. Hospital data only; the locked test set is not loaded. Per-ECG arrays stay
in the git-ignored phase_6b_threshold_sampling_lstm/data/private/train100_mmclean70/; aggregates only.
From the repository root, PYTHONPATH=. :
    python research_experiments/train100/train100_mmclean70.py --jobs 2 --threads 1      (cached per fold, resumable)
    python research_experiments/train100/train100_mmclean70.py --report
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

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.model_selection import StratifiedGroupKFold  # noqa: E402
from sklearn.pipeline import make_pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import train100_echonext as T1  # noqa: E402  (TRAIN100 head learner head100)
from research_experiments.phase_5.baseline.calibrated_mlp import choose_alpha  # noqa: E402
from research_experiments.phase_5.p5common import corr_ci, decide, make_X  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import mmclean as MC  # noqa: E402
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B  # noqa: E402

warnings.filterwarnings("ignore")
torch.set_num_threads(THREADS)
Y, PAT, nD, CLASSES = B.Y, B.PAT, B.nD, B.CLASSES
PRIV = B.PRIV / "train100_mmclean70"
RES_JSON = HERE / "results" / "train100_mmclean70.json"
CFG, TARGET, BIG_C = "MMC", 0.70, 1e4


def guard(p):
    p = Path(p).resolve()
    if not (PRIV.resolve() in p.parents or p == RES_JSON.resolve()):
        raise PermissionError(f"refusing to write outside the TRAIN100-MMCLEAN-70 folders: {p}")
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def members(rows, pred_sets):
    """The three C1 members (BASES of phase_5.experiments.nested) fitted on `rows` without the brake; inputs built with
    the fit-row statistics (make_X), as the originals. -> {member: [P for each pred set]}, info."""
    fm = np.zeros(nD, bool); fm[rows] = True
    Xf, pf = make_X(["hdr", "ecgf"], fm)
    fus = [T1.head100(Xf, rows, Y, "mlp", s, no_std=pf) for s in B.SEEDS]
    Xl, pl = make_X(["ecgf"], fm)
    lin, li = T1.head100(Xl, rows, Y, "lin", 42, no_std=pl)
    Xh, _ = make_X(["hdr", "feat:clin3"], fm)
    hc = make_pipeline(StandardScaler(), LogisticRegression(C=BIG_C, class_weight="balanced", max_iter=5000)).fit(Xh[rows], Y[rows])
    out = {"FUSION-MLP": [np.mean([f(i) for f, _ in fus], 0) for i in pred_sets], "ECGF-linear": [lin(i) for i in pred_sets],
           "HDRCLIN-LR": [hc.predict_proba(Xh[i]) for i in pred_sets]}
    return out, dict(fusion_epochs=[x["epochs"] for _, x in fus], fusion_train_acc=[x["train_acc"] for _, x in fus],
                     linear_iterations=li["iterations"], linear_train_acc=li["train_acc"])


def fold(rk):
    r, k = rk
    f, fj = PRIV / f"r{r}_k{k}.npz", PRIV / f"r{r}_k{k}.json"
    if f.exists() and fj.exists():
        return r, k
    warnings.filterwarnings("ignore"); torch.set_num_threads(THREADS); t0 = time.time()
    excl = MC.removed(CFG)
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k); yTR = Y[TR]
    split = list(StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k).split(TR, yTR, PAT[TR]))
    keep = ~excl[TR]
    M, info = members(TR[keep], [te, TR])
    Pin_m, inf_in = {m: np.zeros((len(TR), 4)) for m in M}, []
    for ia, ib in split:
        rows = TR[ia][~excl[TR[ia]]]
        Mi, ii = members(rows, [TR[ib]]); inf_in.append(ii)
        for m in Mi:
            Pin_m[m][ib] = Mi[m][0]
    Pte, Ptr = (np.mean([M[m][i] for m in M], 0) for i in (0, 1)); Pin = np.mean(list(Pin_m.values()), 0)   # C1 vote
    prior = np.bincount(yTR[keep], minlength=4) / keep.sum()
    a = choose_alpha(Pin[keep], yTR[keep], prior, "protected")                          # C3 rule (unchanged)
    c3_in, c3_te, c3_tr = (decide(P, "prior", a, prior) for P in (Pin, Pte, Ptr))
    ci, ok = Pin.max(1), c3_in == yTR                                                     # selective threshold (unchanged)
    ts = [t for t in np.unique(np.round(ci, 3)) if (ci >= t).sum() >= 20 and ok[ci >= t].mean() >= TARGET]
    t = ts[0] if ts else 1.01
    k_te, k_tr = Pte.max(1) >= t, Ptr.max(1) >= t
    np.savez(guard(f), te=te, c3_te=c3_te, k_te=k_te, c3_tr=c3_tr, k_tr=k_tr,
             **{f"{m}__te": M[m][0] for m in M}, **{f"{m}__tr": M[m][1] for m in M})
    acc = lambda p_, y_: float(np.mean(p_ == y_)) if len(y_) else None  # noqa: E731
    kf = k_tr & keep
    agg = dict(repeat=r, fold=k, alpha=float(a), threshold=float(t), n_fit_rows=int(keep.sum()), n_flagged_in_training=int((~keep).sum()),
               train=dict(answered_share_fit_rows=float(kf.sum() / keep.sum()), accuracy_answered_fit_rows=acc(c3_tr[kf], yTR[kf]),
                          answered_share_all_rows=float(k_tr.mean()), accuracy_answered_all_rows=acc(c3_tr[k_tr], yTR[k_tr]),
                          c3_accuracy_fit_rows=acc(c3_tr[keep], yTR[keep]), vote_accuracy_fit_rows=acc(Ptr.argmax(1)[keep], yTR[keep]),
                          **{f"{m}_accuracy_fit_rows": acc(M[m][1].argmax(1)[keep], yTR[keep]) for m in M}),
               heldout=dict(answered_share=float(k_te.mean()), accuracy_answered=acc(c3_te[k_te], Y[te][k_te]), c3_accuracy=acc(c3_te, Y[te]),
                            vote_accuracy=acc(Pte.argmax(1), Y[te]), **{f"{m}_accuracy": acc(M[m][0].argmax(1), Y[te]) for m in M}),
               members=dict(outer=info, inner=inf_in), minutes=round((time.time() - t0) / 60, 2))
    guard(fj).write_text(json.dumps(agg, default=float))
    tr_, ho = agg["train"], agg["heldout"]
    log(f"r{r} k{k}: {agg['minutes']} min | train answered {tr_['answered_share_fit_rows']:.2f} acc {tr_['accuracy_answered_fit_rows']} | "
        f"held-out answered {ho['answered_share']:.2f} acc {ho['accuracy_answered']} | C3 all {ho['c3_accuracy']:.3f}")
    return r, k


def report():
    cells_rk = [(r, k) for r in (0, 1, 2) for k in range(5)]
    F = {rk: json.loads((PRIV / f"r{rk[0]}_k{rk[1]}.json").read_text()) for rk in cells_rk}
    Z = {rk: np.load(PRIV / f"r{rk[0]}_k{rk[1]}.npz") for rk in cells_rk}
    reps, cl, c3a, c3c = [], [], [], []
    for r in (0, 1, 2):                                                   # as mmclean.selective
        pred, keep = np.full(nD, -1), np.zeros(nD, bool)
        for k in range(5):
            z = Z[(r, k)]; pred[z["te"]] = z["c3_te"]; keep[z["te"]] = z["k_te"]
            sel = np.zeros(nD, bool); sel[z["te"]] = z["k_te"]
            if sel.any():
                cl.append(float(np.mean(pred[sel] == Y[sel])))
            c3c.append(float(np.mean(z["c3_te"] == Y[z["te"]])))
        y, p = Y[keep], pred[keep]; c3a.append(float(np.mean(pred == Y)))
        reps.append(dict(coverage=float(keep.mean()), accuracy=float(np.mean(p == y)) if keep.any() else None,
                         coverage_by_class={c: round(float(keep[Y == i].mean()), 3) for i, c in enumerate(CLASSES)},
                         recall_covered={c: (round(float(np.mean(p[y == i] == i)), 3) if (y == i).any() else None) for i, c in enumerate(CLASSES)},
                         predicted_mix={c: int((p == i).sum()) for i, c in enumerate(CLASSES)}))
    ok = [x for x in reps if x["accuracy"] is not None]
    S = pd.read_csv(B.RES / "mmclean_selective.csv")
    o = S[(S.model == "c3") & (S.trained == CFG) & (np.isclose(S.target, TARGET))].iloc[0]
    R = pd.read_csv(B.RES / "mmclean_results.csv")
    oc3 = R[(R.config == CFG) & (R.evaluated_on == "all_837") & (R.labels == "ORIG") & (R.trained == CFG) & R.model.str.startswith("C3")].iloc[0]
    mean = lambda part, key: float(np.mean([F[rk][part][key] for rk in cells_rk if F[rk][part][key] is not None]))  # noqa: E731
    acc_h = float(np.mean([x["accuracy"] for x in ok]))
    ep = [e for rk in cells_rk for x in [F[rk]["members"]["outer"]] + F[rk]["members"]["inner"] for e in x["fusion_epochs"]]
    it = [x["linear_iterations"] for rk in cells_rk for x in [F[rk]["members"]["outer"]] + F[rk]["members"]["inner"]]
    out = dict(
        amendment="research_experiments/train100/AMENDMENT_TRAIN100_ECHONEXT.md (Addendum B)",
        model="70s model = MMCLEAN selective: C3 trained on MMC, answering at the 0.70 target (mmclean.selective), TRAIN100 retrain",
        evaluation="Phase-2 patient-grouped 5-fold CV x 3 repeats, 837 development ECGs, original labels; locked test not loaded",
        selective=dict(
            train=dict(accuracy_answered=mean("train", "accuracy_answered_fit_rows"), answered_share=mean("train", "answered_share_fit_rows"),
                       accuracy_answered_all_training_rows=mean("train", "accuracy_answered_all_rows"),
                       answered_share_all_training_rows=mean("train", "answered_share_all_rows"),
                       folds_at_100_percent=int(sum(F[rk]["train"]["accuracy_answered_fit_rows"] == 1.0 for rk in cells_rk))),
            heldout=dict(accuracy_answered=acc_h, accuracy_ci95_corrected=corr_ci(cl) if len(cl) > 2 else None, n_fold_cells=len(cl),
                         accuracy_per_repeat=[round(x["accuracy"], 4) for x in ok], answered_share=float(np.mean([x["coverage"] for x in reps])),
                         per_repeat=reps),
            original=dict(accuracy_answered=float(o.accuracy), ci95_corrected=json.loads(o.accuracy_ci95_corrected), answered_share=float(o.coverage)),
            change_points=round(100 * (acc_h - float(o.accuracy)), 2)),
        c3_mmc_all_patients=dict(train_accuracy=mean("train", "c3_accuracy_fit_rows"), heldout_accuracy=float(np.mean(c3a)),
                                 heldout_ci95_corrected=corr_ci(c3c), original_heldout_accuracy=float(oc3.accuracy)),
        members=dict(**{m: dict(train_accuracy=mean("train", f"{m}_accuracy_fit_rows"), heldout_accuracy=mean("heldout", f"{m}_accuracy"))
                        for m in ("FUSION-MLP", "ECGF-linear", "HDRCLIN-LR")},
                     fusion_epochs=dict(median=float(np.median(ep)), max=int(max(ep)), fits_at_cap=int(np.sum(np.array(ep) >= T1.MLP_CAP))),
                     linear_lbfgs_iterations=dict(median=float(np.median(it)), max=int(max(it)), fits_at_cap=int(np.sum(np.array(it) >= T1.LIN_CAP)))),
        decision_rules_per_fold=[dict(repeat=rk[0], fold=rk[1], alpha=F[rk]["alpha"], threshold=F[rk]["threshold"]) for rk in cells_rk],
        per_fold=[dict(repeat=rk[0], fold=rk[1], train=F[rk]["train"], heldout=F[rk]["heldout"]) for rk in cells_rk],
        minutes_per_fold=float(np.mean([F[rk]["minutes"] for rk in cells_rk])))
    guard(RES_JSON).write_text(json.dumps(out, indent=1, default=float))
    s = out["selective"]
    print(json.dumps(dict(train=s["train"], heldout={k_: s["heldout"][k_] for k_ in ("accuracy_answered", "accuracy_ci95_corrected", "answered_share")},
                          original=s["original"], change=s["change_points"], c3_all=out["c3_mmc_all_patients"], members=out["members"]), indent=1, default=float))
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--jobs", type=int, default=1); ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--cells", default="all"); ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    if a.report:
        report(); return
    MC.flags(); B.table()
    cells = [(r, k) for r in (0, 1, 2) for k in range(5)] if a.cells == "all" else [tuple(map(int, c.split(":"))) for c in a.cells.split(",")]
    log(f"TRAIN100-MMCLEAN-70: {len(cells)} folds, jobs {a.jobs}, threads {THREADS}")
    if a.jobs <= 1:
        for c in cells:
            fold(c)
    else:
        with Pool(a.jobs) as p:
            for r, k in p.imap_unordered(fold, cells):
                log(f"done r{r} k{k}")


if __name__ == "__main__":
    main()
