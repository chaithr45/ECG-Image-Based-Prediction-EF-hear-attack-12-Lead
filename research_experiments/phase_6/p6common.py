"""Phase-6 shared code: development-set-only data, EF target, storage, nested inner CV, guards.

Re-uses research_experiments.phase_5.p5common (development-only loader, folds, features, metrics, statistics, decision
rules, privacy scan) instead of re-implementing it.

LOCKED-TEST PROTECTION
  * p5common drops every locked-test row from the manifest on import; Phase 6 never sees a locked-test row, label,
    feature, waveform or embedding. Phase 6 code never runs a locked-test ECG through any model (not even unlabelled).
  * The EF target table (phase_4/data/private/cohorts.csv) also contains locked-test rows: they are dropped
    immediately on read (only development ids are kept; asserted).
  * Forbidden imports (they load the locked test): phase_2.cv_features, phase_2.accuracy_tradeoff, phase_3.p3common,
    phase_3.p3_robust, phase_4.p4common, and anything importing them (e.g. phase_3.clinical_features).
    tests/test_phase6.py asserts that none is loaded.
STORAGE (additive; refuses to write outside phase_6/)
  aggregate results  -> phase_6/results/experiments/<run_id>.json      (committed; aggregates only)
  per-ECG outputs    -> phase_6/data/private/runs/<run_id>/oof.csv     (git-ignored; never committed)
  inner-CV caches    -> phase_6/data/private/inner/...                 (git-ignored)
  run markers        -> phase_6/results/running/<run_id>.json          (git-ignored; exist only while a run executes)
Phase 6 never writes a Phase 1-5 aggregate. Regenerated Phase-5 per-ECG outputs (MISSING_PRIVATE) are written to the
Phase-6 private store and checked against the committed Phase-5 aggregates (oof_sha256 and metrics).
A run is DONE only if both the aggregate and the private predictions exist and the prediction hash matches.
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedGroupKFold

from research_experiments.phase_5 import p5common as p5
from research_experiments.phase_5.p5common import (  # noqa: F401  (re-exported for Phase-6 modules)
    ALPHAS, BIASES, CLASSES, D, IDS, N_TE, N_TR, PAT, PCOLS, RECALL_FLOOR, REPEATS, ROOT, SEEDS, Y, blockstd, cells,
    config_hash, confusion, corr_ci, corr_t, decide, ecgf_f6, fold_vec, folds, git_head, header_numeric, holm, load_oof_csv,
    make_X, metrics, nD, sha256_file, split_indices)

P6 = ROOT / "research_experiments" / "phase_6"
PRIV6 = P6 / "data" / "private"
RUNS6, FEAT6, INNER6, EMB6 = PRIV6 / "runs", PRIV6 / "features", PRIV6 / "inner", PRIV6 / "embeddings"
RES6 = P6 / "results"
EXP6 = RES6 / "experiments"
RUN_MARK6 = RES6 / "running"
CONF6 = P6 / "configs"
P5DIR = ROOT / "research_experiments" / "phase_5"
EXP5 = P5DIR / "results" / "experiments"
FORBIDDEN = ("research_experiments.phase_2.cv_features", "research_experiments.phase_2.accuracy_tradeoff",
             "research_experiments.phase_3.p3common", "research_experiments.phase_3.p3_robust",
             "research_experiments.phase_4.p4common", "research_experiments.phase_3.clinical_features")
R_REF = 0.66            # Phase-4 reference correlation (P4-9): MLP regression header || ECGF f6, cohort A
R_PASS = 0.69           # pre-registered Stage-1 screen (PHASE6_PLAN.md)


def guard6(path) -> Path:
    path = Path(path).resolve()
    if P6.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_6/: {path}")
    return path


def forbidden_modules_loaded():
    import sys
    return [m for m in FORBIDDEN if m in sys.modules]


# ------------------------------------------------------------------ EF target (analysis / training target only)
_EF = None


def ef_target():
    """Nearest-in-time linked echo EF (label-agnostic, Phase 4 definition) for the 837 development ECGs, in IDS order.
    TARGET ONLY - never a model input. Locked-test rows are dropped on read."""
    global _EF
    if _EF is None:
        c = pd.read_csv(ROOT / "research_experiments" / "phase_4" / "data" / "private" / "cohorts.csv",
                        usecols=["ecg_id", "split", "ef", "G_hc", "C_14d"])
        c = c[c.split != "test"].drop(columns="split")
        c = c[c.ecg_id.isin(set(IDS))].set_index("ecg_id")
        assert len(c) == nD and set(c.index) == set(IDS), "EF table does not cover exactly the development set"
        c = c.loc[IDS]
        _EF = dict(ef=c.ef.values.astype(float), G=c.G_hc.values.astype(bool), C=c.C_14d.values.astype(bool))
    return _EF


def ef_stats(efhat, mask=None):
    ef = ef_target()["ef"]; m = np.isfinite(ef) & np.isfinite(efhat)
    if mask is not None:
        m &= mask
    a, b = ef[m], efhat[m]
    r = float(np.corrcoef(a, b)[0, 1])
    from scipy.stats import spearmanr
    return dict(n=int(m.sum()), pearson=r, spearman=float(spearmanr(a, b)[0]), mae=float(np.mean(np.abs(a - b))),
                rmse=float(np.sqrt(np.mean((a - b) ** 2))), r2=float(1 - np.sum((a - b) ** 2) / np.sum((a - a.mean()) ** 2)),
                calib_slope=float(np.polyfit(b, a, 1)[0]))


# ------------------------------------------------------------------ storage, run markers
def agg_path(run_id):
    return EXP6 / f"{run_id}.json"


def oof_path(run_id):
    return RUNS6 / run_id / "oof.csv"


def done(run_id):
    a, o = agg_path(run_id), oof_path(run_id)
    if not (a.exists() and o.exists()):
        return False
    return json.loads(a.read_text()).get("oof_sha256") in (None, sha256_file(o))


def mark_running(run_id, cmd):
    guard6(RUN_MARK6).mkdir(parents=True, exist_ok=True)
    (RUN_MARK6 / f"{run_id}.json").write_text(json.dumps(dict(run_id=run_id, pid=os.getpid(), started=time.strftime("%F %T"),
                                                             command=cmd)))


def clear_running(run_id):
    f = RUN_MARK6 / f"{run_id}.json"
    if f.exists():
        f.unlink()


def save_run(run_id, P, pred, repeat, config, extra=None, efhat=None, overwrite=False):
    """Private per-ECG predictions (+ optional EF estimate) and committed aggregate. Never overwrites a completed run."""
    if agg_path(run_id).exists() and not overwrite:
        raise FileExistsError(f"{run_id} already completed; create a new version instead of overwriting")
    d = guard6(RUNS6 / run_id); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    if efhat is not None:
        o["efhat"] = efhat
    o.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    m = metrics(Y, P, pred)
    agg = dict(run_id=run_id, repeat=repeat, config=config, config_hash=config_hash(config), created=time.strftime("%F %T"),
               git=git_head(), n_ecgs=nD, n_patients=int(pd.Series(PAT).nunique()), metrics=m, confusion=confusion(Y, pred).tolist(),
               per_fold=cells(P, pred, repeat), oof_sha256=sha256_file(d / "oof.csv"), **(extra or {}))
    if efhat is not None:
        efr = pd.read_csv(d / "oof.csv").efhat.values            # stats on the stored (rounded) values
        E = ef_target()
        agg["ef_regression"] = dict(A=ef_stats(efr), G=ef_stats(efr, E["G"]), C=ef_stats(efr, E["C"]),
                                    per_fold_pearson=[ef_stats(efr, fold_vec(repeat) == k)["pearson"] for k in range(5)])
    guard6(EXP6).mkdir(parents=True, exist_ok=True)
    agg_path(run_id).write_text(json.dumps(agg, indent=1, default=float))
    clear_running(run_id)
    return agg


def load_run(run_id):
    return load_oof_csv(oof_path(run_id))


def load_efhat(run_id):
    o = pd.read_csv(oof_path(run_id)).set_index("ecg_id").loc[IDS]
    return o.efhat.values


def load_agg(run_id):
    return json.loads(agg_path(run_id).read_text())


def load_agg5(run_id):
    return json.loads((EXP5 / f"{run_id}.json").read_text())


# ------------------------------------------------------------------ nested inner CV (same protocol as phase_5.nested)
def inner_oof(base, r, k, spec, fold_vector=None, tag=""):
    """Training-portion-only out-of-fold predictions for outer fold k of repeat r (Phase-5 nested protocol:
    patient-grouped 4-fold inner CV, random_state 1000 + 10 r + k; every inner model uses its own grouped 80/20 split,
    random_state 2000 + 10 k + j). spec = (blocks, kind, cfg, seeds). Cached in phase_6/data/private/inner/."""
    from research_experiments.phase_5.experiments.nested import fit_predict
    f = guard6(INNER6 / (base + tag) / f"r{r}_k{k}.npz")
    if f.exists():
        z = np.load(f); return z["TR"], z["P"]
    blocks, kind, cfg, seeds = spec
    fv = fold_vec(r) if fold_vector is None else fold_vector
    TR = np.flatnonzero(fv != k)
    inner = StratifiedGroupKFold(4, shuffle=True, random_state=1000 + 10 * r + k)
    P = np.zeros((len(TR), 4))
    for j, (ia, ib) in enumerate(inner.split(TR, Y[TR], PAT_GROUP()[TR])):
        assert not set(PAT_GROUP()[TR[ia]]) & set(PAT_GROUP()[TR[ib]]), "inner patient leakage"
        P[ib] = np.mean([fit_predict(blocks, kind, cfg, TR[ia], TR[ib], s, 2000 + 10 * k + j) for s in seeds], 0)
    f.parent.mkdir(parents=True, exist_ok=True)
    np.savez(f, TR=TR, P=P)
    return TR, P


_GROUP = None


def PAT_GROUP():
    """Grouping key used for inner splits (default: the Phase 1-5 name-derived patient key)."""
    return PAT if _GROUP is None else _GROUP


def privacy_scan(paths):
    return p5.privacy_scan(paths)
