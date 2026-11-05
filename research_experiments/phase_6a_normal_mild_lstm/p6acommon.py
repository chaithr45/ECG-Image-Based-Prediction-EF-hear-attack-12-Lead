"""Phase-6A shared code (Normal/Mild boundary + LSTM study). Development set only; re-uses phase_5.p5common (via
phase_6.p6common): loader without locked-test rows, folds, features, metrics, statistics, decision rules.

Label configurations (PHASE6A_PLAN.md section 4):
  B1   original folder labels
  B2   Mild -> Normal for clearly defined borderline Mild ECGs (EF >= 50 by a nearest echo that is itself Mild)
  B3   original labels; uncertain Normal/Mild borderline ECGs removed from inner-train only
  A270 original labels; inner-train Normal raised to 270 by moving whole Normal patients from inner-val
Echo EF / link data are used ONLY to define these label configurations and for analysis - never as model inputs.
Storage (refuses to write outside phase_6a_normal_mild_lstm/):
  results/experiments/<run_id>.json  aggregates (committed)      data/private/runs/<run_id>/oof.csv  per-ECG (ignored)
  data/private/fold_cache/, inner/, checkpoints/, logs/, labels.csv  (git-ignored)
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd

from research_experiments.phase_6.p6common import (  # noqa: F401  (re-exported)
    CLASSES, D, IDS, N_TE, N_TR, PAT, PCOLS, REPEATS, ROOT, SEEDS, Y, config_hash, confusion, corr_ci, corr_t, decide,
    fold_vec, forbidden_modules_loaded, git_head, holm, load_oof_csv, make_X, metrics, nD, privacy_scan, sha256_file,
    split_indices)

P6A = ROOT / "research_experiments" / "phase_6a_normal_mild_lstm"
PRIV = P6A / "data" / "private"
RUNS, FOLDC, INNER, CKPT, LOGS = PRIV / "runs", PRIV / "fold_cache", PRIV / "inner", PRIV / "checkpoints", PRIV / "logs"
RES = P6A / "results"
EXP = RES / "experiments"
RUN_MARK = PRIV / "running"
CONF = P6A / "configs"
PLOTS = P6A / "plots"
CONFIGS = ("B1", "B2", "B3", "A270")
B2_BOUNDARY = 50.0
B3_BAND = 2.0
A_TARGET = 270


def guard(path) -> Path:
    path = Path(path).resolve()
    if P6A.resolve() not in path.parents:
        raise PermissionError(f"refusing to write outside phase_6a_normal_mild_lstm/: {path}")
    return path


# ------------------------------------------------------------------ label configurations (private, development only)
_LAB = None


def _links_dev():
    L = pd.read_csv(ROOT / "research_experiments" / "phase_4" / "data" / "private" / "ecg_echo_links.csv",
                    usecols=["ecg_id", "split", "ef", "derived"])
    L = L[L.split != "test"].drop(columns="split")
    return L[L.ecg_id.isin(set(IDS))]


def label_table():
    """Per development ECG: original label, nearest-echo EF and link facts, B2 label, B3 'uncertain' flag."""
    global _LAB
    if _LAB is not None:
        return _LAB
    c = pd.read_csv(ROOT / "research_experiments" / "phase_4" / "data" / "private" / "cohorts.csv")
    c = c[c.split != "test"]
    c = c[c.ecg_id.isin(set(IDS))].set_index("ecg_id").loc[IDS]
    assert (c.label_idx.values == Y).all(), "label table does not match the development labels"
    male = c.sex_ecg.astype(str).str.upper().str.startswith("M").values
    bnd = np.where(male, 52.0, 54.0)
    lab = c.label.values
    T = pd.DataFrame(dict(label=lab, y=Y, male=male, ef=c.ef.values, derived=c.derived.values, n_links=c.n_links.values,
                          ef_spread=c.ef_spread_all_links.values, score=c.score.values, abs_days=c.abs_days.values,
                          G=c.G_hc.values.astype(bool), patient=PAT), index=pd.Index(IDS, name="ecg_id"))
    nm = np.isin(lab, ["Normal", "Mild"])
    # B2: clearly defined borderline Mild -> Normal
    T["b2_move"] = (lab == "Mild") & (T.derived.values == "Mild") & (T.ef.values >= B2_BOUNDARY)
    T["y_b2"] = np.where(T.b2_move, 0, Y)
    # B3: uncertain Normal/Mild borderline (band around the sex-specific boundary, or a conflicting Normal/Mild link)
    L = _links_dev()
    other = {"Normal": "Mild", "Mild": "Normal"}
    lab_of = pd.Series(lab, index=IDS)
    conf_ids = set(L[L.apply(lambda r: other.get(lab_of[r.ecg_id]) == r.derived, axis=1)].ecg_id)
    T["b3_band"] = nm & (np.abs(T.ef.values - bnd) <= B3_BAND)
    T["b3_conflict"] = nm & T.index.isin(conf_ids)
    T["b3_uncertain"] = T.b3_band | T.b3_conflict
    T["sex_boundary"] = bnd
    _LAB = T
    return T


def y_train(cfg):
    return label_table().y_b2.values.astype(int) if cfg == "B2" else Y


# ------------------------------------------------------------------ splits per configuration
def a270_split(r, k):
    """Inner-train Normal raised to exactly 270 by moving whole Normal patients from inner-val (seeded)."""
    itr, iva, ote = split_indices(r, k)
    n_norm = int((Y[itr] == 0).sum())
    va_norm_pats = pd.unique(PAT[iva][Y[iva] == 0])
    rng = np.random.default_rng(6270 + 10 * r + k)
    move = []
    for p in rng.permutation(va_norm_pats):
        rows = iva[PAT[iva] == p]
        if (Y[rows] != 0).any():                      # patient also has a non-Normal ECG in inner-val: keep grouping intact
            continue
        if n_norm + len(rows) <= A_TARGET:
            move.extend(rows.tolist()); n_norm += len(rows)
        if n_norm == A_TARGET:
            break
    move = np.array(sorted(move), int)
    itr2 = np.sort(np.r_[itr, move]); iva2 = np.setdiff1d(iva, move)
    assert not set(PAT[itr2]) & set(PAT[iva2]), "patient in both inner-train and inner-val"
    assert not set(PAT[itr2]) & set(PAT[ote]) and not set(PAT[iva2]) & set(PAT[ote])
    return itr2, iva2, ote, dict(normal_inner_train=int((Y[itr2] == 0).sum()), moved_ecgs=int(len(move)),
                                 normal_inner_val=int((Y[iva2] == 0).sum()))


def split(cfg, r, k):
    """-> (inner_train, inner_val, outer_test) index arrays for a configuration."""
    if cfg == "A270":
        return a270_split(r, k)[:3]
    itr, iva, ote = split_indices(r, k)
    if cfg == "B3":
        itr = itr[~label_table().b3_uncertain.values[itr]]
    return itr, iva, ote


# ------------------------------------------------------------------ storage
def agg_path(rid):
    return EXP / f"{rid}.json"


def oof_path(rid):
    return RUNS / rid / "oof.csv"


def done(rid):
    a, o = agg_path(rid), oof_path(rid)
    return a.exists() and o.exists() and json.loads(a.read_text()).get("oof_sha256") in (None, sha256_file(o))


def mark_running(rid, cmd):
    guard(RUN_MARK).mkdir(parents=True, exist_ok=True)
    (RUN_MARK / f"{rid}.json").write_text(json.dumps(dict(run_id=rid, pid=os.getpid(), started=time.strftime("%F %T"), command=cmd)))


def clear_running(rid):
    f = RUN_MARK / f"{rid}.json"
    if f.exists():
        f.unlink()


def save_run(rid, P, pred, repeat, cfg_name, config, extra=None):
    """Private per-ECG predictions + committed aggregate. Metrics against the ORIGINAL labels always; B2 models are
    additionally scored against the B2 labels. Never overwrites a completed run."""
    if agg_path(rid).exists():
        raise FileExistsError(f"{rid} already completed")
    d = guard(RUNS / rid); d.mkdir(parents=True, exist_ok=True)
    o = pd.DataFrame(P, columns=PCOLS); o.insert(0, "ecg_id", IDS); o.insert(1, "fold", fold_vec(repeat)); o["pred"] = pred
    o.to_csv(d / "oof.csv", index=False, float_format="%.6f")
    from research_experiments.phase_6.p6common import cells
    agg = dict(run_id=rid, repeat=repeat, label_config=cfg_name, config=config, config_hash=config_hash(config),
               created=time.strftime("%F %T"), git=git_head(), n_ecgs=nD, metrics=metrics(Y, P, pred),
               confusion=confusion(Y, pred).tolist(), per_fold=cells(P, pred, repeat), oof_sha256=sha256_file(d / "oof.csv"),
               **(extra or {}))
    yb2 = label_table().y_b2.values.astype(int)
    agg["metrics_vs_B2_labels"] = metrics(yb2, P, pred)
    agg["confusion_vs_B2_labels"] = confusion(yb2, pred).tolist()
    guard(EXP).mkdir(parents=True, exist_ok=True)
    agg_path(rid).write_text(json.dumps(agg, indent=1, default=float))
    clear_running(rid)
    return agg


def load_run(rid):
    return load_oof_csv(oof_path(rid))
