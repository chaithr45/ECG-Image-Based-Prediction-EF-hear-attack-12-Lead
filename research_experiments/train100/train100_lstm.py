"""NOT RUN: withdrawn at the owner's request (AMENDMENT_TRAIN100.md, note C). Kept as declared.
TRAIN100 (AMENDMENT_TRAIN100.md): the LSTM family retrained until (nearly) 100 % training accuracy, no early stopping.
Same networks as Phases 6A/6B (LSTM-fixed = L3: aligned 26-channel 10-s input at 100 Hz, BiLSTM-64; CNN-LSTM = L5:
3 x Conv1d k7 s2 -> BiLSTM-64), same head, class-balanced CE, AdamW lr 1e-3, batch 32, grad clip 1.0. Weight decay is
removed and training continues until training accuracy >= 99.5 % (checked every 5 epochs, cap 400 epochs).
Phase-2 patient-grouped 5-fold CV x 3 repeats, original labels, development ECGs only. Aggregates only.
    PYTHONPATH=<repo> python3 train100_lstm.py --out <dir> [--jobs 4]
"""
from __future__ import annotations

import argparse
import json
import time
from multiprocessing import Pool

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from research_experiments.phase_2.p2heads import class_w
from research_experiments.phase_5.p5common import corr_ci
from research_experiments.phase_6a_normal_mild_lstm import lstm as L
from research_experiments.phase_6a_normal_mild_lstm import signals
from research_experiments.phase_7 import p7common as C

B = C.B
Y = C.Y
MODELS = {"LSTM-fixed (L3)": "L3", "CNN-LSTM (L5)": "L5",                         # main run (3 repeats)
          "LSTM L0 (rhythm strip, 1 layer, uni)": "L0", "LSTM L1 (aligned, 1 layer, uni)": "L1",   # addendum B (repeat 0)
          "LSTM L2 (aligned, 2 layers, uni)": "L2"}                                    # L4 dropped (owner's request)
DEFAULT = ["LSTM-fixed (L3)", "CNN-LSTM (L5)"]
TARGET, MAX_EP, CHECK = 0.995, 400, 5
_X = {}


def X(kind="aligned"):
    if kind not in _X:
        _X[kind] = torch.from_numpy(signals.load(kind))
    return _X[kind]


@torch.no_grad()
def predict(m, idx, bs=128, kind="aligned"):
    m.eval(); x = X(kind)
    return torch.cat([m(x[idx[i:i + bs]]).argmax(1) for i in range(0, len(idx), bs)]).numpy()


def cell(task):
    name, r, k = task
    torch.set_num_threads(1); t0 = time.time()
    spec = L.VARIANTS[MODELS[name]]; kind = spec["input"]; x = X(kind); yt = torch.tensor(Y)
    fv = B.fold_vec(r); TR = np.flatnonzero(fv != k); te = np.flatnonzero(fv == k)
    torch.manual_seed(1000 * 42 + 10 * r + k)
    m = L.LSTMNet(x.shape[2], spec["layers"], spec["bi"], spec["inter_dropout"], spec["cnn"])
    opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=0.0)
    cw = class_w(Y[TR], "balanced"); g = torch.Generator().manual_seed(1000 * 42 + 10 * r + k)
    ep, acc, hist = 0, 0.0, []
    while ep < MAX_EP and acc < TARGET:
        m.train(); perm = TR[torch.randperm(len(TR), generator=g).numpy()]
        for i in range(0, len(perm), 32):
            j = perm[i:i + 32]
            loss = F.cross_entropy(m(x[j]), yt[j], weight=cw)
            opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(m.parameters(), 1.0); opt.step()
        ep += 1
        if ep % CHECK == 0:
            acc = float(np.mean(predict(m, TR, kind=kind) == Y[TR]))
            hist.append((ep, acc, float(np.mean(predict(m, te, kind=kind) == Y[te]))))
            if ep % 50 == 0:
                print(f"  [{name} r{r} k{k}] epoch {ep}: train {acc:.3f} held-out {hist[-1][2]:.3f} "
                      f"({(time.time() - t0) / 60:.0f} min)", flush=True)
    p_tr, p_te = predict(m, TR, kind=kind), predict(m, te, kind=kind); yte = Y[te]
    out = dict(model=name, repeat=r, fold=k, epochs=ep, minutes=(time.time() - t0) / 60, train_acc=float(np.mean(p_tr == Y[TR])),
               heldout_acc=float(np.mean(p_te == yte)), heldout_recall=[float(np.mean(p_te[yte == c] == c)) for c in range(4)],
               heldout_confusion=np.array([[np.sum((yte == i) & (p_te == j)) for j in range(4)] for i in range(4)]).tolist(), curve=hist)
    print(f"[{name} r{r} k{k}] {ep} epochs, {out['minutes']:.1f} min | train {out['train_acc']:.3f} held-out {out['heldout_acc']:.3f}", flush=True)
    return out


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--jobs", type=int, default=4)
    ap.add_argument("--models", nargs="+", default=DEFAULT); ap.add_argument("--repeats", nargs="+", type=int, default=[0, 1, 2])
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    X()
    tasks = [(n, r, k) for n in a.models for r in a.repeats for k in range(5)]
    with Pool(a.jobs) as pool:
        cells = list(pool.imap_unordered(cell, tasks))
    S = {}
    for n in a.models:
        c = [x for x in cells if x["model"] == n]
        S[n] = dict(train_accuracy=dict(mean=float(np.mean([x["train_acc"] for x in c])), ci95_corrected=[float(v) for v in corr_ci([x["train_acc"] for x in c])]),
                    heldout_accuracy=dict(mean=float(np.mean([x["heldout_acc"] for x in c])), ci95_corrected=[float(v) for v in corr_ci([x["heldout_acc"] for x in c])]),
                    heldout_recall_N_Mi_Mo_S=np.mean([x["heldout_recall"] for x in c], 0).round(3).tolist(),
                    epochs=dict(median=float(np.median([x["epochs"] for x in c])), max=int(max(x["epochs"] for x in c))),
                    n_cells=len(c), repeats=a.repeats)
        print(f"{n}: train {S[n]['train_accuracy']['mean']:.3f} | held-out {S[n]['heldout_accuracy']['mean']:.3f} "
              f"{np.round(S[n]['heldout_accuracy']['ci95_corrected'], 3).tolist()} | epochs median {S[n]['epochs']['median']}")
    with open(f"{a.out}/train100_lstm{a.tag}.json", "w") as f:
        json.dump(dict(amendment="AMENDMENT_TRAIN100.md", summary=S, cells=sorted(cells, key=lambda c: (c["model"], c["repeat"], c["fold"]))),
                  f, indent=1, default=float)


if __name__ == "__main__":
    main()
