"""What happens if the ECGFounder MLP head is trained for 150 epochs without early stopping?
Same inputs (frozen ECGFounder f6 embedding), same folds (Phase-2 patient-grouped 5-fold CV, repeat 0, original
labels), same head as the project (Dropout 0.5 -> Linear 1024->64 -> GELU -> Dropout 0.5 -> Linear 64->4,
class-balanced cross-entropy, AdamW lr 1e-3, batch 64). After every epoch: accuracy on the training ECGs and on the
held-out fold, both measured in eval mode (no dropout). Variant B removes all regularisation (512 hidden units,
no dropout, no weight decay) to show pure memorisation. Aggregates only (per-epoch means over the 5 folds).
    PYTHONPATH=<repo> python3 epochs150_demo.py <outdir>
"""
import json
import sys

import numpy as np
import torch

from ecg_ef.metrics import evaluate
from research_experiments.phase_2.p2common import s3
from research_experiments.phase_2.p2heads import Head, class_w
from research_experiments.phase_5.p5common import ecgf_f6
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

OUT = sys.argv[1]
EPOCHS, PATIENCE = 150, 25
X = ecgf_f6().astype(np.float32); Y = B.Y
torch.set_num_threads(4)

VARIANTS = {
    "A": dict(label="Project head (64 units, dropout 0.5, weight decay 0.1)", hidden=64, dropout=0.5, wd=0.1),
    "B": dict(label="No regularisation (512 units, no dropout, no weight decay)", hidden=512, dropout=0.0, wd=0.0),
}


def acc(m, x, y):
    with torch.no_grad():
        return float((m.probs(x).argmax(1).numpy() == y).mean())


res = {}
for key, v in VARIANTS.items():
    tr_curves, te_curves, va_curves, stop_eps, stop_te, stop_tr = [], [], [], [], [], []
    for k in range(5):
        itr, iva, ote = B.split("ORIG", 0, k)
        mu, sd = X[itr].mean(0), X[itr].std(0) + 1e-6
        T = lambda idx: torch.tensor((X[idx] - mu) / sd)
        xtr, xva, xte = T(itr), T(iva), T(ote); ytr = torch.tensor(Y[itr])
        cw = class_w(Y[itr], "balanced")
        torch.manual_seed(42)
        m = Head(X.shape[1], "ce", "mlp", v["hidden"], v["dropout"])
        opt = torch.optim.AdamW(m.parameters(), lr=1e-3, weight_decay=v["wd"])
        g = torch.Generator().manual_seed(42)
        tr, va, te = [], [], []
        best, best_ep, bad, stopped = -1, 0, 0, None
        for ep in range(EPOCHS):
            m.train(); idx = torch.randperm(len(itr), generator=g)
            for i in range(0, len(idx), 64):
                b = idx[i:i + 64]
                lo = m.loss(xtr[b], ytr[b], cw, {}); opt.zero_grad(); lo.backward(); opt.step()
            m.eval()
            tr.append(acc(m, xtr, Y[itr])); te.append(acc(m, xte, Y[ote])); va.append(acc(m, xva, Y[iva]))
            with torch.no_grad():
                sc = s3(evaluate(Y[iva], m.probs(xva).numpy()))      # the project's early-stopping score (inner validation)
            if stopped is None:
                if sc > best + 1e-6:
                    best, best_ep, bad = sc, ep, 0
                else:
                    bad += 1
                if bad >= PATIENCE:
                    stopped = best_ep
        stop_ep = stopped if stopped is not None else best_ep
        tr_curves.append(tr); te_curves.append(te); va_curves.append(va)
        stop_eps.append(stop_ep + 1); stop_te.append(te[stop_ep]); stop_tr.append(tr[stop_ep])
        print(f"{key} fold {k}: early stop at epoch {stop_ep + 1}: train {tr[stop_ep]:.3f} held-out {te[stop_ep]:.3f} | "
              f"epoch 150: train {tr[-1]:.3f} held-out {te[-1]:.3f}", flush=True)
    res[key] = dict(label=v["label"], train=np.mean(tr_curves, 0).tolist(), heldout=np.mean(te_curves, 0).tolist(),
                    train_min=np.min(tr_curves, 0).tolist(), train_max=np.max(tr_curves, 0).tolist(),
                    heldout_min=np.min(te_curves, 0).tolist(), heldout_max=np.max(te_curves, 0).tolist(),
                    early_stop_epoch_mean=float(np.mean(stop_eps)), early_stop_epochs=stop_eps,
                    early_stop_train=float(np.mean(stop_tr)), early_stop_heldout=float(np.mean(stop_te)),
                    epoch150_train=float(np.mean([c[-1] for c in tr_curves])), epoch150_heldout=float(np.mean([c[-1] for c in te_curves])),
                    best_heldout_any_epoch=float(np.max(np.mean(te_curves, 0))))
json.dump(dict(setup=__doc__.split("\n")[0], folds="Phase-2 patient-grouped 5-fold CV, repeat 0", n_ecgs=int(len(Y)), variants=res),
          open(f"{OUT}/epochs150_results.json", "w"), indent=1)
for key, r in res.items():
    print(key, r["label"], "| early stop epoch (mean)", round(r["early_stop_epoch_mean"], 1), "| at early stop: train",
          round(r["early_stop_train"], 3), "held-out", round(r["early_stop_heldout"], 3), "| at 150: train",
          round(r["epoch150_train"], 3), "held-out", round(r["epoch150_heldout"], 3))
