"""TRAIN100-B (AMENDMENT_TRAIN100.md, addendum B): the networks of the original project and Phase 1 retrained until
(nearly) 100 % training accuracy on the ORIGINAL FIXED SPLIT (669 training / 168 validation ECGs, patient-disjoint),
as in the upper part of report Table 3.3. The third part of that split (210 ECGs) is the locked test set: it is never
loaded here (only the 'train' and 'val' rows of the manifest are read; images were rendered for train + val only).

Change against the original recipe, the same for every network: no early stopping, no augmentation, weight decay 0,
no label smoothing, no EMA, constant learning rate after the original warm-up; train until training accuracy
(evaluation mode, all 669 training ECGs) >= 99.5 % or the epoch cap. Architecture, input, batch size, learning rate,
class-balanced loss and gradient clipping as the original. Aggregates only; nothing per ECG is committed.

    PYTHONPATH=<repo> python3 train100_fixedsplit.py --out <dir> --models ResNet-18 ECGNet ... [--threads 4]
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F

TARGET = 0.995
SPLITS = ("train", "val")                         # never "test" (= locked test set)
IMAGE_NETS = {"ResNet-18": "resnet18", "ECGNet": "ecgnet", "EfficientNet-B0": "efficientnet_b0",
              "MobileNetV3-Large": "mobilenetv3", "ConvNeXt-Atto": "convnext_atto"}
# EfficientNet-B2 / ConvNeXt-Tiny (IMG70 architectures) left out at the owner's request (addendum B)
CAP = {"image": 100, "1d": 150, "ecgf": 60}
CAP_BIG = {}
ORIGINAL_TAG = {"ResNet-18": "resnet18_filtered", "ECGNet": "ecgnet_filtered", "EfficientNet-B0": "efficientnet_b0_filtered",
                "MobileNetV3-Large": "mobilenetv3_filtered", "ConvNeXt-Atto": "convnext_atto_filtered",
                "1-D SE-ResNet": "resnet1d", "HOG + RBF-SVM": "hog_svm_filtered", "XGBoost (waveform features)": "xgb_waveform",
                "Logistic regression (header)": "logreg_header", "Header 0.6 + ResNet-18 0.4": "fusion_header_resnet18"}


def cw_of(y):
    c = torch.bincount(y, minlength=4).float(); w = c.sum() / (4 * c)
    return w / w.mean()


@torch.no_grad()
def probs(model, x, prep, bs=32):
    model.eval()
    return torch.cat([F.softmax(model(prep(x[i:i + bs]))[:, :4], 1) for i in range(0, len(x), bs)])


def fit_deep(name, model, xtr, ytr, xva, yva, prep, groups, bs, cap, warm_epochs, freeze_eval=None):
    cw = cw_of(ytr)
    opt = torch.optim.AdamW(groups, weight_decay=0.0)
    spe = len(xtr) // bs
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / (warm_epochs * spe)))
    params = [p for p in model.parameters() if p.requires_grad]
    hist, t0, ep, tr_acc = [], time.time(), 0, 0.0
    while ep < cap and tr_acc < TARGET:
        ep += 1; model.train()
        if freeze_eval is not None:
            freeze_eval.eval()
        perm = torch.randperm(len(xtr))
        for i in range(spe):
            idx = perm[i * bs:(i + 1) * bs]
            loss = F.cross_entropy(model(prep(xtr[idx]))[:, :4], ytr[idx], weight=cw)
            opt.zero_grad(set_to_none=True); loss.backward(); nn.utils.clip_grad_norm_(params, 5.0); opt.step(); sched.step()
        ptr, pva = probs(model, xtr, prep), probs(model, xva, prep)
        tr_acc = float((ptr.argmax(1) == ytr).float().mean()); va_acc = float((pva.argmax(1) == yva).float().mean())
        tr_loss = float(F.nll_loss(ptr.clamp_min(1e-8).log(), ytr)); va_loss = float(F.nll_loss(pva.clamp_min(1e-8).log(), yva))
        hist.append(dict(epoch=ep, train_acc=tr_acc, val_acc=va_acc, train_loss=tr_loss, val_loss=va_loss,
                         minutes=(time.time() - t0) / 60))
        print(f"[{name}] ep {ep:3d} train acc {tr_acc:.3f} val acc {va_acc:.3f} | loss {tr_loss:.3f}/{va_loss:.3f} "
              f"| {hist[-1]['minutes']:.1f} min", flush=True)
    return dict(epochs=ep, reached_target=tr_acc >= TARGET, train_acc=tr_acc, val_acc=hist[-1]["val_acc"],
                train_loss=hist[-1]["train_loss"], val_loss=hist[-1]["val_loss"], minutes=hist[-1]["minutes"],
                best_val_acc_any_epoch=max(h["val_acc"] for h in hist), curve=hist,
                n_params=sum(p.numel() for p in model.parameters()),
                n_trainable=sum(p.numel() for p in model.parameters() if p.requires_grad)), ptr, pva


# ------------------------------------------------------------------ the networks
def image_net(name, seed=42):
    import timm
    from ecg_ef.data import Normalizer, load_split
    from ecg_ef.models import build_model
    (xtr, ytr, _), (xva, yva, _) = (load_split(s, "filtered", (256, 512)) for s in SPLITS)
    norm = Normalizer.fit(xtr)
    torch.manual_seed(seed); np.random.seed(seed)
    key = IMAGE_NETS[name]
    model = (timm.create_model(key[5:], pretrained=False, in_chans=1, num_classes=4, drop_rate=0.3) if key.startswith("timm:")
             else build_model(key, 4, pretrained=False)).to(memory_format=torch.channels_last)
    prep = lambda xb: norm(xb.float() / 255).contiguous(memory_format=torch.channels_last)
    return fit_deep(name, model, xtr, ytr, xva, yva, prep, [{"params": model.parameters(), "lr": 1e-3}], 16, CAP_BIG.get(name, CAP["image"]), 2)


def resnet1d(name, seed=42):
    from ecg_ef.train1d import ResNet1d, load_signal_split
    (ltr, _, ytr, _), (lva, _, yva, _) = (load_signal_split(s) for s in SPLITS)
    torch.manual_seed(seed); np.random.seed(seed)
    net = ResNet1d(4, rhythm=False)

    class W(nn.Module):
        def __init__(self):
            super().__init__(); self.net = net

        def forward(self, x):
            return self.net(x, None)
    model = W()
    return fit_deep(name, model, ltr, ytr, lva, yva, lambda xb: xb, [{"params": model.parameters(), "lr": 1e-3}], 32, CAP["1d"], 3)


def ecgf_finetune(name, seed=42):
    from research_experiments.common import ROOT, load_meta
    from research_experiments.ecgfounder.inputs import build_all
    from research_experiments.train_deep import FounderFT
    meta = load_meta(); z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    idx = {e: i for i, e in enumerate(z["ecg_id"])}
    d = {}
    for s in SPLITS:
        ms = meta[meta.split == s]; ii = np.array([idx[e] for e in ms.ecg_id])
        d[s] = (torch.from_numpy(build_all(z["leads"][ii], z["rhythm"][ii], "aligned")), torch.tensor(ms.label_idx.values))
    torch.manual_seed(seed); np.random.seed(seed)
    model = FounderFT("ce", [5, 6])
    bb = [p for p in model.base.parameters() if p.requires_grad]
    groups = [{"params": bb, "lr": 1e-4}, {"params": list(model.fc.parameters()), "lr": 1e-3}]
    return fit_deep(name, model, *d["train"], *d["val"], lambda xb: xb, groups, 16, CAP["ecgf"], 1, freeze_eval=model.base)


def classical(out):
    """HOG + RBF-SVM, XGBoost on waveform features, logistic regression on the header, and the original default
    pipeline (header 0.6 + ResNet-18 0.4), each fitted without the regularisation / early stopping of the original."""
    import xgboost as xgb
    from sklearn.decomposition import PCA
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler
    from sklearn.svm import SVC
    from ecg_ef.baseline import hog_features
    from ecg_ef.data import load_split
    from ecg_ef.tabular import balanced_weights, feature_sets
    res = {}

    def rec(name, ptr, ytr, pva, yva, **kw):          # ptr / pva: probabilities, or predicted labels (1-D)
        lab = lambda p: p if p.ndim == 1 else p.argmax(1)
        res[name] = dict(train_acc=float((lab(ptr) == ytr).mean()), val_acc=float((lab(pva) == yva).mean()), **kw)
        print(f"[{name}] train acc {res[name]['train_acc']:.3f} val acc {res[name]['val_acc']:.3f}", flush=True)
    (xtr, ytr, _), (xva, yva, _) = (load_split(s, "filtered", (128, 256)) for s in SPLITS)
    ftr, fva = hog_features(xtr), hog_features(xva); ytr, yva = ytr.numpy(), yva.numpy()
    svm = make_pipeline(StandardScaler(), PCA(150, random_state=0),
                        SVC(C=1e4, kernel="rbf", class_weight="balanced", probability=True, random_state=0)).fit(ftr, ytr)
    # accuracy from the SVM's own decision (predict); the argmax of its Platt-scaled probabilities (internal 5-fold
    # calibration, as the original evaluation used) is recorded too, because the two can disagree
    rec("HOG + RBF-SVM", svm.predict(ftr), ytr, svm.predict(fva), yva, C=1e4,
        proba_argmax_train_acc=float((svm.predict_proba(ftr).argmax(1) == ytr).mean()),
        proba_argmax_val_acc=float((svm.predict_proba(fva).argmax(1) == yva).mean()))
    h, sets = feature_sets(); tr, va = (h.split == "train").values, (h.split == "val").values
    y = h.label_idx.values
    w = sets["waveform"]
    xg = xgb.XGBClassifier(objective="multi:softprob", n_estimators=2000, learning_rate=0.03, max_depth=4, subsample=1.0,
                           colsample_bytree=1.0, min_child_weight=0, reg_lambda=0.0, tree_method="hist", n_jobs=4, random_state=0)
    xg.fit(w[tr], y[tr], sample_weight=balanced_weights(y[tr]))
    rec("XGBoost (waveform features)", xg.predict_proba(w[tr]), y[tr], xg.predict_proba(w[va]), y[va], n_estimators=2000)
    Xh = sets["header"]
    lr = make_pipeline(StandardScaler(), LogisticRegression(C=1e4, class_weight="balanced", max_iter=20000)).fit(Xh[tr], y[tr])
    phtr, phva = lr.predict_proba(Xh[tr]), lr.predict_proba(Xh[va])
    rec("Logistic regression (header)", phtr, y[tr], phva, y[va], C=1e4)
    rn = out / "resnet18_probs_private.npz"                     # scratch only (per-ECG; never committed)
    if rn.exists():
        z = np.load(rn, allow_pickle=False)
        # header rows are in manifest order within split, as the image loader -> same order (checked by ids)
        assert (z["train_ids"] == np.array(h.ecg_id[tr].tolist(), dtype=str)).all() and (z["val_ids"] == np.array(h.ecg_id[va].tolist(), dtype=str)).all()
        rec("Header 0.6 + ResNet-18 0.4", 0.6 * phtr + 0.4 * z["train"], y[tr], 0.6 * phva + 0.4 * z["val"], y[va])
    return res


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--out", required=True); ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--models", nargs="+", required=True)
    a = ap.parse_args()
    from pathlib import Path
    out = Path(a.out); torch.set_num_threads(a.threads)
    for name in a.models:
        f = out / f"train100_fixedsplit__{name.replace(' ', '_').replace('/', '-')}.json"
        if f.exists():
            print("done already:", name); continue
        t0 = time.time()
        if name == "classical":
            r = classical(out)
        else:
            fn = image_net if name in IMAGE_NETS else resnet1d if name == "1-D SE-ResNet" else ecgf_finetune
            r, ptr, pva = fn(name)
            if name == "ResNet-18":                                  # keep probabilities (scratch only) for the fusion
                from research_experiments.common import load_meta
                m = load_meta()
                np.savez(out / "resnet18_probs_private.npz", train=ptr.numpy(), val=pva.numpy(),
                         train_ids=np.array(m[m.split == "train"].ecg_id.tolist(), dtype=str), val_ids=np.array(m[m.split == "val"].ecg_id.tolist(), dtype=str))
            r = {name: r}
        json.dump(dict(amendment="AMENDMENT_TRAIN100.md (addendum B)", split="original fixed split: 669 train / 168 val; "
                       "locked test not loaded", results=r, wall_minutes=(time.time() - t0) / 60), open(f, "w"), indent=1)


if __name__ == "__main__":
    main()
