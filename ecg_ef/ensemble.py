"""Soft-voting ensemble: average the class probabilities of several trained CNNs.

    python -m ecg_ef.ensemble                       # all CNN checkpoints in models/
    python -m ecg_ef.ensemble --members ecgnet_filtered resnet18_filtered

Writes results/ensemble/ in the same format as a single model, so compare.py lists it.
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
import torch

from .config import CLASSES, HF_TARGET_SENSITIVITY, MODELS_DIR, RESULTS_DIR
from .data import Normalizer, load_split
from .metrics import evaluate, pick_hf_threshold
from .predict import load_checkpoint
from .train import plot_confusion, predict_probs


def member_probs(tag: str):
    model, ckpt = load_checkpoint(MODELS_DIR / f"{tag}.pt")
    norm = Normalizer(**ckpt["normalizer"])
    out = {}
    for split in ("val", "test"):
        x, y, df = load_split(split, ckpt["variant"], tuple(ckpt["img_hw"]))
        out[split] = (predict_probs(model, x, norm), y.numpy(), df)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="*", default=None, help="checkpoint tags (default: all)")
    ap.add_argument("--name", default="ensemble")
    args = ap.parse_args()
    torch.set_num_threads(4)
    tags = args.members or sorted(p.stem for p in MODELS_DIR.glob("*.pt"))
    print("members:", tags)

    probs = {s: [] for s in ("val", "test")}
    for t in tags:
        r = member_probs(t)
        for s in probs:
            probs[s].append(r[s][0])
        yva, yte, dte = r["val"][1], r["test"][1], r["test"][2]
    pva, pte = np.mean(probs["val"], 0), np.mean(probs["test"], 0)

    hf_thr = pick_hf_threshold(yva, pva, HF_TARGET_SENSITIVITY)
    val_m, test_m = evaluate(yva, pva, hf_thr), evaluate(yte, pte, hf_thr)
    out = RESULTS_DIR / args.name
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(tag=args.name, model="ensemble", variant="mixed", pretrained=False,
                   img_hw=[256, 512], params=None, members=tags, hf_threshold=hf_thr,
                   val=val_m, test=test_m)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(test_m["confusion_matrix"],
                   f"Ensemble ({len(tags)} CNNs) - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    pred = pd.DataFrame(pte, columns=[f"p_{c}" for c in CLASSES])
    pred.insert(0, "ecg_id", dte.ecg_id.values); pred.insert(1, "true", dte.label.values)
    pred.insert(2, "pred", [CLASSES[i] for i in pte.argmax(1)])
    pred["hf_likelihood"] = 1 - pte[:, 0]
    pred.round(4).to_csv(out / "test_predictions.csv", index=False)
    print(f"[{args.name}] VAL macro-F1 {val_m['macro_f1']:.3f} | TEST acc {test_m['accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}  "
          f"HF-AUC {test_m['hf_screen']['auc']:.3f}")


if __name__ == "__main__":
    main()
