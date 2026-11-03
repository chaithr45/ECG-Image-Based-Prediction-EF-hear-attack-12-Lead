"""Train one CNN on the ECG images and evaluate it on the held-out test patients.

Examples
--------
    python -m ecg_ef.train --model ecgnet
    python -m ecg_ef.train --model resnet18 --pretrained      # needs internet for weights
    python -m ecg_ef.train --model ecgnet --variant raw        # no grid filtering
"""
from __future__ import annotations

import argparse
import copy
import json
import math
import random
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CLASSES, IMG_H, IMG_W, MODELS_DIR, RESULTS_DIR
from .data import Normalizer, augment, load_split
from .metrics import evaluate
from .models import MODEL_NAMES, build_model


def seed_all(seed: int):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)


@torch.no_grad()
def predict_probs(model, imgs_u8, norm, bs=32):
    model.eval()
    out = []
    for i in range(0, len(imgs_u8), bs):
        x = norm(imgs_u8[i:i + bs].float() / 255).contiguous(memory_format=torch.channels_last)
        out.append(F.softmax(model(x), 1))
    return torch.cat(out).numpy()


def plot_confusion(cm, title, path):
    cm = np.asarray(cm)
    frac = cm / cm.sum(1, keepdims=True).clip(min=1)
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(frac, cmap="Blues", vmin=0, vmax=1)
    ax.set_xticks(range(4), CLASSES); ax.set_yticks(range(4), CLASSES)
    ax.set_xlabel("Predicted"); ax.set_ylabel("True (echo)")
    for i in range(4):
        for j in range(4):
            ax.text(j, i, f"{cm[i, j]}\n{frac[i, j]:.0%}", ha="center", va="center",
                    fontsize=8, color="white" if frac[i, j] > 0.55 else "black")
    ax.set_title(title, fontsize=10)
    fig.tight_layout(); fig.savefig(path, dpi=130); plt.close(fig)


def plot_history(hist: pd.DataFrame, path):
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(hist.epoch, hist.train_loss, label="train")
    axes[0].plot(hist.epoch, hist.val_loss, label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(hist.epoch, hist.train_acc, label="train acc")
    axes[1].plot(hist.epoch, hist.val_acc, label="val acc")
    axes[1].plot(hist.epoch, hist.val_macro_f1, label="val macro-F1")
    axes[1].set_title("Validation"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout(); fig.savefig(path, dpi=110); plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="ecgnet", choices=MODEL_NAMES)
    ap.add_argument("--variant", default="filtered", choices=["filtered", "raw"])
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--patience", type=int, default=12)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=None)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--label-smoothing", type=float, default=0.05)
    ap.add_argument("--img-size", type=int, nargs=2, default=[IMG_H, IMG_W], metavar=("H", "W"))
    ap.add_argument("--pretrained", action="store_true", help="ImageNet init (downloads weights)")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--threads", type=int, default=4)
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()

    torch.set_num_threads(args.threads)
    seed_all(args.seed)
    tag = args.tag or f"{args.model}_{args.variant}" + ("_pretrained" if args.pretrained else "")
    lr = args.lr or (3e-4 if args.pretrained else 1e-3)
    out_dir = RESULTS_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(exist_ok=True)
    img_hw = tuple(args.img_size)

    xtr, ytr, _ = load_split("train", args.variant, img_hw)
    xva, yva, _ = load_split("val", args.variant, img_hw)
    xte, yte, dte = load_split("test", args.variant, img_hw)
    norm = Normalizer.fit(xtr)
    print(f"[{tag}] train {len(xtr)}  val {len(xva)}  test {len(xte)}  img {img_hw}  lr {lr}")

    # Class imbalance (Normal 46% ... Severe 11%): inverse-sqrt-frequency loss weights.
    counts = torch.bincount(ytr, minlength=len(CLASSES)).float()
    class_w = (counts.sum() / (len(CLASSES) * counts)).sqrt()
    class_w = class_w / class_w.mean()
    print("class weights", dict(zip(CLASSES, class_w.round(decimals=2).tolist())))

    model = build_model(args.model, len(CLASSES), pretrained=args.pretrained)
    model = model.to(memory_format=torch.channels_last)
    n_params = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=args.wd)
    steps_per_epoch = len(xtr) // args.batch_size
    total, warm = args.epochs * steps_per_epoch, 2 * steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: (s + 1) / warm if s < warm else
        0.5 * (1 + math.cos(math.pi * (s - warm) / max(1, total - warm))))

    best, best_state, best_epoch, hist, bad = -1.0, None, 0, [], 0
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        perm = torch.randperm(len(xtr))
        tl, tc, tn = 0.0, 0, 0
        for i in range(steps_per_epoch):
            idx = perm[i * args.batch_size:(i + 1) * args.batch_size]
            x = augment(xtr[idx].float() / 255)
            x = norm(x).contiguous(memory_format=torch.channels_last)
            y = ytr[idx]
            logits = model(x)
            loss = F.cross_entropy(logits, y, weight=class_w, label_smoothing=args.label_smoothing)
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step(); sched.step()
            tl += loss.item() * len(y); tc += (logits.argmax(1) == y).sum().item(); tn += len(y)

        pva = predict_probs(model, xva, norm)
        vloss = F.nll_loss(torch.log(torch.from_numpy(pva).clamp_min(1e-8)), yva).item()
        vm = evaluate(yva.numpy(), pva)
        hist.append(dict(epoch=epoch, lr=sched.get_last_lr()[0], train_loss=tl / tn,
                         train_acc=tc / tn, val_loss=vloss, val_acc=vm["accuracy"],
                         val_bal_acc=vm["balanced_accuracy"], val_macro_f1=vm["macro_f1"],
                         val_qwk=vm["qwk"], val_hf_auc=vm["hf_screen"]["auc"],
                         elapsed_min=(time.time() - t0) / 60))
        score = vm["macro_f1"] + 1e-3 * vm["qwk"]          # select on val macro-F1
        flag = ""
        if score > best:
            best, best_state, best_epoch, bad, flag = score, copy.deepcopy(model.state_dict()), epoch, 0, " *"
        else:
            bad += 1
        h = hist[-1]
        print(f"ep {epoch:2d}  loss {h['train_loss']:.3f}/{h['val_loss']:.3f}  "
              f"acc {h['train_acc']:.3f}/{h['val_acc']:.3f}  vF1 {h['val_macro_f1']:.3f}  "
              f"vQWK {h['val_qwk']:.3f}  vHF-AUC {h['val_hf_auc']:.3f}  "
              f"{h['elapsed_min']:.1f}m{flag}", flush=True)
        if bad >= args.patience:
            print(f"early stop (no val improvement for {args.patience} epochs)")
            break

    train_min = (time.time() - t0) / 60
    model.load_state_dict(best_state)
    pva = predict_probs(model, xva, norm)
    pte = predict_probs(model, xte, norm)
    val_m, test_m = evaluate(yva.numpy(), pva), evaluate(yte.numpy(), pte)
    test_m["per_year_accuracy"] = {
        str(y): float((pte.argmax(1)[dte.year.values == y] == yte.numpy()[dte.year.values == y]).mean())
        for y in sorted(dte.year.unique())}

    ckpt = dict(model_name=args.model, variant=args.variant, img_hw=img_hw, classes=CLASSES,
                normalizer=norm.state(), state_dict=best_state, best_epoch=best_epoch,
                val_metrics=val_m, args=vars(args))
    torch.save(ckpt, MODELS_DIR / f"{tag}.pt")

    summary = dict(tag=tag, model=args.model, variant=args.variant, pretrained=args.pretrained,
                   img_hw=img_hw, params=n_params, epochs_run=len(hist), best_epoch=best_epoch,
                   train_minutes=round(train_min, 1), lr=lr, val=val_m, test=test_m)
    (out_dir / "metrics.json").write_text(json.dumps(summary, indent=2))
    hist = pd.DataFrame(hist); hist.to_csv(out_dir / "history.csv", index=False)
    plot_history(hist, out_dir / "training_curves.png")
    plot_confusion(test_m["confusion_matrix"],
                   f"{tag} - test (acc {test_m['accuracy']:.1%})", out_dir / "confusion_matrix.png")
    pred = pd.DataFrame(pte, columns=[f"p_{c}" for c in CLASSES])
    pred.insert(0, "ecg_id", dte.ecg_id); pred.insert(1, "true", dte.label)
    pred.insert(2, "pred", [CLASSES[i] for i in pte.argmax(1)])
    pred["hf_likelihood"] = 1 - pte[:, 0]
    pred.round(4).to_csv(out_dir / "test_predictions.csv", index=False)

    print(f"\n[{tag}] best epoch {best_epoch}  ({train_min:.1f} min)")
    print(f"  TEST  acc {test_m['accuracy']:.3f}  bal-acc {test_m['balanced_accuracy']:.3f}  "
          f"macro-F1 {test_m['macro_f1']:.3f}  QWK {test_m['qwk']:.3f}")
    hf = test_m["hf_screen"]
    print(f"  HF screen (reduced EF vs normal): AUC {hf['auc']:.3f}  sens {hf['sensitivity']:.3f}  "
          f"spec {hf['specificity']:.3f}")
    print(f"  EF<=40% detection: AUC {test_m['ef_le_40']['auc']:.3f}")


if __name__ == "__main__":
    main()
