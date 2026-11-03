"""Predict LVEF category + heart-failure likelihood for new 12-lead ECG PDFs/images.

    python -m ecg_ef.predict path/to/ecg.pdf
    python -m ecg_ef.predict some_folder/ --csv predictions.csv --gradcam heatmaps/
    python -m ecg_ef.predict ecg.png --model models/resnet18_filtered.pt

Output per ECG:
  * LVEF category: Normal / Mild / Moderate / Severe (+ probability of each)
  * Heart-failure likelihood = P(Mild) + P(Moderate) + P(Severe) = 1 - P(Normal)
  * Further cardiac evaluation needed: YES if likelihood >= 50%, else NO

This is a research screening aid, not a diagnosis; confirm with echocardiography.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

from .config import CLASSES, FURTHER_EVALUATION, HF_THRESHOLD, RESULTS_DIR, ROOT
from .data import Normalizer
from .models import build_model
from .preprocess import load_waveform, prepare_for_model

SUPPORTED = {".pdf", ".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"}


def default_checkpoint() -> Path:
    best = RESULTS_DIR / "best_model.json"
    if best.exists():
        return ROOT / json.loads(best.read_text())["checkpoint"]
    raise SystemExit("No results/best_model.json - pass --model models/<name>.pt")


def load_checkpoint(path: Path):
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(ckpt["model_name"], len(ckpt["classes"]))
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def gradcam(model, x, target: int) -> np.ndarray:
    """Grad-CAM on the last convolutional feature map -> [h, w] in [0, 1]."""
    feats = model.forward_features(x)
    feats.retain_grad()
    logits = model.forward_head(feats)
    model.zero_grad(set_to_none=True)
    logits[0, target].backward()
    w = feats.grad.mean((2, 3), keepdim=True)
    cam = F.relu((w * feats).sum(1))[0].detach().numpy()
    return cam / cam.max() if cam.max() > 0 else cam


def save_gradcam(src: Path, cam: np.ndarray, title: str, out_path: Path):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    rgb = load_waveform(src)
    h, w = rgb.shape[:2]
    heat = cv2.resize(cam, (w, h), interpolation=cv2.INTER_CUBIC)
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.imshow(rgb); ax.imshow(heat, cmap="jet", alpha=0.35, vmin=0, vmax=1)
    ax.set_title(title, fontsize=11); ax.axis("off")
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def iter_inputs(paths):
    for p in map(Path, paths):
        if p.is_dir():
            yield from sorted(q for q in p.rglob("*") if q.suffix.lower() in SUPPORTED)
        else:
            yield p


def main():
    ap = argparse.ArgumentParser(description="Predict LVEF category from 12-lead ECG images")
    ap.add_argument("inputs", nargs="+", help="ECG PDF/image files or folders")
    ap.add_argument("--model", type=Path, default=None, help="checkpoint (default: best model)")
    ap.add_argument("--csv", type=Path, default=None, help="write all predictions to CSV")
    ap.add_argument("--gradcam", type=Path, default=None, help="folder for Grad-CAM heatmaps")
    args = ap.parse_args()

    ckpt_path = args.model or default_checkpoint()
    model, ckpt = load_checkpoint(ckpt_path)
    norm = Normalizer(**ckpt["normalizer"])
    variant, img_hw = ckpt["variant"], tuple(ckpt["img_hw"])
    print(f"model: {ckpt_path.name}  ({ckpt['model_name']}, {variant} images, {img_hw})\n")
    if args.gradcam:
        args.gradcam.mkdir(parents=True, exist_ok=True)

    rows = []
    for path in iter_inputs(args.inputs):
        im = prepare_for_model(path, variant, img_hw)
        x = norm(torch.from_numpy(im)[None, None].float() / 255)
        with torch.no_grad():
            probs = F.softmax(model(x), 1)[0].numpy()
        k = int(probs.argmax())
        cat = CLASSES[k]
        hf = float(1 - probs[0])
        needs = hf >= HF_THRESHOLD
        rec = FURTHER_EVALUATION[cat] if needs == (cat != "Normal") else (
            "YES - borderline: most probable class is Normal but reduced-EF probability "
            ">= 50%; consider echo" if needs else
            "NO  - borderline: reduced-EF probability < 50%; clinical judgement")
        rows.append(dict(file=str(path), category=cat, confidence=float(probs[k]),
                         hf_likelihood=hf, further_evaluation="YES" if needs else "NO",
                         recommendation=rec, **{f"p_{c}": float(p) for c, p in zip(CLASSES, probs)}))
        print(f"{path.name}")
        print(f"  LVEF category          : {cat}  ({probs[k]:.0%})   "
              + "  ".join(f"{c} {p:.0%}" for c, p in zip(CLASSES, probs)))
        print(f"  Heart-failure likelihood: {hf:.0%}")
        print(f"  Further evaluation     : {rec}\n")

        if args.gradcam:
            cam = gradcam(model, x.requires_grad_(False), k)
            save_gradcam(path, cam, f"{path.stem[:60]}  ->  {cat} ({probs[k]:.0%}),  "
                         f"HF likelihood {hf:.0%}", args.gradcam / f"{path.stem}_gradcam.png")

    if args.csv:
        pd.DataFrame(rows).to_csv(args.csv, index=False)
        print(f"saved {len(rows)} predictions -> {args.csv}")


if __name__ == "__main__":
    main()
