"""Collect every model's results into one comparison table + chart, and pick the model
used by predict.py.

For each model two 4-class decision rules are reported:
  * argmax   - the most probable class
  * ordinal  - expected severity sum_k k*p_k cut into 4 bands; the 3 cut-points are fit
               on the VALIDATION set (macro-F1). Argmax almost never outputs the middle
               classes (Mild/Moderate) because their probability is shared with the
               neighbouring classes; the ordinal rule respects the Normal<Mild<Moderate<Severe
               ordering.
Model selection uses the VALIDATION set only (mean of macro-F1, QWK and heart-failure AUC),
so the test numbers stay an unbiased estimate.

    python -m ecg_ef.compare
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import RESULTS_DIR, ROOT
from .ensemble import load_predictions
from .metrics import evaluate, fit_ordinal_cutpoints
from .train import plot_confusion, selection_score

PRETTY = {"ecgnet": "ECGNet (custom SE-ResNet)", "resnet18": "ResNet-18", "resnet34": "ResNet-34",
          "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121",
          "mobilenetv3": "MobileNetV3-Large", "convnext_atto": "ConvNeXt-Atto",
          "hog_svm": "HOG + SVM (classical)", "ensemble": "Ensemble (mean of all CNNs)"}

INK, MUTED, GRID, BAR, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def model_name(m):
    name = PRETTY.get(m["model"], m["model"])
    if m["variant"] == "raw":
        name += ", no grid filter"
    if m["model"] == "ecgnet" and tuple(m["img_hw"]) != (256, 512):
        name += f", {m['img_hw'][0]}x{m['img_hw'][1]}"
    if m.get("pretrained"):
        name += ", ImageNet"
    return name


def load_rows():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        tag = m["tag"]
        yv, pv, _ = load_predictions(tag, "val")
        yt, pt, _ = load_predictions(tag, "test")
        cut = fit_ordinal_cutpoints(yv, pv)
        ov = evaluate(yv, pv, m["hf_threshold"], cut)
        ot = evaluate(yt, pt, m["hf_threshold"], cut)
        (f.parent / "ordinal.json").write_text(json.dumps(
            {"cutpoints": cut, "val": ov, "test": ot}, indent=2))
        plot_confusion(ot["confusion_matrix"], f"{tag} - test, ordinal decoding "
                       f"(macro-F1 {ot['macro_f1']:.2f})", f.parent / "confusion_matrix_ordinal.png")
        t, v = m["test"], m["val"]
        rows.append({
            "tag": tag, "Model": model_name(m),
            "Params (M)": round(m["params"] / 1e6, 1) if m.get("params") else None,
            "Train (min)": m.get("train_minutes"),
            "Val score": selection_score(v),
            "Accuracy": t["accuracy"], "Balanced acc": t["balanced_accuracy"],
            "Macro-F1": t["macro_f1"], "QWK": t["qwk"],
            "Ord. accuracy": ot["accuracy"], "Ord. balanced acc": ot["balanced_accuracy"],
            "Ord. macro-F1": ot["macro_f1"], "Ord. QWK": ot["qwk"],
            "HF AUC": t["hf_screen"]["auc"], "HF sensitivity": t["hf_screen"]["sensitivity"],
            "HF specificity": t["hf_screen"]["specificity"], "EF<=40% AUC": t["ef_le_40"]["auc"],
            "checkpoints": (None if m["model"] == "hog_svm" else
                            [f"models/{x}.pt" for x in m["members"]] if m["model"] == "ensemble"
                            else [f"models/{tag}.pt"]),
            "hf_threshold": m["hf_threshold"], "cutpoints": cut,
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path):
    df = df.sort_values("HF AUC")
    metrics = [("Accuracy", "4-class accuracy (argmax)"),
               ("Ord. macro-F1", "4-class macro-F1 (ordinal)"),
               ("HF AUC", "Heart-failure screen AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13.5, 0.42 * len(df) + 1.5), sharey=True,
                             facecolor=SURFACE)
    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        ax.barh(df.Model, df[col], color=BAR, height=0.62)
        for y, v in enumerate(df[col]):
            ax.text(v + 0.012, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axes[0].tick_params(axis="y", colors=INK, labelsize=9)
    fig.suptitle("Held-out test patients (n = 210); sorted by heart-failure screen AUC",
                 x=0.01, ha="left", fontsize=11.5, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def to_md(df: pd.DataFrame) -> str:
    md = df.copy()
    for c in md.columns:
        if md[c].dtype == float and c not in ("Params (M)", "Train (min)"):
            md[c] = md[c].map(lambda x: f"{x:.3f}" if pd.notna(x) else "")
    return md.fillna("").to_markdown(index=False)


def main():
    df = load_rows()
    if df.empty:
        raise SystemExit("no results/*/metrics.json yet")
    df = df.sort_values("HF AUC", ascending=False)
    show = df.drop(columns=["tag", "checkpoints", "hf_threshold", "cutpoints"])
    show.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    (RESULTS_DIR / "comparison.md").write_text(to_md(show) + "\n")
    plot(df, RESULTS_DIR / "comparison.png")

    cnn = df[df.checkpoints.notna()]
    best = cnn.sort_values("Val score", ascending=False).iloc[0]
    if all((ROOT / c).exists() for c in best.checkpoints):
        (RESULTS_DIR / "best_model.json").write_text(json.dumps(
            {"tag": best.tag, "model": best.Model, "checkpoints": best.checkpoints,
             "hf_threshold": best.hf_threshold, "decoding": "ordinal",
             "ordinal_cutpoints": best.cutpoints,
             "selected_by": "validation score = mean(macro-F1, QWK, HF AUC)",
             "val_score": best["Val score"]}, indent=2))
    print(to_md(show))
    print(f"\nselected for predict.py (best validation score): {best.Model}  -> {best.checkpoints}")


if __name__ == "__main__":
    main()
