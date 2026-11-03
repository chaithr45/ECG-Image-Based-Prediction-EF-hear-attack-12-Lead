"""Collect results/*/metrics.json into one comparison table + chart, and pick the
model used by predict.py. The model is chosen on the VALIDATION set (macro-F1) so the
test numbers stay an unbiased estimate.

    python -m ecg_ef.compare
"""
from __future__ import annotations

import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from .config import MODELS_DIR, RESULTS_DIR, ROOT

PRETTY = {"ecgnet": "ECGNet (custom SE-ResNet)", "resnet18": "ResNet-18", "resnet34": "ResNet-34",
          "efficientnet_b0": "EfficientNet-B0", "densenet121": "DenseNet-121",
          "mobilenetv3": "MobileNetV3-L", "convnext_atto": "ConvNeXt-Atto",
          "hog_svm": "HOG + SVM (classical)"}

INK, MUTED, GRID, BAR, SURFACE = "#0b0b0b", "#52514e", "#e4e3df", "#2a78d6", "#fcfcfb"


def load_rows():
    rows = []
    for f in sorted(RESULTS_DIR.glob("*/metrics.json")):
        m = json.loads(f.read_text())
        t, v = m["test"], m["val"]
        name = PRETTY.get(m["model"], m["model"])
        if m["variant"] == "raw":
            name += " - no grid filter"
        if m.get("pretrained"):
            name += " - ImageNet"
        rows.append({
            "tag": m["tag"], "Model": name, "Input": f"{m['variant']} {m['img_hw'][0]}x{m['img_hw'][1]}",
            "Params (M)": round(m["params"] / 1e6, 1) if m.get("params") else None,
            "Train (min)": m.get("train_minutes"),
            "Val macro-F1": v["macro_f1"],
            "Test accuracy": t["accuracy"], "Test balanced acc": t["balanced_accuracy"],
            "Test macro-F1": t["macro_f1"], "Test QWK": t["qwk"],
            "Within 1 class": t["within_one_category"],
            "HF-screen AUC": t["hf_screen"]["auc"], "HF sensitivity": t["hf_screen"]["sensitivity"],
            "HF specificity": t["hf_screen"]["specificity"], "EF<=40% AUC": t["ef_le_40"]["auc"],
            "checkpoint": f"models/{m['tag']}.pt" if m["model"] != "hog_svm" else None,
        })
    return pd.DataFrame(rows)


def plot(df: pd.DataFrame, path):
    df = df.sort_values("Test accuracy")
    metrics = [("Test accuracy", "4-class accuracy"), ("Test macro-F1", "4-class macro-F1"),
               ("HF-screen AUC", "Heart-failure screen AUC")]
    fig, axes = plt.subplots(1, 3, figsize=(13, 0.45 * len(df) + 1.6), sharey=True,
                             facecolor=SURFACE)
    for ax, (col, title) in zip(axes, metrics):
        ax.set_facecolor(SURFACE)
        ax.barh(df.Model, df[col], color=BAR, height=0.62)
        for y, v in enumerate(df[col]):
            ax.text(v + 0.01, y, f"{v:.2f}", va="center", fontsize=8.5, color=INK)
        ax.set_xlim(0, 1.0)
        ax.set_title(title, fontsize=10.5, color=INK, loc="left")
        ax.xaxis.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(GRID)
        ax.tick_params(colors=MUTED, labelsize=8.5, length=0)
    axes[0].tick_params(axis="y", colors=INK, labelsize=9)
    fig.suptitle("Model comparison on held-out test patients (n=210)", x=0.01, ha="left",
                 fontsize=12, color=INK)
    fig.tight_layout(); fig.savefig(path, dpi=130, facecolor=SURFACE); plt.close(fig)


def main():
    df = load_rows()
    if df.empty:
        raise SystemExit("no results/*/metrics.json yet")
    df = df.sort_values("Test accuracy", ascending=False)
    show = df.drop(columns=["tag", "checkpoint"])
    show.to_csv(RESULTS_DIR / "comparison.csv", index=False)
    fmt = {c: "{:.3f}" for c in show.columns if show[c].dtype == float
           and c not in ("Params (M)", "Train (min)")}
    md = show.copy()
    for c, f in fmt.items():
        md[c] = md[c].map(lambda x: f.format(x) if pd.notna(x) else "")
    (RESULTS_DIR / "comparison.md").write_text(md.to_markdown(index=False) + "\n")
    plot(df, RESULTS_DIR / "comparison.png")

    cnn = df[df.checkpoint.notna()]
    best = cnn.sort_values("Val macro-F1", ascending=False).iloc[0]
    if (ROOT / best.checkpoint).exists():
        (RESULTS_DIR / "best_model.json").write_text(json.dumps(
            {"tag": best.tag, "checkpoint": best.checkpoint, "selected_by": "validation macro-F1",
             "val_macro_f1": best["Val macro-F1"], "test_accuracy": best["Test accuracy"]}, indent=2))
    print(md.to_markdown(index=False))
    print(f"\nselected for predict.py (best val macro-F1): {best.Model}  -> {best.checkpoint}")


if __name__ == "__main__":
    main()
