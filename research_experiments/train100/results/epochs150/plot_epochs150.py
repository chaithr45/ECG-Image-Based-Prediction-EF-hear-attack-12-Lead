import json, sys
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
plt.rcParams.update({"font.family": "Latin Modern Roman", "font.size": 10})
INK, MUTED, GRID = "#1f2328", "#5b6470", "#e3e7ec"
BLUE, ORANGE = "#2a78d6", "#eb6834"
D = json.load(open(sys.argv[1]))["variants"]
fig, axes = plt.subplots(1, 2, figsize=(10.2, 5.0), sharey=True)
ep = np.arange(1, 151)
for ax, key in zip(axes, ("A", "B")):
    r = D[key]
    ax.fill_between(ep, r["train_min"], r["train_max"], color=BLUE, alpha=0.12, lw=0)
    ax.fill_between(ep, r["heldout_min"], r["heldout_max"], color=ORANGE, alpha=0.12, lw=0)
    ax.plot(ep, r["train"], color=BLUE, lw=2, solid_capstyle="round")
    ax.plot(ep, r["heldout"], color=ORANGE, lw=2, solid_capstyle="round")
    ax.axhline(0.459, color=MUTED, lw=1, ls=":")
    ax.text(150, 0.459 - 0.012, "always say Normal: 45.9 %", ha="right", va="top", fontsize=8, color=MUTED)
    es = r["early_stop_epoch_mean"]
    ax.axvline(es, color=MUTED, lw=1, ls="--")
    ax.text(es + 2, 0.70, f"early stopping\n(mean epoch {es:.0f})", ha="left", va="center", fontsize=8.5, color=MUTED)
    ax.text(150, r["epoch150_train"] - 0.02, f"Training: {r['epoch150_train']:.0%} at epoch 150", ha="right", va="top", fontsize=9.5, color=INK)
    ax.text(150, max(r["heldout"][-30:]) + 0.025, f"New patients (held-out): {r['epoch150_heldout']:.1%}", ha="right", va="bottom", fontsize=9.5, color=INK)
    ax.set_title(r["label"], fontsize=10, color=INK, loc="left")
    ax.set_xlim(1, 150); ax.set_ylim(0.40, 1.02)
    ax.set_xlabel("Epoch", color=MUTED)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0, decimals=0))
    ax.grid(axis="y", color=GRID, lw=0.8); ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9cfd6")
    ax.tick_params(colors=MUTED, length=0)
axes[0].set_ylabel("Four-class accuracy", color=MUTED)
h = [plt.Line2D([], [], color=BLUE, lw=2, label="Training ECGs"), plt.Line2D([], [], color=ORANGE, lw=2, label="Held-out (new) patients")]
fig.legend(handles=h, loc="upper left", bbox_to_anchor=(0.01, 0.815), ncol=2, frameon=False, fontsize=9.5)
fig.suptitle("Training for 150 epochs: training accuracy reaches 100 %, new-patient accuracy stays near 52 %",
             x=0.015, ha="left", fontsize=13, fontweight="bold", color=INK, y=0.985)
fig.text(0.015, 0.895, "ECGFounder MLP head on the frozen ECGFounder embedding; patient-grouped 5-fold cross-validation (repeat 0, 837 ECGs, original labels).\n"
         "Lines: mean of the 5 folds; shaded: lowest to highest fold. Accuracy measured after each epoch with dropout off.",
         ha="left", va="top", fontsize=9, color=MUTED, linespacing=1.35)
fig.subplots_adjust(left=0.07, right=0.985, top=0.70, bottom=0.11, wspace=0.06)
out = sys.argv[2]
fig.savefig(out + ".png", dpi=220, facecolor="white"); fig.savefig(out + ".pdf", facecolor="white")
