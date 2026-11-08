"""TRAIN100 figure + table: training vs held-out accuracy per model after training to (nearly) 100 %, next to the
original held-out accuracy. Reads the two aggregate JSON files written by train100_features.py / train100_lstm.py.
    python3 make_report.py <results_dir> <out_dir>
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

RES, OUT = Path(sys.argv[1]), Path(sys.argv[2])
# original held-out accuracy (development CV, 5 folds x 3 repeats), from the committed run records
ORIGINAL = {
    "ECGF-linear": (0.530, "phase_5/results/phase5_baseline_results.csv (P5A-import-ecgflindist-v1)"),
    "ECGF-MLP": (0.542, "phase_6b.../results/mildfeat_models.json (ECGF-MLP)"),
    "HDR-LR": (0.489, "phase5_baseline_results.csv (P5A-import-hdrlr-v1)"),
    "HDRCLIN-LR": (0.501, "phase5_baseline_results.csv (P5A-import-hdrclinlr-v1)"),
    "CLIN-LR": (0.509, "mildfeat_models.json (CLIN-LR)"),
    "FUSION-MLP": (0.544, "phase5_baseline_results.csv (P5A-import-mlpfusion-v1)"),
    "LGBM-FEAT": (0.524, "phase5_all_models.csv (P5D-E2-lgbm-eng-v1)"),
    "LSTM-fixed (L3)": (0.467, "report 45, Table 4.5 (L3)"),
    "CNN-LSTM (L5)": (0.484, "report 45, Table 4.5 (L5)"),
    "C1": (0.547, "phase5_baseline_results.csv (P5A-import-vote-v1)"),
    "C3": (0.561, "phase5_all_models.csv (P5D-E5-vote-prior-protected-v1)"),
    "IMG-vote": (0.541, "mildfeat_models.json (IMG-vote)"),
    "IMG-C3": (0.558, "mildfeat_models.json (IMG-C3)"),
    "AGT-img triage": (0.694, "report 45, Table 6.1 (answers 56 % of patients)"),
}
ORDER = ["HDR-LR", "HDRCLIN-LR", "CLIN-LR", "LGBM-FEAT", "ECGF-linear", "ECGF-MLP", "FUSION-MLP", "LSTM-fixed (L3)",
         "CNN-LSTM (L5)", "C1", "C3", "IMG-vote", "IMG-C3", "AGT-img triage"]

S = {}
for f in ("train100_features.json", "train100_lstm.json"):
    if (RES / f).exists():
        S.update({k: v for k, v in json.load(open(RES / f))["summary"].items() if not k.startswith("_")})
rows = []
for m in ORDER:
    if m not in S:
        continue
    s = S[m]
    if m == "AGT-img triage":
        tr, te, ci = s["train_accuracy_answered"]["mean"], s["heldout_accuracy_answered"]["mean"], s["heldout_accuracy_answered"]["ci95_corrected"]
        extra = f"answers {s['heldout_coverage']['mean']:.0%} held-out, {s['train_coverage']['mean']:.0%} training"
    else:
        tr, te, ci = s["train_accuracy"]["mean"], s["heldout_accuracy"]["mean"], s["heldout_accuracy"]["ci95_corrected"]
        extra = ""
    rows.append(dict(model=m, train=tr, heldout=te, ci=ci, original=ORIGINAL[m][0], source=ORIGINAL[m][1], note=extra))

# ---------------------------------------------------------------- figure: dot plot, one row per model
for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    try:
        matplotlib.font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
        plt.rcParams["font.family"] = "Latin Modern Roman"
    except Exception:
        pass
INK, MUTED, GRID, SURF = "#1f2328", "#5b6470", "#e3e6ea", "#ffffff"
C_TR, C_TE, C_OR = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3 (validated default palette)
n = len(rows)
fig, ax = plt.subplots(figsize=(8.4, 0.42 * n + 1.9))
y = np.arange(n)[::-1]
for yi, r in zip(y, rows):
    ax.plot([r["heldout"], r["train"]], [yi, yi], color=GRID, lw=2, zorder=1)
    ax.plot([r["ci"][0], r["ci"][1]], [yi, yi], color=C_TE, lw=2, alpha=0.45, zorder=2, solid_capstyle="round")
    ax.scatter(r["original"], yi, s=70, marker="D", facecolor=SURF, edgecolor=C_OR, lw=2, zorder=3)
    ax.scatter(r["heldout"], yi, s=70, color=C_TE, edgecolor=SURF, lw=1.5, zorder=4)
    ax.scatter(r["train"], yi, s=70, color=C_TR, edgecolor=SURF, lw=1.5, zorder=4)
    ax.text(r["train"] + 0.012, yi, f"{r['train']:.0%}", va="center", ha="left", fontsize=9, color=INK)
    lo = min(r["heldout"], r["original"], r["ci"][0])
    ax.text(lo - 0.012, yi, f"{r['heldout']:.1%}", va="center", ha="right", fontsize=9, color=INK)
ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in rows], fontsize=10, color=INK)
ax.set_xlim(0.35, 1.07); ax.set_xticks(np.arange(0.4, 1.01, 0.1))
ax.set_xticklabels([f"{t:.0%}" for t in np.arange(0.4, 1.01, 0.1)], fontsize=9, color=MUTED)
ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_xlabel("Accuracy (four classes; triage: on the patients it answers)", fontsize=9.5, color=MUTED)
h = [plt.Line2D([], [], marker="o", ls="", color=C_TR, ms=8, label="Training accuracy, trained to ~100 %"),
     plt.Line2D([], [], marker="o", ls="", color=C_TE, ms=8, label="Held-out accuracy, same model (bar = 95 % CI)"),
     plt.Line2D([], [], marker="D", ls="", mfc=SURF, mec=C_OR, mew=2, ms=7, label="Held-out accuracy, original model (early stopping)")]
ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, 1.0), ncol=1, frameon=False, fontsize=9, labelcolor=INK)
fig.suptitle("Training to 100 % does not raise held-out accuracy", x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.02, 0.935, "Development data, patient-grouped 5-fold CV x 3 repeats (837 ECGs). Locked test not used.",
         fontsize=9, color=MUTED, ha="left")
fig.subplots_adjust(left=0.2, right=0.97, top=1 - 1.45 / (0.42 * n + 1.9), bottom=0.45 / (0.42 * n + 1.9) + 0.03)
fig.savefig(OUT / "train100_training_vs_heldout.png", dpi=220, facecolor=SURF)
fig.savefig(OUT / "train100_training_vs_heldout.pdf", facecolor=SURF)

# ---------------------------------------------------------------- table (markdown)
L = ["| Model | Training accuracy (TRAIN100) | Held-out accuracy (TRAIN100) [95 % CI] | Held-out accuracy (original) | Change |",
     "|---|---|---|---|---|"]
for r in rows:
    d = r["heldout"] - r["original"]
    L.append(f"| {r['model']} | {r['train']:.1%} | {r['heldout']:.1%} [{r['ci'][0]:.1%}, {r['ci'][1]:.1%}]"
             f"{' — ' + r['note'] if r['note'] else ''} | {r['original']:.1%} | {d * 100:+.1f} points |")
(OUT / "train100_table.md").write_text("\n".join(L) + "\n")
json.dump(dict(rows=rows), open(OUT / "train100_table.json", "w"), indent=1)
print("\n".join(L))
