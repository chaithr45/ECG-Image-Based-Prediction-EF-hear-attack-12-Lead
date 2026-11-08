"""TRAIN100 figures + tables: training vs held-out accuracy per model after training to (nearly) 100 %, next to the
original held-out accuracy. Reads only the aggregate JSON files written by the three TRAIN100 scripts.
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
# ---------------------------------------------------------------- originals (committed run records / report 45)
ORIGINAL_CV = {        # held-out accuracy, development CV (5 folds x 3 repeats)
    "HDR-LR": (0.489, "phase_5/results/phase5_baseline_results.csv, P5A-import-hdrlr-v1"),
    "HDRCLIN-LR": (0.501, "phase5_baseline_results.csv, P5A-import-hdrclinlr-v1"),
    "CLIN-LR": (0.509, "phase_6b.../results/mildfeat_models.json, CLIN-LR"),
    "LGBM-FEAT": (0.524, "phase_5/results/phase5_all_models.csv, P5D-E2-lgbm-eng-v1"),
    "ECGF-linear": (0.530, "phase5_baseline_results.csv, P5A-import-ecgflindist-v1"),
    "ECGF-MLP": (0.542, "mildfeat_models.json, ECGF-MLP"),
    "FUSION-MLP": (0.544, "phase5_baseline_results.csv, P5A-import-mlpfusion-v1"),
    "C1": (0.547, "phase5_baseline_results.csv, P5A-import-vote-v1"),
    "C3": (0.561, "phase5_all_models.csv, P5D-E5-vote-prior-protected-v1"),
    "IMG-vote": (0.541, "mildfeat_models.json, IMG-vote"),
    "IMG-C3": (0.558, "mildfeat_models.json, IMG-C3"),
    "AGT-img triage": (0.694, "report 45 Table 6.1 (answers 56 % of patients)"),
}
ORIGINAL_FIXED = {     # (validation accuracy at the selected epoch, training accuracy at that epoch, source)
    "ResNet-18": (0.524, 0.515, "results/resnet18_filtered"), "ECGNet": (0.476, 0.523, "results/ecgnet_filtered"),
    "EfficientNet-B0": (0.351, 0.447, "results/efficientnet_b0_filtered"),
    "MobileNetV3-Large": (0.185, 0.331, "results/mobilenetv3_filtered"),
    "ConvNeXt-Atto": (0.452, 0.404, "results/convnext_atto_filtered"),
    "1-D SE-ResNet": (0.482, 0.553, "results/resnet1d"),
    "ECGFounder fine-tuned (stages 5-6)": (0.524, None, "report 45 Table 3.3"),
    "HOG + RBF-SVM": (0.458, None, "results/hog_svm_filtered"),
    "XGBoost (waveform features)": (0.500, None, "results/xgb_waveform"),
    "Logistic regression (header)": (0.500, None, "results/logreg_header"),
    "Header 0.6 + ResNet-18 0.4": (0.565, None, "results/fusion_header_resnet18"),
}

# ---------------------------------------------------------------- collect TRAIN100 aggregates
cv = {}
for f in sorted(RES.glob("train100_features.json")):          # the LSTM family was withdrawn (note C)
    cv.update({k: v for k, v in json.load(open(f))["summary"].items() if not k.startswith("_")})
fixed = {}
for f in sorted(RES.glob("train100_fixedsplit__*.json")):
    fixed.update(json.load(open(f))["results"])

rows_cv = []
for m in ORIGINAL_CV:
    if m not in cv:
        continue
    s = cv[m]
    if m == "AGT-img triage":
        tr, te, ci = s["train_accuracy_answered"]["mean"], s["heldout_accuracy_answered"]["mean"], s["heldout_accuracy_answered"]["ci95_corrected"]
        note = f"answers {s['heldout_coverage']['mean']:.0%} of held-out patients"
    else:
        tr, te, ci = s["train_accuracy"]["mean"], s["heldout_accuracy"]["mean"], s["heldout_accuracy"]["ci95_corrected"]
        note = f"{s['n_cells']} folds" if "n_cells" in s and s["n_cells"] != 15 else ""
    rows_cv.append(dict(model=m, train=tr, heldout=te, ci=ci, original=ORIGINAL_CV[m][0], source=ORIGINAL_CV[m][1], note=note))
rows_fx = []
for m, (ov, otr, src) in ORIGINAL_FIXED.items():
    if m not in fixed:
        continue
    r = fixed[m]
    note = "" if r.get("reached_target", True) else f"cap reached ({r['epochs']} epochs)"
    rows_fx.append(dict(model=m, train=r["train_acc"], heldout=r["val_acc"], ci=None, original=ov, original_train=otr,
                        epochs=r.get("epochs"), source=src, note=note))

# ---------------------------------------------------------------- figure (one dot-plot per evaluation)
for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    try:
        matplotlib.font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
        plt.rcParams["font.family"] = "Latin Modern Roman"
    except Exception:
        pass
INK, MUTED, GRID, SURF = "#1f2328", "#5b6470", "#e3e6ea", "#ffffff"
C_TR, C_TE, C_OR = "#2a78d6", "#eb6834", "#1baf7a"            # categorical slots 1-3 (validated default palette)


def dotplot(rows, path, title, subtitle, held_label, orig_label, xlo):
    n = len(rows); H = 0.42 * n + 2.0
    fig, ax = plt.subplots(figsize=(8.6, H))
    y = np.arange(n)[::-1]
    for yi, r in zip(y, rows):
        ax.plot([r["heldout"], r["train"]], [yi, yi], color=GRID, lw=2, zorder=1)
        if r["ci"]:
            ax.plot(r["ci"], [yi, yi], color=C_TE, lw=2, alpha=0.45, zorder=2, solid_capstyle="round")
        if r["original"] is not None:
            ax.scatter(r["original"], yi, s=70, marker="D", facecolor=SURF, edgecolor=C_OR, lw=2, zorder=3)
        ax.scatter(r["heldout"], yi, s=70, color=C_TE, edgecolor=SURF, lw=1.5, zorder=4)
        ax.scatter(r["train"], yi, s=70, color=C_TR, edgecolor=SURF, lw=1.5, zorder=4)
        ax.text(r["train"] + 0.012, yi, f"{r['train']:.0%}", va="center", ha="left", fontsize=9, color=INK)
        lo = min([r["heldout"]] + ([r["original"]] if r["original"] is not None else []) + ([r["ci"][0]] if r["ci"] else []))
        ax.text(lo - 0.012, yi, f"{r['heldout']:.1%}", va="center", ha="right", fontsize=9, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([r["model"] for r in rows], fontsize=9.5, color=INK)
    ticks = np.arange(np.ceil(xlo * 10) / 10, 1.001, 0.1)
    ax.set_xlim(xlo, 1.07); ax.set_xticks(ticks); ax.set_xticklabels([f"{t:.0%}" for t in ticks], fontsize=9, color=MUTED)
    ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True); ax.tick_params(length=0)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_xlabel("Accuracy (four classes; triage: on the patients it answers)", fontsize=9.5, color=MUTED)
    h = [plt.Line2D([], [], marker="o", ls="", color=C_TR, ms=8, label="Training accuracy, trained to ~100 %"),
         plt.Line2D([], [], marker="o", ls="", color=C_TE, ms=8, label=held_label),
         plt.Line2D([], [], marker="D", ls="", mfc=SURF, mec=C_OR, mew=2, ms=7, label=orig_label)]
    ax.legend(handles=h, loc="lower left", bbox_to_anchor=(-0.02, 1.0), ncol=1, frameon=False, fontsize=9, labelcolor=INK)
    fig.suptitle(title, x=0.02, ha="left", fontsize=13.5, fontweight="bold", color=INK)
    fig.text(0.02, 1 - 0.5 / H, subtitle, fontsize=9, color=MUTED, ha="left", va="top")
    fig.subplots_adjust(left=0.29, right=0.97, top=1 - 1.75 / H, bottom=0.5 / H + 0.02)
    fig.savefig(str(path) + ".png", dpi=220, facecolor=SURF); fig.savefig(str(path) + ".pdf", facecolor=SURF)
    plt.close(fig)


if rows_cv:
    dotplot(rows_cv, OUT / "train100_cv_training_vs_heldout", "Training to 100 % does not raise held-out accuracy",
            "Development data, patient-grouped 5-fold CV x 3 repeats (837 ECGs). Locked test not used.",
            "Held-out accuracy, same model (bar = corrected 95 % CI)", "Held-out accuracy, original model (early stopping)", 0.35)
if rows_fx:
    lo = min(min(r["heldout"] for r in rows_fx), min(r["original"] for r in rows_fx if r["original"] is not None)) - 0.08
    dotplot(rows_fx, OUT / "train100_fixedsplit_training_vs_validation", "Same picture for the image and signal networks",
            "Original fixed split: trained on 669 ECGs, measured on 168 validation ECGs. Locked test not loaded.",
            "Validation accuracy, same model", "Validation accuracy, original model (selected epoch)", max(0.0, np.floor(lo * 10) / 10))

# ---------------------------------------------------------------- tables (markdown + json)
L = ["**Development cross-validation** (5 folds × 3 repeats unless stated)", "",
     "| Model | Training accuracy (TRAIN100) | Held-out accuracy (TRAIN100) [corrected 95 % CI] | Held-out accuracy (original) | Change |",
     "|---|---|---|---|---|"]
for r in rows_cv:
    L.append(f"| {r['model']} | {r['train']:.1%} | {r['heldout']:.1%} [{r['ci'][0]:.1%}, {r['ci'][1]:.1%}]"
             f"{' — ' + r['note'] if r['note'] else ''} | {r['original']:.1%} | {(r['heldout'] - r['original']) * 100:+.1f} points |")
L += ["", "**Original fixed split** (trained on 669 ECGs, measured on 168 validation ECGs)", "",
      "| Model | Epochs | Training accuracy (TRAIN100) | Validation accuracy (TRAIN100) | Original: training / validation accuracy | Change |",
      "|---|---|---|---|---|---|"]
for r in rows_fx:
    o = "—" if r["original"] is None else (f"{r['original_train']:.1%} / {r['original']:.1%}" if r["original_train"] is not None
                                           else f"— / {r['original']:.1%}")
    d = "—" if r["original"] is None else f"{(r['heldout'] - r['original']) * 100:+.1f} points"
    L.append(f"| {r['model']} | {r['epochs'] or '—'} | {r['train']:.1%}{' (' + r['note'] + ')' if r['note'] else ''} | "
             f"{r['heldout']:.1%} | {o} | {d} |")
(OUT / "train100_table.md").write_text("\n".join(L) + "\n")
json.dump(dict(cv=rows_cv, fixed_split=rows_fx), open(OUT / "train100_table.json", "w"), indent=1)
print("\n".join(L))
