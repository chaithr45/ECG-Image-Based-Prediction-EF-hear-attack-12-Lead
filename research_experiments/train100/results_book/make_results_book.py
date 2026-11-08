"""TRAIN100 results book: every result of the train-to-100 % experiment in one PDF (aggregates only).
Main models ranked by held-out accuracy with macro-F1, Mild F1, balanced accuracy and per-class recall; how each model is
built (inputs, layers, parameters, epochs); confusion matrices; bar charts; fixed-split networks; variants; label
studies; the owner's laptop run with EchoNext (numbers as reported by that session); training accuracy of all models.
    python3 make_results_book.py      (reads ../results, writes figures/ tables/ and compiles TRAIN100_results_book.pdf)
"""
import json
import subprocess
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RES = HERE.parent / "results"
FIG, TAB = HERE / "figures", HERE / "tables"
FIG.mkdir(exist_ok=True); TAB.mkdir(exist_ok=True)
for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    try:
        font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
        plt.rcParams["font.family"] = "Latin Modern Roman"
    except Exception:
        pass
INK, MUTED, GRID, SURF = "#1f2328", "#5b6470", "#e3e6ea", "#ffffff"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"          # categorical slots 1-4 (validated default palette)
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CL = ["Normal", "Mild", "Moderate", "Severe"]


def metrics(C):
    C = np.asarray(C, float)
    rec = np.diag(C) / np.maximum(C.sum(1), 1)
    prec = np.divide(np.diag(C), C.sum(0), out=np.zeros(4), where=C.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return dict(bal=rec.mean(), mf1=f1.mean(), f1=f1, rec=rec, prec=prec)


def pooled(cells, get):
    """F1 / recall from the confusion matrix pooled within each repeat, averaged over the 3 repeats (project convention)."""
    reps = [metrics(np.sum([get(c) for c in cells if c["repeat"] == r], 0)) for r in (0, 1, 2)]
    return {k: np.mean([x[k] for x in reps], 0) for k in reps[0]}


def f3(x):
    return "--" if x is None else f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def colspec(ws):
    return "".join(r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}" % w for w in ws)


def write_table(name, caption, widths, header, rows, size=r"\small"):
    L = [r"\begin{table}[htbp]", r"\centering", r"\caption{%s}\label{tab:%s}" % (caption, name), size, r"\begin{tabular}{%s}" % colspec(widths),
         r"\toprule", " & ".join(r"\textbf{%s}" % h for h in header) + r" \\", r"\midrule"]
    for r in rows:
        if r == "MID":
            L.append(r"\midrule")
        elif isinstance(r, str):
            L.append(r"\multicolumn{%d}{l}{\textit{%s}} \\" % (len(header), r))
        else:
            L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TAB / f"{name}.tex").write_text("\n".join(L) + "\n")


# ================================================================ data
F = json.load(open(RES / "train100_features.json")); V = json.load(open(RES / "train100_variants_base.json"))
LB = json.load(open(RES / "train100_variants_labels.json"))
FX = {}
for f in sorted(RES.glob("train100_fixedsplit__*.json")):
    FX.update(json.load(open(f))["results"])
SF = F["summary"]
EP_MLP = np.median(sum([c["info"]["ECGF-MLP"]["epochs"] for c in F["cells"]], []))
EP_FUS = np.median(sum([c["info"]["FUSION-MLP"]["epochs"] for c in F["cells"]], []))

# original results (committed run records of report 45: accuracy, macro-F1, Mild F1)
ORIG = {"HDR-LR": (.489, .393, .198), "HDRCLIN-LR": (.501, .424, .269), "CLIN-LR": (.509, .417, .205), "LGBM-FEAT": (.524, .416, .228),
        "ECGF-linear": (.530, .444, .266), "ECGF-MLP": (.542, .446, .239), "FUSION-MLP": (.544, .440, .186), "C1": (.547, .455, .255),
        "C3": (.561, .448, .221), "IMG-vote": (.541, .450, .264), "IMG-C3": (.558, .441, .227), "AGT-img triage": (.694, None, None)}
NAME = {"HDR-LR": "HDR-LR: header measurements, logistic regression", "HDRCLIN-LR": "HDRCLIN-LR: header + clinical features, logistic regression",
        "CLIN-LR": "CLIN-LR: image-only clinician features, logistic regression", "LGBM-FEAT": "LGBM-FEAT: LightGBM on header + engineered features",
        "ECGF-linear": "ECGF-linear: ECGFounder + linear head", "ECGF-MLP": "ECGF-MLP: ECGFounder + MLP head",
        "FUSION-MLP": "FUSION-MLP: header + ECGFounder, MLP head", "C1": "C1: vote of FUSION-MLP, ECGF-linear, HDRCLIN-LR",
        "C3": "C3: C1 + nested prior rule", "IMG-vote": "IMG-vote: vote of ECGF-MLP, ECGF-linear", "IMG-C3": "IMG-C3: IMG-vote + nested prior rule",
        "AGT-img triage": "Image-only triage (AGT-img)"}
SHORT = {k: k for k in NAME}; SHORT["AGT-img triage"] = "Triage"

main = []
for m in ORIG:
    if m == "AGT-img triage":
        s = SF[m]; p = pooled(F["cells"], lambda c: c["triage"]["heldout_confusion"])
        main.append(dict(m=m, acc=s["heldout_accuracy_answered"]["mean"], ci=s["heldout_accuracy_answered"]["ci95_corrected"],
                         tr=s["train_accuracy_answered"]["mean"], cov=s["heldout_coverage"]["mean"], **p,
                         cm=np.sum([c["triage"]["heldout_confusion"] for c in F["cells"]], 0) / 3,
                         ref=np.sum([c["triage"]["heldout_referred"] for c in F["cells"]], 0) / 3))
    else:
        s = SF[m]; p = pooled(F["cells"], lambda c, m=m: c["models"][m]["heldout_confusion"])
        main.append(dict(m=m, acc=s["heldout_accuracy"]["mean"], ci=s["heldout_accuracy"]["ci95_corrected"], tr=s["train_accuracy"]["mean"],
                         cov=1.0, **p, cm=np.sum([c["models"][m]["heldout_confusion"] for c in F["cells"]], 0) / 3, ref=None))
main.sort(key=lambda d: -d["acc"])
four = [d for d in main if d["m"] != "AGT-img triage"]

# ================================================================ tables
rows = []
for i, d in enumerate(main, 1):
    nm = SHORT[d["m"]] + (r" $^{\dagger}$" if d["m"] == "AGT-img triage" else "")
    rows.append([str(i), nm, f"{f3(d['acc'])} {ci(d['ci'])}", f3(d["mf1"]), f3(d["f1"][1]), f3(d["bal"])]
                + [f"{x:.2f}" for x in d["rec"]] + [f3(d["tr"])])
    if d["m"] == "AGT-img triage":
        rows.append("MID")
write_table("main_ranked", r"Main models trained to (nearly) 100\,\% training accuracy, ranked by held-out accuracy (development cross-validation, "
            r"5 folds $\times$ 3 repeats, 837 ECGs). Held-out accuracy: mean of the 15 test folds with the corrected 95\,\% CI. Macro-F1, Mild F1, "
            r"balanced accuracy and recall: from the confusion matrix pooled within each repeat, averaged over the 3 repeats. Rec. = recall of Normal, Mild, Moderate and Severe. "
            r"$^{\dagger}$ Selective: scored on the 48\,\% of patients it answers.",
            [0.045, 0.165, 0.2, 0.08, 0.07, 0.07, 0.065, 0.065, 0.065, 0.065, 0.085],
            ["\\#", "Model", "Held-out accuracy [95\\,\\% CI]", "Macro-F1", "Mild F1", "Bal. acc.", "Rec. N", "Rec. Mi", "Rec. Mo", "Rec. S", "Training acc."], rows, r"\footnotesize")

rows = []
for d in main:
    o = ORIG[d["m"]]
    arrow = lambda a, b: f"{f3(a)} $\\rightarrow$ {f3(b)}" + ("" if a is None else f" ({100 * (b - a):+.1f})")
    rows.append([SHORT[d["m"]], arrow(o[0], d["acc"]), arrow(o[1], d["mf1"]), arrow(o[2], d["f1"][1])])
write_table("main_vs_original", r"The same models before (original, early-stopped, report 45) and after training to (nearly) 100\,\%. Change in percentage points. "
            r"Triage: original macro-F1 and Mild F1 were not recorded in its run summary.",
            [0.16, 0.28, 0.28, 0.28],
            ["Model", "Accuracy: original $\\rightarrow$ now (change)", "Macro-F1: original $\\rightarrow$ now (change)", "Mild F1: original $\\rightarrow$ now (change)"],
            rows, r"\footnotesize")

E = f"{EP_MLP:.0f}"; EF = f"{EP_FUS:.0f}"
build = [
    ("HDR-LR", "16 header measurements (age, sex, rate, intervals, axes)", "1 linear layer (16 $\\rightarrow$ 4)", "68", "solver (L-BFGS)"),
    ("HDRCLIN-LR", "16 header + 28 clinical ECG features (44 inputs)", "1 linear layer (44 $\\rightarrow$ 4)", "180", "solver (L-BFGS)"),
    ("CLIN-LR", "118 image-only clinician features", "1 linear layer (118 $\\rightarrow$ 4)", "476", "solver (L-BFGS)"),
    ("LGBM-FEAT", "16 header + 403 engineered waveform features (419)", "2,000 boosting rounds $\\times$ 4 classes, 31 leaves per tree", "8,000 trees", "2,000 trees"),
    ("ECGF-linear", "frozen ECGFounder embedding (1,024)", "frozen backbone + 1 linear layer (1,024 $\\rightarrow$ 4)", "4.1 k trained", "100 L-BFGS iter."),
    ("ECGF-MLP", "frozen ECGFounder embedding (1,024)", "frozen backbone + MLP: dropout, linear 1,024 $\\rightarrow$ 64, GELU, dropout, linear 64 $\\rightarrow$ 4 (2 layers); 3 seeds", "65.9 k trained", E),
    ("FUSION-MLP", "16 header + ECGFounder embedding (1,040)", "MLP head as ECGF-MLP (2 linear layers); 3 seeds", "67.0 k trained", EF),
    ("C1", "FUSION-MLP + ECGF-linear + HDRCLIN-LR", "average of 3 members' probabilities", "members", EF + " (FUSION-MLP)"),
    ("C3", "C1", "C1 + nested prior rule (one $\\alpha$ chosen on inner OOF)", "members + 1", EF + " (FUSION-MLP)"),
    ("IMG-vote", "ECGF-MLP + ECGF-linear (image only)", "average of 2 members' probabilities", "members", E + " (ECGF-MLP)"),
    ("IMG-C3", "IMG-vote", "IMG-vote + nested prior rule", "members + 1", E + " (ECGF-MLP)"),
    ("AGT-img triage", "ECGFounder + 118 clinician features $\\rightarrow$ EF; IMG-C3", "ridge + LightGBM EF regressors $\\rightarrow$ EF grade (3 cut-points) + margin; answers when the grade agrees with IMG-C3", "regressors + 4", "4,000 trees"),
]
write_table("main_build", r"How the main models are built. ECGFounder (Net1D, 30.7 M parameters: 61 convolution and 40 linear layers in 7 stages of "
            r"20 residual blocks) is pre-trained on 10.8 million ECGs and kept frozen; only the heads are trained on the hospital ECGs. "
            r"Epochs: median over the 15 folds to reach 100\,\% training accuracy; models without epochs show what replaces them.",
            [0.12, 0.25, 0.35, 0.12, 0.16], ["Model", "Inputs", "Layers / structure", "Parameters", "Epochs"],
            [[SHORT[a] if a in SHORT else a, b, c, d, e] for a, b, c, d, e in build], r"\footnotesize")

# fixed split
LAY = {"ResNet-18": "20 conv + 1 FC (18 weight layers)", "ECGNet": "29 conv (incl. SE) + 1 FC", "EfficientNet-B0": "81 conv (16 MBConv) + 1 FC",
       "MobileNetV3-Large": "63 conv (15 bottlenecks) + 1 FC", "ConvNeXt-Atto": "40 conv (12 blocks) + 1 FC", "1-D SE-ResNet": "37 conv + 1 FC",
       "ECGFounder fine-tuned (stages 5-6)": "61 conv + 40 linear (stages 5--6 trained) + 1 FC", "HOG + RBF-SVM": "PCA-150 + RBF kernel",
       "XGBoost (waveform features)": "2,000 trees", "Logistic regression (header)": "1 linear layer", "Header 0.6 + ResNet-18 0.4": "weighted average"}
OFX = {"ResNet-18": .524, "ECGNet": .476, "EfficientNet-B0": .351, "MobileNetV3-Large": .185, "ConvNeXt-Atto": .452, "1-D SE-ResNet": .482,
       "ECGFounder fine-tuned (stages 5-6)": .524, "HOG + RBF-SVM": .458, "XGBoost (waveform features)": .500, "Logistic regression (header)": .500,
       "Header 0.6 + ResNet-18 0.4": .565}
fx = sorted(FX.items(), key=lambda kv: -kv[1]["val_acc"])
rows = [[m.replace("(stages 5-6)", "(stages 5--6)"), LAY[m], f"{r['n_params'] / 1e6:.1f} M" if "n_params" in r else "--",
         str(r["epochs"]) if "epochs" in r else "--", f3(r["train_acc"]), f3(r["val_acc"]), f3(r.get("best_val_acc_any_epoch")), f3(OFX[m]),
         f"{100 * (r['val_acc'] - OFX[m]):+.1f}"] for m, r in fx]
write_table("fixed", r"Image and signal networks on the original fixed split (trained on 669 ECGs, validated on 168; the locked test is not loaded), "
            r"ranked by validation accuracy. Best = highest validation accuracy at any epoch on the way to 100\,\%.",
            [0.17, 0.22, 0.075, 0.075, 0.08, 0.08, 0.09, 0.09, 0.08],
            ["Model", "Layers", "Params", "Epochs", "Train.", "Valid.", "Best valid.", "Orig. valid.", "Change"], rows, r"\scriptsize")

# variants
VS = V["summary"]
OV = {"E1 MLP, all feature families": .554, "E2 LightGBM (eng + ECGF-PCA)": .544, "E3 weighted average": .547, "E3 stacking (LR, balanced)": .546,
      "E4 hierarchical MLP": .567, "E6 signal-quality gating": .546, "E7 mixup MLP": .552, "Multitask MLP (class + EF, lambda 1)": .546,
      "HIER (two-stage)": .570, "ENH M2 synthetic 10-s -> ECGFounder": .528, "ENH M3 median-beat shape": .467, "ENH M4 MiniRocket": .504,
      "ENH M5 LR on HIER inputs": .543, "ENH M6 EF ridge -> bands": .534, "ENH-vote": .569, "ULT-stack (no EchoNext)": .571,
      "SUPER-TOP10-ORD without EchoNext": VS["SUPER-TOP10-ORD without EchoNext, ORIGINAL recipe (reference)"]["heldout_acc"]["mean"]}
var = []
for m, s in VS.items():
    if "heldout_acc" not in s or m.endswith("(reference)"):
        continue
    p = pooled(V["cells"], lambda c, m=m: c["base"][m]["heldout_confusion"])
    var.append(dict(m=m, acc=s["heldout_acc"]["mean"], ci=s["heldout_acc"]["ci95_corrected"], tr=s.get("train_acc", {}).get("mean"), **p, o=OV.get(m)))
var.sort(key=lambda d: -d["acc"])
lab = lambda m: m.replace("->", "$\\rightarrow$").replace(", lambda 1", "")
rows = [[lab(d["m"]), f3(d["tr"]), f"{f3(d['acc'])} {ci(d['ci'])}", f3(d["mf1"]), f3(d["f1"][1]), f3(d["o"]),
         "--" if d["o"] is None else f"{100 * (d['acc'] - d['o']):+.1f}"] for d in var]
write_table("variants", r"Variants of every phase (Phases 4, 5 and 6B, development cross-validation), ranked by held-out accuracy. "
            r"SUPER-TOP10-ORD: hospital-only members; its original is the same recipe on the original members (the published 0.587 also used EchoNext).",
            [0.36, 0.1, 0.2, 0.08, 0.08, 0.09, 0.09], ["Model", "Training acc.", "Held-out acc. [95\\,\\% CI]", "Macro-F1", "Mild F1", "Original", "Change"],
            rows, r"\footnotesize")

L = LB["summary"]["label_studies (C3 recipe)"]
spec = [("VER: 15 labels corrected", "VER", "all_837", "all 837", .565), ("HC30: reliable links only", "HC30", "HC_395", "395 reliable", .594),
        ("CLEAR: no border ECGs", "CLEAR", "clear_cut", "582 clear-cut", .668), ("MMCLEAN: suspect Mild/Mod. dropped", "MMC", "kept_MMC", "620 kept", .672),
        ("NF: noise-free", "NF", "NF_clear", "264 clean", .683), ("NFS: noise-free, Severe kept", "NFS", "NF_clear", "264 clean", .702)]
spec.sort(key=lambda s: -L[s[1]][f"accuracy_{s[2]}_orig_labels"])
rows = [[n, p, f"{L[c]['fusion_mlp_epochs_median']:.0f}", f3(L[c]["train_acc"]), f3(L[c][f"accuracy_{s}_orig_labels"]), f3(o),
         f"{100 * (L[c][f'accuracy_{s}_orig_labels'] - o):+.1f}"] for n, c, s, p, o in spec]
write_table("labels", r"Label-quality studies with the C3 recipe trained to (nearly) 100\,\%, scored on the same subsets as report Table 4.7, ranked by accuracy. "
            r"Epochs: median of the FUSION-MLP member.", [0.3, 0.15, 0.08, 0.11, 0.11, 0.11, 0.1],
            ["Study", "Scored on", "Epochs", "Training acc.", "Accuracy", "Original", "Change"], rows, r"\footnotesize")

# laptop run with EchoNext (numbers as reported by the owner's laptop session; screenshots)
lap = [("SUPER-TOP10-ORD (best for 50s)", .748, .558, "[0.530, 0.587]", .587), ("SUPER (greedy)", .790, .552, "[0.507, 0.596]", .583),
       ("STAGED-4 (pool X)", .722, .548, "[0.516, 0.581]", .585)]
rows = [[n, f3(t), f"{f3(h)} {c}", f3(o), f"{100 * (h - o):+.1f}"] for n, t, h, c, o in lap]
write_table("echonext", r"Top-3 ensembles retrained to (nearly) 100\,\% with EchoNext on the owner's laptop (numbers as reported by that session; "
            r"EchoNext data never left the laptop).", [0.36, 0.14, 0.24, 0.13, 0.13],
            ["Model", "Training acc.", "Held-out acc. [95\\,\\% CI]", "Original", "Change"], rows)
members = [("EN-MiMo-GBM (Mild vs Moderate)$^{\\ddagger\\circ}$", 1.0, .694, ".586"), ("EN-MiMo-LR (Mild vs Moderate)$^{\\ddagger\\circ}$", 1.0, .604, ".577"),
           ("H-MiMo-GBM (Mild vs Moderate)$^{\\ddagger}$", 1.0, None, ".569"), ("H-TAB-GBM", 1.0, None, ".548"), ("M9", 1.0, None, ".546"),
           ("H-X-GBM", 1.0, None, ".546"), ("EN-gbm-X$^{\\circ}$", 1.0, .756, ".539"), ("M10", 1.0, None, ".539"), ("ECGF-MLP", .996, None, ".536"),
           ("EN-gbm (H / plain)$^{\\circ}$", 1.0, .756, ".522 / .522"), ("M5", 1.0, None, ".507"), ("PIX-ZS (CNN on GPU)$^{\\circ}$", .999, .725, ".505"),
           ("ECGF-lin", 1.0, None, ".498"), ("EN-cls-all-X$^{\\circ}$", .996, .694, ".497"), ("M8", 1.0, None, ".496"),
           ("PIX-ZS-band$^{\\circ}$", .929, .642, ".492"), ("EN-cls-all (H / plain)$^{\\circ}$", .999, .673, ".490 / .464"), ("EF-band", None, None, ".479"),
           ("EN-cls$^{\\circ}$", 1.0, .663, ".479"), ("EN-band$^{\\circ}$", .781, .520, ".479"), ("M6 (EF bands)", .557, None, ".477"),
           ("H-TAB-LR", 1.0, None, ".473"), ("EN-band-all (H / plain)$^{\\circ}$", .820, .525, ".458 / .465"), ("EN-mu-mid$^{\\circ}$", .999, .463, ".409"),
           ("M12", 1.0, None, ".346")]
rows = [[n, f3(t), f3(v), "0" + h if not h.startswith("0") else h] for n, t, v, h in members]
rows = [[r[0], r[1], r[2], r[3].replace(" / .", " / 0.")] for r in rows]
write_table("echonext_members", r"Members of the three ensembles (laptop run), ranked by hospital held-out accuracy. $^{\ddagger}$ Two-class task. "
            r"$^{\circ}$ Trained on EchoNext only: training and validation accuracy on EchoNext, held-out accuracy on the hospital ECGs (H / plain input). "
            r"-- = not reported.", [0.46, 0.16, 0.18, 0.2], ["Member", "Training acc.", "EchoNext validation", "Hospital held-out acc."], rows, r"\footnotesize")


# ================================================================ figures
def save(fig, name):
    fig.savefig(FIG / f"{name}.pdf", facecolor=SURF); fig.savefig(FIG / f"{name}.png", dpi=170, facecolor=SURF); plt.close(fig)


def style(ax):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors=MUTED); ax.grid(axis="x", color=GRID, lw=0.8); ax.set_axisbelow(True)


# 1. ranking: accuracy, macro-F1, Mild F1 (four-class models)
ms = four[::-1]; y = np.arange(len(ms)); h = 0.26
fig, ax = plt.subplots(figsize=(8.2, 6.6))
for k, (key, col, lab_) in enumerate(((lambda d: d["acc"], S1, "Held-out accuracy"), (lambda d: d["mf1"], S2, "Macro-F1"), (lambda d: d["f1"][1], S3, "Mild F1"))):
    vals = [key(d) for d in ms]
    ax.barh(y + (1 - k) * h, vals, height=h - 0.03, color=col, label=lab_, edgecolor=SURF, lw=0.6)
    for yi, v in zip(y, vals):
        ax.text(v + 0.005, yi + (1 - k) * h, f"{v:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([SHORT[d["m"]] for d in ms], color=INK, fontsize=9); ax.set_xlim(0, 0.66)
ax.set_xlabel("Score (held-out, four classes)", color=MUTED); style(ax)
ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Main models ranked by held-out accuracy", loc="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(); save(fig, "fig_ranking")

# 2. before vs after: accuracy and macro-F1
fig, axs = plt.subplots(1, 2, figsize=(9.6, 5.6), sharey=True)
ms = four[::-1]; y = np.arange(len(ms)); h = 0.38
for ax, (title, ki, now) in zip(axs, (("Held-out accuracy", 0, lambda d: d["acc"]), ("Macro-F1", 1, lambda d: d["mf1"]))):
    o = [ORIG[d["m"]][ki] for d in ms]; n = [now(d) for d in ms]
    ax.barh(y + h / 2, o, height=h - 0.03, color=S3, label="Original (early stopping)")
    ax.barh(y - h / 2, n, height=h - 0.03, color=S2, label="Trained to ~100 %")
    for yi, a, b in zip(y, o, n):
        ax.text(max(a, b) + 0.006, yi, f"{100 * (b - a):+.1f}", va="center", fontsize=7.5, color="#b42318" if b < a else "#1a7f37")
    ax.set_xlim(0.3, 0.62); ax.set_title(title, loc="left", fontsize=11, fontweight="bold", color=INK); style(ax)
axs[0].set_yticks(y); axs[0].set_yticklabels([SHORT[d["m"]] for d in ms], color=INK, fontsize=9)
axs[1].legend(loc="lower right", frameon=False, fontsize=8)
fig.suptitle("Before and after training to 100 % (change in points)", x=0.01, ha="left", fontsize=12.5, fontweight="bold", color=INK)
fig.tight_layout(); save(fig, "fig_before_after")

# 3. per-class recall heat-map
fig, ax = plt.subplots(figsize=(6.4, 5.6))
R = np.array([d["rec"] for d in main])
ax.imshow(R, cmap=CMAP, vmin=0, vmax=1, aspect="auto")
for i in range(R.shape[0]):
    for j in range(4):
        ax.text(j, i, f"{R[i, j]:.2f}", ha="center", va="center", fontsize=8.5, color="white" if R[i, j] > 0.55 else INK)
ax.set_xticks(range(4)); ax.set_xticklabels(CL, fontsize=9); ax.xaxis.set_ticks_position("top")
ax.set_yticks(range(len(main))); ax.set_yticklabels([SHORT[d["m"]] for d in main], fontsize=9)
ax.tick_params(length=0)
for s in ax.spines.values():
    s.set_visible(False)
ax.set_title("Recall of each class (held-out)", loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
fig.tight_layout(); save(fig, "fig_recall")


# 4. confusion matrices (small multiples)
def draw_cm(ax, M, title, ref=None):
    M = np.asarray(M, float); n = M.sum(1); P = M / np.maximum(n[:, None], 1e-9)
    ax.imshow(P, cmap=CMAP, vmin=0, vmax=1, extent=(-0.5, 3.5, 3.5, -0.5))
    for i in range(4):
        for j in range(4):
            dark = P[i, j] > 0.5
            ax.text(j, i - 0.13, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=8.5, fontweight="bold" if i == j else "normal", color="white" if dark else INK)
            ax.text(j, i + 0.22, f"{P[i, j]:.0%}", ha="center", va="center", fontsize=6, color="white" if dark else MUTED)
    for k in range(5):
        ax.axhline(k - 0.5, color="white", lw=1.5); ax.axvline(k - 0.5, color="white", lw=1.5)
    lab_ = ["N", "Mi", "Mo", "S"]
    if ref is not None:
        for i in range(4):
            ax.add_patch(plt.Rectangle((3.6, i - 0.45), 0.9, 0.9, facecolor="#eef0f2", edgecolor="none"))
            ax.text(4.05, i, f"{ref[i]:.0f}", ha="center", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(-0.5, 4.55); ax.set_xticks(range(5)); ax.set_xticklabels(lab_ + ["Ref."], fontsize=7.5)
    else:
        ax.set_xticks(range(4)); ax.set_xticklabels(lab_, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(lab_, fontsize=7.5); ax.tick_params(length=0)
    ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9, color=INK, pad=14)


fig, axs = plt.subplots(4, 3, figsize=(9.0, 12.4))
for ax, d in zip(axs.ravel(), main):
    draw_cm(ax, d["cm"], f"{SHORT[d['m']]}  (acc. {d['acc']:.3f}, macro-F1 {d['mf1']:.3f})", d["ref"])
fig.suptitle("Held-out confusion matrices (rows = true class, columns = predicted; counts per repeat, row %)", x=0.01, ha="left",
             fontsize=11, fontweight="bold", color=INK)
fig.text(0.01, 0.005, "N = Normal, Mi = Mild, Mo = Moderate, S = Severe. Triage: answered patients only; Ref. = referred to echo.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.015, 1, 0.975)); save(fig, "fig_confusion")

# 5. variants bars
vv = var[::-1]; y = np.arange(len(vv)); h = 0.38
fig, ax = plt.subplots(figsize=(8.4, 9.6))
ax.barh(y + h / 2, [d["o"] if d["o"] else 0 for d in vv], height=h - 0.03, color=S3, label="Original")
ax.barh(y - h / 2, [d["acc"] for d in vv], height=h - 0.03, color=S2, label="Trained to ~100 %")
for yi, d in zip(y, vv):
    ax.text(d["acc"] + 0.004, yi - h / 2, f"{d['acc']:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([d["m"].replace(" -> ", " → ").replace(", lambda 1", "") for d in vv], fontsize=9.5, color=INK)
ax.set_xlim(0.3, 0.62); style(ax); ax.legend(loc="lower right", frameon=False, fontsize=8.5)
ax.set_title("Variants of every phase: held-out accuracy", loc="left", fontsize=12, fontweight="bold", color=INK)
fig.tight_layout(); save(fig, "fig_variants")

# 6. fixed split bars
ff = fx[::-1]; y = np.arange(len(ff)); h = 0.27
fig, ax = plt.subplots(figsize=(8.4, 7.0))
ax.barh(y + h, [r["train_acc"] for _, r in ff], height=h - 0.03, color=S1, label="Training")
ax.barh(y, [r["val_acc"] for _, r in ff], height=h - 0.03, color=S2, label="Validation (now)")
ax.barh(y - h, [OFX[m] for m, _ in ff], height=h - 0.03, color=S3, label="Validation (original)")
ax.set_yticks(y); ax.set_yticklabels([m for m, _ in ff], fontsize=9.5, color=INK); ax.set_xlim(0, 1.05); style(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=3, frameon=False, fontsize=9)
ax.set_title("Image and signal networks (fixed split)", loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
fig.tight_layout(); save(fig, "fig_fixed")

# copy existing figures
import shutil
for src, dst in ((RES / "train100_cv_training_vs_heldout.pdf", "fig_train_vs_heldout.pdf"),
                 (RES / "epochs150" / "epochs150_training_vs_heldout.pdf", "fig_epochs150.pdf")):
    shutil.copy(src, FIG / dst)

# all-models training accuracy: longtable from the rows of the standalone table (figures/all_training_source.tex)
src = (FIG / "all_training_source.tex").read_text()
body = src[src.index("\\midrule") + len("\\midrule"):src.index("\\end{tabular}")].strip()
body = body.replace("\\bottomrule", "")
(TAB / "all_training.tex").write_text("\n".join([
    r"\begin{small}", r"\begin{longtable}{>{\raggedright\arraybackslash}p{10.2cm}>{\raggedright\arraybackslash}p{2.8cm}>{\raggedright\arraybackslash}p{2.2cm}}",
    r"\caption{Training accuracy and epochs of every model, sorted from highest to lowest within each group.}\label{tab:alltrain}\\",
    r"\toprule \textbf{Model} & \textbf{Epochs (median)} & \textbf{Training accuracy} \\ \midrule \endfirsthead",
    r"\toprule \textbf{Model} & \textbf{Epochs (median)} & \textbf{Training accuracy} \\ \midrule \endhead",
    r"\bottomrule \endfoot", body, r"\end{longtable}", r"\end{small}"]) + "\n")

# key numbers for the text
best = four[0]
(TAB / "keys.tex").write_text("\n".join([
    r"\newcommand{\bestname}{%s}" % SHORT[best["m"]], r"\newcommand{\bestacc}{%.3f}" % best["acc"],
    r"\newcommand{\bestfone}{%s}" % SHORT[max(four, key=lambda d: d['mf1'])["m"]], r"\newcommand{\bestfoneval}{%.3f}" % max(d["mf1"] for d in four),
    r"\newcommand{\bestmild}{%s}" % SHORT[max(four, key=lambda d: d['f1'][1])["m"]], r"\newcommand{\bestmildval}{%.3f}" % max(d["f1"][1] for d in four)]) + "\n")
print("tables and figures written")
r = subprocess.run("latexmk -pdf -interaction=nonstopmode -halt-on-error TRAIN100_results_book.tex > build.log 2>&1", shell=True, cwd=HERE)
print("latex exit", r.returncode)
