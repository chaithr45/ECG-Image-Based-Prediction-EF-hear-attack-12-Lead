"""Final report: every results table and figure of Chapters 3-6 built from the models trained to (nearly) 100 % training
accuracy (TRAIN100). Rows are sorted by training accuracy (highest first), ties by held-out accuracy.
Reads aggregates only (../train100/results, incl. train100_scores.json); writes latex/tables and latex/figures.
    python3 make_current.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
TEX = HERE / "latex"
RES = HERE.parent / "train100" / "results"
TABD, FIGD = TEX / "tables", TEX / "figures"
for f in ("lmroman10-regular", "lmroman10-bold", "lmroman10-italic"):
    try:
        font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf")
        plt.rcParams["font.family"] = "Latin Modern Roman"
    except Exception:
        pass
INK, MUTED, GRID, SURF = "#1f2328", "#5b6470", "#e3e6ea", "#ffffff"
S1, S2, S3, S4 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CL = ["Normal", "Mild", "Moderate", "Severe"]


# ================================================================ helpers
def f3(x):
    return "--" if x is None else f"{x:.3f}"


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def colspec(ws):
    return "".join(r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}" % w for w in ws)


def group(n, text):
    return r"\multicolumn{%d}{>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr\linewidth-2\tabcolsep\relax}}{\textit{%s}} \\" % (n, text)


def write(label, caption, widths, header, rows, size=r"\small", compact=False):
    """rows: list of lists (cells), 'MID' for a rule, or ('G', text) for a group heading."""
    body = []
    for r in rows:
        if r == "MID":
            body.append(r"\midrule")
        elif isinstance(r, tuple):
            body.append(group(len(header), r[1]))
        else:
            body.append(" & ".join(r) + r" \\")
    L = [r"\begin{table}[htbp]", r"\caption{%s}\label{tab:%s}" % (caption, label),
         size + (r"\setlength{\tabcolsep}{3pt}" if compact or size != r"\small" else ""), r"\begin{tabular}{%s}" % colspec(widths),
         r"\toprule", " & ".join(r"\textbf{%s}" % h for h in header) + r" \\", r"\midrule", "\n".join(body),
         r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TABD / f"tab_{label.replace('.', '_')}.tex").write_text("\n".join(L) + "\n")


def bykey(tr, ho):
    """Sort key: training accuracy, highest first; ties by held-out accuracy, highest first; missing values last."""
    return (-(tr if tr is not None else -1.0), -(ho if ho is not None else -1.0))


def metrics(C):
    C = np.asarray(C, float)
    rec = np.diag(C) / np.maximum(C.sum(1), 1)
    prec = np.divide(np.diag(C), C.sum(0), out=np.zeros(4), where=C.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return dict(bal=rec.mean(), mf1=f1.mean(), f1=f1, rec=rec, prec=prec)


def pooled(cells, get):
    reps = [metrics(np.sum([get(c) for c in cells if c["repeat"] == r], 0)) for r in (0, 1, 2)]
    return {k: np.mean([x[k] for x in reps], 0) for k in reps[0]}


def save(fig, name):
    fig.savefig(FIGD / f"{name}.pdf", facecolor=SURF); plt.close(fig)


def style(ax, axis="x"):
    for s in ("top", "right", "left"):
        ax.spines[s].set_visible(False)
    ax.tick_params(length=0, colors=MUTED); ax.grid(axis=axis, color=GRID, lw=0.8); ax.set_axisbelow(True)


# ================================================================ data
F = json.load(open(RES / "train100_features.json")); V = json.load(open(RES / "train100_variants_base.json"))
LB = json.load(open(RES / "train100_variants_labels.json"))["summary"]["label_studies (C3 recipe)"]
FX = {}
for f in sorted(RES.glob("train100_fixedsplit__*.json")):
    FX.update(json.load(open(f))["results"])
SC = json.load(open(RES / "train100_scores.json")) if (RES / "train100_scores.json").exists() else None
SF, VS = F["summary"], V["summary"]
EP_MLP = int(np.median(sum([c["info"]["ECGF-MLP"]["epochs"] for c in F["cells"]], [])))
EP_FUS = int(np.median(sum([c["info"]["FUSION-MLP"]["epochs"] for c in F["cells"]], [])))

MAIN = {"HDR-LR": "HDR-LR (header measurements, logistic regression)", "HDRCLIN-LR": "HDRCLIN-LR (header + clinical features, logistic regression)",
        "CLIN-LR": "CLIN-LR (image-only clinician features, logistic regression)", "LGBM-FEAT": "LGBM-FEAT (LightGBM, header + engineered features)",
        "ECGF-linear": "ECGF-linear (ECGFounder + linear head)", "ECGF-MLP": "ECGF-MLP (ECGFounder + MLP head)",
        "FUSION-MLP": "FUSION-MLP (header + ECGFounder, MLP head)", "C1": "C1 (vote: FUSION-MLP, ECGF-linear, HDRCLIN-LR)",
        "C3": "C3 (C1 + nested prior rule)", "IMG-vote": "IMG-vote (image-only vote: ECGF-MLP, ECGF-linear)",
        "IMG-C3": "IMG-C3 (IMG-vote + nested prior rule)", "AGT-img triage": "Image-only triage (AGT-img)$^{\\dagger}$"}
EPO = {"HDR-LR": "solver", "HDRCLIN-LR": "solver", "CLIN-LR": "solver", "LGBM-FEAT": "2,000 trees", "ECGF-linear": "100 iter.",
       "ECGF-MLP": str(EP_MLP), "FUSION-MLP": str(EP_FUS), "C1": f"{EP_FUS}$^{{b}}$", "C3": f"{EP_FUS}$^{{b}}$", "IMG-vote": f"{EP_MLP}$^{{c}}$",
       "IMG-C3": f"{EP_MLP}$^{{c}}$", "AGT-img triage": "4,000 trees"}
main = []
for m in MAIN:
    if m == "AGT-img triage":
        s = SF[m]; p = pooled(F["cells"], lambda c: c["triage"]["heldout_confusion"])
        main.append(dict(m=m, acc=s["heldout_accuracy_answered"]["mean"], ci=s["heldout_accuracy_answered"]["ci95_corrected"],
                         tr=s["train_accuracy_answered"]["mean"], cov=s["heldout_coverage"]["mean"], trcov=s["train_coverage"]["mean"], **p,
                         cm=np.sum([c["triage"]["heldout_confusion"] for c in F["cells"]], 0) / 3,
                         ref=np.sum([c["triage"]["heldout_referred"] for c in F["cells"]], 0) / 3))
    else:
        s = SF[m]; p = pooled(F["cells"], lambda c, m=m: c["models"][m]["heldout_confusion"])
        main.append(dict(m=m, acc=s["heldout_accuracy"]["mean"], ci=s["heldout_accuracy"]["ci95_corrected"], tr=s["train_accuracy"]["mean"],
                         cov=1.0, **p, cm=np.sum([c["models"][m]["heldout_confusion"] for c in F["cells"]], 0) / 3, ref=None))
main.sort(key=lambda d: bykey(d["tr"], d["acc"]))
four = [d for d in main if d["m"] != "AGT-img triage"]
M = {d["m"]: d for d in main}
AUC = (SC or {}).get("screen_auc", {})

var = {}
for m, s in VS.items():
    if "heldout_acc" not in s or m.endswith("(reference)"):
        continue
    p = pooled(V["cells"], lambda c, m=m: c["base"][m]["heldout_confusion"])
    var[m] = dict(m=m, acc=s["heldout_acc"]["mean"], ci=s["heldout_acc"]["ci95_corrected"], tr=s["train_acc"]["mean"], **p)
VEP = {"E1 MLP, all feature families": "64", "E2 LightGBM (eng + ECGF-PCA)": "2,000 trees", "E3 weighted average": f"{EP_FUS}$^{{b}}$",
       "E3 stacking (LR, balanced)": f"{EP_FUS}$^{{b}}$", "E4 hierarchical MLP": "52 / 39", "E6 signal-quality gating": f"{EP_FUS}$^{{b}}$",
       "E7 mixup MLP": "144", "Multitask MLP (class + EF, lambda 1)": "131", "HIER (two-stage)": f"{EP_MLP}$^{{c}}$",
       "ENH M2 synthetic 10-s -> ECGFounder": "solver", "ENH M3 median-beat shape": "solver", "ENH M4 MiniRocket": "solver",
       "ENH M5 LR on HIER inputs": "solver", "ENH M6 EF ridge -> bands": "exact", "ENH-vote": f"{EP_MLP}$^{{c}}$",
       "ULT M8 LR multi-scale ECGF": "solver", "ULT M9 kNN (k 15, distance)": "none (kNN)", "ULT M10 gradient boosting": "2,000 iter.",
       "ULT M12 zone LR": "solver", "ULT-stack (no EchoNext)": f"{EP_MLP}$^{{c}}$", "SUPER-TOP10-ORD without EchoNext": f"{EP_MLP}$^{{c}}$"}
VNAME = {"E1 MLP, all feature families": "E1 MLP, header + ECGFounder + feature families", "E2 LightGBM (eng + ECGF-PCA)": "E2 LightGBM, features + ECGFounder PCA",
         "E3 weighted average": "E3 weighted average of the C1 members", "E3 stacking (LR, balanced)": "E3 stacking (logistic meta-learner)",
         "E4 hierarchical MLP": "E4 hierarchical MLP", "E6 signal-quality gating": "E6 signal-quality gating", "E7 mixup MLP": "E7 mixup MLP",
         "Multitask MLP (class + EF, lambda 1)": "Multitask MLP (class + EF)", "HIER (two-stage)": "HIER two-stage (Normal / middle / Severe, then Mild vs Moderate)",
         "ENH M2 synthetic 10-s -> ECGFounder": "ENH M2 synthetic 10-s ECG $\\rightarrow$ ECGFounder", "ENH M3 median-beat shape": "ENH M3 median-beat shape model",
         "ENH M4 MiniRocket": "ENH M4 MiniRocket", "ENH M5 LR on HIER inputs": "ENH M5 logistic regression on the HIER inputs",
         "ENH M6 EF ridge -> bands": "ENH M6 EF ridge $\\rightarrow$ EF bands", "ENH-vote": "ENH-vote (6 members, tuned class weights)",
         "ULT M8 LR multi-scale ECGF": "ULT M8 logistic regression, multi-scale ECGFounder", "ULT M9 kNN (k 15, distance)": "ULT M9 $k$-nearest neighbours ($k$ = 15)",
         "ULT M10 gradient boosting": "ULT M10 gradient boosting", "ULT M12 zone LR": "ULT M12 zone logistic regression",
         "ULT-stack (no EchoNext)": "ULT-stack (10 members + predicted EF)", "SUPER-TOP10-ORD without EchoNext": "SUPER-TOP10-ORD, hospital-trained members only"}

# laptop run with EchoNext (numbers as reported by the owner's computer; EchoNext data never left it)
ENS = [("SUPER-TOP10-ORD: ten best members, ordinal meta-learner", .748, .558, (.530, .587), "10"),
       ("SUPER: greedy member selection, logistic meta-learner", .790, .552, (.507, .596), "greedy"),
       ("STAGED-4: extended pool, staged meta-learner", .722, .548, (.516, .581), "10")]
LAP = [  # name, training acc, EchoNext validation, hospital held-out (H; plain), kind: E = EchoNext-trained, H = hospital-trained, 2 = two-class
    ("EN-MiMo-GBM (Mild vs Moderate)", 1.0, .694, .586, None, "E2"), ("EN-MiMo-LR (Mild vs Moderate)", 1.0, .604, .577, None, "E2"),
    ("H-MiMo-GBM (Mild vs Moderate)", 1.0, None, .569, None, "H2"), ("H-TAB-GBM", 1.0, None, .548, None, "H"), ("M9", 1.0, None, .546, None, "H"),
    ("H-X-GBM", 1.0, None, .546, None, "H"), ("EN-gbm-X", 1.0, .756, .539, None, "E"), ("M10", 1.0, None, .539, None, "H"),
    ("ECGF-MLP", .996, None, .536, None, "H"), ("EN-gbm", 1.0, .756, .522, .522, "E"), ("M5", 1.0, None, .507, None, "H"),
    ("PIX-ZS (pixel CNN, EchoNext printouts)", .999, .725, .505, None, "E"), ("ECGF-lin", 1.0, None, .498, None, "H"),
    ("EN-cls-all-X", .996, .694, .497, None, "E"), ("M8", 1.0, None, .496, None, "H"), ("PIX-ZS-band (pixel CNN, EF bands)", .929, .642, .492, None, "E"),
    ("EN-cls-all", .999, .673, .490, .464, "E"), ("EF-band", None, None, .479, None, "H"), ("EN-cls", 1.0, .663, .479, None, "E"),
    ("EN-band", .781, .520, .479, None, "E"), ("M6 (EF bands)", .557, None, .477, None, "H"), ("H-TAB-LR", 1.0, None, .473, None, "H"),
    ("EN-band-all", .820, .525, .458, .465, "E"), ("EN-mu-mid", .999, .463, .409, None, "E"), ("M12", 1.0, None, .346, None, "H")]

FXN = {"ECGFounder fine-tuned (stages 5-6)": "ECGFounder fine-tuned (stages 5--6)"}
LAY = {"ResNet-18": "20 conv + 1 FC (18 weight layers); ImageNet start", "ECGNet": "29 conv (incl. SE) + 1 FC", "EfficientNet-B0": "81 conv (16 MBConv) + 1 FC",
       "MobileNetV3-Large": "63 conv (15 bottlenecks) + 1 FC", "ConvNeXt-Atto": "40 conv (12 blocks) + 1 FC", "1-D SE-ResNet": "37 conv + 1 FC; digitised waveforms",
       "ECGFounder fine-tuned (stages 5-6)": "61 conv + 40 linear (stages 5--6 trained) + 1 FC", "HOG + RBF-SVM": "HOG + PCA-150 + RBF kernel",
       "XGBoost (waveform features)": "2,000 trees on 263 waveform features", "Logistic regression (header)": "1 linear layer on the header",
       "Header 0.6 + ResNet-18 0.4": "weighted average (default pipeline of the initial project)"}
fx = sorted(FX.items(), key=lambda kv: bykey(kv[1]["train_acc"], kv[1]["val_acc"]))

LABS = [("VER: 15 labels corrected to a high-confidence echo", "VER", "all", "all 837", "label noise costs little"),
        ("HC30: train on exact-name links within 30 days", "HC30", "HC_395", "395 reliable links", "reliable links give about 0.58"),
        ("CLEAR: train without the 255 border ECGs", "CLEAR", "clear_cut", "582 clear-cut ECGs", "clear cases are graded well"),
        ("MMCLEAN: drop 217 suspect Mild or Moderate labels", "MMC", "kept_MMC", "620 kept ECGs", "accuracy follows label clarity"),
        ("NF: noise-free training (all classes)", "NF", "NF_clear", "264 clean, clear-cut ECGs", "clean labels are graded well"),
        ("NFS: noise-free training, Severe kept", "NFS", "NF_clear", "264 clean, clear-cut ECGs", "best on clean labels")]
lab = []
for n, c, s, p, fnd in LABS:
    key = "accuracy_all_837_orig_labels" if s == "all" else f"accuracy_{s}_orig_labels"
    lab.append(dict(n=n, c=c, p=p, tr=LB[c]["train_acc"], acc=LB[c][key], ep=LB[c]["fusion_mlp_epochs_median"], fnd=fnd))
lab.sort(key=lambda d: bykey(d["tr"], d["acc"]))

# ================================================================ Chapter 3
rows = [("G", "Networks of the initial project and Phase 1, fixed split (669 training / 168 validation ECGs)")]
for m, r in fx:
    if "epochs" not in r:
        continue
    rows.append([FXN.get(m, m), str(r["epochs"]), f3(r["train_acc"]), f3(r["val_acc"]), f"{r['train_loss']:.3f} / {r['val_loss']:.3f}"])
rows.append(("G", "Frozen-feature heads, development cross-validation (mean of the 15 folds)"))
for m in sorted(("ECGF-linear", "ECGF-MLP", "FUSION-MLP", "HDRCLIN-LR"), key=lambda m: bykey(M[m]["tr"], M[m]["acc"])):
    rows.append([MAIN[m], EPO[m].replace("$^{b}$", "").replace("$^{c}$", ""), f3(M[m]["tr"]), f3(M[m]["acc"]), "--"])
write("3.3", r"Training against validation accuracy of the main networks, every network trained until its training accuracy reached (nearly) 100 \% "
      r"(no early stopping, augmentation, weight decay, label smoothing or weight averaging). Training accuracy is measured on all training ECGs in evaluation mode. "
      r"Sorted by training accuracy within each part. Epochs: until the target was reached (CNNs $\ge$ 99.5 \%); solver / iter. = models not trained in epochs.",
      [0.42, 0.12, 0.13, 0.15, 0.18], ["Network", "Epochs", "Training accuracy", "Validation / held-out accuracy", "Train / val loss"], rows, r"\footnotesize")

build = [
    ("LGBM-FEAT", "16 header + 403 engineered waveform features (419)", "2,000 boosting rounds $\\times$ 4 classes, 31 leaves per tree", "8,000 trees"),
    ("ECGF-linear", "frozen ECGFounder embedding (1,024)", "frozen backbone + 1 linear layer (1,024 $\\rightarrow$ 4)", "4.1 k trained"),
    ("ECGF-MLP", "frozen ECGFounder embedding (1,024)", "frozen backbone + MLP: dropout, linear 1,024 $\\rightarrow$ 64, GELU, dropout, linear 64 $\\rightarrow$ 4; 3 seeds", "65.9 k trained"),
    ("FUSION-MLP", "16 header + ECGFounder embedding (1,040)", "MLP head as ECGF-MLP; 3 seeds", "67.0 k trained"),
    ("C1", "FUSION-MLP + ECGF-linear + HDRCLIN-LR", "average of the 3 members' probabilities", "members"),
    ("C3", "C1", "C1 + nested prior rule (one $\\alpha$ chosen on inner out-of-fold predictions)", "members + 1"),
    ("IMG-vote", "ECGF-MLP + ECGF-linear (image only)", "average of the 2 members' probabilities", "members"),
    ("IMG-C3", "IMG-vote", "IMG-vote + nested prior rule", "members + 1"),
    ("AGT-img triage", "ECGFounder + 118 clinician features $\\rightarrow$ EF; IMG-C3", "ridge + LightGBM EF regressors $\\rightarrow$ EF grade (3 cut-points) + margin; answers when the grade agrees with IMG-C3", "regressors + 4"),
    ("HDR-LR", "16 header measurements (age, sex, rate, intervals, axes)", "1 linear layer (16 $\\rightarrow$ 4)", "68"),
    ("HDRCLIN-LR", "16 header + 28 clinical ECG features (44)", "1 linear layer (44 $\\rightarrow$ 4)", "180"),
    ("CLIN-LR", "118 image-only clinician features", "1 linear layer (118 $\\rightarrow$ 4)", "476")]
build.sort(key=lambda b: bykey(M[b[0]]["tr"], M[b[0]]["acc"]))
write("3.5", r"How the twelve main models are built, sorted by training accuracy. ECGFounder (Net1D; 61 convolution and 40 linear layers in 7 stages of 20 residual blocks; "
      r"30.7 M parameters) was pre-trained on 10.8 million ECGs and is kept frozen, so only the heads are trained on the hospital ECGs. "
      r"Epochs: median over the 15 folds to reach 100 \% training accuracy; $^{b}$ epochs of the member FUSION-MLP; $^{c}$ epochs of the member ECGF-MLP.",
      [0.14, 0.2, 0.3, 0.12, 0.12, 0.12], ["Model", "Inputs", "Layers / structure", "Params", "Epochs", "Training acc."],
      [[b[0].replace("AGT-img triage", "Triage$^{\\dagger}$"), b[1], b[2], b[3], EPO[b[0]], f3(M[b[0]]["tr"])] for b in build], r"\footnotesize")

# ================================================================ Chapter 4
rows = [[FXN.get(m, m), LAY[m], f"{r['n_params'] / 1e6:.1f} M" if "n_params" in r else "--", str(r.get("epochs", "--")),
         f3(r["train_acc"]), f3(r["val_acc"]), f3(r.get("best_val_acc_any_epoch"))] for m, r in fx]
write("4.2", r"Initial project and Phase 1: image and signal networks on the fixed split (trained on 669 ECGs, validated on 168), every network trained "
      r"until its training accuracy reached (nearly) 100 \%; sorted by training accuracy, ties by validation accuracy. Best valid. = highest validation accuracy at any epoch "
      r"on the way to 100 \%. On 168 ECGs the 95 \% interval of an accuracy is about $\pm$0.075.",
      [0.17, 0.23, 0.11, 0.11, 0.12, 0.13, 0.13], ["Model", "Layers", "Params", "Epochs", "Training acc.", "Validation acc.", "Best valid."], rows, r"\footnotesize")

hdr = ["Model", "Training acc.", "Held-out acc. [95 \\% CI]", "Macro F1", "Mild F1", "Bal. acc.", "HF AUC", "Recall N / Mi / Mo / S"]
rows = []
for d in main:
    rows.append([MAIN[d["m"]].split(" (")[0] + ("$^{\\dagger}$" if d["m"] == "AGT-img triage" else ""),
                 f3(d["tr"]) + (f" ({100 * d['trcov']:.0f} \\%)" if d["m"] == "AGT-img triage" else ""),
                 f"{f3(d['acc'])} {ci(d['ci'])}" + (f" ({100 * d['cov']:.0f} \\%)" if d["m"] == "AGT-img triage" else ""),
                 f3(d["mf1"]), f3(d["f1"][1]), f3(d["bal"]), f3(AUC.get(d["m"])) if d["m"] in AUC else "--", " / ".join(f"{x:.2f}" for x in d["rec"])])
write("4.3", r"Phases 2--3 and 5: the twelve main models in development cross-validation (5 folds $\times$ 3 repeats, 837 ECGs), each trained to (nearly) 100 \% training accuracy; "
      r"sorted by training accuracy, ties by held-out accuracy. Held-out accuracy: mean of the 15 test folds with the corrected 95 \% CI; F1, balanced accuracy and recall from the confusion "
      r"matrix pooled within each repeat. HF AUC = heart-failure screen, reduced EF against Normal (C3 and IMG-C3 use the probabilities of C1 and IMG-vote). $^{\dagger}$ Selective: scored on the patients it answers (share in brackets).",
      [0.15, 0.1, 0.2, 0.08, 0.075, 0.075, 0.075, 0.245], hdr, rows, r"\scriptsize", compact=True)

e = ["E1 MLP, all feature families", "E2 LightGBM (eng + ECGF-PCA)", "E3 weighted average", "E3 stacking (LR, balanced)", "E4 hierarchical MLP",
     "E6 signal-quality gating", "E7 mixup MLP"]
items = [dict(n=VNAME[m], ep=VEP[m], **var[m]) for m in e] + [dict(n="E5 vote + nested prior rule (C3)", ep=EPO["C3"], **{k: M["C3"][k] for k in ("tr", "acc", "ci", "rec", "mf1")})]
items.sort(key=lambda d: bykey(d["tr"], d["acc"]))
rows = [[d["n"], d["ep"], f3(d["tr"]), f"{f3(d['acc'])} {ci(d['ci'])}", f"{min(d['rec']):.3f}", f3(d["mf1"]),
         "admissible" if min(d["rec"]) >= 0.15 else "a recall below 0.15"] for d in items]
write("4.4", r"Phase 5 experiments in development cross-validation, every model trained to (nearly) 100 \% training accuracy; sorted by training accuracy. "
      r"Min recall = lowest recall of the four classes (admissible if $\ge$ 0.15). Epochs: median over the 15 folds; $^{b}$ epochs of the member FUSION-MLP.",
      [0.26, 0.1, 0.12, 0.18, 0.09, 0.09, 0.16], ["Experiment", "Epochs", "Training acc.", "Held-out acc. [95 \\% CI]", "Min recall", "Macro F1", "Note"], rows, r"\footnotesize")

efm, efr, mt = VS["EF regression MLP (hdr + ECGF)"], VS["EF regression ridge + LightGBM (triage)"], var["Multitask MLP (class + EF, lambda 1)"]
fft = FX["ECGFounder fine-tuned (stages 5-6)"]
items = [("EF regression, ridge + LightGBM (triage regressor)", "EF (correlation $r$)", "4,000 trees", efr["train_r"]["mean"], efr["heldout_r"]["mean"],
          f"MAE {efr['train_mae']['mean']:.1f} / {efr['heldout_mae']['mean']:.1f} EF points"),
         ("Multitask MLP (four classes + EF, header + ECGFounder)", "four-class accuracy", "131", mt["tr"], mt["acc"], f"held-out 95 \\% CI {ci(mt['ci'])}"),
         ("ECGFounder fine-tuned (stages 5--6), fixed split", "four-class accuracy", str(fft["epochs"]), fft["train_acc"], fft["val_acc"],
          "168 validation ECGs ($\\pm$0.075)"),
         ("EF regression MLP (header + ECGFounder)", "EF (correlation $r$)", "3,000 (cap)", efm["train_r"]["mean"], efm["heldout_r"]["mean"],
          f"MAE {efm['train_mae']['mean']:.1f} / {efm['heldout_mae']['mean']:.1f} EF points")]
items.sort(key=lambda d: bykey(d[3], d[4]))
write("4.5", r"Phases 4 and 6: EF regression, multitask learning and fine-tuning, every model trained to (nearly) 100 \% training accuracy (EF regressors: training "
      r"correlation $\approx$ 1); sorted by the training score. Development cross-validation unless stated.",
      [0.3, 0.15, 0.11, 0.12, 0.12, 0.2], ["Model", "Measure", "Epochs", "Training", "Held-out", "Note"],
      [[a, b, c, f3(d), f3(e_), f_] for a, b, c, d, e_, f_ in items], r"\footnotesize")

rows = [[d["n"], d["p"], f"{d['ep']:.0f}$^{{b}}$", f3(d["tr"]), f3(d["acc"]), d["fnd"]] for d in lab]
write("4.6", r"Label-quality and cohort studies with the C3 recipe trained to (nearly) 100 \% training accuracy (development cross-validation); sorted by training accuracy. "
      r"Accuracy on the subset named, with the recorded labels. Epochs: median of the member FUSION-MLP.",
      [0.29, 0.17, 0.11, 0.12, 0.12, 0.19], ["Study", "Scored on", "Epochs", "Training acc.", "Accuracy", "Finding"], rows, r"\footnotesize")

img = ["HIER (two-stage)", "ENH M2 synthetic 10-s -> ECGFounder", "ENH M3 median-beat shape", "ENH M4 MiniRocket", "ENH M5 LR on HIER inputs",
       "ENH M6 EF ridge -> bands", "ENH-vote", "ULT M8 LR multi-scale ECGF", "ULT M9 kNN (k 15, distance)", "ULT M10 gradient boosting", "ULT M12 zone LR",
       "ULT-stack (no EchoNext)"]
items = sorted([var[m] for m in img], key=lambda d: bykey(d["tr"], d["acc"]))
rows = [[VNAME[d["m"]], VEP[d["m"]], f3(d["tr"]), f"{f3(d['acc'])} {ci(d['ci'])}", " / ".join(f"{x:.2f}" for x in d["rec"]), f3(d["mf1"])] for d in items]
write("4.7", r"Image-only four-class models (no printed header; all 837 ECGs, development cross-validation), every model trained to (nearly) 100 \% training accuracy; "
      r"sorted by training accuracy. $^{c}$ Epochs of the member ECGF-MLP; solver = fitted to convergence.",
      [0.31, 0.11, 0.12, 0.18, 0.18, 0.1], ["Model", "Epochs", "Training acc.", "Held-out acc. [95 \\% CI]", "Recall N / Mi / Mo / S", "Macro F1"], rows, r"\footnotesize")

# ================================================================ Chapter 5
en = sorted([x for x in LAP if x[5].startswith("E")], key=lambda x: bykey(x[1], x[3]))
rows = [[x[0] + ("$^{\\ddagger}$" if x[5] == "E2" else ""), f3(x[1]), f3(x[2]), f3(x[3]) + ("" if x[4] is None else f" / {x[4]:.3f}")] for x in en]
write("5.1", r"Models trained only on EchoNext (every model trained to (nearly) 100 \% training accuracy, on the owner's computer) and applied to the 837 hospital ECGs; "
      r"sorted by training accuracy. Training and validation accuracy on EchoNext; hospital accuracy with only the decision thresholds tuned on the hospital training folds "
      r"(harmonised / plain input where both were run). $^{\ddagger}$ Two-class task (Mild against Moderate). Numbers as reported by that run.",
      [0.42, 0.16, 0.18, 0.24], ["Model", "Training acc. (EchoNext)", "Validation acc. (EchoNext)", "Hospital held-out acc."], rows, r"\footnotesize")

ens = [dict(n=n + " (with EchoNext members)", tr=t, acc=h, ci=c, mem=k) for n, t, h, c, k in ENS]
for m, nm, k in (("SUPER-TOP10-ORD without EchoNext", "SUPER-TOP10-ORD, hospital-trained members only", "10"),
                 ("ULT-stack (no EchoNext)", "ULT-stack, image-only members", "10 + EF"), ("ENH-vote", "ENH-vote, image-only members", "6")):
    ens.append(dict(n=nm, tr=var[m]["tr"], acc=var[m]["acc"], ci=var[m]["ci"], mem=k))
ens.append(dict(n="C3 (Phase 5 selected model, reads the printed header)", tr=M["C3"]["tr"], acc=M["C3"]["acc"], ci=M["C3"]["ci"], mem="3"))
ens.sort(key=lambda d: bykey(d["tr"], d["acc"]))
write("5.2", r"Ensembles of image-derived members (837 ECGs, development cross-validation, corrected 95 \% CI), every member trained to (nearly) 100 \% training accuracy "
      r"and every selection and meta-learner chosen inside the training folds; sorted by training accuracy. Ensembles with EchoNext members were run on the owner's computer.",
      [0.47, 0.14, 0.14, 0.25], ["Ensemble", "Members", "Training acc.", "Held-out acc. [95 \\% CI]"],
      [[d["n"], d["mem"], f3(d["tr"]), f"{f3(d['acc'])} {ci(d['ci'])}"] for d in ens], r"\footnotesize")

px = [("PIX-ZS: ResNet-18 trained on EchoNext printouts, applied to hospital pages", .999, .725, .505, "cross-validation (thresholds tuned)"),
      ("PIX-ZS-band: the same network grading EF bands", .929, .642, .492, "cross-validation (thresholds tuned)")]
px += [(f"{m}: hospital pages only", r["train_acc"], None, r["val_acc"], "fixed split, 168 validation ECGs")
       for m, r in FX.items() if m in ("ResNet-18", "EfficientNet-B0", "MobileNetV3-Large", "ConvNeXt-Atto", "ECGNet")]
px.sort(key=lambda x: bykey(x[1], x[3]))
write("5.3", r"Pixel models that read the page image directly, every network trained to (nearly) 100 \% training accuracy; sorted by training accuracy. "
      r"EchoNext accuracy: validation on EchoNext printouts.",
      [0.42, 0.12, 0.13, 0.12, 0.21], ["Model", "Training acc.", "EchoNext valid. acc.", "Hospital acc.", "Evaluation"],
      [[a, f3(b), f3(c), f3(d), e_] for a, b, c, d, e_ in px], r"\footnotesize")

# ================================================================ Chapter 6
T, IM = M["AGT-img triage"], M["IMG-C3"]
head = [("Best on all patients", "SUPER-TOP10-ORD: ten-member ensemble with ordinal meta-learner (with EchoNext members)", .748, .558, (.530, .587), "all 837 development ECGs"),
        ("Best staged model", "STAGED-4 on the extended pool (with EchoNext members)", .722, .548, (.516, .581), "all 837 development ECGs"),
        ("Best single model", "E2: LightGBM on header + engineered features + ECGFounder PCA", var["E2 LightGBM (eng + ECGF-PCA)"]["tr"],
         var["E2 LightGBM (eng + ECGF-PCA)"]["acc"], var["E2 LightGBM (eng + ECGF-PCA)"]["ci"], "all 837 development ECGs"),
        ("Best image-only model", "ENH-vote: six image-only members, tuned class weights", var["ENH-vote"]["tr"], var["ENH-vote"]["acc"], var["ENH-vote"]["ci"],
         "all 837 development ECGs"),
        ("Best in the 60s", "Image-only triage model", T["tr"], T["acc"], T["ci"], f"the {100 * T['cov']:.0f} \\% of patients it answers"),
        ("Best on clean labels", "NFS: C3 recipe trained on noise-free labels", LB["NFS"]["train_acc"], LB["NFS"]["accuracy_NF_clear_orig_labels"], None,
         "264 clean, clear-cut ECGs")]
head.sort(key=lambda h: bykey(h[2], h[3]))
rows = [[f"\\textbf{{{a}}}", b, f3(c), f"\\textbf{{{d:.3f}}}" + ("" if e_ is None else f" {ci(e_)}"), p] for a, b, c, d, e_, p in head]
write("6.1", r"Headline results of Phase I (image-derived inputs only, recorded four-class labels, development cross-validation), every model trained to (nearly) 100 \% "
      r"training accuracy; sorted by training accuracy. Ensembles: training accuracy of the meta-learner on its training folds; triage: on the training patients it answers.",
      [0.15, 0.29, 0.15, 0.22, 0.19], ["Result", "Model", "Training acc.", "Held-out acc. [95 \\% CI]", "Population"], rows)

# all models (Tables 6.2-6.4)
allm = []
for d in main:
    allm.append(("a", MAIN[d["m"]], EPO[d["m"]], d["tr"], d["acc"]))
allm.append(("a", "MMCLEAN selective model$^{\\dagger\\,*}$ (answers 7 \\%)", "n/r", .999, .699))
for m, d in var.items():
    allm.append(("b", VNAME[m], VEP[m], d["tr"], d["acc"]))
for d in lab:
    allm.append(("c", d["n"] + f" [{d['p']}]", f"{d['ep']:.0f}$^{{b}}$", d["tr"], d["acc"]))
for m, r in FX.items():
    allm.append(("d", FXN.get(m, m), str(r.get("epochs", "24 (ResNet-18)" if m.startswith("Header 0.6") else "solver" if "regression" in m or "SVM" in m
                                              else "2,000 trees")), r["train_acc"], r["val_acc"]))
for n, t, h, c, k in ENS:
    allm.append(("e", n.split(":")[0] + " ensemble", "n/r", t, h))
for x in LAP:
    allm.append(("e", x[0] + ("$^{\\circ}$" if x[5].startswith("E") else "") + ("$^{\\ddagger}$" if x[5].endswith("2") else ""), "n/r", x[1], x[3]))
GT = {"a": "Main models, development cross-validation (held-out accuracy)", "b": "Variants of every phase, development cross-validation (held-out accuracy)",
      "c": "Label-quality studies, C3 recipe (accuracy on the subset in brackets)", "d": "Image and signal networks, fixed split (validation accuracy)",
      "e": "Run with EchoNext on the owner's computer$^{*}$ (hospital held-out accuracy)"}
note = (r" Epochs: median over the 15 folds; solver / trees / iter. = models not trained in epochs; $^{b}$ epochs of the member FUSION-MLP; "
        r"$^{c}$ epochs of the member ECGF-MLP; $^{\dagger}$ selective model, accuracies on the patients it answers; $^{\ddagger}$ two-class task (Mild against Moderate); "
        r"$^{\circ}$ trained on EchoNext only (training accuracy on EchoNext); $^{*}$ numbers from the owner's own runs; n/r = not reported.")
for lab_, gs, ttl in (("6.2", "ab", "main models and variants of every phase"), ("6.3", "cd", "label studies and networks on the fixed split"),
                      ("6.4", "e", "the run with EchoNext")):
    rows = []
    for g in gs:
        rows.append(("G", GT[g]))
        rows += [[n, ep, f3(t), f3(h)] for _, n, ep, t, h in sorted([a for a in allm if a[0] == g], key=lambda a: bykey(a[3], a[4]))]
    write(lab_, r"All models trained to (nearly) 100 \% training accuracy: " + ttl + ", sorted by training accuracy (highest first), ties by held-out accuracy."
          + (note if lab_ == "6.2" else r" Symbols as in Table~\ref{tab:6.2}."),
          [0.56, 0.16, 0.14, 0.14], ["Model", "Epochs (median)", "Training acc.", "Held-out acc."], rows, r"\footnotesize")

# ================================================================ figures
# Fig 3.2: training curves of six networks (TRAIN100, fixed split)
six = ["ResNet-18", "ECGNet", "EfficientNet-B0", "ConvNeXt-Atto", "1-D SE-ResNet", "ECGFounder fine-tuned (stages 5-6)"]
fig, axs = plt.subplots(2, 3, figsize=(9.4, 6.4))
for ax, m in zip(axs.ravel(), six):
    c = FX[m]["curve"]; ep = [x["epoch"] for x in c]
    ax.plot(ep, [x["train_acc"] for x in c], color=S1, lw=1.8, label="training accuracy")
    ax.plot(ep, [x["val_acc"] for x in c], color=S2, lw=1.8, label="validation accuracy")
    ax.set_ylim(0, 1.02); ax.set_xlabel("epoch", color=MUTED, fontsize=10); style(ax, "y"); ax.tick_params(labelsize=9)
    ax2 = ax.twinx()
    ax2.plot(ep, [x["train_loss"] for x in c], color=S1, lw=1.1, ls=":", label="training loss")
    ax2.plot(ep, [x["val_loss"] for x in c], color=S2, lw=1.1, ls=":", label="validation loss")
    ax2.tick_params(length=0, colors=MUTED, labelsize=9)
    for s in ax2.spines.values():
        s.set_visible(False)
    ax.set_title(FXN.get(m, m).replace("--", "–").replace("ECGFounder fine-tuned", "ECGFounder FT") + f" ({len(c)} ep.)", fontsize=11, color=INK, loc="left")
h1, l1 = axs[0, 0].get_legend_handles_labels(); h2, l2 = ax2.get_legend_handles_labels()
fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=4, frameon=False, fontsize=10.5)
fig.tight_layout(rect=(0, 0.05, 1, 1)); save(fig, "fig_3_2")

# Fig 4.1: main models, training accuracy, held-out accuracy, macro-F1 and Mild F1 (sorted by training accuracy)
ms = four[::-1]; y = np.arange(len(ms)); h = 0.2
fig, ax = plt.subplots(figsize=(7.2, 7.6))
for k, (key, col, lb) in enumerate(((lambda d: d["tr"], S1, "Training accuracy"), (lambda d: d["acc"], S2, "Held-out accuracy"),
                                     (lambda d: d["mf1"], S3, "Macro F1"), (lambda d: d["f1"][1], S4, "Mild F1"))):
    vals = [key(d) for d in ms]
    ax.barh(y + (1.5 - k) * h, vals, height=h - 0.02, color=col, label=lb, edgecolor=SURF, lw=0.5)
    for yi, v in zip(y, vals):
        ax.text(v + 0.006, yi + (1.5 - k) * h, f"{v:.3f}", va="center", fontsize=7.5, color=INK)
ax.set_yticks(y); ax.set_yticklabels([d["m"] for d in ms], color=INK, fontsize=10.5); ax.set_xlim(0, 1.12)
ax.set_xlabel("Score (development cross-validation)", color=MUTED); style(ax)
ax.legend(loc="lower center", bbox_to_anchor=(0.4, 1.0), ncol=2, frameon=False, fontsize=10)
fig.tight_layout(); save(fig, "fig_4_1")

# Fig 5.1: ensembles and image-only variants: held-out accuracy with CI, training accuracy (sorted by training accuracy)
pts = [(d["n"].replace(" (with EchoNext members)", " (EchoNext)").split(":")[0] if ":" in d["n"] else d["n"], d["tr"], d["acc"], d["ci"]) for d in ens]
pts += [(VNAME[m].split(" (")[0], var[m]["tr"], var[m]["acc"], var[m]["ci"]) for m in img if m not in ("ULT-stack (no EchoNext)", "ENH-vote")]
pts += [(VNAME[m].split(" (")[0].split(",")[0], var[m]["tr"], var[m]["acc"], var[m]["ci"]) for m in e]
pts.sort(key=lambda p: bykey(p[1], p[2])); pts = pts[::-1]
fig, ax = plt.subplots(figsize=(7.4, 8.8)); y = np.arange(len(pts))
ax.hlines(y, [p[3][0] for p in pts], [p[3][1] for p in pts], color=S2, lw=2.2, alpha=0.45)
ax.plot([p[2] for p in pts], y, "o", color=S2, ms=5.5, label="Held-out accuracy (95 % CI)")
ax.plot([p[1] for p in pts], y, "D", color=S1, ms=4.5, label="Training accuracy")
for yi, p in zip(y, pts):
    ax.text(p[3][1] + 0.008, yi, f"{p[2]:.3f}", va="center", fontsize=7, color=INK)
ax.set_yticks(y); ax.set_yticklabels([p[0] for p in pts], fontsize=9.5, color=INK); ax.set_xlim(0.28, 1.04); style(ax)
ax.set_xlabel("Four-class accuracy (837 ECGs, development cross-validation)", color=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.45, 1.0), ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); save(fig, "fig_5_1")


# Fig 6.1: confusion matrices of the twelve main models (sorted by training accuracy)
def draw_cm(ax, Mx, title, ref=None):
    Mx = np.asarray(Mx, float); n = Mx.sum(1); P = Mx / np.maximum(n[:, None], 1e-9)
    ax.imshow(P, cmap=CMAP, vmin=0, vmax=1, extent=(-0.5, 3.5, 3.5, -0.5))
    for i in range(4):
        for j in range(4):
            dark = P[i, j] > 0.5
            ax.text(j, i - 0.13, f"{Mx[i, j]:.0f}", ha="center", va="center", fontsize=10, fontweight="bold" if i == j else "normal", color="white" if dark else INK)
            ax.text(j, i + 0.22, f"{P[i, j]:.0%}", ha="center", va="center", fontsize=7.5, color="white" if dark else MUTED)
    for k in range(5):
        ax.axhline(k - 0.5, color="white", lw=1.5); ax.axvline(k - 0.5, color="white", lw=1.5)
    lb = ["N", "Mi", "Mo", "S"]
    if ref is not None:
        for i in range(4):
            ax.add_patch(plt.Rectangle((3.6, i - 0.45), 0.9, 0.9, facecolor="#eef0f2", edgecolor="none"))
            ax.text(4.05, i, f"{ref[i]:.0f}", ha="center", va="center", fontsize=8, color=MUTED)
        ax.set_xlim(-0.5, 4.55); ax.set_xticks(range(5)); ax.set_xticklabels(lb + ["Ref."], fontsize=7.5)
    else:
        ax.set_xticks(range(4)); ax.set_xticklabels(lb, fontsize=7.5)
    ax.set_yticks(range(4)); ax.set_yticklabels(lb, fontsize=7.5); ax.tick_params(length=0); ax.xaxis.set_ticks_position("top")
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(title, fontsize=9.5, color=INK, pad=14)


fig, axs = plt.subplots(4, 3, figsize=(8.6, 11.2))
for ax, d in zip(axs.ravel(), main):
    draw_cm(ax, d["cm"], f"{'Triage' if d['m'] == 'AGT-img triage' else d['m']}  (train {d['tr']:.3f}, held-out {d['acc']:.3f})", d["ref"])
fig.text(0.01, 0.005, "Rows = true class, columns = predicted; counts per repeat and row %. N = Normal, Mi = Mild, Mo = Moderate, S = Severe. "
         "Triage: answered patients only; Ref. = referred to echo.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.015, 1, 1)); save(fig, "fig_6_1")

# Fig 6.2: all-patient development-CV four-class models (main, variants, EchoNext ensembles), sorted by training accuracy
pts = [(d["m"], d["tr"], d["acc"], "a") for d in four] + [(VNAME[m].split(" (")[0], v["tr"], v["acc"], "b") for m, v in var.items()]
pts += [(n.split(":")[0] + " (EchoNext)", t, h_, "e") for n, t, h_, c, k in ENS]
pts = sorted(pts, key=lambda p: bykey(p[1], p[2]))[::-1]
clean = lambda s: s.replace("$\\rightarrow$", "→").replace("$k$", "k").replace("--", "–")
fig, ax = plt.subplots(figsize=(7.6, 10.4)); y = np.arange(len(pts))
GC = {"a": S2, "b": S4, "e": "#d6455d"}
ax.hlines(y, [p[2] for p in pts], [p[1] for p in pts], color=GRID, lw=1.4)
ax.scatter([p[1] for p in pts], y, marker="D", s=22, color=S1, zorder=3, label="Training accuracy")
for g, lb in (("a", "Held-out: main models"), ("b", "Held-out: variants"), ("e", "Held-out: ensembles with EchoNext")):
    q = [(yi, p) for yi, p in zip(y, pts) if p[3] == g]
    ax.scatter([p[2] for _, p in q], [yi for yi, _ in q], s=30, color=GC[g], zorder=3, label=lb)
for yi, p in zip(y, pts):
    ax.text(p[2] - 0.008, yi, f"{p[2]:.3f}", va="center", ha="right", fontsize=7.5, color=MUTED)
ax.set_yticks(y); ax.set_yticklabels([clean(p[0]) for p in pts], fontsize=9, color=INK); ax.set_xlim(0.25, 1.03); ax.set_ylim(-0.8, len(pts) - 0.2)
style(ax); ax.set_xlabel("Accuracy (837 ECGs, development cross-validation)", color=MUTED)
ax.legend(loc="lower center", bbox_to_anchor=(0.35, 1.0), ncol=2, frameon=False, fontsize=9)
fig.tight_layout(); save(fig, "fig_6_2")

# Fig 6.4: results by evaluation population
groups = [("All patients, development CV", [(a[1], a[4]) for a in allm if a[0] in "ab" and "selective" not in a[1] and "triage" not in a[1]]
           + [(n, h) for n, t, h, c, k in ENS]),
          ("Fixed split, 168 validation ECGs", [(a[1], a[4]) for a in allm if a[0] == "d"]),
          ("Answered patients only (selective)", [("Image-only triage", T["acc"]), ("MMCLEAN selective*", .699)]),
          ("Clean or reliable labels (subsets)", [(d["n"], d["acc"]) for d in lab if d["c"] != "VER"])]
fig, ax = plt.subplots(figsize=(8.6, 4.4)); rng = np.random.default_rng(0)
for gi, (g, vals) in enumerate(groups):
    v = np.array([x[1] for x in vals]); yy = gi + rng.uniform(-0.18, 0.18, len(v))
    ax.scatter(v, yy, s=22, color=[S1, "#7c5cc4", S2, S3][gi], alpha=0.85, zorder=3)
    b = int(np.argmax(v)); ax.scatter([v[b]], [gi], marker="*", s=170, color=INK, zorder=4)
    ax.text(v[b] + 0.006, gi + 0.22, f"{v[b]:.3f}", fontsize=8, color=INK)
    ax.text(1.0, gi, f"n = {len(v)}", fontsize=8, color=MUTED, va="center", ha="right")
ax.axvline(0.55, color=MUTED, lw=0.8, ls="--"); ax.text(0.552, 3.45, "0.55", fontsize=7.5, color=MUTED)
ax.set_yticks(range(len(groups))); ax.set_yticklabels([g for g, _ in groups], fontsize=9, color=INK); ax.set_ylim(-0.6, 3.6); ax.set_xlim(0.3, 1.0)
style(ax); ax.set_xlabel("Accuracy (each dot is one model; star = best of the group)", color=MUTED)
fig.tight_layout(); save(fig, "fig_6_4")

# Fig 6.6: Mild recall against accuracy (four-class models, development CV)
fig, ax = plt.subplots(figsize=(7.6, 5.4))
xs = [(d["rec"][1], d["acc"], d["m"], S2) for d in four] + [(v["rec"][1], v["acc"], VNAME[m].split(" (")[0].split(",")[0], S4) for m, v in var.items()]
for x, y_, n, c in xs:
    ax.scatter(x, y_, s=26, color=c, zorder=3, edgecolor=SURF, lw=0.5)
for (x, y_, n, c), off in zip(sorted(xs, key=lambda t: -t[1])[:1] + sorted(xs, key=lambda t: -t[0])[:1], ((6, 4), (-6, 6))):
    ax.annotate(n, (x, y_), xytext=off, textcoords="offset points", fontsize=8, color=INK, ha="left" if off[0] > 0 else "right")
if SC and "mild_offset_imgvote" in SC:
    cv = SC["mild_offset_imgvote"]
    ax.plot([p["mild_recall"] for p in cv], [p["accuracy"] for p in cv], color=S1, lw=1.6, marker="o", ms=2.5, label="IMG-vote with a growing Mild offset")
    ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.scatter([], [], color=S2, s=26, label="main models"); ax.scatter([], [], color=S4, s=26, label="variants")
ax.legend(loc="upper right", frameon=False, fontsize=8.5)
ax.set_xlabel("Mild detected (recall of the Mild class)", color=MUTED); ax.set_ylabel("Four-class held-out accuracy", color=MUTED)
style(ax, "both"); ax.spines["left"].set_visible(True); ax.spines["left"].set_color(GRID)
fig.tight_layout(); save(fig, "fig_6_6")

# Figures that need the held-out probabilities (train100_scores.json): 6.3 screen AUC, 6.5 coverage, 6.7 simulation, 6.8 error by distance
if SC:
    au = sorted([(m, v) for m, v in SC["screen_auc"].items()], key=lambda kv: -kv[1])[::-1]
    fig, ax = plt.subplots(figsize=(7.8, 5.2)); y = np.arange(len(au))
    ax.barh(y, [v for _, v in au], color=S1, height=0.6)
    for yi, (_, v) in zip(y, au):
        ax.text(v + 0.004, yi, f"{v:.3f}", va="center", fontsize=8, color=INK)
    ax.set_yticks(y); ax.set_yticklabels([m for m, _ in au], fontsize=9, color=INK); ax.set_xlim(0.5, 0.92); style(ax)
    ax.set_xlabel("Heart-failure screen AUC (reduced EF against Normal, held-out)", color=MUTED)
    fig.tight_layout(); save(fig, "fig_6_3")

    fig, ax = plt.subplots(figsize=(7.2, 4.8))
    for (m, cv), col in zip(SC["coverage"].items(), (S1, S2, S3, S4)):
        ax.plot([p["coverage"] for p in cv], [p["accuracy"] for p in cv], color=col, lw=1.8, marker="o", ms=3, label=m)
    ax.scatter([T["cov"]], [T["acc"]], marker="*", s=180, color=INK, zorder=4, label=f"image-only triage ({T['acc']:.3f} on {100 * T['cov']:.0f} %)")
    ax.set_xlabel("Share of ECGs answered (most confident first)", color=MUTED); ax.set_ylabel("Accuracy on the answered ECGs", color=MUTED)
    ax.invert_xaxis(); style(ax, "both"); ax.legend(frameon=False, fontsize=8.5, loc="upper left")
    fig.tight_layout(); save(fig, "fig_6_5")

    sim = SC["simulation"]
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    ax.plot([p["sd"] for p in sim], [p["accuracy"] for p in sim], color=S1, lw=2, label="perfect grader with EF error of this SD")
    ax.axhline(0.7, color=S2, lw=0.9, ls="--"); ax.text(0.3, 0.705, "70 % target", color=S2, fontsize=8.5)
    ax.plot([p["sd"] for p in sim], [p["accuracy_echo_consistent"] for p in sim], color=S3, lw=1.6, ls="--", label="echo-consistent labels")
    best, sdp = .558, SC["ef_sd_population"]
    sd_mlp = sdp * np.sqrt(1 - VS["EF regression MLP (hdr + ECGF)"]["heldout_r"]["mean"] ** 2)
    sd_b = np.interp(-best, [-p["accuracy"] for p in sim], [p["sd"] for p in sim])
    ax.scatter([sd_b], [best], marker="*", s=180, color=INK, zorder=4)
    ax.annotate(f"best model ({best:.3f})", (sd_b, best), xytext=(8, 6), textcoords="offset points", fontsize=8.5, color=INK)
    ax.axvline(sd_mlp, color=MUTED, lw=0.9, ls=":"); ax.text(sd_mlp + 0.2, 0.43, f"our EF regressor\n(r = 0.624, SD {sd_mlp:.1f})", fontsize=8, color=MUTED)
    sd70 = np.interp(-0.70, [-p["accuracy"] for p in sim], [p["sd"] for p in sim])
    ax.annotate(f"70 % needs SD $\\leq$ {sd70:.1f}\n(r $\\approx$ {np.sqrt(1 - (sd70 / sdp) ** 2):.2f})", (sd70, 0.70), xytext=(-10, -40), textcoords="offset points",
                fontsize=8.5, color=INK, arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    sec = ax.secondary_xaxis("top", functions=(lambda s_: s_, lambda s_: s_))
    ticks_r = [0.95, 0.9, 0.8, 0.7, 0.6, 0.5]
    sec.set_xticks([sdp * np.sqrt(1 - r_ ** 2) for r_ in ticks_r]); sec.set_xticklabels([f"{r_:.2f}" for r_ in ticks_r], fontsize=8, color=MUTED)
    sec.set_xlabel(f"equivalent correlation r with echo EF (EF SD {sdp:.1f})", color=MUTED, fontsize=9); sec.tick_params(length=0)
    ax.set_xlabel("SD of the EF estimation error (EF points)", color=MUTED); ax.set_ylabel("Expected four-class accuracy", color=MUTED)
    ax.set_ylim(0.4, 1.0); style(ax, "both"); ax.legend(frameon=False, fontsize=8.5, loc="upper right")
    fig.tight_layout(); save(fig, "fig_6_7")

    db = SC["distance_bins"]
    fig, ax = plt.subplots(figsize=(6.8, 4.2)); x = np.arange(len(db["labels"])); w = 0.38
    for k, (m, col) in enumerate((("C3", S2), ("IMG-C3", S1))):
        ax.bar(x + (k - 0.5) * w, db[m], width=w - 0.03, color=col, label=m + (" (with header)" if m == "C3" else " (image only)"))
    ax.set_xticks(x); ax.set_xticklabels([f"{l}\n(n = {n})".replace("$\\ge$ 10 points", "10 points or more").replace("--", "–") for l, n in zip(db["labels"], db["n"])], fontsize=9)
    ax.set_ylabel("Error rate", color=MUTED); ax.set_xlabel("Distance of the echo EF from the nearest class limit", color=MUTED)
    ax.set_ylim(0, 1); style(ax, "y"); ax.legend(frameon=False, fontsize=8.5)
    fig.tight_layout(); save(fig, "fig_6_8")

print("tables and figures written; probability-based figures:", "yes" if SC else "NO (train100_scores.json missing)")
