"""Chapter-7 tables of the TRAIN100 edition of the report, generated from the aggregate result files (no number typed
by hand except the ORIGINAL values below, each with its source).
    python3 make_tables.py <results_dir> <latex_dir>
"""
import json
import sys
from pathlib import Path

import numpy as np

RES, TEX = Path(sys.argv[1]), Path(sys.argv[2])


def load(name):
    f = RES / name
    return json.load(open(f)) if f.exists() else None


def pct(x, d=3):
    """Accuracy as a fraction (report convention, e.g. 0.587); d=0 gives a whole percentage for coverage."""
    return "--" if x is None else (f"{100 * x:.0f}" if d == 0 else f"{x:.{d}f}")


def ci(c):
    return f"[{c[0]:.3f}, {c[1]:.3f}]"


def chg(new, old):
    return "--" if new is None or old is None else f"{100 * (new - old):+.1f}"


def colspec(widths):
    return "".join(r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}" % w for w in widths)


def table(label, caption, widths, header, rows, groups=None):
    L = [r"\begin{table}[htbp]", r"\caption{%s}\label{%s}" % (caption, label), r"\small", r"\begin{tabular}{%s}" % colspec(widths),
         r"\toprule", " & ".join(r"\textbf{%s}" % h for h in header) + r" \\", r"\midrule"]
    for i, r in enumerate(rows):
        if r == "MIDRULE":
            L.append(r"\midrule"); continue
        if isinstance(r, str):
            L.append(r"\multicolumn{%d}{>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr\linewidth-2\tabcolsep\relax}}{\textit{%s}} \\" % (len(header), r)); continue
        L.append(" & ".join(r) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "tables" / f"tab_{label.split(':')[1].replace('.', '_')}.tex").write_text("\n".join(L) + "\n")


# ---------------------------------------------------------------- originals (committed run records / report 45)
O_CV = {"HDR-LR": .489, "HDRCLIN-LR": .501, "CLIN-LR": .509, "LGBM-FEAT": .524, "ECGF-linear": .530, "ECGF-MLP": .542,
        "FUSION-MLP": .544, "C1": .547, "C3": .561, "IMG-vote": .541, "IMG-C3": .558, "AGT-img triage": .694}
O_FX = {"ResNet-18": (.515, .524), "ECGNet": (.523, .476), "EfficientNet-B0": (.447, .351), "MobileNetV3-Large": (.331, .185),
        "ConvNeXt-Atto": (.404, .452), "1-D SE-ResNet": (.553, .482), "ECGFounder fine-tuned (stages 5-6)": (None, .524),
        "HOG + RBF-SVM": (None, .458), "XGBoost (waveform features)": (None, .500), "Logistic regression (header)": (None, .500),
        "Header 0.6 + ResNet-18 0.4": (None, .565)}
O_VAR = {"E1 MLP, all feature families": .554, "E2 LightGBM (eng + ECGF-PCA)": .544, "E3 weighted average": .547,
         "E3 stacking (LR, balanced)": .546, "E4 hierarchical MLP": .567, "E6 signal-quality gating": .546, "E7 mixup MLP": .552,
         "Multitask MLP (class + EF, lambda 1)": .546, "HIER (two-stage)": .570, "ENH M2 synthetic 10-s -> ECGFounder": .528,
         "ENH M3 median-beat shape": .467, "ENH M4 MiniRocket": .504, "ENH M5 LR on HIER inputs": .543, "ENH M6 EF ridge -> bands": .534,
         "ENH-vote": .569, "ULT-stack (no EchoNext)": .571, "SUPER-TOP10-ORD without EchoNext": None,
         "SUPER-TOP10-ORD without EchoNext, ORIGINAL recipe (reference)": None}
SHOW = {"E1 MLP, all feature families": "E1 MLP, header + ECGFounder + feature families", "E2 LightGBM (eng + ECGF-PCA)": "E2 LightGBM, features + ECGFounder PCA",
        "E3 weighted average": "E3 weighted average of the C1 members", "E3 stacking (LR, balanced)": "E3 stacking (logistic meta-learner)",
        "E4 hierarchical MLP": "E4 hierarchical MLP", "E6 signal-quality gating": "E6 signal-quality gating", "E7 mixup MLP": "E7 mixup MLP",
        "Multitask MLP (class + EF, lambda 1)": "Multitask MLP (class + EF)", "HIER (two-stage)": "HIER two-stage (ext. grid)",
        "ENH M2 synthetic 10-s -> ECGFounder": "ENH M2 synthetic 10-s ECG $\\rightarrow$ ECGFounder", "ENH M3 median-beat shape": "ENH M3 median-beat shape",
        "ENH M4 MiniRocket": "ENH M4 MiniRocket", "ENH M5 LR on HIER inputs": "ENH M5 logistic regression", "ENH M6 EF ridge -> bands": "ENH M6 EF ridge $\\rightarrow$ bands",
        "ENH-vote": "ENH-vote", "ULT M8 LR multi-scale ECGF": "ULT M8 multi-scale ECGFounder LR", "ULT M9 kNN (k 15, distance)": "ULT M9 kNN",
        "ULT M10 gradient boosting": "ULT M10 gradient boosting", "ULT M12 zone LR": "ULT M12 zone LR", "ULT-stack (no EchoNext)": "ULT-stack",
        "SUPER-TOP10-ORD without EchoNext": "SUPER-TOP10-ORD, hospital members only",
        "SUPER-TOP10-ORD without EchoNext, ORIGINAL recipe (reference)": "same, original recipe (no retraining)"}

# ---------------------------------------------------------------- Table 7.2: development-CV models
F = load("train100_features.json")
EPN = r" Epochs: median over the 15 folds; $^{a}$ boosted trees (no epochs); $^{b}$ epochs of the neural member FUSION-MLP; $^{c}$ epochs of the neural member ECGF-MLP; $^{d}$ full-batch L-BFGS iterations; $^{e}$ solved to convergence (no epochs)."
EP_MLP = EP_FUS = EP_LIN = "--"
if F:
    S = F["summary"]; rows = []
    EP_MLP = f"{np.median(sum([c['info']['ECGF-MLP']['epochs'] for c in F['cells']], [])):.0f}"
    EP_FUS = f"{np.median(sum([c['info']['FUSION-MLP']['epochs'] for c in F['cells']], [])):.0f}"
    EP_LIN = f"{np.median([c['info']['ECGF-linear']['steps'] for c in F['cells']]):.0f}"
    EP72 = {"HDR-LR": "solver$^{e}$", "HDRCLIN-LR": "solver$^{e}$", "CLIN-LR": "solver$^{e}$", "LGBM-FEAT": "2,000 trees$^{a}$",
            "ECGF-linear": EP_LIN + " iter.$^{d}$", "ECGF-MLP": EP_MLP, "FUSION-MLP": EP_FUS, "C1": EP_FUS + "$^{b}$", "C3": EP_FUS + "$^{b}$",
            "IMG-vote": EP_MLP + "$^{c}$", "IMG-C3": EP_MLP + "$^{c}$", "AGT-img triage": "4,000 trees$^{a}$"}
    for m, o in O_CV.items():
        s = S[m]
        if m == "AGT-img triage":
            rows.append(["Image-only triage (AGT-img)", EP72[m], pct(s["train_accuracy_answered"]["mean"]) + f" ({pct(s['train_coverage']['mean'], 0)} \\% ans.)",
                         pct(s["heldout_accuracy_answered"]["mean"]) + " " + ci(s["heldout_accuracy_answered"]["ci95_corrected"]) +
                         f" ({pct(s['heldout_coverage']['mean'], 0)} \\% ans.)", pct(o) + " (56 \\% ans.)", chg(s["heldout_accuracy_answered"]["mean"], o)])
        else:
            rows.append([m, EP72[m], pct(s["train_accuracy"]["mean"]), pct(s["heldout_accuracy"]["mean"]) + " " + ci(s["heldout_accuracy"]["ci95_corrected"]),
                         pct(o), chg(s["heldout_accuracy"]["mean"], o)])
    table("tab:7.2", "Development cross-validation (5 folds $\\times$ 3 repeats, 837 ECGs): every model trained to (nearly) 100 \\% "
          "training accuracy (TRAIN100; training and held-out accuracy) against the original early-stopped model (original held-out accuracy). Held-out 95 \\% CI corrected for "
          "repeated CV; change = TRAIN100 minus original, in percentage points." + EPN,
          [0.20, 0.12, 0.14, 0.25, 0.15, 0.14], ["Model", "Epochs", "Training", "Held-out [95 \\% CI]", "Original", "Change (points)"], rows)

# ---------------------------------------------------------------- Table 7.3: fixed-split networks
fx = {}
for f in sorted(RES.glob("train100_fixedsplit__*.json")):
    fx.update(json.load(open(f))["results"])
if fx:
    rows = []
    for m, (otr, ova) in O_FX.items():
        if m not in fx:
            continue
        r = fx[m]
        ep = (str(r["epochs"]) + ("" if r.get("reached_target", True) else " (cap)") if "epochs" in r else
              {"HOG + RBF-SVM": "solver", "XGBoost (waveform features)": "2,000 trees", "Logistic regression (header)": "solver",
               "Header 0.6 + ResNet-18 0.4": str(fx.get("ResNet-18", {}).get("epochs", "--")) + " (ResNet-18)"}.get(m, "--"))
        rows.append([m.replace("(stages 5-6)", "(stages 5--6)"), ep, pct(r["train_acc"]), pct(r["val_acc"]),
                     f"{pct(otr)} / {pct(ova)}", chg(r["val_acc"], ova)])
    table("tab:7.3", "Image and signal networks on the original fixed split (trained on 669 ECGs, measured on 168 validation ECGs; "
          "the locked test is not loaded). TRAIN100: no early stopping, augmentation, weight decay, label smoothing or EMA, trained "
          "until training accuracy $\\ge$ 99.5 \\%. Original training accuracy was measured during the epoch with augmentation and "
          "dropout active.",
          [0.21, 0.14, 0.14, 0.17, 0.20, 0.14], ["Model", "Epochs", "Training accuracy", "Validation accuracy",
                                                 "Original: training / validation", "Change (points)"], rows)

# ---------------------------------------------------------------- Table 7.4: variants of every phase
V = load("train100_variants_base.json")
if V:
    S = V["summary"]; rows = []
    groups = [("Phase 5 variants", ["E1 MLP, all feature families", "E2 LightGBM (eng + ECGF-PCA)", "E3 weighted average",
                                    "E3 stacking (LR, balanced)", "E4 hierarchical MLP", "E6 signal-quality gating", "E7 mixup MLP"]),
              ("Phase 4", ["Multitask MLP (class + EF, lambda 1)"]),
              ("Phase 6B image-only models", ["HIER (two-stage)", "ENH M2 synthetic 10-s -> ECGFounder", "ENH M3 median-beat shape",
                                              "ENH M4 MiniRocket", "ENH M5 LR on HIER inputs", "ENH M6 EF ridge -> bands", "ENH-vote",
                                              "ULT M8 LR multi-scale ECGF", "ULT M9 kNN (k 15, distance)", "ULT M10 gradient boosting",
                                              "ULT M12 zone LR", "ULT-stack (no EchoNext)"]),
              ("Best-for-50s recipe without EchoNext (original column: same recipe on the original members; the published 0.587 also used EchoNext members)",
               ["SUPER-TOP10-ORD without EchoNext"])]
    ref = S.get("SUPER-TOP10-ORD without EchoNext, ORIGINAL recipe (reference)", {}).get("heldout_acc", {}).get("mean")
    cl = lambda m, k: sum([(c["base"][m][k] if isinstance(c["base"][m][k], list) else [c["base"][m][k]]) for c in V["cells"]], [])
    md = lambda m, k: f"{np.median(cl(m, k)):.0f}"
    EP74 = {"E1 MLP, all feature families": md("E1 MLP, all feature families", "epochs"), "E2 LightGBM (eng + ECGF-PCA)": "2,000 trees$^{a}$",
            "E3 weighted average": EP_FUS + "$^{b}$", "E3 stacking (LR, balanced)": EP_FUS + "$^{b}$",
            "E4 hierarchical MLP": md("E4 hierarchical MLP", "epochs_stage1") + " / " + md("E4 hierarchical MLP", "epochs_stage2"),
            "E6 signal-quality gating": EP_FUS + "$^{b}$", "E7 mixup MLP": md("E7 mixup MLP", "epochs"),
            "Multitask MLP (class + EF, lambda 1)": md("Multitask MLP (class + EF, lambda 1)", "epochs"), "HIER (two-stage)": EP_MLP + "$^{c}$",
            "ENH M2 synthetic 10-s -> ECGFounder": "solver$^{e}$", "ENH M3 median-beat shape": "solver$^{e}$", "ENH M4 MiniRocket": "solver$^{e}$",
            "ENH M5 LR on HIER inputs": "solver$^{e}$", "ENH M6 EF ridge -> bands": "exact$^{e}$", "ENH-vote": EP_MLP + "$^{c}$",
            "ULT M8 LR multi-scale ECGF": "solver$^{e}$", "ULT M9 kNN (k 15, distance)": "none (kNN)", "ULT M10 gradient boosting": "2,000 iter.$^{a}$",
            "ULT M12 zone LR": "solver$^{e}$", "ULT-stack (no EchoNext)": EP_MLP + "$^{c}$", "SUPER-TOP10-ORD without EchoNext": EP_MLP + "$^{c}$"}
    for g, ms in groups:
        rows.append(g)
        for m in ms:
            if m not in S:
                continue
            s = S[m]; o = O_VAR.get(m)
            if m == "SUPER-TOP10-ORD without EchoNext":
                o = ref
            tr = s.get("train_acc", {}).get("mean")
            rows.append([SHOW.get(m, m), EP74.get(m, "--"), pct(tr), pct(s["heldout_acc"]["mean"]) + " " + ci(s["heldout_acc"]["ci95_corrected"]),
                         "--" if m.endswith("(reference)") else pct(o), "--" if m.endswith("(reference)") else chg(s["heldout_acc"]["mean"], o)])
    table("tab:7.4", "Variants of every phase, trained to (nearly) 100 \\% training accuracy in the same development cross-validation. "
          "Members without an original stand-alone result show --." + EPN.replace("over the 15 folds;", "over the 15 folds (E4: stage 1 / stage 2; members of combiners as in Table~\\ref{tab:7.2}, same learners, seeds and rows);"),
          [0.23, 0.12, 0.14, 0.24, 0.14, 0.13], ["Model", "Epochs", "Training", "Held-out [95 \\% CI]", "Original", "Change (points)"], rows)
    ef = S.get("EF regression MLP (hdr + ECGF)"); ef2 = S.get("EF regression ridge + LightGBM (triage)")
    if ef:
        r2 = [["EF regression MLP, header + ECGFounder", f"{int(np.median(cl('EF regression MLP (hdr + ECGF)', 'epochs'))):,} (cap)", f"{ef['train_r']['mean']:.3f}", f"{ef['heldout_r']['mean']:.3f}",
               f"{ef['train_mae']['mean']:.1f} / {ef['heldout_mae']['mean']:.1f}", "0.653"]]
        if ef2:
            r2.append(["EF ridge + LightGBM (triage regressor)", "4,000 trees", f"{ef2['train_r']['mean']:.3f}", f"{ef2['heldout_r']['mean']:.3f}",
                       f"{ef2['train_mae']['mean']:.1f} / {ef2['heldout_mae']['mean']:.1f}", "--"])
        table("tab:7.5", "EF regression (Phase 4) trained without early stopping: correlation with echo EF on the training rows and "
              "held out, and mean absolute error in EF points. The MLP reached the 3,000-epoch cap in every fold (training $R^2$ just below the 0.99 target).",
              [0.25, 0.15, 0.14, 0.13, 0.18, 0.15], ["Regressor", "Epochs", "Training $r$", "Held-out $r$", "MAE training / held-out", "Original held-out $r$"], r2)

# ---------------------------------------------------------------- Table 7.6: label studies (C3 recipe)
Lb = load("train100_variants_labels.json")
if Lb:
    L = Lb["summary"]["label_studies (C3 recipe)"]
    spec = [("VER: 15 labels corrected to a high-confidence echo", "VER", "all_837", "all 837 (original labels)", .565),
            ("HC30: train on reliable links only", "HC30", "HC_395", "395 reliable links", .594),
            ("CLEAR: train without border ECGs", "CLEAR", "clear_cut", "582 clear-cut ECGs", .668),
            ("MMCLEAN: drop suspect Mild or Moderate labels", "MMC", "kept_MMC", "620 kept ECGs", .672),
            ("NF: noise-free training (all classes)", "NF", "NF_clear", "264 clean, clear-cut ECGs", .683),
            ("NFS: noise-free training, Severe kept", "NFS", "NF_clear", "264 clean, clear-cut ECGs", .702)]
    rows = []
    for name, cfg, sub, pop, o in spec:
        d = L[cfg]; a = d[f"accuracy_{sub}_orig_labels"]
        epl = f"{d['fusion_mlp_epochs_median']:.0f}$^{{b}}$" if "fusion_mlp_epochs_median" in d else "--"
        rows.append([name, pop, epl, pct(d["train_acc"]), pct(a), pct(o), chg(a, o)])
    table("tab:7.6", "Label-quality studies (report Table 4.7) with the C3 recipe trained to (nearly) 100 \\% training accuracy, "
          "scored on the same subsets as the original. Epochs: median over the 15 folds; $^{b}$ epochs of the neural member FUSION-MLP.",
          [0.17, 0.14, 0.12, 0.14, 0.15, 0.14, 0.14], ["Study", "Scored on", "Epochs", "Training", "Accuracy", "Original", "Change (points)"], rows)
print("tables written:", sorted(p.name for p in (TEX / "tables").glob("tab_7_*.tex")))
