"""Bring every remaining table and chart of the TRAIN100 results book into Chapter 7 of the final report.
Reads the aggregate results (../train100/results) and the results-book tables/figures; writes report-style tables
(labels tab:7.N, figure files fig_7_N.pdf) into latex/. Run from this folder: python3 add_results_book.py
"""
import ast
import json
import re
import shutil
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
TEX = HERE / "latex"
RES = HERE.parent / "train100" / "results"
RB = HERE.parent / "train100" / "results_book"


def colspec(ws):
    return "".join(r">{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr %.3f\linewidth-2\tabcolsep\relax}" % w for w in ws)


def group(n, text):
    return r"\multicolumn{%d}{>{\raggedright\arraybackslash\hspace{0pt}}p{\dimexpr\linewidth-2\tabcolsep\relax}}{\textit{%s}} \\" % (n, text)


def write(label, caption, widths, header, body, size=r"\small"):
    L = [r"\begin{table}[htbp]", r"\caption{%s}\label{tab:%s}" % (caption, label), size, r"\begin{tabular}{%s}" % colspec(widths),
         r"\toprule", " & ".join(r"\textbf{%s}" % h for h in header) + r" \\", r"\midrule", body.strip(), r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    (TEX / "tables" / f"tab_{label.replace('.', '_')}.tex").write_text("\n".join(L) + "\n")


def body_of(name):
    s = (RB / "tables" / name).read_text()
    return s[s.index(r"\midrule") + len(r"\midrule"):s.index(r"\bottomrule")].strip()


def metrics(C):
    C = np.asarray(C, float)
    rec = np.diag(C) / np.maximum(C.sum(1), 1)
    prec = np.divide(np.diag(C), C.sum(0), out=np.zeros(4), where=C.sum(0) > 0)
    f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
    return f1


def pooled_f1(cells, m):
    return np.mean([metrics(np.sum([c["base"][m]["heldout_confusion"] for c in cells if c["repeat"] == r], 0)) for r in (0, 1, 2)], 0)


# ---- Table 7.11: main models ranked, with balanced accuracy and per-class recall (results book Table 1)
write("7.11", r"All twelve main models after training to (nearly) 100 \% training accuracy, ranked by held-out accuracy, with macro-F1, Mild F1, "
      r"balanced accuracy and the recall of each class. Held-out accuracy: mean of the 15 test folds with the corrected 95 \% CI; the other columns "
      r"come from the confusion matrix pooled within each repeat and averaged over the 3 repeats. Rec. = recall of Normal, Mild, Moderate and Severe. "
      r"$^{\dagger}$ Selective: scored on the 48 \% of patients it answers.",
      [0.045, 0.165, 0.2, 0.08, 0.07, 0.07, 0.065, 0.065, 0.065, 0.065, 0.085],
      ["\\#", "Model", "Held-out accuracy [95 \\% CI]", "Macro F1", "Mild F1", "Bal. acc.", "Rec. N", "Rec. Mi", "Rec. Mo", "Rec. S", "Training acc."],
      body_of("main_ranked.tex").replace("Triage $^{\\dagger}$", "Image-only triage$^{\\dagger}$"), r"\footnotesize")

# ---- Table 7.5: networks with layers, parameters and best validation accuracy at any epoch (results book Table 4)
write("7.5", r"Image and signal networks on the original fixed split (trained on 669 ECGs, validated on 168; the locked test is not loaded), ranked by "
      r"validation accuracy. TRAIN100: no early stopping, augmentation, weight decay, label smoothing or EMA, trained until training accuracy $\ge$ 99.5 \%. "
      r"Best valid. = highest validation accuracy at any epoch on the way to 100 \%; original training accuracy was measured during the epoch with augmentation "
      r"and dropout active (Table~\ref{tab:3.3}).",
      [0.17, 0.22, 0.075, 0.075, 0.08, 0.08, 0.09, 0.09, 0.08],
      ["Model", "Layers", "Params", "Epochs", "Train.", "Valid.", "Best valid.", "Orig. valid.", "Change"], body_of("fixed.tex"), r"\scriptsize")

# ---- Table 7.6: add macro-F1 and Mild F1 columns to the variants table
src = (HERE.parent / "train100" / "report_nitw" / "make_tables.py").read_text()
show = ast.literal_eval(src[src.index("SHOW = {") + len("SHOW = "):src.index("}\n", src.index("SHOW = {")) + 1])
inv = {v: k for k, v in show.items()}
V = json.load(open(RES / "train100_variants_base.json"))
t = (TEX / "tables" / "tab_7_6.tex").read_text()
rows = []
for line in t[t.index(r"\midrule") + len(r"\midrule"):t.index(r"\bottomrule")].strip().split("\n"):
    if line.startswith(r"\multicolumn"):
        rows.append(re.sub(r"\\multicolumn\{6\}", r"\\multicolumn{8}", line)); continue
    cells = [c.strip() for c in line.rstrip("\\").rstrip().rstrip("\\").split(" & ")]
    key = inv.get(cells[0])
    f1 = pooled_f1(V["cells"], key) if key else None
    extra = ["--", "--"] if f1 is None else [f"{f1.mean():.3f}", f"{f1[1]:.3f}"]
    rows.append(" & ".join(cells[:4] + extra + cells[4:]) + r" \\")
cap = t[t.index(r"\caption{") + len(r"\caption{"):t.index(r"}\label{tab:7.6}")]
write("7.6", cap + r" Macro-F1 and Mild F1 from the held-out confusion matrices pooled within each repeat.",
      [0.2, 0.09, 0.1, 0.19, 0.08, 0.08, 0.1, 0.12],
      ["Model", "Epochs", "Training", "Held-out [95 \\% CI]", "Macro F1", "Mild F1", "Original", "Change (points)"], "\n".join(rows), r"\footnotesize")

# ---- Table 7.12: members of the three EchoNext ensembles (laptop run)
write("7.12", r"Members of the three ensembles of Table~\ref{tab:7.9} after training to (nearly) 100 \%, on the owner's computer, ranked by hospital held-out accuracy "
      r"(numbers as reported by that run). $^{\ddagger}$ Two-class task (Mild against Moderate). $^{\circ}$ Trained on EchoNext only: training and validation accuracy "
      r"on EchoNext, held-out accuracy on the hospital ECGs (H / plain = harmonised / plain hospital input). -- = not reported.",
      [0.46, 0.16, 0.18, 0.2], ["Member", "Training accuracy", "EchoNext validation", "Hospital held-out accuracy"], body_of("echonext_members.tex"), r"\footnotesize")

# ---- Tables 7.13-7.15: training accuracy and epochs of every model (results book Table 9), split by group
s = (RB / "tables" / "all_training.tex").read_text()
body = s[s.index(r"\endfoot") + len(r"\endfoot"):s.index(r"\end{longtable}")].strip()
blocks, cur = [], None
for line in body.split("\n"):
    m = re.match(r"\\multicolumn\{3\}\{l\}\{\\textbf\{(.*)\}\} \\\\", line.strip())
    if m:
        cur = [m.group(1), []]; blocks.append(cur); continue
    if line.strip() in ("", r"\midrule"):
        continue
    cur[1].append(line.strip())
parts = [("7.13", blocks[0:2]), ("7.14", blocks[2:4]), ("7.15", blocks[4:5])]
note = (r" Epochs: median over the 15 folds; solver / trees / iter. = models not trained in epochs; $^{b}$ epochs of the neural member FUSION-MLP; "
        r"$^{c}$ epochs of the neural member ECGF-MLP; $^{\dagger}$ selective model, training accuracy on the patients it answers; $^{\ddagger}$ two-class task; "
        r"$^{\circ}$ trained on EchoNext only (training accuracy on EchoNext); $^{*}$ numbers from the owner's other or laptop session; n/r = not reported; "
        r"$^{\S}$ not visible in the shared screenshot.")
titles = {"7.13": "main models and variants of every phase", "7.14": "label studies and networks on the fixed split", "7.15": "the laptop run with EchoNext"}
for lab, blks in parts:
    b = "\n".join(group(3, re.sub(r"^\([a-e]\) ", "", t0)) + "\n" + "\n".join(rs) for t0, rs in blks)
    write(lab, r"Training accuracy and epochs of every model after training to (nearly) 100 \%: " + titles[lab] + ", sorted from highest to lowest within each group."
          + (note if lab == "7.13" else r" Symbols as in Table~\ref{tab:7.13}."),
          [0.62, 0.2, 0.18], ["Model", "Epochs (median)", "Training accuracy"], b, r"\footnotesize")

# ---- Figures 7.5-7.9 from the results book
for src_, dst in (("fig_ranking", "fig_7_5"), ("fig_before_after", "fig_7_6"), ("fig_recall", "fig_7_7"), ("fig_fixed", "fig_7_8"), ("fig_variants", "fig_7_9")):
    shutil.copy(RB / "figures" / f"{src_}.pdf", TEX / "figures" / f"{dst}.pdf")
print("tables 7.5, 7.6, 7.11-7.15 and figures 7.5-7.9 written")

# ---- compact layout for the three widest tables (12-pt report class): smaller font and narrower column gaps
for lab, size, widths in (("7.11", r"\scriptsize", None), ("7.6", r"\scriptsize", None),
                          ("7.5", r"\scriptsize", [0.15, 0.185, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095, 0.095])):
    f = TEX / "tables" / f"tab_{lab.replace('.', '_')}.tex"
    t = f.read_text().split("\n")
    t[2] = size + r"\setlength{\tabcolsep}{3pt}"
    if widths:
        t[3] = r"\begin{tabular}{%s}" % colspec(widths)
    f.write_text("\n".join(t))
print("compact layout applied to Tables 7.5, 7.6 and 7.11")
