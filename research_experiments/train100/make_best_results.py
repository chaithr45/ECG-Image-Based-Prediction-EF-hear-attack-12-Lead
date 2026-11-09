"""BEST_RESULTS from the models trained to (nearly) 100 % training accuracy: result JSONs and confusion matrices for
best_for_50s / best_for_60s / best_for_70s (aggregates only), and the copies of the code (marked COPY FOR READING).
Reads train100/results; writes BEST_RESULTS/. Run from anywhere: python3 make_best_results.py
"""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

HERE = Path(__file__).resolve().parent
RES = HERE / "results"
REPO = HERE.parents[1]
BR = REPO / "BEST_RESULTS"
for f in ("lmroman10-regular", "lmroman10-bold"):
    try:
        font_manager.fontManager.addfont(f"/usr/share/texmf/fonts/opentype/public/lm/{f}.otf"); plt.rcParams["font.family"] = "Latin Modern Roman"
    except Exception:
        pass
INK, MUTED = "#1f2328", "#5b6470"
CMAP = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
CL = ["Normal", "Mild", "Moderate", "Severe"]


def pooled(cells, get):
    reps = []
    for r in (0, 1, 2):
        C = np.sum([get(c) for c in cells if c["repeat"] == r], 0).astype(float)
        rec = np.diag(C) / np.maximum(C.sum(1), 1); prec = np.divide(np.diag(C), C.sum(0), out=np.zeros(4), where=C.sum(0) > 0)
        f1 = np.divide(2 * prec * rec, prec + rec, out=np.zeros(4), where=(prec + rec) > 0)
        reps.append(dict(recall=rec, precision=prec, f1=f1))
    m = {k: np.mean([x[k] for x in reps], 0) for k in reps[0]}
    return dict(macro_f1=float(m["f1"].mean()), balanced_accuracy=float(m["recall"].mean()),
                recall=dict(zip(CL, m["recall"].round(3).tolist())), precision=dict(zip(CL, m["precision"].round(3).tolist())),
                f1=dict(zip(CL, m["f1"].round(3).tolist())))


def r3(x):
    return None if x is None else round(float(x), 3)


F = json.load(open(RES / "train100_features.json")); V = json.load(open(RES / "train100_variants_base.json"))
L = json.load(open(RES / "train100_variants_labels.json"))["summary"]["label_studies (C3 recipe)"]
E = json.load(open(RES / "train100_echonext.json")); S = json.load(open(RES / "train100_scores.json"))
CARD = json.load(open(RES / "testkit_frozen_card_train100.json"))
SF, VS = F["summary"], V["summary"]
NOTE = ("Every model trained to (nearly) 100 % training accuracy (research_experiments/train100). Development "
        "cross-validation: patient-grouped 5 folds x 3 repeats on the 837 development ECGs; held-out accuracy = mean of the 15 "
        "test folds with the corrected 95 % CI. Aggregates only. Sorted by training accuracy.")


def cv_model(name, key, src="features"):
    if src == "features":
        s = SF[key]; cells = F["cells"]; get = lambda c: c["models"][key]["heldout_confusion"]  # noqa: E731
        tr, acc, ci = s["train_accuracy"]["mean"], s["heldout_accuracy"]["mean"], s["heldout_accuracy"]["ci95_corrected"]
    else:
        s = VS[key]; cells = V["cells"]; get = lambda c: c["base"][key]["heldout_confusion"]  # noqa: E731
        tr, acc, ci = s["train_acc"]["mean"], s["heldout_acc"]["mean"], s["heldout_acc"]["ci95_corrected"]
    out = dict(model=name, training_accuracy=r3(tr), heldout_accuracy=r3(acc), heldout_accuracy_ci95=[r3(x) for x in ci], **pooled(cells, get),
               confusion_mean_per_repeat=(np.sum([get(c) for c in cells], 0) / 3).round(1).tolist())
    if src == "features" and key in S["screen_auc"]:
        out["heart_failure_screen_auc"] = r3(S["screen_auc"][key])
    return out


def ens(name, key):
    m = E["models"][key]; h = m["heldout"]
    return dict(model=name, training_accuracy=r3(m["train_accuracy"]["mean"]), heldout_accuracy=r3(h["accuracy"]),
                heldout_accuracy_ci95=[r3(x) for x in h["accuracy_ci95_corrected"]], macro_f1=r3(h["macro_f1"]),
                recall={c: r3(h[f"recall_{c}"]) for c in CL}, precision={c: r3(h[f"precision_{c}"]) for c in CL},
                confusion_mean_per_repeat=h["confusion_mean_per_repeat"], note="run on the owner's computer (EchoNext data stay there)")


# ------------------------------------------------------------------ 50s: all patients
r50 = [ens("SUPER-TOP10-ORD: ten best members, ordinal meta-learner (with EchoNext members), highest all-patient score", "SUPER-TOP10-ORD"),
       ens("SUPER: greedy member selection (with EchoNext members)", "SUPER (greedy)"),
       ens("STAGED-4: extended pool, staged meta-learner (with EchoNext members)", "STAGED-4 (pool X)"),
       cv_model("E2: LightGBM on header + engineered features + ECGFounder PCA, best single model", "E2 LightGBM (eng + ECGF-PCA)", "variants"),
       cv_model("ENH-vote: six image-only members, best model without printed measurements", "ENH-vote", "variants"),
       cv_model("IMG-C3: image-only vote of the ECGFounder heads + prior rule, the main image-only classifier", "IMG-C3")]
r50.sort(key=lambda d: (-d["training_accuracy"], -d["heldout_accuracy"]))
(BR / "best_for_50s" / "best_for_50s_results.json").write_text(json.dumps(dict(note=NOTE, population="all 837 development ECGs", models=r50), indent=1))

# ------------------------------------------------------------------ 60s: answered / clean patients
st = SF["AGT-img triage"]; tcm = np.array(st["heldout_confusion_mean_per_repeat"])
tri = dict(model="Image-only triage (AGT-img): answers when the image EF grade agrees with IMG-C3 and the EF estimate is far from every class cut",
           training_accuracy_on_answered=r3(st["train_accuracy_answered"]["mean"]), training_answered_share=r3(st["train_coverage"]["mean"]),
           heldout_accuracy_on_answered=r3(st["heldout_accuracy_answered"]["mean"]),
           heldout_accuracy_ci95=[r3(x) for x in st["heldout_accuracy_answered"]["ci95_corrected"]],
           heldout_answered_share=r3(st["heldout_coverage"]["mean"]), heldout_answered_share_ci95=[r3(x) for x in st["heldout_coverage"]["ci95_corrected"]],
           **pooled(F["cells"], lambda c: c["triage"]["heldout_confusion"]),
           confusion_answered_mean_per_repeat=tcm.round(1).tolist(), referred_mean_per_repeat=st["heldout_referred_mean_per_repeat"],
           normal_answers_with_ef_at_least_41=r3(tcm[:2, 0].sum() / tcm[:, 0].sum()))
labs = [dict(model=f"{n} (C3 recipe)", scored_on=p, training_accuracy=r3(L[c]["train_acc"]), accuracy=r3(L[c][k]))
        for n, c, p, k in (("NFS: noise-free training, Severe kept", "NFS", "264 clean, clear-cut ECGs", "accuracy_NF_clear_orig_labels"),
                           ("CLEAR: trained without border ECGs", "CLEAR", "582 clear-cut ECGs", "accuracy_clear_cut_orig_labels"),
                           ("MMCLEAN: suspect Mild / Moderate labels dropped", "MMC", "620 kept ECGs", "accuracy_kept_MMC_orig_labels"))]
labs.sort(key=lambda d: (-d["training_accuracy"], -d["accuracy"]))
cov = {m: [dict(answered_share=p["coverage"], accuracy=r3(p["accuracy"])) for p in v if p["coverage"] in (1.0, 0.75, 0.5, 0.25)] for m, v in S["coverage"].items()}
(BR / "best_for_60s" / "best_for_60s_results.json").write_text(json.dumps(dict(
    note=NOTE, triage=tri, clean_label_studies=labs,
    accuracy_by_share_answered=dict(note="most confident first, descriptive (cut-off not nested)", **cov)), indent=1))

# ------------------------------------------------------------------ 70s: the frozen triage model for new ECGs (test kit)
shutil.copy(RES / "testkit_frozen_card_train100.json", BR / "best_for_70s" / "best_for_70s_frozen_model_card.json")


# ------------------------------------------------------------------ confusion matrices
def draw(ax, M, title, ref=None):
    M = np.asarray(M, float); P = M / np.maximum(M.sum(1)[:, None], 1e-9)
    ax.imshow(P, cmap=CMAP, vmin=0, vmax=1, extent=(-0.5, 3.5, 3.5, -0.5))
    for i in range(4):
        for j in range(4):
            dark = P[i, j] > 0.5
            ax.text(j, i - 0.13, f"{M[i, j]:.0f}", ha="center", va="center", fontsize=11, fontweight="bold" if i == j else "normal", color="white" if dark else INK)
            ax.text(j, i + 0.22, f"{P[i, j]:.0%}", ha="center", va="center", fontsize=8, color="white" if dark else MUTED)
    for k in range(5):
        ax.axhline(k - 0.5, color="white", lw=1.5); ax.axvline(k - 0.5, color="white", lw=1.5)
    if ref is not None:
        for i in range(4):
            ax.add_patch(plt.Rectangle((3.6, i - 0.45), 0.9, 0.9, facecolor="#eef0f2", edgecolor="none"))
            ax.text(4.05, i, f"{ref[i]:.0f}", ha="center", va="center", fontsize=10, color=MUTED)
        ax.set_xlim(-0.5, 4.55); ax.set_xticks(range(5)); ax.set_xticklabels(CL + ["Referred"], fontsize=9)
    else:
        ax.set_xticks(range(4)); ax.set_xticklabels(CL, fontsize=9)
    ax.set_yticks(range(4)); ax.set_yticklabels(CL, fontsize=9); ax.tick_params(length=0); ax.xaxis.set_ticks_position("top")
    ax.set_ylabel("True class", color=MUTED); ax.set_title(title, fontsize=10.5, color=INK, pad=24, loc="left")
    for s in ax.spines.values():
        s.set_visible(False)


def save(fig, path):
    fig.savefig(f"{path}.png", dpi=170, facecolor="white"); fig.savefig(f"{path}.pdf", facecolor="white"); plt.close(fig)


best, img = r50[[d["model"].startswith("SUPER-TOP10") for d in r50].index(True)], r50[[d["model"].startswith("IMG-C3") for d in r50].index(True)]
fig, axs = plt.subplots(1, 2, figsize=(10.4, 4.9))
draw(axs[0], best["confusion_mean_per_repeat"], f"SUPER-TOP10-ORD: training {best['training_accuracy']:.3f}, held-out {best['heldout_accuracy']:.3f}")
draw(axs[1], img["confusion_mean_per_repeat"], f"IMG-C3 (image only): training {img['training_accuracy']:.3f}, held-out {img['heldout_accuracy']:.3f}")
fig.text(0.01, 0.01, "Held-out, development cross-validation; counts per repeat (each of the 837 ECGs predicted once per repeat), colour = row % (recall). "
         "Columns = predicted class.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, BR / "best_for_50s" / "best_for_50s_confusion_matrix")

fig, ax = plt.subplots(figsize=(6.6, 5.0))
draw(ax, tcm, f"Image-only triage: training {tri['training_accuracy_on_answered']:.3f} on answered, held-out {tri['heldout_accuracy_on_answered']:.3f} "
              f"on the {100 * tri['heldout_answered_share']:.0f} % answered", np.array(st["heldout_referred_mean_per_repeat"]))
fig.text(0.01, 0.01, "Held-out, development cross-validation; counts per repeat; colour = row % among answered. Referred = sent to echo.", fontsize=8, color=MUTED)
fig.tight_layout(rect=(0, 0.04, 1, 1)); save(fig, BR / "best_for_60s" / "best_for_60s_confusion_matrix")

# ------------------------------------------------------------------ code copies (for reading; run the originals)
COPIES = {"best_for_50s": [("best_for_50s_code_2_main_models_IMG-C3_trained_to_100.py", "train100/train100_features.py"),
                           ("best_for_50s_code_3_variants_E2_ENH-vote_trained_to_100.py", "train100/train100_variants.py"),
                           ("best_for_50s_code_4_highest_score_ensemble_SUPER-TOP10-ORD_trained_to_100.py", "train100/train100_echonext.py")],
          "best_for_60s": [("best_for_60s_code_2_triage_model_trained_to_100.py", "train100/train100_features.py"),
                           ("best_for_60s_code_3_confidence_and_error_analysis.py", "train100/train100_scores.py")],
          "best_for_70s": [("best_for_70s_code_1_freeze_final_model_trained_to_100.py", "train100/train100_freeze.py"),
                           ("best_for_70s_code_2_frozen_model.py", "phase_7/testkit/kit.py"),
                           ("best_for_70s_code_3_score_new_ECGs.py", "phase_7/testkit/score.py")]}
for folder, items in COPIES.items():
    for dst, src in items:
        body = (REPO / "research_experiments" / src).read_text()
        mod = "research_experiments." + src[:-3].replace("/", ".")
        run = (f"PYTHONPATH=. python research_experiments/{src}" if src.startswith("train100") else f"python -m {mod}")
        head = (f"# >>> COPY FOR READING. Original (run this one; its imports need the project): research_experiments/{src}\n"
                f"# >>> Run from the repository root: {run} (see the docstring below for the options)\n"
                f"# >>> Part of BEST_RESULTS/{folder}; see the README there.\n")
        (BR / folder / dst).write_text(head + body)
print("BEST_RESULTS written")
