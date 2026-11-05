"""Exploratory MMC / MMNC configurations (AMENDMENT_MMCLEAN.md): the top-3 existing recipes (MLP fusion, C1 vote, C3)
trained without suspect Mild / Moderate labels (MMC), and without suspect Normal / Mild / Moderate labels (MMNC).
Severe ECGs are never removed (owner request).
Suspect = echo evidence against the label only (never a model prediction): folder label != nearest-echo class, a linked
echo of another class, EF spread > 10 points across links, or nearest-echo EF within 3 points of a class boundary.
Name-match score < 100 and |ECG - echo| > 30 days are counted but NOT used (they flag 47 % / 23 % of ECGs evenly across
classes: weak-link indicators, not evidence of a wrong label).
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --counts
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --run --jobs 2
    python -m research_experiments.phase_6b_threshold_sampling_lstm.mmclean --report
"""
from __future__ import annotations

import argparse
import json
import warnings
from multiprocessing import Pool

import numpy as np
import pandas as pd

from research_experiments.phase_5.p5common import cells, corr_ci, corr_t, holm
from research_experiments.phase_6b_threshold_sampling_lstm import p6bcommon as B

CONFIGS = ("MMC", "MMNC")
BAND, SPREAD, DAYS = 3.0, 10.0, 30
MODELS = (("mlpfusion", "MLP fusion"), ("vote", "C1 vote"), ("c3", "C3 (vote + nested prior rule)"))
ORIG_ID = {"mlpfusion": "P6A-mlpfusion-B1-v1__s{s}_r{r}", "vote": "P6A-vote-B1-v1__s0_r{r}", "c3": "P6A-c3-B1-v1__s0_r{r}"}
_F = None


def flags():
    """Per-ECG boolean flags (development set). Echo evidence only; computed once."""
    global _F
    if _F is not None:
        return _F
    from research_experiments.phase_6a_normal_mild_lstm.p6acommon import _links_dev
    from research_experiments.phase_6b_threshold_sampling_lstm.audit6b import true_interval
    T = B.table(); ef = T.ef.values; lab = T.label.values; IDS = B.IDS
    L = _links_dev()
    other = set(L.assign(lab=L.ecg_id.map(pd.Series(lab, index=IDS))).query("derived != 'Unknown' and derived != lab").ecg_id)
    days = true_interval()
    F = pd.DataFrame(dict(
        mild_moderate=np.isin(lab, ["Mild", "Moderate"]),
        normal=lab == "Normal",
        inconsistent=T.derived.values != lab,
        near_boundary=(np.abs(ef - T.sex_boundary.values) <= BAND) | (np.abs(ef - 41) <= BAND) | (np.abs(ef - 31) <= BAND),
        name_score_lt_100=T.score.values < 100,
        interval_gt_30d=~(days <= DAYS),
        other_class_link=np.isin(IDS, list(other)),
        ef_spread_gt_10=T.ef_spread.values > SPREAD))
    F["suspect_any"] = F.inconsistent | F.near_boundary | F.other_class_link | F.ef_spread_gt_10
    F["border_any"] = F.inconsistent | F.near_boundary                     # = AMENDMENT_CLEAR border definition
    F["MMC"] = F.mild_moderate & F.suspect_any
    F["MMNC"] = (F.mild_moderate | F.normal) & F.suspect_any               # Severe never removed
    _F = F
    return F


def removed(cfg):
    return flags()[cfg].values.copy()


def _excl(cfg, r, k, _orig=B.train_exclude):
    return removed(cfg) if cfg in CONFIGS else _orig(cfg, r, k)


def work(cfg):
    from research_experiments.phase_6b_threshold_sampling_lstm import existing6b as E
    B.train_exclude = _excl; E.train_exclude = _excl                      # patched inside the worker (spawned process)
    flags()
    return E.run_config(cfg)


def counts():
    F = flags(); lab = B.table().label.values
    out = {}
    for c in ("suspect_any", "border_any") + CONFIGS + ("inconsistent", "near_boundary", "name_score_lt_100", "interval_gt_30d",
                                                       "other_class_link", "ef_spread_gt_10"):
        m = F[c].values
        out[c] = dict(total=int(m.sum()), **{k: int((m & (lab == k)).sum()) for k in B.CLASSES})
    out["kept"] = {c: dict(total=int((~F[c].values).sum()), **{k: int((~F[c].values & (lab == k)).sum()) for k in B.CLASSES})
                   for c in CONFIGS}
    return out


# ------------------------------------------------------------------ evaluation
def preds(m, src, r):
    """Probabilities and predictions of model m; src = 'ORIG' (Phase 6A B1 runs, regenerated on this machine) or a config."""
    if src == "ORIG":
        ids = [ORIG_ID[m].format(s=s, r=r) for s in (B.SEEDS if m == "mlpfusion" else (0,))]
    else:
        ids = [f"P6B-{m}-{src}-v1__s{s}_r{r}" for s in (B.SEEDS if m == "mlpfusion" else (0,))]
    Ps = [B.load_run(i) for i in ids]
    P = np.mean([p[0] for p in Ps], 0)
    return P, (Ps[0][1] if m == "c3" else P.argmax(1))


def summary(m, src, mask):
    ms, cl = [], []
    for r in (0, 1, 2):
        P, pred = preds(m, src, r)
        ms.append(B.metrics(B.Y[mask], P[mask], pred[mask]))
        cl += [c["accuracy"] for c in cells(P, pred, r, mask=mask)]
    o = {k: float(np.mean([x[k] for x in ms])) for k in ("accuracy", "balanced_accuracy", "macro_f1", "recall_Normal", "recall_Mild",
                                                          "recall_Moderate", "recall_Severe", "min_recall", "f1_Mild")}
    present = [k for i, k in enumerate(B.CLASSES) if (B.Y[mask] == i).any()]
    o["classes_present"] = present
    o["min_recall"] = float(min(o[f"recall_{k}"] for k in present))          # over the classes present in the subset
    o["accuracy_ci95_corrected"] = corr_ci(cl)
    o["accuracy_per_repeat"] = [round(x["accuracy"], 4) for x in ms]
    o["admissible"] = bool(o["min_recall"] >= 0.15)
    o["confusion_mean"] = np.mean([B.confusion(B.Y[mask], preds(m, src, r)[1][mask]) for r in (0, 1, 2)], 0).round(1).tolist()
    return o


def paired(m, src, mask, metric="accuracy"):
    d, dr = [], []
    for r in (0, 1, 2):
        Pa, pa = preds(m, src, r); Pb, pb = preds(m, "ORIG", r)
        d += [x[metric] - z[metric] for x, z in zip(cells(Pa, pa, r, mask=mask), cells(Pb, pb, r, mask=mask))]
        dr.append(B.metrics(B.Y[mask], Pa[mask], pa[mask])[metric] - B.metrics(B.Y[mask], Pb[mask], pb[mask])[metric])
    _, p = corr_t(d)
    return dict(model=m, config=src, metric=metric, delta=float(np.mean(dr)), delta_per_repeat=[round(x, 4) for x in dr],
                ci95_corrected_delta=corr_ci(d), p_corrected_t=p,
                same_sign_all_repeats=bool(all(x > 0 for x in dr) or all(x < 0 for x in dr)))


def hc_mask():
    """AMENDMENT_MMCLEAN.md addition: MMNC-kept minus weak-link Normal / Mild / Moderate ECGs (Severe untouched)."""
    F = flags(); lab = B.table().label.values
    return ~F.MMNC.values & ~((F.name_score_lt_100 | F.interval_gt_30d).values & (lab != "Severe"))


def coverage_curve(m, src, levels=(1.0, 0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3)):
    out = {}
    for c in levels:
        acc, mix = [], []
        for r in (0, 1, 2):
            P, pred = preds(m, src, r); o = np.argsort(-P.max(1), kind="stable")[: int(round(c * B.nD))]
            acc.append(np.mean(pred[o] == B.Y[o])); mix.append(np.bincount(B.Y[o], minlength=4) / len(o))
        out[f"{c:.1f}"] = dict(accuracy=float(np.mean(acc)), accuracy_per_repeat=[round(x, 4) for x in acc],
                               true_class_mix={k: round(float(v), 3) for k, v in zip(B.CLASSES, np.mean(mix, 0))})
    return out


def report():
    F = flags(); allm = np.ones(B.nD, bool); hc = hc_mask()
    subsets = {"all_837": lambda c: allm, "kept": lambda c: ~F[c].values, "removed": lambda c: F[c].values,
               "HC": lambda c: hc}
    out = dict(counts=counts(), results=[], comparisons=[])
    for c in CONFIGS:
        for sname, sf in subsets.items():
            mask = sf(c)
            for m, label in MODELS:
                for src in ("ORIG", c):
                    out["results"].append(dict(config=c, evaluated_on=sname, n_ecgs=int(mask.sum()), model=label, trained=src,
                                               **summary(m, src, mask)))
                for metric in ("accuracy", "macro_f1", "recall_Mild"):
                    out["comparisons"].append(dict(evaluated_on=sname, n_ecgs=int(mask.sum()), **paired(m, c, mask, metric)))
    out["coverage_curve"] = {f"{label} [{src}]": coverage_curve(m, src) for m, label in MODELS for src in ("ORIG",) + CONFIGS}
    adj = holm([x["p_corrected_t"] for x in out["comparisons"]])
    for x, a in zip(out["comparisons"], adj):
        x["p_holm"] = float(a)
        x["genuine_improvement"] = bool(x["delta"] > 0 and a < 0.05 and x["same_sign_all_repeats"])
    (B.guard(B.RES) / "mmclean_exploratory.json").write_text(json.dumps(out, indent=1))
    rows = pd.DataFrame(out["results"])
    rows["accuracy_ci95_corrected"] = rows.accuracy_ci95_corrected.map(lambda v: f"[{v[0]:.3f}, {v[1]:.3f}]")
    rows.drop(columns=["confusion_mean", "accuracy_per_repeat"]).to_csv(B.guard(B.RES / "mmclean_results.csv"), index=False,
                                                                        float_format="%.4f")
    pd.DataFrame(out["comparisons"]).to_csv(B.guard(B.RES / "mmclean_comparisons.csv"), index=False, float_format="%.4f")
    plot(out)
    return out


def plot(out):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    R = out["results"]
    fig, axes = plt.subplots(1, 3, figsize=(16, 5), sharey=True)
    cols = {"ORIG": "#9aa5b1", "MMC": "#4c78a8", "MMNC": "#f58518"}
    for ax, sname, title in zip(axes, ("all_837", "kept", "removed"),
                                ("All 837 ECGs (original labels; primary)", "Kept ECGs of each configuration (descriptive)",
                                 "Removed ECGs of each configuration (descriptive)")):
        x = 0; ticks, tl = [], []
        for c in CONFIGS:
            for m, label in MODELS:
                for j, src in enumerate(("ORIG", c)):
                    rr = next(z for z in R if z["config"] == c and z["evaluated_on"] == sname and z["model"] == label and z["trained"] == src)
                    lo, hi = rr["accuracy_ci95_corrected"]
                    ax.bar(x + 0.4 * j, rr["accuracy"], 0.38, color=cols[src], yerr=[[rr["accuracy"] - lo], [hi - rr["accuracy"]]], capsize=2)
                    ax.text(x + 0.4 * j, hi + 0.01, f"{rr['accuracy']:.3f}", ha="center", fontsize=6.5, rotation=90)
                ticks.append(x + 0.2); tl.append(f"{label.split(' (')[0]}\n{c} (n={rr['n_ecgs']})")
                x += 1.1
            x += 0.4
        ax.set_xticks(ticks); ax.set_xticklabels(tl, fontsize=7, rotation=45, ha="right"); ax.set_title(title, fontsize=9)
        ax.axhline(0.6, ls=":", c="k", lw=0.8); ax.set_ylim(0, 1)
    axes[0].set_ylabel("accuracy (mean of 3 repeats, corrected 95 % CI)")
    from matplotlib.patches import Patch
    axes[0].legend(handles=[Patch(color=v, label=f"trained on {k}") for k, v in cols.items()], fontsize=7, loc="upper left")
    fig.suptitle("Top-3 recipes trained without suspect Mild/Moderate labels (MMC) or Normal/Mild/Moderate labels (MMNC); "
                 "development CV, exploratory", fontsize=10)
    fig.tight_layout(); B.guard(B.PLOTS).mkdir(parents=True, exist_ok=True)
    fig.savefig(B.PLOTS / "mmclean_accuracy.png", dpi=150); plt.close(fig)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for name, cur in out["coverage_curve"].items():
        xs = [float(k) for k in cur]; ys = [v["accuracy"] for v in cur.values()]
        ax.plot(xs, ys, marker="o", ms=3, lw=1.2 if "C3" in name else 0.8, ls="-" if "[ORIG]" in name else "--", label=name)
    ax.axhline(0.7, ls=":", c="k", lw=0.8); ax.invert_xaxis()
    ax.set_xlabel("coverage (share of the 837 ECGs answered; the rest referred to echo)")
    ax.set_ylabel("accuracy on answered ECGs (mean of 3 repeats)")
    ax.set_title("Accuracy vs coverage, ranked by model confidence (descriptive)", fontsize=9); ax.legend(fontsize=6)
    fig.tight_layout(); fig.savefig(B.PLOTS / "mmclean_coverage_curve.png", dpi=150); plt.close(fig)


def main():
    warnings.filterwarnings("ignore", message="Only one class is present")   # per-fold AUROC on small subsets (unused)
    ap = argparse.ArgumentParser()
    ap.add_argument("--counts", action="store_true"); ap.add_argument("--run", action="store_true")
    ap.add_argument("--report", action="store_true"); ap.add_argument("--jobs", type=int, default=2)
    a = ap.parse_args()
    if a.counts:
        print(json.dumps(counts(), indent=1))
    if a.run:
        B.table()
        with Pool(min(a.jobs, len(CONFIGS))) as pool:
            for c in pool.imap_unordered(work, CONFIGS):
                print("done", c, flush=True)
    if a.report:
        out = report()
        for x in out["results"]:
            print(f"{x['config']:5s} {x['evaluated_on']:8s} n={x['n_ecgs']:3d} {x['model'][:10]:10s} trained {x['trained']:5s} "
                  f"acc {x['accuracy']:.3f} {np.round(x['accuracy_ci95_corrected'], 3).tolist()} macroF1 {x['macro_f1']:.3f} "
                  f"Mild rec {x['recall_Mild']:.3f} min rec {x['min_recall']:.3f}")


if __name__ == "__main__":
    main()
