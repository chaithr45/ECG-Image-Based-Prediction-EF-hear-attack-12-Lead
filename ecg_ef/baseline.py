"""Classical (non-deep) baseline: HOG features + RBF-SVM on the filtered ECG images.

    python -m ecg_ef.baseline
"""
from __future__ import annotations

import json
import time

import numpy as np
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import HF_TARGET_SENSITIVITY, RESULTS_DIR
from .data import load_split
from .metrics import evaluate, pick_hf_threshold
from .train import plot_confusion, save_predictions


def hog_features(imgs_u8) -> np.ndarray:
    return np.stack([hog(im[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                         feature_vector=True) for im in imgs_u8.numpy()])


def main():
    t0 = time.time()
    hw = (128, 256)  # HOG works on edge orientations; this keeps ~17k features
    (xtr, ytr, _), (xva, yva, dva), (xte, yte, dte) = (
        load_split(s, "filtered", hw) for s in ("train", "val", "test"))
    ftr, fva, fte = hog_features(xtr), hog_features(xva), hog_features(xte)
    print("HOG features:", ftr.shape)

    best = None
    for C in (0.3, 1, 3, 10, 30):
        clf = make_pipeline(StandardScaler(), PCA(150, random_state=0),
                            SVC(C=C, kernel="rbf", class_weight="balanced", probability=True,
                                random_state=0))
        clf.fit(ftr, ytr.numpy())
        vm = evaluate(yva.numpy(), clf.predict_proba(fva))
        print(f"C={C:<4}  val acc {vm['accuracy']:.3f}  macro-F1 {vm['macro_f1']:.3f}")
        if best is None or vm["macro_f1"] > best[0]["macro_f1"]:
            best = (vm, C, clf)
    _, C, clf = best
    pva, pte = clf.predict_proba(fva), clf.predict_proba(fte)
    hf_thr = pick_hf_threshold(yva.numpy(), pva, HF_TARGET_SENSITIVITY)
    val_m = evaluate(yva.numpy(), pva, hf_thr)
    test_m = evaluate(yte.numpy(), pte, hf_thr)
    out = RESULTS_DIR / "hog_svm_filtered"
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(tag="hog_svm_filtered", model="hog_svm", variant="filtered", pretrained=False,
                   img_hw=hw, params=None, epochs_run=None, best_epoch=None,
                   train_minutes=round((time.time() - t0) / 60, 1), C=C, hf_threshold=hf_thr,
                   val=val_m, test=test_m)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(test_m["confusion_matrix"], f"HOG+SVM - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    save_predictions(pte, dte, out / "test_predictions.csv")
    save_predictions(pva, dva, out / "val_predictions.csv")
    print(f"[HOG+SVM C={C}] TEST acc {test_m['accuracy']:.3f}  macro-F1 {test_m['macro_f1']:.3f}  "
          f"QWK {test_m['qwk']:.3f}  HF-AUC {test_m['hf_screen']['auc']:.3f}")


if __name__ == "__main__":
    main()
