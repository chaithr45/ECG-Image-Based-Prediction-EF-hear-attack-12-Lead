"""Classical (non-deep) baseline: HOG features + RBF-SVM on the filtered ECG images.

    python -m ecg_ef.baseline
"""
from __future__ import annotations

import json
import time

import numpy as np
import pandas as pd
from skimage.feature import hog
from sklearn.decomposition import PCA
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from .config import CLASSES, RESULTS_DIR
from .data import load_split
from .metrics import evaluate
from .train import plot_confusion


def hog_features(imgs_u8) -> np.ndarray:
    return np.stack([hog(im[0], orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2),
                         feature_vector=True) for im in imgs_u8.numpy()])


def main():
    t0 = time.time()
    hw = (128, 256)  # HOG works on edge orientations; this keeps ~17k features
    (xtr, ytr, _), (xva, yva, _), (xte, yte, dte) = (
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
    val_m, C, clf = best
    pte = clf.predict_proba(fte)
    test_m = evaluate(yte.numpy(), pte)
    out = RESULTS_DIR / "hog_svm_filtered"
    out.mkdir(parents=True, exist_ok=True)
    summary = dict(tag="hog_svm_filtered", model="hog_svm", variant="filtered", pretrained=False,
                   img_hw=hw, params=None, epochs_run=None, best_epoch=None,
                   train_minutes=round((time.time() - t0) / 60, 1), C=C, val=val_m, test=test_m)
    (out / "metrics.json").write_text(json.dumps(summary, indent=2))
    plot_confusion(test_m["confusion_matrix"], f"HOG+SVM - test (acc {test_m['accuracy']:.1%})",
                   out / "confusion_matrix.png")
    pred = pd.DataFrame(pte, columns=[f"p_{c}" for c in CLASSES])
    pred.insert(0, "ecg_id", dte.ecg_id); pred.insert(1, "true", dte.label)
    pred.insert(2, "pred", [CLASSES[i] for i in pte.argmax(1)])
    pred["hf_likelihood"] = 1 - pte[:, 0]
    pred.round(4).to_csv(out / "test_predictions.csv", index=False)
    print(f"[HOG+SVM C={C}] TEST acc {test_m['accuracy']:.3f}  macro-F1 {test_m['macro_f1']:.3f}  "
          f"QWK {test_m['qwk']:.3f}  HF-AUC {test_m['hf_screen']['auc']:.3f}")


if __name__ == "__main__":
    main()
