# TRAIN100: the 70s model retrained to 100 % training accuracy

Declared before running in `AMENDMENT_TRAIN100_ECHONEXT.md`, Addendum A. Code: `train100_triage70.py`.
Results (aggregates only): `results/train100_triage70.json`.

**The 70s model** is the frozen header-free triage model AGT-img (`BEST_RESULTS/best_for_60s`, `best_for_70s`). It
answers only when its EF grade agrees with the image-only classifier IMG-C3 and the EF estimate is far from every class
cut; everyone else is referred to echo. Here every part of it is retrained without the brake: no early stopping, no
weight decay, the ECGFounder heads trained until >= 99.5 % training accuracy, EF ridge alpha 1e-6, and EF LightGBM with
4,000 trees and no early stopping. The decision rules (prior rule, EF-variant choice, grade cuts, answer threshold) are
unchanged and still chosen inside the training folds.

**Evaluation:** the same patient-grouped 5-fold cross-validation x 3 repeats on the 837 development ECGs as its original
0.694. The locked test set was **not** loaded or rescored (owner's rule), so the 0.747 locked-test number of the frozen
original model has no TRAIN100 counterpart.

## Table

| Model | Training (now) | Held-out (now) | Held-out (original) |
|---|---|---|---|
| ECGF-MLP (3 seeds) | 99.6 % | 53.6 % | 54.2 % |
| ECGF-linear | 100 % | 49.8 % | 53.0 % |
| IMG-vote (mean of the two) | 100 % | 50.3 % | 54.1 % |
| IMG-C3 (IMG-vote + prior rule) | 100 % | 54.4 % [51.2, 57.7] | 55.8 % |
| EF regressor (RMSE, EF points; lower is better) | 4.6 | 14.1 | 13.6 |
| **Triage (the 70s model)** | **100 %**, answering 59 % | **70.4 % [65.4, 75.8]**, answering 53 % | **69.4 % [65.5, 73.6]**, answering 56 % |

Training = everything fitted on the outer training rows and applied to those same rows (mean of 15 folds); for the
triage it is the accuracy on the training ECGs it answers (100 % in every one of the 15 folds). Held-out = accuracy on
the answered held-out ECGs, per repeat over all 837, mean of 3 repeats; 95 % CI corrected for repeated CV.
Change for the triage: **+1.0 point** (inside both confidence intervals; not a real difference).

Answered held-out ECGs, mean per repeat: 441 of 837. Recall on answered ECGs, Normal / Mild / Moderate / Severe:
0.99 / 0.03 / 0.36 / 0.18 (original 0.98 / 0.02 / 0.45 / 0.17); macro-F1 0.386 (original 0.389). Answers per repeat:
Normal 383, Mild 6, Moderate 39, Severe 12. Mild is still almost never called Mild.

## What this shows

* Every part now fits its training ECGs (almost) perfectly: 99.6-100 % for the classifiers, EF error 4.6 points on the
  training ECGs against 14.1 on held-out ECGs.
* The held-out parts are the same or lower than the original (ECGF-linear -3.2 points, IMG-vote -3.8, IMG-C3 -1.4; EF
  error +0.6 points). This is overfitting, as expected before the run.
* The triage keeps its held-out accuracy (70.4 % vs 69.4 %) because its answer rule is tuned on inner out-of-fold
  predictions, not on the memorised training fit. It answers slightly fewer patients (53 % vs 56 %).
* The 100 % training accuracy is a memorisation number, not an estimate of accuracy on new patients. The honest number
  is the held-out 70.4 % on the 53 % it answers.
