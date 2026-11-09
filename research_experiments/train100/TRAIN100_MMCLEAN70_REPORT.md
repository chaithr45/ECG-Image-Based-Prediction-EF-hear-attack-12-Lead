# TRAIN100: the 70s model (MMCLEAN selective) retrained to 100 % training accuracy

Declared before running in `AMENDMENT_TRAIN100_ECHONEXT.md`, Addendum B. Code: `train100_mmclean70.py`.
Results (aggregates only): `results/train100_mmclean70.json`.

**The 70s model** (owner's choice) is MMCLEAN selective: C3 trained on MMC (never fitted on the 217 ECGs whose Mild /
Moderate label has echo evidence against it), answering only when its vote confidence passes a threshold that is set
inside the training folds for 0.70 accuracy; the rest are referred to echo. Here its three members are retrained without
the brake: FUSION-MLP and ECGF-linear without early stopping or weight decay until >= 99.5 % training accuracy, and
HDRCLIN-LR with C = 10,000. The C3 prior rule and the selective threshold rule are unchanged.

**Evaluation:** the same patient-grouped 5-fold cross-validation x 3 repeats on the 837 development ECGs (original labels)
as the original 0.713. Hospital data only; the locked test set was not loaded.

## Table

| Model | Training (now) | Held-out (now) | Held-out (original) |
|---|---|---|---|
| **MMCLEAN selective (the 70s model)** | **99.9 %**, answering 11 % | **69.9 % [41.7, 91.7]**, answering 7 % | **71.3 % [63.7, 82.0]**, answering 22 % |

Training = the model fitted on the outer training rows and applied to those same (non-flagged) rows, accuracy on the
ones it answers (mean over the folds that answered; 100 % in 8 of them). Held-out = accuracy on the answered held-out
ECGs, per repeat over all 837, mean of 3 repeats; 95 % CI corrected for repeated CV over the fold cells with answers.
Change: **-1.4 points** in accuracy and **-15 points** in the share answered.

## What this shows

* **The model fits its training ECGs almost perfectly** (99.9 % on the ones it answers; the C3 vote 95.2 % on all of
  them). HDRCLIN-LR is the exception: a logistic regression on the low-dimensional header + clinical features cannot
  separate its training ECGs even with C = 10,000 (64 % training accuracy).
* **It now answers far fewer patients** (7 % instead of 22 %; per repeat 5.5 / 6.5 / 8.6 %), and **6 of the 15 folds
  answer nobody.** The reason: the members trained to 100 % are almost always very confident, so on the inner
  out-of-fold predictions the confidence no longer separates right from wrong answers, and in those 6 folds no
  threshold reaches 70 % inner accuracy with at least 20 answered ECGs. The rule then refers everyone.
* **Accuracy on the few it answers stays near 70 %** (69.9 %), but on so few ECGs that the CI is very wide (42-92 %).
  Answers per repeat: about 47 Normal, 0 Mild, 3 Moderate, 7 Severe; Mild is never called.
* **Four-class accuracy on all patients is unchanged:** C3 [MMC] 54.7 % [51.9, 57.6] held-out, the same as the original
  54.7 %, against 95.2 % training.
* The 100 % training accuracy is memorisation. For selective prediction it costs coverage: the confidence that the
  70 % rule depends on is no longer informative after the brake is removed.
