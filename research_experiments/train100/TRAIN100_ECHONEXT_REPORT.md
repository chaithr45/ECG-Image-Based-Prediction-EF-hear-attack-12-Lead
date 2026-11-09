# TRAIN100-EchoNext: the top 3 all-patient models retrained to (nearly) 100 % training accuracy

Declared before running in `AMENDMENT_TRAIN100_ECHONEXT.md` (commit 2a2acc6). Code: `train100_echonext.py`.
Results (aggregates only): `results/train100_echonext.json`.

The three best all-patient models of report 45 (Tables 5.2 and 6.1), all from `public_ef/en_super.py`, with **every
member and every meta-learner retrained without the training brake** (no early stopping, weight decay / L2 0, logistic
regression C = 10,000, boosting 2,000 trees with min_child_samples 1 and L2 0, ridge alpha 1e-6, kNN k 15, networks
until >= 99.5 % training accuracy; the pixel CNN retrained on the laptop GPU). Inputs, original four-class labels,
Phase-2 patient-grouped 5-fold CV x 3 repeats, inner splits, seeds and decision rules (top-10 / greedy selection,
floor rule, STAGED-4 thresholds) unchanged. Hospital data and EchoNext used as in the original pipelines.

## Models

| Model | Training accuracy | Held-out accuracy [95 % CI] | Recall N / Mi / Mo / S | Macro-F1 | Original [95 % CI] | Change (points) |
|---|---|---|---|---|---|---|
| SUPER-TOP10-ORD | 0.748 | 0.558 [0.530, 0.587] | 0.90 / 0.16 / 0.39 / 0.25 | 0.420 | 0.587 [0.565, 0.610] | -2.9 |
| STAGED-4 (pool X) | 0.722 | 0.548 [0.516, 0.581] | 0.86 / 0.15 / 0.45 / 0.23 | 0.416 | 0.585 [0.557, 0.613] | -3.7 |
| SUPER (greedy) | 0.790 | 0.552 [0.507, 0.596] | 0.87 / 0.15 / 0.46 / 0.21 | 0.413 | 0.583 [0.540, 0.626] | -3.1 |

Training accuracy = the fitted pipeline (members fitted on all outer training rows, then the meta-learner) applied to
those same training rows, mean of the 15 folds (per-fold range 0.66-0.88, 0.58-0.87 and 0.60-1.00). Held-out accuracy
as in the original: per repeat over all 837 development ECGs, mean of 3 repeats, corrected repeated-CV 95 % CI
(`phase_5.p5common.corr_ci`) over the 15 fold accuracies. Original = committed `public_ef/results/en_super_original*.json`.

## Members

| Member | Trained on | Training accuracy | Held-out accuracy | Held-out on | Accuracy on the 837 hospital ECGs |
|---|---|---|---|---|---|
| ECGF-MLP | hospital | 0.996 | 0.536 [0.501, 0.571] | hospital outer folds | - |
| ECGF-lin | hospital | 1.000 | 0.498 [0.454, 0.542] | hospital outer folds | - |
| M5 | hospital | 1.000 | 0.507 [0.452, 0.561] | hospital outer folds | - |
| M6 | hospital | 0.557 | 0.477 [0.458, 0.496] | hospital outer folds | - |
| M8 | hospital | 1.000 | 0.496 [0.459, 0.532] | hospital outer folds | - |
| M9 | hospital | 1.000 | 0.546 [0.520, 0.571] | hospital outer folds | - |
| M10 | hospital | 1.000 | 0.539 [0.496, 0.582] | hospital outer folds | - |
| M12 | hospital zone rows | 1.000 | 0.346 [0.302, 0.390] | hospital outer folds | - |
| H-TAB-LR | hospital | 1.000 | 0.473 [0.445, 0.501] | hospital outer folds | - |
| H-TAB-GBM | hospital | 1.000 | 0.548 [0.497, 0.599] | hospital outer folds | - |
| H-X-GBM | hospital | 1.000 | 0.546 [0.494, 0.598] | hospital outer folds | - |
| H-MiMo-GBM (Mild vs Moderate) | hospital Mild/Moderate rows | 1.000 | 0.569 [0.512, 0.626] | hospital outer folds | - |
| EF-band | hospital | 0.567 | 0.479 [0.460, 0.498] | hospital outer folds | - |
| EN-cls | EchoNext, 3853 rows | 1.000 | 0.663 | EchoNext validation, 771 rows | 0.479 |
| EN-band | EchoNext, 3853 rows | 0.781 (RMSE 6.4) | 0.520 (RMSE 19.8) | EchoNext validation, 771 rows | 0.479 |
| EN-cls-all-H | EchoNext, 3853 rows | 0.999 | 0.673 | EchoNext validation, 771 rows | 0.490 |
| EN-gbm-H | EchoNext, 3853 rows | 1.000 | 0.756 | EchoNext validation, 771 rows | 0.522 |
| EN-band-all-H | EchoNext, 3853 rows | 0.820 (RMSE 5.4) | 0.525 (RMSE 25.9) | EchoNext validation, 771 rows | 0.458 |
| EN-cls-all-plain | EchoNext, 3853 rows | 0.999 | 0.673 | EchoNext validation, 771 rows | 0.464 |
| EN-gbm-plain | EchoNext, 3853 rows | 1.000 | 0.756 | EchoNext validation, 771 rows | 0.522 |
| EN-band-all-plain | EchoNext, 3853 rows | 0.820 (RMSE 5.4) | 0.525 (RMSE 25.9) | EchoNext validation, 771 rows | 0.465 |
| PIX-ZS | EchoNext, 3467 rows | 0.999 | 0.725 | EchoNext validation, 386 rows | 0.505 |
| PIX-ZS-band | EchoNext, 3467 rows | 0.929 (RMSE 4.0) | 0.642 (RMSE 11.9) | EchoNext validation, 386 rows | 0.492 |
| EN-cls-all-X | EchoNext, 3853 rows | 0.996 | 0.694 | EchoNext validation, 771 rows | 0.497 |
| EN-gbm-X | EchoNext, 3853 rows | 1.000 | 0.756 | EchoNext validation, 771 rows | 0.539 |
| EN-MiMo-LR (Mild vs Moderate) | EchoNext, 661 rows | 1.000 | 0.604 | EchoNext validation, 134 rows | 0.577 |
| EN-MiMo-GBM (Mild vs Moderate) | EchoNext, 661 rows | 1.000 | 0.694 | EchoNext validation, 134 rows | 0.586 |
| EN-mu-mid | EchoNext, 2319 rows | 0.999 (RMSE 0.0) | 0.463 (RMSE 18.5) | EchoNext validation, 460 rows | 0.409 |

Hospital members: training = in-sample argmax accuracy on their own training rows (outer training rows; zone rows for
M12; Mild / Moderate rows for H-MiMo-GBM), mean of 15 folds; held-out = argmax accuracy over the outer folds, mean of 3
repeats [corrected CI]. M6 / EF-band are scored by the argmax of their band probabilities. EchoNext members: training =
the model used for the hospital ECGs, on all its EchoNext training rows; validation = the same recipe fitted on the
pipeline's grouped 80 % part and scored on the 20 % validation part (the image CNN: trained on 3,467, scored on 386);
EF regressors as graded four-class accuracy with RMSE in EF points; last column = accuracy on the hospital ECGs, which
these members never saw in training. Each "-H" / "-plain" pair is one EchoNext model applied to the harmonised and the
as-recorded hospital inputs. **EchoNext data:** all 4,827 rows of the pipeline cache come from EchoNext's published test
file; the pipeline's EchoNext locked test (974 rows, 20 % of patients by hash) was neither used nor scored.

Members chosen (folds out of 15): SUPER-TOP10-ORD: M10, EN-gbm-H, ECGF-MLP, EN-gbm-plain, H-TAB-GBM 15; PIX-ZS 14;
M9 12; M8 11; M5 10; PIX-ZS-band 10; H-TAB-LR 9. STAGED-4 (pool X): EN-gbm-X, EN-gbm-H, ECGF-MLP, EN-gbm-plain,
H-TAB-GBM, H-X-GBM 15; M10 14; PIX-ZS 13. SUPER (greedy): 1-4 members per fold; ECGF-MLP 7, EN-gbm-plain 6, M10 4,
PIX-ZS 4.

## Plain-English summary

* **Every classifier member now fits its training data (almost) perfectly**: all 21 reach 99.6-100 % training accuracy
  (the MLP heads after a median of 52 epochs, none at the cap; the pixel CNN after 13 epochs). Of the 7 EF regressors,
  the hospital ridge fits training EF exactly (RMSE 0.00005 EF points) but is scored through Gaussian bands whose
  width is its out-of-fold error (17.7 points), so its training accuracy stays at 0.56-0.57; the EchoNext EF
  regressors reach 0.78-1.00 graded accuracy on their training rows (RMSE 0.0-6.4).
* **Every model got worse on new patients.** The three ensembles lose 2.9-3.7 points (0.558 / 0.548 / 0.552 against
  0.587 / 0.585 / 0.583), and the members' held-out accuracies are 0.35-0.57. This is overfitting, as expected before
  the run: the extra fit is memorisation, not signal.
* **The ensembles' own training accuracy is only 0.72-0.79**, although their members are at 100 %. The meta-learner is
  fitted on the members' honest out-of-fold predictions, so it learns to trust the members that generalise, mostly
  EchoNext-trained ones (EN-gbm, PIX-ZS chosen in nearly every fold) that never saw the hospital ECGs and are therefore
  no more accurate on the training ECGs than on new ones.
* **EchoNext validation accuracies rise for the boosting and image members** (EN-gbm 0.756, PIX-ZS 0.725 on EchoNext's
  own validation patients), but that does not carry over to the hospital ECGs (0.46-0.54 zero-shot).
* Mild remains the weak class (recall 0.15-0.16).

## Run notes

* Smoke test (one fold, tiny budgets) first; then the full run. Time: EchoNext tabular members 2.4 min, pixel CNN
  5.4 min (GPU), hospital members about 10 min per fold and the three meta-learners about 2 min per fold; the 15 folds
  ran 3 at a time in about 63 minutes.
* While this run waited, the owner asked for the "70s model" first: see Addenda A and B of the amendment and
  `TRAIN100_TRIAGE70_REPORT.md` / `TRAIN100_MMCLEAN70_REPORT.md`. Nothing in this run was changed after seeing results.
* Aggregates only. EchoNext rows, per-record predictions, embeddings, images and all weights stay on the owner's laptop
  in git-ignored folders; the locked hospital test set was not loaded.
