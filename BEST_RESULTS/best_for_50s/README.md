# Best for 50s: four-class accuracy on ALL patients

## The two results in this band
| | Model | Accuracy [95 % CI] | Evaluated on | Mild recall | Macro-F1 |
|---|---|---|---|---|---|
| **Confirmed on unseen patients** | IMG-C3 (image-only four-class classifier) | **0.553** [0.485, 0.621] | locked test, 199 ECGs, scored once | 0.32 | 0.453 |
| same model, development | IMG-C3 | 0.558 [0.504, 0.611] | cross-validation, 837 ECGs | 0.19 | 0.441 |
| **Highest score** | SUPER-TOP10-ORD (ensemble) | **0.587** [0.565, 0.610] | cross-validation, 837 ECGs | 0.17 | 0.467 |

- **Use IMG-C3 when asked for "the" confusion matrix:** it is the only four-class model scored on unseen patients.
- **SUPER-TOP10-ORD is the highest number**, but it was picked after trying many variants, it is not significantly
  better than the alternatives (Holm p = 1.0) and it was never scored on the locked test. Its full confusion matrix
  was not stored, only recall (N / Mi / Mo / S 0.89 / 0.17 / 0.52 / 0.31) and precision (0.71 / 0.31 / 0.47 / 0.44).

## Confusion matrices
- `best_for_50s_confusion_matrix.png`: IMG-C3, locked test (rows = true class, columns = predicted).
- `best_for_50s_confusion_matrix_cross_validation.png`: IMG-C3, cross-validation (mean of 3 repeats).

## How IMG-C3 works
1. The waveform is digitised exactly from the vector PDF and passed through the frozen ECGFounder foundation model
   (stage-6 embedding, 1,024 numbers) — `code_1`.
2. Two small heads on that embedding (an MLP over 3 seeds and a linear model) predict the four classes — `code_2`.
3. IMG-C3 averages the two heads and applies a class-prior decision rule whose strength is chosen inside the
   training folds (nested) — `code_3`.
4. The locked test was scored once with the frozen model — `code_4`.

## How SUPER-TOP10-ORD works
- About 21 image-derived models (ECGFounder heads, engineered image features, EchoNext-pre-trained members, a pixel
  model) are pooled; inside each training fold the 10 best are kept and an ordinal meta-learner combines them — `code_5`.

## Files
| File | What it is | Original |
|---|---|---|
| `best_for_50s_code_1_ECGFounder_features.py` | ECGFounder embedding of the digitised waveform | [`research_experiments/phase_2/ecgf_features.py`](../../research_experiments/phase_2/ecgf_features.py) |
| `best_for_50s_code_2_ECGFounder_heads_training.py` | trains the MLP / linear heads (cross-validation) | [`research_experiments/phase_2/cv_features.py`](../../research_experiments/phase_2/cv_features.py) |
| `best_for_50s_code_3_image_only_classifier_IMG-C3.py` | IMG-C3: vote + nested decision rule | [`research_experiments/phase_6b_threshold_sampling_lstm/mildmodels.py`](../../research_experiments/phase_6b_threshold_sampling_lstm/mildmodels.py) |
| `best_for_50s_code_4_score_locked_test.py` | one-shot locked-test scorer | [`research_experiments/phase_7/testkit/locked_test.py`](../../research_experiments/phase_7/testkit/locked_test.py) |
| `best_for_50s_code_5_highest_score_ensemble_SUPER-TOP10-ORD.py` | SUPER / TOP10 / ordinal ensembles | [`research_experiments/public_ef/en_super.py`](../../research_experiments/public_ef/en_super.py) |
| `best_for_50s_results_locked_test.json` | locked-test numbers (IMG-C3 and triage) | [`research_experiments/phase_7/results/testkit/locked_test_scores.json`](../../research_experiments/phase_7/results/testkit/locked_test_scores.json) |
| `best_for_50s_results_SUPER-TOP10-ORD.json` | ensemble numbers | [`research_experiments/public_ef/results/en_super_original_top10.json`](../../research_experiments/public_ef/results/en_super_original_top10.json) |

## If asked
- *"What is your four-class accuracy?"* About 0.55–0.59 on all patients; 0.553 on unseen patients.
- *"Why not higher?"* The ECG-based EF estimate has about 13 EF points of error against echo; with class bands
  10–12 points wide that caps four-class accuracy near 0.59 (simulation in the final report). Mild is the hardest class.
