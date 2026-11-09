# Best for 50s: four-class accuracy on ALL patients

Every model here was trained until its training accuracy reached (nearly) 100 %, and is scored on held-out patients in
development cross-validation: patient-grouped 5 folds × 3 repeats on the 837 development ECGs. The models are sorted by
training accuracy.

| Model | Role | Training acc. | Held-out acc. [95 % CI] | Macro-F1 | Mild recall |
|---|---|---|---|---|---|
| E2 | best single model | 1.000 | **0.557** [0.517, 0.597] | 0.422 | 0.16 |
| ENH-vote | best without printed measurements | 1.000 | **0.549** [0.513, 0.585] | 0.418 | 0.18 |
| IMG-C3 | main image-only classifier | 1.000 | **0.532** [0.492, 0.572] | 0.416 | 0.23 |
| SUPER | greedy ensemble | 0.790 | **0.552** [0.507, 0.596] | 0.413 | 0.15 |
| SUPER-TOP10-ORD | **highest all-patient score** | 0.748 | **0.558** [0.530, 0.587] | 0.420 | 0.16 |
| STAGED-4 | staged ensemble | 0.722 | **0.548** [0.516, 0.581] | 0.416 | 0.15 |

- **Highest score:** SUPER-TOP10-ORD, 0.558 [0.530, 0.587]. Its training accuracy stays at 0.748 because its
  meta-learner is fitted on out-of-fold member outputs. It was run on the owner's computer, where the EchoNext data are.
- **Main image-only classifier:** IMG-C3, 0.532 [0.492, 0.572], heart-failure screen AUC
  0.822. It is also the classifier inside the triage model (`../best_for_60s/`).

## Confusion matrices
`best_for_50s_confusion_matrix.png` shows SUPER-TOP10-ORD (left) and IMG-C3 (right).
- Rows = true class, columns = predicted class.
- Counts are per repeat (each of the 837 ECGs is predicted once per repeat), and the colour is the row % (recall).

## How the models work
1. The waveform is digitised exactly from the vector PDF and passed through the frozen ECGFounder foundation model
   (stage-6 embedding, 1,024 numbers). See `code_1`.
2. **IMG-C3:** two heads on that embedding, an MLP (3 seeds) and a linear model, each trained to 100 % training
   accuracy. Their average vote goes through a class-prior decision rule chosen inside the training folds. See `code_2`.
3. **E2 / ENH-vote:** LightGBM on the engineered features with a 32-component projection of the ECGFounder embedding, and
   a vote of six image-only members. See `code_3`.
4. **SUPER-TOP10-ORD:** the image-derived models, including EchoNext-trained ones, are pooled. Inside each training
   fold the 10 best are kept, and an ordinal meta-learner combines them. See `code_4`.

## Files
| File | What it is | Original (run this one) |
|---|---|---|
| `best_for_50s_code_1_ECGFounder_features.py` | ECGFounder embedding of the digitised waveform (no training) | [`research_experiments/phase_2/ecgf_features.py`](../../research_experiments/phase_2/ecgf_features.py) |
| `best_for_50s_code_2_main_models_IMG-C3_trained_to_100.py` | the twelve main models, IMG-C3 included, trained to 100 % | [`research_experiments/train100/train100_features.py`](../../research_experiments/train100/train100_features.py) |
| `best_for_50s_code_3_variants_E2_ENH-vote_trained_to_100.py` | every variant, E2 and ENH-vote included | [`research_experiments/train100/train100_variants.py`](../../research_experiments/train100/train100_variants.py) |
| `best_for_50s_code_4_highest_score_ensemble_SUPER-TOP10-ORD_trained_to_100.py` | the three ensembles with EchoNext members | [`research_experiments/train100/train100_echonext.py`](../../research_experiments/train100/train100_echonext.py) |
| `best_for_50s_results.json` | every number above, with precision, recall, F1 and the confusion matrices | written by [`research_experiments/train100/make_best_results.py`](../../research_experiments/train100/make_best_results.py) |

## If asked
- *"What is your four-class accuracy?"* 0.53–0.56 on all patients: 0.558 for the best ensemble and 0.557 for the best
  single model.
- *"Why not higher?"* Our best EF estimate errs by about 14 EF points (r = 0.62 with echo). With class bands 10–12
  points wide, that caps four-class accuracy near 0.58 (simulation in the final report). Mild is the hardest class.
