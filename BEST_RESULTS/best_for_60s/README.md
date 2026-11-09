# Best for 60s: triage model, 69.6 % on the patients it answers

The image-only triage model was trained until its training accuracy reached 100 %, and is scored on held-out patients
in development cross-validation: patient-grouped 5 folds × 3 repeats on the 837 development ECGs.

| | Value [95 % CI] |
|---|---|
| Training accuracy on the training patients it answers | **1.000** (58 % of them answered) |
| Held-out accuracy on the patients it answers | **0.696** [0.630, 0.762] |
| Share of held-out patients answered | **48 %** [39, 57]; the rest are referred to echocardiography |
| Answers per repeat | Normal 364, Mild 6, Moderate 17, Severe 13 |
| "Normal" answers that are truly Normal | 72 % (89 % have EF ≥ 41 %) |
| Patients with EF < 41 % given a confident "Normal" | 15 % |

**On clean labels (C3 recipe trained to 100 %):** CLEAR 0.670 on 582 clear-cut ECGs (training 0.993); NFS 0.694 on 264 clean, clear-cut ECGs (training 0.985); MMCLEAN 0.669 on 620 kept ECGs (training 0.978).

The same triage model, frozen for scoring new ECGs, is in `../best_for_70s/`.

## How the triage model works
1. The model reads image measurements from the drawing (rhythm, intervals, QRS, ST-T, vector features; see `code_1`)
   and the ECGFounder embedding of the digitised waveform (`../best_for_50s/` code_1).
2. The EF estimate is the average of a ridge model (α 10⁻⁶) and a LightGBM model (4,000 trees), both fitted without
   shrinkage or early stopping. The estimate is graded into the four classes, with the Normal cut at 52 % for men and
   54 % for women. See `code_2`.
3. The model **answers only when** (a) that grade agrees with the image-only classifier IMG-C3 and (b) the EF estimate
   is far enough from every class cut. Otherwise it says "refer to echo". The cut-points and the margin are chosen
   inside the training folds. See `code_2`.
4. How accuracy rises as fewer patients are answered, and how errors depend on the distance from the class limits: `code_3`.

## Confusion matrix
`best_for_60s_confusion_matrix.png`: rows = true class, columns = the model's answer, plus a grey column for the
patients referred to echo. Counts are per repeat (mean of the 3 repeats).

## Files
| File | What it is | Original (run this one) |
|---|---|---|
| `best_for_60s_code_1_image_features.py` | image measurements from the ECG drawing (no training) | [`research_experiments/phase_6b_threshold_sampling_lstm/mildfeat.py`](../../research_experiments/phase_6b_threshold_sampling_lstm/mildfeat.py) |
| `best_for_60s_code_2_triage_model_trained_to_100.py` | IMG-C3 and the triage model, trained to 100 % | [`research_experiments/train100/train100_features.py`](../../research_experiments/train100/train100_features.py) |
| `best_for_60s_code_3_confidence_and_error_analysis.py` | accuracy against share answered, and error rate against distance from the class limits | [`research_experiments/train100/train100_scores.py`](../../research_experiments/train100/train100_scores.py) |
| `best_for_60s_results.json` | every number above, plus the clean-label studies and the accuracy-by-share curve | written by [`research_experiments/train100/make_best_results.py`](../../research_experiments/train100/make_best_results.py) |

## If asked
- *"How can it be right 70 % of the time?"* It answers only the 48 % of patients whose
  ECG gives a clear, consistent picture. The others, mostly near a class limit, are referred to echocardiography.
- *"Is it safe?"* 89 % of its confident "Normal" answers have an echo EF of
  at least 41 %. It rarely answers "Mild": most Mild patients it does answer are called Normal, so the Mild band is
  left to the echo.
