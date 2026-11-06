# Best for 60s: triage model, cross-validation — 69.4 % on the patients it answers

| | Value [95 % CI] |
|---|---|
| Accuracy on answered patients | **0.694** [0.655, 0.736] |
| Share of patients answered | **56 %** [48, 63]; the rest are referred to echocardiography |
| Evaluated on | 837 development ECGs, patient-grouped 5-fold cross-validation × 3 repeats |
| Answers per repeat | Normal 394, Mild 4, Moderate 57, Severe 11 |
| "Normal" answers that are truly Normal | 74 % (90 % have EF ≥ 41 %) |
| Patients with EF < 41 % given a confident "Normal" | 15 % |

**The same model scored once on unseen patients is in `../best_for_70s/` (74.7 %).**

## How the triage model works
1. Image measurements from the drawing (rhythm, intervals, QRS, ST-T, vector features) — `code_1`;
   ECGFounder embedding of the digitised waveform (see `../best_for_50s/` code_1).
2. EF estimate = average of a ridge and a LightGBM regressor; EF is graded into the four classes
   (Normal cut 52 % men / 54 % women) — `code_3`.
3. The model **answers only when** (a) that grade agrees with the image-only classifier IMG-C3 — `code_2` — and
   (b) the EF estimate is far enough from every class cut. Otherwise it says "refer to echo" — `code_3`.
4. All thresholds are chosen inside the training folds (nested) — `code_3`; summary numbers — `code_4`.

## Confusion matrix
`best_for_60s_confusion_matrix.png`: rows = true class, columns = the model's answer, plus a grey column for the
patients referred to echo.
- **Where the counts come from:** the per-patient outputs are private and were not in the cloud copy of the data, so
  the matrix was recomputed from the cached inputs. Repeats 1 and 2 of the 3 cross-validation repeats reproduce
  bit for bit (SHA-256 of the image-only classifier outputs matches the registered hash); repeat 3 does not, so it is
  left out.
- **The counts are the mean of repeats 1–2:** 69.3 % accuracy with 55 % answered, against the official 69.4 % with 56 %
  over all 3 repeats. The stored 3-repeat totals are consistent with these two repeats.
- **What it shows:** true Normal answered → 98 % called Normal; Moderate 44 % right; Mild is almost never called Mild
  (it is either referred, 54 %, or called Normal).

## Files
| File | What it is | Original |
|---|---|---|
| `best_for_60s_code_1_image_features.py` | image measurements from the ECG drawing | [`research_experiments/phase_6b_threshold_sampling_lstm/mildfeat.py`](../../research_experiments/phase_6b_threshold_sampling_lstm/mildfeat.py) |
| `best_for_60s_code_2_image_only_classifier_IMG-C3.py` | IMG-C3 (the agreement check) | [`research_experiments/phase_6b_threshold_sampling_lstm/mildmodels.py`](../../research_experiments/phase_6b_threshold_sampling_lstm/mildmodels.py) |
| `best_for_60s_code_3_triage_model.py` | EF regressors, grading, answer rule (AGT-img) | [`research_experiments/phase_7/e6_cascade.py`](../../research_experiments/phase_7/e6_cascade.py) |
| `best_for_60s_code_4_triage_numbers.py` | the summary numbers in this table | [`research_experiments/phase_7/triage_statement.py`](../../research_experiments/phase_7/triage_statement.py) |
| `best_for_60s_results.json` | accuracy, precision / recall, answered share | [`research_experiments/phase_7/results/triage_scores.json`](../../research_experiments/phase_7/results/triage_scores.json) |
| `best_for_60s_results_detail.json` | composition of the answers, rule-out numbers | [`research_experiments/phase_7/results/triage_statement.json`](../../research_experiments/phase_7/results/triage_statement.json) |

## If asked
- *"Is 69 % four-class accuracy?"* Only on the 56 % of patients the model answers. On all patients it is about 0.56.
- *"Does it find Mild?"* Almost never (4 Mild answers per repeat); Mild patients are mostly referred or called Normal.
