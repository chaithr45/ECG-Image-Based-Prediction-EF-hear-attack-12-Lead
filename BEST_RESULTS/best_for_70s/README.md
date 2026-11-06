# Best for 70s: the frozen triage model on the locked test — 74.7 % on the patients it answers

| | Value [95 % CI] |
|---|---|
| Accuracy on answered patients | **0.747** [0.660, 0.837] (74 of 99) |
| Share answered | **49.7 %** [42.9, 56.7] (99 of 199 unseen ECGs); 100 referred to echo |
| Answers | Normal 88, Mild 0, Moderate 10, Severe 1 |
| "Normal" answers with EF ≥ 41 % | 85 % [77, 92] |
| Patients with EF < 41 % given a confident "Normal" | 21 % [11, 31] |
| Scored | once, owner-authorised, 2026-11-06; model frozen before scoring (SHA-256 09808d12…) |

**This is the same model as `../best_for_60s/` (69.4 % in cross-validation).** Every cross-validation value lies inside
its locked-test confidence interval.

## Confusion matrix
`best_for_70s_confusion_matrix.png`: rows = true class, columns = the model's answer, plus a grey column for the
patients referred to echo. It is rebuilt exactly from `best_for_70s_results_locked_test.json` (answers per class,
recalls and the make-up of the "Normal" answers fix every cell; the cells add up to 74 of 99).

## Files
| File | What it is | Original |
|---|---|---|
| `best_for_70s_code_1_freeze_final_model.py` | fits and freezes the final model on all 837 development ECGs | [`research_experiments/phase_7/testkit/freeze.py`](../../research_experiments/phase_7/testkit/freeze.py) |
| `best_for_70s_code_2_frozen_model.py` | the frozen model: PDF → inputs → answer / refer | [`research_experiments/phase_7/testkit/kit.py`](../../research_experiments/phase_7/testkit/kit.py) |
| `best_for_70s_code_3_score_locked_test.py` | one-shot locked-test scorer (refuses a second run) | [`research_experiments/phase_7/testkit/locked_test.py`](../../research_experiments/phase_7/testkit/locked_test.py) |
| `best_for_70s_code_4_score_new_ECGs.py` | scores a new set of ECG PDFs with the frozen model | [`research_experiments/phase_7/testkit/score.py`](../../research_experiments/phase_7/testkit/score.py) |
| `best_for_70s_results_locked_test.json` | all locked-test numbers | [`research_experiments/phase_7/results/testkit/locked_test_scores.json`](../../research_experiments/phase_7/results/testkit/locked_test_scores.json) |
| `best_for_70s_frozen_model_card.json` | frozen model settings and checks | [`research_experiments/phase_7/results/testkit_frozen_card.json`](../../research_experiments/phase_7/results/testkit_frozen_card.json) |
| `best_for_70s_test_kit_instructions.md` | how to test the frozen model on new ECGs | [`research_experiments/phase_7/testkit/TESTKIT.md`](../../research_experiments/phase_7/testkit/TESTKIT.md) |

## If asked
- *"What does 74.7 % mean?"* Of the half of new patients the model is confident about, three in four get the right
  class; the other half go to echo.
- *"Can it replace echo?"* No: 21 % of patients with EF < 41 % still got a confident "Normal", and it never calls Mild.
- *"Was the test set used before?"* The frozen model never saw it; the same 210 ECGs had been scored by earlier
  projects, so a new prospective set is the stronger confirmation (test kit, `code_4`).
