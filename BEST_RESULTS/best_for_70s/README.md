# Best for 70s: the triage model frozen for new ECGs (test kit)

This folder holds the image-only triage model of `../best_for_60s/`, **frozen** for use on new patients, and the kit
that scores it on ECG PDFs you collect.
- Every part was trained on all 837 development ECGs until its training accuracy reached 100 %.
- The cross-validation expectation is 0.696 [0.630, 0.762] on the 48 % of patients it answers.
- A score in the 70s needs a new test set: the kit reports it with 95 % CIs next to that expectation.

| | Value |
|---|---|
| Frozen | 2026-11-09, file SHA-256 `d1b4ca2c…` (the file itself is private and git-ignored) |
| Training accuracy | MLP heads 277, 168, 225 epochs, linear head 100 L-BFGS steps, all **1.000**; IMG-C3 1.000; triage 1.000 on the 57 % of training ECGs it answers |
| Prior rule | α = 1.4 |
| EF grading | Normal above the sex boundary -3.5 (48.5 % men / 50.5 % women), Mild 42 up to that, Moderate 28.5–42, Severe below 28.5 |
| Answers when | the grade agrees with IMG-C3 and the EF estimate is at least 4.75 points from every cut |
| Cross-validation expectation | 48 % answered at 0.696 [0.630, 0.762]; IMG-C3 0.532 [0.492, 0.572] on all patients |
| Checks | gives exactly the report's cross-validated model on a test fold; 25 development PDFs read from scratch give identical inputs and decisions; end-to-end pipeline test passed |

## How to use it
See `best_for_70s_test_kit_instructions.md`. In short:
- run `python -m research_experiments.phase_7.testkit.score --skeleton <folder of PDFs> --out <manifest.csv>`;
- fill the manifest (sex, echo EF, echo gap, patient ID);
- run `python -m research_experiments.phase_7.testkit.score --manifest <manifest.csv> --name <run name>`.

The scorer uses this model by default (`--model train100`).

## Files
| File | What it is | Original (run this one) |
|---|---|---|
| `best_for_70s_code_1_freeze_final_model_trained_to_100.py` | freezes the triage model trained to 100 % on all 837 ECGs; self-test | [`research_experiments/train100/train100_freeze.py`](../../research_experiments/train100/train100_freeze.py) |
| `best_for_70s_code_2_frozen_model.py` | PDF → image inputs, the frozen model, guards | [`research_experiments/phase_7/testkit/kit.py`](../../research_experiments/phase_7/testkit/kit.py) |
| `best_for_70s_code_3_score_new_ECGs.py` | one-shot scorer for a test set you collect | [`research_experiments/phase_7/testkit/score.py`](../../research_experiments/phase_7/testkit/score.py) |
| `best_for_70s_frozen_model_card.json` | every chosen value, the training accuracy, the checks and the CV expectation | [`research_experiments/train100/results/testkit_frozen_card_train100.json`](../../research_experiments/train100/results/testkit_frozen_card_train100.json) |
| `best_for_70s_test_kit_instructions.md` | how to collect and score a test set | [`research_experiments/phase_7/testkit/TESTKIT.md`](../../research_experiments/phase_7/testkit/TESTKIT.md) |
