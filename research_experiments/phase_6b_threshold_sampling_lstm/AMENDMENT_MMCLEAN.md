# Phase 6B exploratory amendment: MMC / MMNC (train without suspect Mild / Moderate / Normal labels)

**Date:** 2026-11-05. Written before any MMC / MMNC run (laptop session), at the owner's request:
- "few cases in the dataset are wrong ... the difficult Mild ... train by removing them in Mild and Moderate and the
  clear-cut border cases too ... no LSTM this time, our top 3 models";
- "keep Severe as it is ... only change for Mild, Moderate and Normal".

Code: `mmclean.py`.

- **Suspect label:** an ECG that meets at least one of these:
  - the class of the nearest echo differs from the folder label;
  - a linked echo is of another class;
  - the EF spread across linked echoes is > 10 points;
  - the nearest-echo EF is within 3.0 points of a class boundary (52 M / 54 F, 41 %, 31 %).

  The selection uses echo evidence only and never a model prediction. Removing ECGs because a model misclassifies
  them is circular: see the circularity check in `ADDENDUM_LABEL_EXPERIMENTS.md`.
- **Not used:** name-match score < 100 and an ECG–echo interval > 30 days. They flag 391 and 192 of 837 ECGs, evenly
  across classes (Normal 183 and 95). They show a weak link, not a wrong label. Using them would leave 33 Mild and
  35 Moderate ECGs.
- **MMC:** suspect Mild and Moderate ECGs removed. That is 217 ECGs (Mild 106, Moderate 111); kept: Normal 384,
  Mild 79, Moderate 66, Severe 91 (620).
- **MMNC:** suspect Normal, Mild and Moderate ECGs removed. That is 249 ECGs (Normal 32, Mild 106, Moderate 111);
  kept: Normal 352, Mild 79, Moderate 66, Severe 91 (588).
- **Severe ECGs are never removed** (owner request).
- **Training:** removed ECGs are excluded from every fit: the outer members, the nested inner models and the
  prior / rule fitting. This is the same mechanism as `AMENDMENT_CLEAR.md`. Labels are original; folds, inner-val and
  evaluation populations are unchanged.
- **Models:** the top 3 admissible-family recipes on the original labels, with no LSTM (owner request):
  - MLP fusion (3 seeds), 0.544;
  - C1 vote, 0.548;
  - C3 (vote + nested prior rule), 0.561.

  C2 (0.568) is not admissible (Mild recall < 0.15) and is not included.
- **Evaluation:**
  1. All 837 ECGs, original labels. This is primary and comparable with every other result.
  2. The kept ECGs of the configuration. Descriptive only:
     - it is not deployable, because the echo EF is unknown at prediction time;
     - keeping ECGs far from the boundaries makes the task easier by design (spectrum bias).
  3. The removed ECGs.

  On every subset, the same recipe trained on the original labels (Phase 6A B1 runs) is scored on the same ECGs. Each
  comparison uses a paired corrected repeated-CV t-test on accuracy, macro-F1 and Mild recall, with Holm correction
  over all tests (54; 72 with the HC subset of the addition below). A genuine improvement needs Holm p < 0.05, the same sign in all 3 repeats, and admissibility
  (every class recall ≥ 0.15).
- **Status:** EXPLORATORY and reported separately. It is not eligible as an improvement of the original task unless it
  meets the criterion on all 837 ECGs.

## Addition (2026-11-05, written before computing either analysis; owner request "bring accuracy above 70")
Both analyses are descriptive. Neither changes the models.
- **HC subset (evaluation only):** the MMNC-kept ECGs minus the Normal / Mild / Moderate ECGs with a weak echo link
  (name-match score < 100 or ECG–echo interval > 30 days). Severe is untouched. That leaves 333 ECGs: Normal 174,
  Mild 33, Moderate 35, Severe 91. The ORIG, MMC and MMNC models are all scored on it.
  - The class mix differs from the full set (Severe 27 % vs 11 %).
  - Its accuracy is a property of the selected ECGs, not of the model.
- **Coverage curve (all 837 ECGs):**
  - ECGs are ranked by the model's confidence (maximum vote probability); prediction = the model's decision (the C3
    rule for C3).
  - Accuracy is reported at 100 / 90 / 80 / 70 / 60 / 50 / 40 / 30 % coverage, with the unanswered ECGs referred to
    echo.
  - The coverage at which a target is reached is read from the same data, so it is optimistic.
- Neither selection criterion is tuned after seeing a result.
