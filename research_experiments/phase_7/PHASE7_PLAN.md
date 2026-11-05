# Phase 7 plan (pre-registration)

**Date:** 2026-11-05 (laptop session, RTX 3060 6 GB). Written before any Phase 7 run. Owner brief (verbatim intent):
"genuine improvement in four-class LVEF classification, with reliable Mild detection".
- **Recurrent models are discontinued:** LSTM, BiLSTM, GRU, CNN-LSTM, ECGFounder-LSTM fusion and any other recurrent
  network. None is running (checked 2026-11-05: no Python process, GPU idle) and none will be launched.
- **Scope:** development set only (837 ECGs, 794 patient keys). The locked 210-ECG test cohort is not loaded, inspected
  or scored without explicit owner authorisation.

## 1. Primary task, targets and reference
- **Primary task:** the original four-class EF definitions (Normal M > 52 / F ≥ 54, Mild ≥ 41, Moderate 31–< 41,
  Severe < 31), scored on all 837 ECGs. The evaluation population is never changed. Alternative thresholds and subsets
  are sensitivity analyses only and are reported separately.
- **Owner research targets** (not permission to manipulate data or evaluation):

  | Metric | Target |
  |---|---|
  | Accuracy | ≥ 0.70 |
  | Mild recall | ≥ 0.60 |
  | Mild precision | ≥ 0.50 |
  | Mild F1 | ≥ 0.55 |
  | Macro-F1 | ≥ 0.60 |
  | Balanced accuracy | ≥ 0.60 |

- **Reference:** C3 (C1 vote + nested prior rule), accuracy 0.562 [0.519, 0.605], macro-F1 0.449, Mild P / R / F1
  about 0.31 / 0.18 / 0.22. For pilots, the reference is the same-recipe member, the Phase 2 MLP fusion on [header ‖
  ECGFounder f6], seed 42 (`P6A-mlpfusion-B1-v1__s42_r*`).
- **Prior evidence** (Phase 6B, MMCLEAN, MILDFIX, MILDFIX2):
  - the ECG–echo EF error is about 13 points SD (11.8 on clean links);
  - Mild-vs-Moderate AUROC is about 0.6;
  - a calibrated EF estimator with 13-point error caps four-class accuracy at about 0.63;
  - 70 % needs about 9 points.

  Phase 7 is therefore expected to improve Mild separation only modestly. The targets will be reported as met or not
  met, as found.

## 2. Validation protocol (unchanged from Phases 5–6B)
- **Folds:** Phase 2 patient-grouped 5-fold × 3 repeats (`fold_vec`), identical for every model; paired comparisons on
  the same folds.
- **Within each outer fold:** the Phase 6A split (`split_indices`) into inner-train and inner-val.
  - Inner-val alone selects the hyper-parameters and the early-stopping epoch.
  - Nested inner OOF (4-fold, `existing6b.inner_oof` splits) is used for any decision rule, stacking or fusion weight.
- **Seeds:** 42, 43, 44 for full runs; pilots use seed 42.
- **Statistics:**
  - corrected repeated-CV 95 % CI (Nadeau–Bengio);
  - paired corrected t-test vs C3 (ORIG) on all 837 ECGs (accuracy, macro-F1, Mild F1);
  - Holm correction over all Phase 7 confirmatory comparisons.
  - A genuine improvement needs Holm p < 0.05, the same sign in all 3 repeats, and admissibility (every class recall
    ≥ 0.15).
- **Metrics reported:** accuracy, balanced accuracy, macro-F1, per-class precision / recall / F1, confusion matrices.
- **No Mild boosting:** decision rules may not multiply Mild probabilities or force Mild predictions. The C3 prior rule
  (protected objective) is the only decision rule on fused models, as before.
- **Verified continuous EF:** where EF supervision is used, the target is the nearest-echo EF of clean-link ECGs only.
  These are the NF set of `AMENDMENT_MMCLEAN.md`: exact name match, echo ≤ 30 days (true dates), folder label =
  nearest-echo class, no other-class link, EF spread ≤ 10. That is 366 of 837. Other ECGs contribute their class label
  only.

## 3. Experiments (ordered by expected value / cost; each pilot gates its full run)
**Pilot gate (fixed now).** A pilot (seed 42, 3 repeats, all 837 ECGs) advances to the full run only if, against its
same-recipe reference on the same folds:
- accuracy ≥ +0.010, **or** Mild F1 ≥ +0.03;
- **and** macro-F1 is not lower by more than 0.010.

The gates are screens, not claims.

- **E1, ECGFounder representations (no new training of the backbone).** Features come from the existing ECGFounder
  cache (aligned input; stage 4: 400 × 79, stage 5: 1024 × 40 positions over 10 s). Each candidate replaces f6 in the
  MLP fusion recipe [header ‖ representation]:
  - **E1-msp:** stage-5 mean ‖ max ‖ std over time (3,072).
  - **E1-col:** stage-5 mean per printed column. The four 2.5-s windows are 10 positions each; they are concatenated
    (4,096), so each lead group keeps its own summary.
  - **E1-s45:** stage-4 mean ‖ f6 (1,424): multi-scale.
  - **E1-dx:** ECGFounder's 150 pre-trained diagnostic outputs (sigmoid), one forward pass of the frozen network on
    the same aligned input. These are interpretable labels (conduction, infarct, hypertrophy patterns).
  - Efficient fine-tuning is **not** repeated: Phase 6 A1 / A2 failed their screen (r 0.650 vs 0.652).
- **E2, continuous-EF supervision and Mild separation** (heads on [header ‖ f6]; Phase 2 MLP architecture: hidden 64,
  dropout 0.5, AdamW, early stopping on inner-val S3, L2 from {1e-4, 1e-3, 1e-2}):
  - **E2-mt, multitask:** class-balanced CE + λ · Huber(EF / 10) from the hidden layer, on verified-EF rows only.
    λ ∈ {0.1, 0.5, 1.0} is chosen on inner-val.
  - **E2-ldl, boundary-aware label distribution:** verified-EF rows are trained against soft targets P(class | EF ~
    N(EF_echo, σ²)), integrated over the sex-specific class bands. Other rows use their hard label. σ ∈ {3, 5, 7} EF
    points is chosen on inner-val. This encodes measurement uncertainty: a 41.5 % Mild is partly Moderate.
  - **E2-scl, supervised contrastive (EF-weighted):** CE + β · SupCon on the L2-normalised hidden representation.
    - Positive weight between two ECGs: exp(−|ΔEF| / 5) when both have verified EF; otherwise 1 for the same class and
      0 for a different class.
    - Temperature 0.1; β ∈ {0.1, 0.3} chosen on inner-val.
    - This pulls ECGs with similar EF together, so Mild is learned as a position on the EF scale.
  - **E2-ce, control:** the same trainer with plain class-balanced CE. It separates implementation effects from loss
    effects. E2 gates are applied against both the reference member and E2-ce, and both must pass.
  - E1 also re-runs f6 through its own pipeline (**E1-f6**), as a reproduction check against the reference member.
- **E3, lightweight CNN on the digitised waveforms (GPU).** Raw digital ECGs are not available (`DATA_ACCESS.md`), so
  the waveforms digitised from the PDFs are used. No signal is invented: the "short" construction, each lead's own
  2.5 s.
  - **E3-cnn:** a multi-lead 1D ResNet.
    - Input: 12 × 625 (250 Hz) for the leads, plus a rhythm-II branch of 1 × 2,500.
    - Four residual blocks (32-64-128-128 channels, kernel 7) with global average pooling, then CE + 0.5 · Huber(EF)
      on verified rows.
    - ≤ 60 epochs, early stopping on inner-val S3; class-balanced CE; AdamW 1e-3.
  - **E3-cnn+hdr:** the same network with the header vector concatenated before the head.
  - **Pilot:** repeat 0 only (5 folds, seed 42) against the reference member on the same folds. It advances only if it
    passes the gate on repeat 0 and reaches accuracy ≥ 0.50.
- **E4, fusion (only if complementary).** For each candidate that passes its full run:
  - nested inner OOF is computed;
  - complementarity with the C1 vote is checked on the inner OOF: error correlation < 0.8 and ≥ 10 % disagreement;
  - complementary candidates join the C1 vote (equal weight), and the C3 prior rule is re-fitted nested.
  - Non-complementary candidates are not fused.
- **E5, data quality and data needs (report only).** A summary of the matching, label-reliability and borderline-EF
  analyses (6B audit, VER / NF / MMCLEAN), plus the concrete data required for phase 2. The original dataset stays
  immutable.

## 4. Final-model rule and deliverables
- **Best Phase 7 model:** the admissible confirmatory candidate with the highest mean four-class accuracy on all 837
  ECGs. It is compared with C3 (Holm). If none beats C3 genuinely, C3 remains the best overall model.
- **Deliverables** in `research_experiments/phase_7/`:
  - `PHASE7_EXPERIMENT_INVENTORY.md`;
  - `PHASE7_MODEL_COMPARISON.csv`;
  - confusion matrices, accuracy and macro-F1 graphs, Mild precision / recall / F1 graphs;
  - `PHASE7_CLAIMS_REGISTER.md`;
  - `PHASE7_FINAL_REPORT.md`, stating whether 70 % was reached and which data are needed if not.
- **Storage:** per-ECG predictions only in the git-ignored `phase_7/data/private/`; aggregates only in git.
- **Integrity guard:** `phase_6/integrity6.py` excludes additive work through `PHASE6_DIRS`, in the same way as for
  phase_6a / 6b. `research_experiments/phase_7` is added to that list, so Phase 7 files are not counted as changes.
  Phase 1–5 files and the original data stay fully checked.
- **Checks before every commit:**
  - `phase_7.tests.test_phase7` (pre-registration, write guard, no recurrent layers, no ECG identifiers in tracked
    files);
  - the Phase 6 / 6A / 5 suites and `integrity6 verify phase6_before`.

## Amendments
_None yet._
