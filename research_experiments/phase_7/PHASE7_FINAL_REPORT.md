# Phase 7 final report: ECG-based four-class LVEF classification with reliable Mild detection

**Data:** development set only (837 ECGs, 794 patient keys); the locked 210-ECG test cohort was not loaded or scored.
**Protocol:** Phase 2 patient-grouped 5-fold × 3 repeats, identical folds for every model; nested selection
(inner-val); corrected repeated-CV CIs; Holm correction.
**Pre-registration:** `PHASE7_PLAN.md` (commit 0d14775) and Amendment 1 (commit a1a7eef). Every number below is an
aggregate on all 837 ECGs with the original four-class labels; the evaluation population was never changed.
**Recurrent models** (LSTM / BiLSTM / GRU / CNN-LSTM / ECGFounder-LSTM) were discontinued. None was running, none was
launched, and a test blocks recurrent layers in Phase 7 code.

## 1. Answers
| Question | Answer |
|---|---|
| Best original four-class accuracy | **0.562 [0.519, 0.605]**, C3 (C1 vote + nested prior rule); unchanged by Phase 7 |
| Best overall model | **C3**: balanced accuracy 0.457, macro-F1 0.449 |
| Mild precision / recall / F1 (C3) | **0.300 / 0.178 / 0.224** |
| Best Mild F1 of any model | Header + clinical LR, 0.272 (P 0.305 / R 0.247; accuracy 0.503). Admissible ensemble: C1 vote, 0.257 (accuracy 0.548) |
| Statistical significance | No Phase 7 candidate beats its reference or C3: 44 paired comparisons, smallest Holm p 0.51 |
| 70 % reached? | **No.** None of the six owner targets is met by any model |
| Phase 7 candidate passing its gate | None. Under the pre-registered rule, no full run and no fusion followed |

Per-class performance of C3 (precision / recall / F1):

| Class | Precision | Recall | F1 |
|---|---|---|---|
| Normal | 0.710 | 0.846 | 0.772 |
| Mild | 0.300 | 0.178 | 0.224 |
| Moderate | 0.457 | 0.460 | 0.458 |
| Severe | 0.344 | 0.344 | 0.343 |

## 2. What was tried (pilots: seed 42, 3 repeats; gate = accuracy ≥ +0.010 or Mild F1 ≥ +0.03, macro-F1 ≥ −0.010)
| Exp. | Candidate | Accuracy | Macro-F1 | Mild P / R / F1 | Gate |
|---|---|---|---|---|---|
| ref | MLP fusion [header ‖ f6], seed 42 | 0.544 | 0.436 | 0.283 / 0.132 / 0.179 | — |
| E1 | f6 through the Phase 7 pipeline (reproduction) | 0.544 | 0.436 | identical | Δ 0.000 |
| E1 | stage-5 mean ‖ max ‖ std | 0.517 | 0.409 | 0.263 / 0.146 / 0.187 | fail |
| E1 | per-printed-column stage-5 mean | 0.527 | 0.431 | 0.271 / 0.157 / 0.199 | fail |
| E1 | stage-4 mean ‖ f6 (multi-scale) | 0.529 | 0.425 | 0.250 / 0.141 / 0.178 | fail |
| E1 | 150 pre-trained ECGFounder diagnostic outputs | 0.531 | 0.412 | 0.287 / 0.117 / 0.165 | fail |
| E2 | CE control (own trainer) | 0.541 | 0.432 | 0.280 / 0.112 / 0.159 | — |
| E2 | multitask CE + verified-EF regression | 0.545 | 0.440 | 0.289 / 0.142 / 0.191 | fail |
| E2 | boundary-aware soft labels (EF measurement uncertainty) | 0.548 | 0.433 | 0.303 / 0.123 / 0.174 | fail |
| E2 | EF-weighted supervised contrastive | 0.546 | 0.436 | 0.276 / 0.130 / 0.176 | fail |
| E3 | multi-lead 1D ResNet, digitised waveforms (repeat 0) | 0.496 | 0.384 | 0.264 / 0.178 / 0.213 | fail (ref 0.559 on repeat 0) |
| E3 | 1D ResNet + header (repeat 0) | 0.485 | 0.378 | 0.243 / 0.189 / 0.213 | fail |

**Not repeated, with reasons:**
- ECGFounder fine-tuning: Phase 6 A1 / A2 failed (r 0.650 vs 0.652).
- Image ResNet: 0.40–0.49 in Phases 1–3.
- Mild-probability boosting: excluded by the brief, and shown in Phase 6B to trade accuracy for Mild calls that are
  right only about 1 time in 4.

The full table, including existing models (C2 0.568 is not admissible: Mild recall 0.112), is in
`PHASE7_MODEL_COMPARISON.csv`. Graphs:
- `plots/phase7_accuracy_macro_f1.png`;
- `plots/phase7_mild_performance.png`;
- `plots/phase7_confusion_matrices.png`;
- `plots/phase7_learning_curve.png`.

## 3. Why Mild separation did not improve
- **Mild is spread across all four predictions.** In the C3 confusion matrix, true Mild goes to Normal 43 %, Mild 18 %,
  Moderate 28 % and Severe 11 %. Every Phase 7 representation reproduces the same pattern (38–47 % to Normal).
- **The ECG carries little Mild-specific information at this sample size and label quality.**
  - Pairwise AUROC: Normal-vs-Mild about 0.75; Mild-vs-Moderate about 0.6.
  - The ECG–echo EF error is about 13 points SD (11.8 on clean links), against a Mild band 12 points wide.
  - Richer representations, EF supervision, uncertainty-aware labels and contrastive pulling all redistribute the same
    weak signal. None adds information.
- **More data of the same kind does not fix Mild.** In the learning curve, Mild F1 is flat while accuracy rises
  slowly (next section).

## 4. Learning curve and what additional data are needed
The reference member and the ridge EF regressor were retrained on 25 / 50 / 75 / 100 % of each fold's training
patients (nested subsamples; inner-val and test folds unchanged):

| Training ECGs per fold | Accuracy | Macro-F1 | Mild F1 | EF SD, all ECGs | EF SD, verified-EF ECGs |
|---|---|---|---|---|---|
| 134 | 0.515 | 0.415 | 0.204 | 14.39 | 13.39 |
| 267 | 0.527 | 0.421 | 0.151 | 13.92 | 12.58 |
| 401 | 0.537 | 0.439 | 0.188 | 13.64 | 12.27 |
| 536 | 0.544 | 0.436 | 0.179 | 13.55 | 12.08 |

**Per doubling of training data:** accuracy +1.4 points, macro-F1 +1.2, Mild F1 −0.9 (no trend). EF SD falls by 0.44
on all ECGs and by 0.66 on verified-EF ECGs.

**Indicative extrapolation** (log-linear from 4 points over 2 doublings; uncertain far out):

| Scenario | Accuracy | EF SD |
|---|---|---|
| ≈ 100,000 training ECGs, current label quality | ≈ 0.65 | ≈ 10.3 |
| ≈ 100,000 training ECGs, verified-quality EF targets | — | ≈ 7.1, below the ≈ 9 needed for 70 % |
| 70 % at current label quality | needs ≈ 1 million ECGs | — |
| EF SD 9 with verified-quality labels | — | needs ≈ 13,000 training ECGs |

**Conclusion: label quality matters more than quantity.** The phase-2 dataset should provide, in order of expected
value:
1. **Verified ECG–echo linkage.** Link by patient identifier (not names), with the echo within a few days (ideally the
   same admission) of the ECG. In this dataset only 366 / 837 ECGs (44 %) have a clean link, and echoes > 30 days away
   correlate with the ECG at r 0.42 vs 0.75 for clean links.
2. **A better EF reference.** Biplane Simpson EF (guideline method), not Teichholz M-mode. Repeat measurements or
   core-lab reads near the 41 % and 50–54 % boundaries, where the Mild band lies.
3. **Raw digital ECGs** (e.g. vendor XML / DICOM-ECG at 500 Hz, all 12 leads simultaneous for 10 s) instead of
   PDF-digitised traces. Today each lead has only 2.5 s, not simultaneous with the others. ECGFounder was pre-trained
   on simultaneous 10-s ECGs.
4. **Volume:** about 13,000+ ECGs with verified labels, with Mild deliberately well represented. With 100,000 patients
   of verified quality, the learning curve indicates that the EF error could fall below the ≈ 9 points needed for
   70 %. That is a hypothesis to test, not a result.
5. **Keep the protocol:** a pre-registered primary task on the original four classes, patient-grouped splits, and the
   locked test cohort scored once at the end.

## 5. Data quality (summary of Phases 4–6B; the original dataset unchanged)
- **Matching:**
  - every development ECG has ≥ 1 linked echo;
  - the folder label agrees with the nearest echo for 93.4 % (55 inconsistent);
  - 85 ECGs have a linked echo of another class;
  - 77 have an EF spread > 10 points across linked echoes;
  - 391 have a name-match score < 100; 192 have the nearest echo > 30 days away.
- **Borderline EF:** 221 ECGs lie within 3 points of a class boundary.
- **Label cleaning, relabelling and noise-free training** (VER, BRES, ECHO, CLEAR, MMC / MMNC / MN240 / MN200,
  NF / NFS) change the C3 recipe's accuracy on all 837 ECGs by −0.021 to +0.005. Label noise is real but explains
  little of the gap.
- **Raw digital ECG:** no access is recorded (`DATA_ACCESS.md`). All waveform work uses traces digitised from the
  printed PDFs.

## 6. Reproducibility
- **Code:**
  - `p7common.py`;
  - `e1_ecgf.py` (`--pilot`), `e2_heads.py` (`--pilot`), `e3_cnn.py` (`--pilot`);
  - `e5_curve.py`;
  - `report7.py` (CSV, statistics, plots, inventory).
- **Inputs:** the existing ECGFounder cache (Phase 6), ECGFounder f6 (Phase 2), header and clinical features, and
  `data/signals.npz` (development rows only).
- **Outputs:**
  - aggregates in `results/` and `results/experiments/`;
  - per-ECG predictions in the git-ignored `data/private/`.
- **Pre-commit checks:** `tests/test_phase7.py` (pre-registration present, write guard, no recurrent layers, no ECG or
  patient identifier in tracked files), plus the Phase 6 / 6A / 5 suites and `integrity6 verify phase6_before`.
- **Integrity guard:** `phase_7` was added to the additive-work exclusions in `phase_6/integrity6.py`, as was done
  for 6a / 6b. Phase 1–5 files and the data stay fully checked.
