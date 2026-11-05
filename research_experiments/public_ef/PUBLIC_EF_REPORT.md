# Public-EF study: EchoNext → hospital four-class LVEF, image-derived inputs only (overnight report)

**Date:** 2026-11-06, about 05:45. All numbers are aggregates.
- **Hospital data:** the 837 development ECGs, Phase-2 folds 5 × 3, original four-class labels. The locked 210-ECG
  test set was not used.
- **EchoNext data:** the test file only (owner). Its 20 % locked-test patients were never used for any choice.
- **Inputs:** image-derived only, i.e. traces or pixels from the page. No header, machine measurements, age or sex.
- **Pre-registration:** every configuration was declared and pushed before it ran (PUBLIC_EF_PLAN Amendments 1–9,
  IMG_PLAN Amendments 1–3).

## Answer
**70 % four-class accuracy on all patients was not reached.**

**The best honest number** is the pre-declared headline ensemble SUPER, at **0.583** [0.540, 0.626].

The best of all the variants run, chosen after the fact, is SUPER-TOP10 with the ordinal meta-learner: **0.587**
[0.565, 0.610], macro-F1 0.467. It keeps 10 members per fold.

**Against the references:**
- C3, which uses the header: 0.562.
- stack-BASE: 0.574.
- The ensembles beat both by 1–2.5 points, but no difference is statistically significant (Holm p = 1.0 throughout).

**Why the ceiling is near 0.59:**
- **EF error.** Simulation says that grading by an ECG-based EF estimate whose error matches ours (RMSE 13.3, r 0.66)
  gives about 0.59. Reaching 0.70 needs an error SD of 7.7 or less, which means r ≈ 0.90 with the echo. Published
  AI-ECG EF models report r ≈ 0.6–0.7.
- **Labels.** 55 of 837 labels (6.6 %) do not match their own recorded echo EF band. Even a perfect EF predictor would
  stop at 0.934.

## 1. Every variant run, on the 837 hospital ECGs (four classes, original labels)
| Model | Pre-registered | Accuracy [corrected 95 % CI] | Recall N / Mi / Mo / S | Precision N / Mi / Mo / S | Macro-F1 | Admissible |
|---|---|---|---|---|---|---|
| stack-BASE (ENH members) | plan | 0.574 [0.541, 0.607] | 0.91 / 0.17 / 0.42 / 0.28 | 0.70 / 0.27 / 0.47 / 0.39 | 0.442 | yes |
| stack-PLUS (+ EN-cls, EN-band) | plan | 0.566 [0.531, 0.600] | 0.89 / 0.19 / 0.41 / 0.27 | | | yes |
| stack-ALL (+ all-feature EchoNext members, H) | A1 | 0.572 [0.534, 0.609] | 0.90 / 0.17 / 0.43 / 0.27 | | | yes |
| stack-ALL-plain | A1 | 0.570 [0.535, 0.606] | 0.91 / 0.16 / 0.41 / 0.28 | | | yes |
| EN zero-shot (no hospital training) | plan | 0.530 [0.486, 0.575] | 0.84 / 0.18 / 0.35 / 0.26 | | | yes |
| EN-all zero-shot (no hospital training) | A1 | 0.551 [0.504, 0.597] | 0.86 / 0.17 / 0.37 / 0.38 | | | yes |
| **SUPER** (greedy pruning; headline of A4) | A4 | **0.583 [0.540, 0.626]** | 0.90 / 0.15 / 0.50 / 0.27 | 0.72 / 0.30 / 0.45 / 0.42 | 0.451 | yes |
| STAGED-4 (staged decision on SUPER members) | A5 | 0.581 [0.535, 0.627] | 0.89 / 0.17 / 0.50 / 0.25 | 0.72 / 0.31 / 0.45 / 0.41 | 0.450 | yes |
| SUPER-ORD (ordinal meta-learner) | A7 | 0.578 [0.537, 0.620] | 0.89 / 0.17 / 0.49 / 0.28 | 0.72 / 0.29 / 0.44 / 0.41 | 0.450 | yes |
| SUPER-TOP10 | A9 | 0.577 [0.556, 0.598] | 0.88 / 0.15 / 0.52 / 0.26 | 0.71 / 0.29 / 0.45 / 0.41 | 0.447 | yes |
| SUPER-TOP10 STAGED-4 | A5 + A9 | 0.581 [0.555, 0.607] | 0.89 / 0.16 / 0.50 / 0.29 | 0.72 / 0.28 / 0.46 / 0.42 | 0.454 | yes |
| **SUPER-TOP10 ORD** (best after the fact) | A7 + A9 | **0.587 [0.565, 0.610]** | 0.89 / 0.17 / **0.52** / **0.31** | 0.71 / 0.31 / 0.47 / 0.44 | **0.467** | yes |
| SUPER-X (+ FEATX, Mild / Moderate members) | A8 | 0.575 [0.540, 0.611] | 0.89 / 0.16 / 0.48 / 0.25 | 0.71 / 0.30 / 0.44 / 0.39 | 0.442 | yes |
| SUPER-X STAGED-4 | A5 + A8 | 0.575 [0.541, 0.610] | 0.88 / 0.17 / 0.50 / 0.25 | | 0.444 | yes |
| SUPER-X ORD | A7 + A8 | 0.577 [0.545, 0.610] | 0.90 / 0.18 / 0.46 / 0.27 | | 0.448 | yes |
| SUPER-X-TOP10 | A8 + A9 | 0.579 [0.556, 0.603] | 0.91 / 0.15 / 0.47 / 0.26 | | 0.443 | **no** (Mild < 0.15) |
| SUPER-X-TOP10 STAGED-4 | A5 + A8 + A9 | 0.585 [0.557, 0.613] | 0.89 / 0.18 / 0.51 / 0.26 | | 0.455 | yes |
| SUPER-X-TOP10 ORD | A7 + A8 + A9 | 0.576 [0.552, 0.600] | 0.89 / 0.16 / 0.49 / 0.27 | | 0.449 | yes |
| *C3 (with header; reference)* | | 0.562 [0.519, 0.605] | 0.85 / 0.18 / 0.46 / 0.34 | 0.71 / 0.30 / 0.46 / 0.34 | 0.449 | yes |

**Pixel models** (IMG_REPORT.md; IMG70 in progress):

| Pixel model | Accuracy [95 % CI] | Recall N / Mi / Mo / S | Macro-F1 |
|---|---|---|---|
| ResNet-18 FT-EN | 0.530 [0.481, 0.578] | 0.82 / 0.23 / 0.34 / 0.29 | 0.416 |
| ResNet-18 ZS | 0.524 | 0.86 / 0.18 / 0.27 / 0.32 | 0.403 |
| ResNet-18 FT-IN | 0.492 | 0.75 / 0.20 / 0.36 / 0.24 | 0.386 |
| IMG70 E2, ConvNeXt-T 384 × 768 | 0.535 [0.484, 0.586] | 0.80 / 0.22 / 0.40 / 0.31 | 0.433 |

**Tests** (corrected repeated-CV t, Holm within each run):
- SUPER vs stack-BASE +0.009; SUPER vs C3 +0.021;
- SUPER-TOP10-ORD vs SUPER-TOP10 +0.010;
- all Holm p = 1.0. No genuine improvement by the pre-registered standard.

## 2. What each idea contributed
- **EchoNext transfer:** genuine signal on the hospital ECGs, but no gain once combined.
  - The all-feature EchoNext model (EN-all) gives EF ≤ 45 AUROC 0.85, Normal-vs-Mild 0.77, Mild-vs-Moderate 0.62 and
    EF r 0.62.
  - With no hospital training it reaches 0.551.
  - In the pruned ensembles, EchoNext members are kept in most folds: EN-gbm-X 14 / 15, EN-gbm-H 13 / 15. The pixel
    EF band PIX-ZS-band is kept in 9 / 15 folds of SUPER.
- **Pruning to 8–10 members (TOP10):** halves the width of the CI (±0.02 instead of ±0.04), so the ensembles are more
  stable. It is not clearly more accurate.
- **Staging** (Normal / {Mild, Moderate} / Severe, then Mild vs Moderate): 0.575–0.585. No better than the flat stack.
- **Ordinal meta-learner:** the best single number (0.587) and the best macro-F1 (0.467), within noise of the others.
- **FEATX:** 73 new features (QRS / T loop shape, TCRT, TMD, dispersion, beat-to-beat, per-lead QRS shape) and 7
  Mild / Moderate-targeted members.
  - Selected often, and EchoNext's own held-out Mild-vs-Moderate AUROC is 0.688.
  - No gain in hospital accuracy or Mild recall, which stays at 0.15–0.18.
- **Moderate recall** improved most: 0.42 (stack-BASE) → 0.50–0.52. **Mild** stays at the 0.15–0.18 floor.

## 3. EchoNext side (aggregates)
**Data:** the test file, 4,827 ECGs with LVEF. Normal 3,518, Mild 547, Moderate 298, Severe 464.
- **Recovered millivolts:** EchoNext and hospital ECGs agree. Median QRS 82 / 82 ms; QTc 415 / 416 ms; R in V5
  0.94 / 0.93 mV.
- **Waveform / metadata alignment:** heart rate measured on each waveform vs the metadata column, r = 0.96, median
  difference 0.4 bpm.

**EchoNext locked test (974 patients, never used for any choice):**

| Model | Accuracy | EF r |
|---|---|---|
| All-feature trace model | 0.648 | 0.629 |
| Pixel ResNet-18 | 0.582 | 0.587 |
| Pixel ConvNeXt-T | 0.640 | 0.606 |

## 4. Sensitivity and pending items
**Echo-consistent labels (Amendment 4 B / 6):** the 55 mismatching labels are re-derived from their own echo EF. The
retraining was still running at the time of this report; results follow in `results/en_super_echo.json`. Simulation
expectation: about +3–4 points against the corrected labels.

**IMG70:**

| Step | Status |
|---|---|
| E1 (train split) | deferred by the owner |
| E2 (ConvNeXt-T chosen over EfficientNet-B2 on the EchoNext hold-out) | done: 0.535 |
| E3 (EF-led pre-training + EF grading) | running |
| E4 (recipe, 3-seed ensemble), final model, E5, E6 | follow |

They will be reported in `IMG70_REPORT.md`.

**Problems found and fixed overnight (code only; no result changed):**
- The laptop ran out of commit memory several times. Fix: parallelism reduced, and per-fold cleanup added to `img70`.
- A bug in `en_featx` re-decompressed the signal file once per ECG.
- A missing-value bug: the G5 block was not imputed.
- A hung first FT-IN attempt was holding the GPU.

## 5. For Phase 2 (what would move the number)
The analysis points to **data, not modelling**:
- **EF information:** to reach 70 %, the ECG-based EF estimate needs r ≈ 0.9 with the echo.
- **The biggest levers:**
  - raw 10-s 12-lead digital ECGs instead of printouts;
  - hundreds of thousands of ECG–echo pairs (the full EchoNext train split plus the ~100,000-patient Phase 2 data);
  - tighter ECG–echo pairing;
  - a biplane-Simpson EF reference instead of Teichholz;
  - fixing the 6.6 % label / EF mismatches at source.
