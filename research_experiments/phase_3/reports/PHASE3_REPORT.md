# Phase 3 report: pushing 4-class LVEF prediction toward 60–65 % accuracy

## Basis of all numbers

- **Dataset:** unchanged. 1,047 ECGs / 993 patients; development set D = 837 ECGs / 794 patients;
  locked test T = 210 ECGs / 199 patients; 0 patients shared.
- **Locked test: NOT scored.** No phase-3 decision used it.
- **Protocol:** the phase-2 patient-grouped 5-fold CV on D (`phase_2/configs/folds_D.csv`), with
  3 CV repeats × 3 seeds for the leading models. Models are fitted on inner-train and selected or
  early-stopped on inner-val.
- **Calibration:** decision thresholds are **fully nested**.
- **Statistics:** corrected repeated-CV t-test (Bouckaert & Frank), Holm within each metric family.
- **Labels in this report:** every statement is tagged **[MEASURED]**, **[HYPOTHESIS]** or
  **[ESTIMATE]**.
- **S3** = (accuracy + balanced accuracy + macro-F1) / 3, the pre-registered selection score (§9).

---

## 0. Final decision report

| # | Question | Answer |
|---|---|---|
| 1 | Did we cross 60 % accuracy? | **No.** [MEASURED] The best robust (3-repeat) development-CV accuracy is **0.568 ± 0.007**: header + ECGFounder MLP fusion with the nested, protected prior calibration. The best uncalibrated accuracy is **0.547 ± 0.002** (vote ensemble). VOTE_CAL_PLACEHOLDER |
| 2 | Best validated accuracy? | 0.568 (calibrated MLP fusion; Mild F1 0.159) or 0.547 (vote ensemble, argmax; Mild F1 0.255). Both are development CV; the locked test is not yet scored. |
| 3 | Did balanced accuracy improve? | Marginally. [MEASURED] Best robust value **0.467** (vote ensemble) vs 0.442 (frozen ECGFounder, linear), 0.414 (header numeric) and BAL_R_PLACEHOLDER (header + ResNet-18). |
| 4 | Did macro-F1 improve? | Marginally. [MEASURED] Best **0.455 ± 0.003** (vote) vs 0.434–0.444 (ECGFounder alone) and MF1_R_PLACEHOLDER (header + ResNet-18). |
| 5 | Did Mild F1 improve? | Not robustly beyond phase 2. [MEASURED] Best robust Mild F1 is **0.266–0.269**: ECGFounder linear with distance penalty, or header + clinical LR. Vote ensemble 0.255. **Mild one-vs-rest AUROC stays 0.53–0.55 in every model.** |
| 6 | Did QWK improve? | [MEASURED] Best robust QWK **0.583–0.591** (MLP fusion, uncalibrated / calibrated) vs 0.548 (ECGFounder linear). The target of 0.55 is reached. |
| 7 | Best overall model? | **Vote ensemble:** equal-weight mean of 2-way MLP fusion (3 seeds) + ECGFounder linear with distance penalty + header-numeric + clinical LR. Highest and most stable S3 (0.490 ± 0.002), best macro-F1, Mild F1 0.255, no fitted combination weights. |
| 8 | Best model for publication? | The same vote ensemble, reported **with and without** the nested protected calibration, next to the simplest strong single model, ECGFounder frozen + linear head (S3 0.474, 4,100 trainable parameters). See §8. |
| 9 | Failed experiments and why | Gradient boosting, attention fusion, vector gates, ordinal heads, fine-tuning, stacking, and per-class threshold calibration (collapses Mild). Details in §7. |
| 10 | Next most promising direction | **Label quality and quantity**, not model architecture. All 20+ model families converge to ≈ 0.53–0.55 accuracy and Mild AUROC ≈ 0.55. See §10. |

### Best model per objective (robust development CV; mean over runs)

| Objective | Model | Value | Other metrics |
|---|---|---|---|
| Accuracy | MLP fusion + nested protected prior calibration | **0.568** | balanced 0.463, macro-F1 0.442, Mild F1 0.159 |
| Balanced accuracy | Vote ensemble | **0.467** | accuracy 0.547 |
| Macro-F1 | Vote ensemble | **0.455** | accuracy 0.547 |
| Mild discrimination | ECGFounder linear + distance penalty (Mild F1 0.266, recall 0.247) / header + clinical LR (0.269) | Mild-vs-Normal AUROC best for MLP fusion (0.766) | |
| Overall | **Vote ensemble** | S3 **0.490 ± 0.002** | |

### Ranked top 5 (robust development CV, by S3; seeds × repeats)

| Rank | Model | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|---|
| 1 | Vote ensemble (MLP fusion + ECGFounder linear + header + clinical LR) | 0.547 ± 0.002 | 0.467 ± 0.003 | 0.455 ± 0.003 | 0.255 ± 0.023 | 0.574 | **0.490 ± 0.002** |
| 2 | Weighted-average ensemble (same members, cross-fitted weights) | 0.544 ± 0.003 | 0.462 | 0.447 | 0.235 | 0.579 | 0.485 ± 0.005 |
| 3 | 3-way MLP fusion (header + clinical + ECGFounder), single seed | 0.541 ± 0.011 | 0.466 ± 0.013 | 0.442 | 0.202 | 0.572 | 0.483 |
| 4 | 2-way MLP fusion (header + ECGFounder), 3-seed ensemble | 0.544 ± 0.008 | 0.466 ± 0.018 | 0.440 | 0.186 | 0.583 | 0.483 ± 0.015 |
| 5 | ECGFounder frozen, MLP head | 0.534 ± 0.011 | 0.453 | 0.440 | 0.241 | 0.549 | 0.476 |

Ranks 2–5 are **statistically indistinguishable** from rank 1 under the corrected repeated-CV t-test.
Rank 1 wins on stability (SD 0.002) and macro-F1, not by a significant margin.

---

## 1. The three issues raised before Phase 3

### 1.1 ECGFounder waveform masking

**Coverage** [MEASURED] (`results/masking_audit_features.json`):
- Each printed lead covers 2.5 s of the 10-s acquisition.
- Lead II covers 10 s.
- I, III, aVR, aVL, aVF get a further 2.5 s each by exact Einthoven/Goldberger identities (r ≈ 0.99).
- 70 of 120 lead-seconds were never printed and are masked, not invented.
- The observed fraction is 0.403–0.417 per ECG.

**Validity** (phase-2 audit) [MEASURED]:
- calibration exact;
- lead identity, polarity and R-wave progression physiological;
- heart rate and QRS axis match the printed machine values;
- column timing verified (r = 0.997).

**Is masking limiting performance?** Each construction removes real signal step by step; nothing is
added. Frozen ECGFounder, same linear head, development CV, repeat 0 [MEASURED]:

| Input (all real data) | Observed | Label-free agreement* | Accuracy | Balanced | Macro-F1 | S3 |
|---|---|---|---|---|---|---|
| aligned (current) | 41.7 % | 0.887 | 0.532 | 0.441 | 0.437 | 0.470 |
| **each lead only in its own 2.5-s window** (no derived segments, II only 2.5 s) | 25.0 % | 0.881 | 0.536 | 0.451 | 0.438 | **0.475** |
| measured samples only (no derived segments) | 31.2 % | 0.885 | 0.528 | 0.445 | 0.432 | 0.468 |
| limb leads only | 29.2 % | 0.870 | 0.470 | 0.401 | 0.391 | 0.421 |
| precordial + II | 20.8 % | 0.873 | 0.492 | 0.402 | 0.392 | 0.429 |
| precordial only | 12.5 % | 0.800 | 0.503 | 0.415 | 0.400 | 0.440 |
| lead II only (10 s) | 8.3 % | 0.858 | 0.459 | 0.391 | 0.371 | 0.407 |
| lead II, 2.5 s | 2.1 % | 0.772 | 0.459 | 0.390 | 0.360 | 0.403 |

\* Median AUROC of ECGFounder's pre-trained statement outputs against the statements the ECG machine
printed on the same PDF. This uses no EF labels.

Valid alternatives without fabrication [MEASURED]:

| Alternative | S3 |
|---|---|
| each 2.5-s window fed separately with only its recorded leads, embeddings averaged | 0.445 |
| same, embeddings concatenated | 0.466 |
| multiview: aligned 12-lead + 1-lead model on the true 10-s lead II | 0.453 |
| multiview + windows | 0.461 |

None beats the aligned input.

**Conclusions:**
- [MEASURED] **Lead coverage matters; observed time does not.** Dropping 17 % of the observed time
  (aligned → each lead only in its window) changes S3 by +0.005. Removing whole lead groups costs
  0.03–0.07. So **the 58 % time masking is not the limiting factor** for this task.
- [MEASURED] A naive linear fit across all constructions gives +0.017 S3 per 10 % observed. That
  slope reflects lead diversity, not time, and **must not be extrapolated** to full 10-s 12-lead
  recordings.
- [HYPOTHESIS] Fully simultaneous 10-s 12-lead signals (a different export format) might add
  beat-to-beat information (e.g. multiple QRS complexes per lead). These printouts cannot test it,
  and the window experiment suggests the effect would be small.

### 1.2 Dataset accounting [MEASURED] (`results/dataset_accounting.json`)

- **Inclusion:** all **1,047** PDFs in the four label folders are in the dataset. None was excluded,
  and the manifest has no rows without a PDF.
- **Not used:** the 4 spreadsheets are present but not opened.
- **Duplicates:** no byte-identical PDFs, no duplicate acquisition timestamps. 5 files carry a "(1)"
  copy suffix but are different recordings.
- **Per year:** Normal = 120 in every year (a property of the folder construction).
  - 2020: Mild 48, Moderate 36, Severe 28
  - 2021: Mild 74, Moderate 69, Severe 31
  - 2022: Mild 42, Moderate 43, Severe 21
  - 2024: Mild 68, Moderate 73, Severe 34
- **Class totals:** Normal 480 / Mild 232 / Moderate 221 / Severe 114.
- **Patients:** 993 in total. 47 patients have several ECGs (101 ECGs); 39 span years, and 31 changed
  EF class between ECGs. All ECGs of a patient are in one split.
- **The 669 figure** is the original **training split** (635 patients), never the total.
  - Original split: 669 / 168 / 210 ECGs (635 / 159 / 199 patients).
  - Phase 2/3 development set = train + validation = 837 ECGs (794 patients).
  - Locked test = 210 ECGs (199 patients); **0** patients overlap.

### 1.3 Statistical reporting

The phase-1 claim "header + ECGFounder vs header, p < 0.001" is **retracted**: the corrected
resampled t-test gives p = 0.12.
- `reports/ERRATUM_phase1.md` documents this. The phase-1 report is kept unchanged as the historical
  record.
- `phase_3/reports/CLAIMS_REGISTER.md` tracks every claim as VALID / CORRECTED / RETRACTED /
  HYPOTHESIS.
- The retracted value is not used anywhere in this report.

---

## 2. Experiment prioritisation (decided before running Phase 3)

| Rank | Experiment | Expected gain [ESTIMATE before running] | Outcome [MEASURED] |
|---|---|---|---|
| 1 | Nested, constrained decision calibration | +1–3 accuracy points | **+2.5** (protected prior); only rule without balanced-accuracy loss |
| 2 | Clinical ECG features + boosting | +0.5–2 | LR: +1.2 accuracy on header-only (n.s.); boosting worse; no gain inside the fusion |
| 3 | Gated / attention fusion | 0–1.5 | no gain (S3 0.458–0.482 vs 0.481) |
| 4 | Complementary ensembles / stacking | +0.5–2 | vote: +0.5 accuracy, +0.015 macro-F1, +0.07 Mild F1 vs MLP fusion, n.s. but most stable; stacking unstable |
| 5 | Supervised contrastive / self-supervised adaptation | 0–1 | **not run**: ranks 1–4 showed every representation plateaus at the same ceiling; fine-tuning (phases 1–2) gave nothing |

---

## 3. Direction 1: clinical ECG feature engineering

**Features** (`clinical_features.py`; label-free, deterministic, per ECG):
- frontal QRS-T angle, JT, JTc, QRS ≥ 120 ms (from the printed measurements);
- from the digitised waveforms: limb and precordial QRS voltage (max, sum, low-voltage flags),
  Sokolow–Lyon, Cornell, log R/S V1–V6, transition lead, RR variability (SD, RMSSD, CV, max/min, beat
  count), T-wave amplitude in II/V5/V6, lateral T inversion.

28 features in total. Plausibility check: mean QRS-T angle 64°, QRS ≥ 120 ms in 13 %, lateral T
inversion in 6.6 %.

Robust results (3 repeats; corrected t-test on 15 matched cells) [MEASURED]:

| Model | Accuracy | Balanced | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|
| Header numeric LR | 0.489 ± 0.009 | 0.414 | 0.393 | 0.198 | 0.493 | 0.432 |
| **Header numeric + clinical LR** | 0.501 ± 0.021 | 0.440 | 0.424 | 0.269 | 0.504 | 0.455 |
| Clinical only LR (repeat 0) | 0.480 | 0.413 | 0.389 | 0.207 | 0.481 | 0.427 |
| Header + clinical, boosting (repeat 0) | 0.490 | 0.416 | 0.402 | 0.205 | 0.500 | 0.436 |
| Header + clinical + 264 engineered features, boosting (repeat 0) | 0.499 | 0.390 | 0.386 | 0.232 | 0.480 | 0.425 |

- Header + clinical vs header: Δ accuracy +0.012 (p = 0.58), Δ balanced +0.026 (p = 0.28),
  Δ macro-F1 +0.029 (p = 0.24), Δ Mild F1 +0.068 (p = 0.21).
  **Consistent direction, not statistically significant.**
- Boosting underperforms LR at n ≈ 535 per fold. **Negative result.**

### The 3-way vs 2-way question (identical folds, seeds 42/43/44 × repeats 0/1/2, same MLP configuration and block scaling) [MEASURED]

| | Accuracy | Balanced | Macro-F1 | Mild F1 | Mild recall | QWK | S3 |
|---|---|---|---|---|---|---|---|
| 2-way MLP: header + ECGFounder (9 runs) | 0.542 ± 0.010 | 0.465 ± 0.018 | 0.436 ± 0.019 | 0.176 ± 0.043 | 0.131 | 0.579 | 0.481 |
| 3-way MLP: + clinical (9 runs) | 0.541 ± 0.011 | 0.466 ± 0.013 | 0.442 ± 0.017 | 0.202 ± 0.031 | 0.159 | 0.572 | 0.483 |
| **Δ (3-way − 2-way), 15 matched cells** | −0.002 (p = 0.90) | +0.001 (p = 0.98) | +0.006 (p = 0.74) | +0.028 (p = 0.42) | +0.028 (p = 0.35) | −0.007 (p = 0.73) | +0.002 (p = 0.92) |

**Answer:** adding clinical features to header + ECGFounder provides **no statistically meaningful
incremental information** [MEASURED].
- They help a header-only model (above, n.s.), which suggests ECGFounder already encodes most of what
  the clinical features measure.
- [HYPOTHESIS] Supported by phase-2 `complementarity.py` (decodability of printed measurements from
  the ECGFounder embedding): COMPLEMENTARITY_PLACEHOLDER

---

## 4. Direction 2: gated and attention multimodal fusion

**Architecture** (`exp_p3.py`, ≈ 35–70 k trainable parameters on top of the frozen 30.8 M ECGFounder):

```
header numeric (16) ─► dropout ─► Linear(16→h) ─► GELU ─┐
clinical (28)       ─► dropout ─► Linear(28→h) ─► GELU ─┼─► gate g_m = σ(W[z_1..z_M]) ─► Σ g_m·z_m ─► dropout ─► Linear(h→4)
ECGFounder f6 (1024)─► dropout ─► Linear(1024→h) ─► GELU┘   (attention variant: softmax(q·K z_m / √h) over modality tokens)
```

Training: AdamW, class-weighted CE (± distance penalty), weight decay from {0.01, 0.1} on inner-val,
early stopping (patience 25).

Results, repeat 0 (single seed), plus 3 seeds × 3 repeats for the best variant [MEASURED]:

| Variant | Accuracy | Balanced | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|
| scalar gate, header + ECGFounder | 0.526 | 0.458 | 0.434 | 0.211 | 0.503 | 0.473 |
| vector gate, header + ECGFounder | 0.515 | 0.435 | 0.424 | 0.215 | 0.519 | 0.458 |
| scalar gate, 3 modalities | 0.533 | 0.449 | 0.437 | 0.250 | 0.509 | 0.473 |
| attention, 3 modalities | 0.526 | 0.440 | 0.429 | 0.218 | 0.522 | 0.465 |
| scalar gate, 3 modalities, h = 64 | 0.527 | 0.446 | 0.433 | 0.199 | 0.538 | 0.469 |
| **scalar gate, 3 modalities + distance penalty** (9 runs) | 0.530 ± 0.010 | 0.452 ± 0.012 | 0.438 ± 0.013 | 0.226 ± 0.028 | 0.546 | 0.473 |

Gated (best) vs 2-way MLP: Δ accuracy −0.013 (p = 0.39), Δ Mild F1 +0.055 (p = 0.27), Δ QWK −0.033
(p = 0.055). **Gating and attention do not beat plain concatenation + MLP** [MEASURED]. With three
modality tokens and ≈ 535 training ECGs, the extra flexibility is not usable.

---

## 5. Direction 3: complementary ensembles and robust classification

### 5.1 Ensembles (`p3_ensembles.py`; members seed-averaged; cross-fitted on each repeat's own folds) [MEASURED]

| Ensemble (3 repeats) | Accuracy | Balanced | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|
| **Vote: MLP fusion + ECGFounder linear (dist.) + header + clinical LR** | 0.547 ± 0.002 | 0.467 ± 0.003 | **0.455 ± 0.003** | 0.255 ± 0.023 | 0.574 | **0.490 ± 0.002** |
| Weighted average, same members | 0.544 ± 0.003 | 0.462 | 0.447 | 0.235 | 0.579 | 0.485 |
| Stacked LR, same members | 0.547 ± 0.019 | 0.473 | 0.442 | 0.196 ± 0.058 | 0.585 | 0.487 ± 0.011 |

ENSEMBLE_EXTRA_PLACEHOLDER

- The equal-weight vote is the most **stable** model of the whole study (SD 0.002 across repeats).
- Cross-fitted weighting and stacking add variance without gain, as expected when weights are fitted
  on about 670 ECGs.
- Vote vs 2-way MLP fusion (3-seed ensemble): +0.003 accuracy, +0.015 macro-F1, +0.07 Mild F1;
  none significant.

### 5.2 Ordinal / imbalance losses (phase 2, not repeated) [MEASURED]

- CORAL and cumulative link lower accuracy in every family. They give the best QWK (0.567), but
  lower S3.
- The distance-penalised CE (λ = 1) is the only "ordinal-aware" loss that helps: ECGFounder linear,
  S3 0.474 vs 0.468.
- Focal, effective-number and balanced sampling are all within ±0.005 S3 of class-weighted CE.
- Unweighted CE raises accuracy (0.552) but collapses Mild (F1 0.11–0.23). Rejected.

### 5.3 Decision calibration (fully nested; `phase_2/nested_calibration.py`, `nested_calib_vote.py`)

How it works:
- Priors come from the training portion of each outer fold.
- The exponent and thresholds are chosen on genuinely out-of-fold predictions from an inner 4-fold
  CV inside that portion.
- Calibrated values are applied to the outer fold; no model involved ever saw it.

[MEASURED] Mean over 3 repeats; Δ vs the same model's argmax:

| Model | Rule | Accuracy | Balanced | Macro-F1 | Mild F1 | Mild recall | QWK | Predicted Normal (true 45.9 %) |
|---|---|---|---|---|---|---|---|---|
| MLP fusion (3-seed ens.) | original | 0.544 | 0.466 | 0.440 | 0.186 | 0.141 | 0.583 | 50.3 % |
| | **prior, protected** | **0.568 (+0.025)** | 0.463 (−0.003) | 0.442 (+0.002) | 0.159 (−0.027) | 0.112 (−0.029) | 0.591 (+0.008) | 55.8 % |
| | prior, max accuracy | 0.572 (+0.029) | 0.443 (−0.023) | 0.423 (−0.016) | 0.140 (−0.046) | 0.092 | 0.571 | 60.4 % |
| | per-class thresholds, max accuracy | 0.571 (+0.028) | 0.443 | 0.399 (−0.041) | **0.031 (−0.155)** | 0.016 | 0.577 | 61.1 % |
| ECGFounder linear | prior, protected | 0.544 (+0.016) | 0.434 (−0.008) | 0.427 (−0.006) | 0.207 (−0.037) | 0.168 | 0.546 | 54.7 % |
| Header numeric LR | prior, protected | 0.498 (+0.009) | 0.408 (−0.006) | 0.390 (−0.004) | 0.184 (−0.014) | 0.142 | 0.480 | 54.2 % |

VOTE_CAL_TABLE_PLACEHOLDER

Confusion matrices before and after calibration, summed over 3 × 837 OOF predictions, are in
`results/calibration_nested_tables.md`. MLP fusion, original vs prior-protected:

```
            original                     prior, protected
true N   922  125   54   51          997   73   57   25
true Mi  214   78  158  105          247   62  179   67
true Mo   95   55  232  149          111   49  266  105
true S    33   20   87  133           46   27   98  102
```

**Is calibration genuinely better, or does it favour Normal?** [MEASURED]
- **Partly the latter.** The +2.5-point gain comes mainly from Normal recall (0.800 → 0.865) and
  Moderate.
- Severe recall falls (133 → 102 correct), Mild recall falls (0.141 → 0.112), and the share
  predicted Normal rises from 50 % to 56 % (true 46 %).
- Balanced accuracy and macro-F1 are unchanged, so it is a re-allocation of decisions, not better
  discrimination. AUROC is unchanged by construction.
- **Per-class threshold rules are rejected:** they buy the same accuracy by abandoning Mild.

---

## 6. Comparison against the required baselines (robust development CV) [MEASURED]

BASELINE_TABLE_PLACEHOLDER

Per-class precision / recall, confidence intervals, SDs and all paired tests:
- `results/p3_robust_tables.md`, `results/p3_robust_summary.csv`, `results/p3_robust_comparisons.csv`
- confusion matrices of the leading models: `results/confusion_matrices_leading.md` and
  `plots/cm_*.png`

---

## 7. Failed / negative experiments (and why)

| Experiment | Result [MEASURED] | Likely reason |
|---|---|---|
| Gradient boosting on header + clinical (+ 264 engineered features) | S3 0.436 / 0.425 vs LR 0.468 | trees overfit at about 535 training ECGs per fold; the signal is mostly linear |
| Attention fusion; vector gates | S3 0.465 / 0.458 | more parameters, no usable cross-modal structure with 3 tokens |
| Clinical features inside the fusion | Δ S3 +0.002 (p = 0.92) | ECGFounder already captures QRS-voltage / conduction / repolarisation information |
| Stacking | unstable (S3 0.500 / 0.482 / 0.480 across repeats) | weights fitted on about 670 OOF rows; high variance |
| Per-class decision thresholds | accuracy +2.8 but Mild F1 → 0.03 | maximising accuracy at this prevalence means "predict Normal" |
| ECGFounder fine-tuning (phase 2) | = frozen (Δ S3 0.000) | too few labels to move 3–27 M parameters usefully |
| The phase-2 "best" MLP fusion (S3 0.505) | robust 0.481 | seed / fold luck in one run: always use repeat × seed means |
| Supervised contrastive / self-supervised adaptation | not run | deprioritised (§2) |

---

## 8. Best model, configuration, reproducibility

**Recommended model:** the vote ensemble. Equal-weight mean of three probability vectors, all on the
frozen ECGFounder embedding and/or the printed header:

| Member | Configuration | Trainable params |
|---|---|---|
| (a) MLP fusion | header numeric (16, block-standardised, ÷√16) ‖ ECGFounder f6 (1024, ÷√1024) → dropout 0.5 → Linear(1040→64) → GELU → dropout 0.5 → Linear(64→4); AdamW lr 1e-3, weight decay = 100·L2 with L2 ∈ {1e-4, 1e-3, 1e-2} on inner-val; class-weighted CE; early stopping (patience 25); 3 seeds averaged | 66,820 × 3 |
| (b) ECGFounder linear | f6 → linear softmax; class-weighted CE + 1.0 × distance penalty; L2 on inner-val; L-BFGS | 4,100 |
| (c) Header + clinical LR | 16 header numbers + 28 clinical features → standardised multinomial LR, class_weight = balanced, C on inner-val | 180 |

Plus:
- **Frozen backbone:** ECGFounder 12-lead checkpoint (Hugging Face revision `d9b17939`, sha256
  `ee199f37…`, MIT).
- **Input:** "aligned" construction, with the masking fix for the 2 PDFs that have coverage gaps.
- **Decision rule:** argmax (primary). Optionally the nested protected prior rule, with its
  Normal-favouring caveat (§5.3).
- **Inference:** about 170 ms/ECG on 4 CPU cores ≈ PDF waveform extraction 60 ms + input build 9 ms
  + ECGFounder 51 ms + header/clinical features ≈ 50 ms; heads < 1 ms (timings from phase 2).

**Reproducible inference:**
- `phase_2/best_model/predict_pdf.py` implements member (a).
- FINAL_MODEL_PLACEHOLDER
- No checkpoint of the full ensemble trained on all development data exists yet; it is the next step
  before any locked-test evaluation.

---

## 9. Why S3, and what it does not do

S3 = (Accuracy + Balanced accuracy + Macro-F1) / 3, where:
- Accuracy = Σₖ πₖ Rₖ, with πₖ the class share and Rₖ the class recall;
- Balanced accuracy = ¼ Σₖ Rₖ;
- Macro-F1 = ¼ Σₖ F1ₖ.

Per class this is Σₖ (πₖ/3 + 1/12) Rₖ + (1/12) Σₖ F1ₖ. On this dataset the recall weights are about
0.24 (Normal), 0.16 (Mild), 0.15 (Moderate) and 0.12 (Severe). It rewards overall accuracy but
penalises buying it with minority classes.
- It is pre-registered, so it cannot be tuned after the fact.
- Its equal weights are a choice, not derived.
- It ignores ordinal severity, so QWK is reported separately.
- It is not a clinical-utility measure.
- Every table here shows the components too, so no conclusion rests on S3 alone.

---

## 10. Why 60–65 % was not reached: limiting factors

**Measured observations:**
1. [MEASURED] **A common ceiling.** More than 20 model families (linear, MLP, gated, attention,
   boosting, fine-tuned, ensembles; three representations) all reach 0.53–0.55 accuracy (argmax).
   Representation and architecture changes move S3 by ≤ 0.01–0.02. The data, not the model, is
   limiting.
2. [MEASURED] **Mild is not separable.** Mild one-vs-rest AUROC is 0.53–0.55 in every model
   (Mild-vs-Normal 0.72–0.77). Mild ECGs spread across Normal, Moderate and Severe (about 39 / 28 / 19 %
   in the MLP fusion). Every 4-class accuracy gain must therefore come from Normal / Moderate / Severe.
3. [MEASURED] **Accuracy beyond about 0.57 requires sacrificing Mild.** Decision calibration lifts
   accuracy by 2.5–3 points only by re-allocating decisions toward Normal.

**Hypotheses and estimates:**
4. [HYPOTHESIS] **Label noise** near the 41 / 52–54 % EF cut-offs (Teichholz EF, variable ECG–echo
   intervals, name-based matching) caps achievable accuracy. Testing it needs the restricted
   spreadsheets (EF values and dates), which were not opened.
5. [HYPOTHESIS] **Data size.** Phase-2 learning curves (`phase_2/analyses.py`, if run) and the
   literature (Ning, Vaid: tens of thousands of ECGs for Mild AUROC ≈ 0.82) suggest an order of
   magnitude more labelled ECGs is needed.
6. [ESTIMATE] Interval-cleaned labels plus EF regression with the authorised EF values: + a few
   accuracy points; more data: larger gains. These are not demonstrated and must not be cited as
   results.

## 11. Research novelty (honest assessment)

**New and defensible:**
- A validated pipeline that applies a public ECG foundation model to **3×4 PDF printouts**, with a
  label-free compatibility check and a masking dose-response showing that lead coverage, not time
  coverage, matters.
- Rigorous small-data benchmarking of more than 20 approaches: repeated patient-grouped CV, nested
  calibration, corrected tests.
- Evidence that printed machine **measurements** (not the machine's text) carry the header signal.

**Not novel / not supported:**
- State-of-the-art accuracy.
- Any significant superiority of fusion over ECGFounder alone (2-way vs ECGFounder MLP: Δ accuracy
  +0.008, p = 0.43).

## 12. What still requires validation

1. One authorised, pre-registered locked-test evaluation of the frozen final ensemble (plus the
   reference R). The test set was exposed in phase 1, so this is not unbiased.
2. External or temporal validation.
3. With permission: interval-based label cleaning and EF regression.
