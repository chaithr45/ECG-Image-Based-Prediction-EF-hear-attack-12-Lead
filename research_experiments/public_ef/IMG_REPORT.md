# Image-only route: EchoNext drawn as hospital printouts → ResNet-18 (IMG_PLAN.md)

**Date:** 2026-11-06. All numbers are aggregates.
- **Hospital data:** the 837 development ECG images, Phase-2 folds 5 × 3, original four-class labels. The locked
  210-ECG test set was not loaded.
- **EchoNext data:** the **test** file only (owner, IMG_PLAN Amendment 1). 20 % of its patients are a locked EchoNext
  test set (Amendment 2).
- **Model input:** pixels of the page's waveform area, with all text removed. No header, machine measurements, age or
  sex.

## Answer
EchoNext image pre-training helped the pixel model, but not significantly.
- **FT-EN** (pre-trained on EchoNext images, then fine-tuned on each training fold): **0.530** [0.481, 0.578].
- **FT-IN** (ImageNet only): **0.492** [0.446, 0.537].
- **Difference:** +0.038, Holm p = 0.30.

Without any hospital training, the EchoNext-only network (ZS) already reaches 0.524. FT-EN has the highest Mild
recall of any model in the project so far (0.23).

The best pixel model stays below the image-derived trace models:
- ULT-stack 0.571 and stack-BASE 0.574 (traces read from the same images);
- C3 with header 0.562; FT-EN vs C3 −0.033, Holm p = 0.30.

## 1. Fidelity of the drawn images (step 4; 30 development ECGs, `img_fidelity.py`)
| | Laptop: median | Laptop: minimum | Cloud reference: median |
|---|---|---|---|
| Pixel correlation, redrawn vs real image | 0.993 | 0.988 | 0.993 |
| Trace-mask IoU (pixels > 127) | 0.966 | 0.949 | 0.967 |

**Structural check of the drawn EchoNext images** (aggregate pixel profiles of 300 images per source; no image was
opened or uploaded):
- trace-row centres at pixel rows 48 / 127 / 207 / 287, against 48 / 128 / 208 / 288 on the hospital images;
- column-profile correlation 0.977;
- ink in the calibration-pulse strip 0.049 vs 0.050.

## 2. EchoNext images and pre-training
- **Images:** 4,827 EchoNext test-file ECGs with LVEF and sex (4,827 patients), drawn at 320 × 640.
  - Classes, project rule: Normal 3,518, Mild 547, Moderate 298, Severe 464.
- **Pre-training** (EchoNext train part only): 3,467 training images and a patient-grouped 10 % early-stopping
  hold-out of 386. 15 epochs, 4.6 min on the RTX 3060.

| EchoNext | n | Accuracy | Macro-F1 | Recall N / Mi / Mo / S | EF r |
|---|---|---|---|---|---|
| Early-stopping hold-out | 386 | 0.642 | 0.438 | 0.74 / 0.25 / 0.38 / 0.47 | 0.575 |
| **Locked test (never used for any choice)** | 974 | **0.582** | 0.400 | 0.68 / 0.22 / 0.30 / 0.52 | 0.587 |

## 3. Hospital evaluation (837 development ECGs, Phase-2 folds 5 × 3, seed 42)
| Model | Accuracy [corrected 95 % CI] | Recall N / Mi / Mo / S | Precision N / Mi / Mo / S | Macro-F1 | Admissible |
|---|---|---|---|---|---|
| ZS: EchoNext only, class weights tuned on outer-training | 0.524 [0.493, 0.556] | 0.86 / 0.18 / 0.27 / 0.32 | 0.63 / 0.27 / 0.42 / 0.37 | 0.403 | yes |
| **FT-EN**: EchoNext pre-trained + fine-tuned | **0.530 [0.481, 0.578]** | 0.82 / **0.23** / 0.34 / 0.29 | 0.71 / 0.29 / 0.38 / 0.29 | 0.416 | yes |
| FT-IN: ImageNet only (pixel baseline) | 0.492 [0.446, 0.537] | 0.75 / 0.20 / 0.36 / 0.24 | 0.69 / 0.23 / 0.37 / 0.25 | 0.386 | yes |
| C3 (with header; reference) | 0.562 [0.519, 0.605] | 0.85 / 0.18 / 0.46 / 0.34 | 0.71 / 0.30 / 0.46 / 0.34 | 0.449 | yes |

**Tests** (corrected repeated-CV t-test over 15 cells, Holm over the two):
- FT-EN vs FT-IN: +0.038 (p 0.248, Holm 0.303);
- FT-EN vs C3: −0.033 (p 0.151, Holm 0.303).

**Context:**
- Best image-derived models so far: ULT-stack 0.571, and on this branch stack-BASE 0.574 (traces read from the
  image).
- The triage model answered 56 % at 0.694 in CV, and 50 % at 0.747 on the locked test.
- The original project's pixel ResNet-18 scored 0.471 in CV, against FT-IN 0.492 here.

## 4. Notes
- **FT-IN run order:** the FT-IN folds were run before EchoNext pre-training, through the identical code path and cache
  files (IMG_PLAN Amendment 1).
- **Seed and status:** one seed (42) and exploratory, as pre-registered. The IMG70 study (Amendment 3) continues with
  stronger backbones, EF-led pre-training and a 3-seed final model.
- **What stays private:** EchoNext images, weights and per-ECG outputs remain in the git-ignored
  `research_experiments/public_ef/data/private/`.

## 5. IMG70 results (Amendment 3; development CV, 837 hospital page images, image only)

| Variant | Accuracy [95 % CI] | Recall N / Mi / Mo / S | Macro-F1 | Inner-val acc. |
|---|---|---|---|---|
| FT-IN: ResNet-18, ImageNet only (320x640) | 0.492 [0.446, 0.537] | 0.75 / 0.20 / 0.36 / 0.24 | 0.386 | -- |
| FT-EN: ResNet-18, EchoNext pre-trained (320x640) | 0.530 [0.481, 0.578] | 0.82 / 0.23 / 0.34 / 0.29 | 0.416 | 0.569 |
| E2: ConvNeXt-Tiny (384x768) | 0.535 [0.484, 0.586] | 0.80 / 0.22 / 0.40 / 0.31 | 0.433 | 0.583 |
| E3: ConvNeXt-Tiny, EF-led + EF grading | 0.540 [0.511, 0.570] | 0.86 / 0.17 / 0.35 / 0.34 | 0.419 | 0.597 |
| **E4: E3 + layer-wise LR, mixup, TTA, 3-seed ensemble (final)** | **0.548 [0.513, 0.582]** | 0.85 / 0.18 / 0.41 / 0.28 | 0.423 | **0.604** |

- **Final model** (pre-declared rule: highest mean inner-val accuracy): E4. Mean of its three single seeds: 0.542
  [0.516, 0.567].
- **Tests** (corrected repeated-CV t-test, Holm): E4 vs C3 (reads the printed header) -0.021 (Holm p 0.717);
  E4 vs FT-EN +0.012 (Holm p 0.717). Neither is significant.
- **E6 triage on the E4 probabilities** (threshold chosen on inner validation for a 0.70 target): answers 60.6 % at
  0.661 (per repeat 59.7 % / 0.670, 69.8 % / 0.630, 52.2 % / 0.684). Descriptive, after the fact: the most confident
  20 / 30 / 40 / 50 % of ECGs are graded with 0.80 / 0.80 / 0.75 / 0.70 accuracy.
- **70 % on all patients:** not reached (pre-registered target). The best image-derived all-patient result remains the
  SUPER ensembles (0.583-0.587, PUBLIC_EF_REPORT.md).
- **Interpretation:** a network reading only the page pixels, with no digitisation, reaches 0.548, within about two
  points of the header-using C3 (0.562) and 5.6 points above ImageNet initialisation. EchoNext pre-training in the
  hospital format, EF-led training and seed ensembling each added a little.
