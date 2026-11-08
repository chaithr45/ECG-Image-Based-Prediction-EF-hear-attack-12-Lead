# TRAIN100: every model trained to (nearly) 100 % training accuracy

**Question (owner's request):** what happens if every model is trained, for as many epochs or iterations as it needs,
until its training accuracy is (nearly) 100 %?

**Answer:** training accuracy reaches 100 % for every flexible model, but **no model becomes reliably better on new
patients, and most become worse**. The early-stopped models of report 45, with training accuracy of about 0.60–0.85,
remain the right ones to use.

The rules were declared before any run in [AMENDMENT_TRAIN100.md](AMENDMENT_TRAIN100.md) (main rules, addenda B and D,
note C). Every number below is aggregate; nothing per ECG is in this branch. The locked test set was not used and no
EchoNext data were used.

## Full write-up

- **Report (NITW template, TRAIN100 edition of report 45):**
  - [report_nitw/Report_TRAIN100_NITW.pdf](report_nitw/Report_TRAIN100_NITW.pdf), 68 pages. New Chapter 7 "Training Every Model to 100 % Training Accuracy".
  - Word version: [report_nitw/word/Report_TRAIN100_NITW.docx](report_nitw/word/Report_TRAIN100_NITW.docx).
  - LaTeX / Overleaf sources: [report_nitw/latex/](report_nitw/latex).
- **Results:** [results/](results).
  - JSON per run.
  - Figures: `train100_cv_training_vs_heldout`, `train100_fixedsplit_training_vs_validation`, and `epochs150/` (the 150-epoch demo).

## Headline numbers

### Development cross-validation (5 folds × 3 repeats, 837 ECGs)

| Model | Training | Held-out [95 % CI] | Original held-out | Change |
|---|---|---|---|---|
| Image-only triage (best in the 60s) | 1.000 | 0.696 [0.630, 0.762], answering 48 % | 0.694, answering 56 % | +0.2, but fewer patients answered |
| LGBM-FEAT | 1.000 | 0.548 [0.512, 0.583] | 0.524 | +2.4 (inside CI) |
| FUSION-MLP | 1.000 | 0.538 [0.499, 0.578] | 0.544 | −0.6 |
| ECGF-MLP | 1.000 | 0.536 [0.506, 0.565] | 0.542 | −0.6 |
| C3 | 0.995 | 0.533 [0.496, 0.569] | 0.561 | −2.8 |
| IMG-C3 | 1.000 | 0.532 [0.492, 0.572] | 0.558 | −2.6 |
| C1 | 1.000 | 0.517 [0.477, 0.558] | 0.547 | −3.0 |
| IMG-vote | 1.000 | 0.503 [0.463, 0.544] | 0.541 | −3.8 |
| ECGF-linear | 1.000 | 0.491 [0.451, 0.532] | 0.530 | −3.9 |
| HDRCLIN-LR | 0.589 | 0.486 [0.436, 0.536] | 0.501 | −1.5 |
| HDR-LR | 0.529 | 0.483 [0.432, 0.535] | 0.489 | −0.6 |
| CLIN-LR | 0.720 | 0.442 [0.390, 0.493] | 0.509 | −6.7 |

The logistic regressions cannot reach 100 % even with almost no regularisation. A linear model on a few dozen values
cannot separate every training ECG; this is a limit of the model type.

### Original fixed split (669 training / 168 validation ECGs)

| Network | Epochs to target | Training | Validation | Original validation |
|---|---|---|---|---|
| ResNet-18 | 24 | 0.997 | 0.423 | 0.524 |
| ECGNet | 21 | 0.999 | 0.435 | 0.476 |
| EfficientNet-B0 | 37 | 1.000 | 0.417 | 0.351 (original under-trained) |
| MobileNetV3-Large | 57 | 0.999 | 0.387 | 0.185 (original under-trained) |
| ConvNeXt-Atto | 53 | 0.996 | 0.345 | 0.452 |
| 1-D SE-ResNet | 32 | 0.999 | 0.435 | 0.482 |
| ECGFounder fine-tuned (stages 5–6) | 12 | 0.996 | 0.560 | 0.524 (single split, ±0.075) |
| HOG + RBF-SVM | — | 1.000 | 0.423 | 0.458 |
| XGBoost (waveform features) | — | 1.000 | 0.494 | 0.500 |
| Header logistic regression | — | 0.674 | 0.405 | 0.500 |
| Header 0.6 + ResNet-18 0.4 | — | 0.972 | 0.458 | 0.565 |

Every CNN was more accurate on validation at an earlier epoch than at the 100 % point.

### Variants of every phase, label studies, best-for-50s recipe without EchoNext

**Variants:** all flexible variants reach about 1.000 training accuracy, with held-out accuracy of 0.53–0.56.
- Phase 5: E1 0.542 (original 0.554); E2 0.557 (0.544); E3 0.530 / 0.532 (0.547 / 0.546); E4 0.540 (0.567); E6 0.542 (0.546); E7 0.534 (0.552).
- HIER: 0.535 (0.570).
- ENH-vote: 0.549 (0.569).
- ULT-stack: 0.537 (0.571).
- Multitask: 0.536 (0.546).
- EF regression MLP: held-out r 0.624 (0.653), training r 0.993.

**Label studies** (C3 recipe): CLEAR 0.670 (0.668), MMCLEAN 0.669 (0.672), NFS 0.694 (0.702), NF 0.673 (0.683),
HC30 0.583 (0.594), VER 0.537 (0.565).

**Best-for-50s recipe (SUPER-TOP10-ORD) on its hospital-only members:**
- original training: 0.565 [0.540, 0.590];
- trained to 100 %: 0.539 [0.509, 0.570].
- The published 0.587 also used EchoNext members, worth about 2 points. A separate laptop run with EchoNext is the owner's
  next step.

## Not retrained

| Model | Reason |
|---|---|
| EchoNext models and ensembles with EchoNext members | The PhysioNet data-use agreement keeps that data on the owner's laptop. |
| Locked-test result | Scored once with authorisation; not rescored. |
| LSTM family | Withdrawn at the owner's request. |
| EfficientNet-B2 and ConvNeXt-Tiny | Dropped at the owner's request. |
| E7 augmentation | Its inputs are not on this machine. |

See report Table 7.7.

## Where to stop training

In the 150-epoch demo (ECGFounder MLP head), training accuracy rises to 1.000 while held-out accuracy stays at about
0.52. With the early-stopping point (about epoch 12), training accuracy is 0.82 and held-out accuracy is 0.52.
The best stopping point is the epoch where accuracy on unseen data stops improving (inner-validation early stopping),
not 100 % training accuracy.
