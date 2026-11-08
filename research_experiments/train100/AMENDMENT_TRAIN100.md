# Amendment TRAIN100: retrain the project's models to (nearly) 100 % training accuracy

Status: **declared before any TRAIN100 run** (this file and the two scripts are committed first; results come later).
Branch: `research/train100`. `main` and every earlier result stay as they are.

## Why this branch exists

The project owner asked for every model to be trained until its training accuracy is (nearly) 100 %, using as many
epochs as needed. The original runs stop early on purpose (inner-validation early stopping, weight decay, dropout,
shrinkage), so their training accuracy is about 60–85 %. This branch shows what changes when that brake is removed.

**What we expect** (written down before running): training accuracy close to 100 %, and held-out accuracy the same or
lower than the original, because the extra fit is memorisation of the 670 training ECGs per fold, not new signal. The
150-epoch pilot (ECGFounder MLP head, repeat 0) already showed this: 82 % → 100 % training, 51.9 % → 52.0 % held-out.

## What is retrained

| Model | Original training | TRAIN100 change |
|---|---|---|
| ECGF-linear (ECGFounder f6, distance-penalised CE) | L-BFGS, L2 chosen by inner validation | L2 = 1e-6, L-BFGS until 100 % (cap 5,000 iterations) |
| ECGF-MLP (3 seeds) | AdamW, early stopping on inner validation | same head, no early stopping, until 100 % (cap 3,000 epochs) |
| FUSION-MLP (header ‖ ECGFounder, 3 seeds) | as ECGF-MLP | as ECGF-MLP |
| HDR-LR, HDRCLIN-LR, CLIN-LR (logistic regression) | C chosen by inner validation | C = 10,000 (almost no regularisation) |
| LGBM-FEAT (engineered feature families) | early stopping on inner validation | 2,000 trees, min_child_samples 1, no L2, no early stopping |
| LSTM-fixed (L3) and CNN-LSTM (L5) | ≤ 60 epochs, early stopping (patience 12) | weight decay 0, no early stopping, until ≥ 99.5 % (cap 400 epochs) |
| C1, IMG-vote (averages of the members above) | — | rebuilt from the TRAIN100 members |
| C3, IMG-C3 (+ nested class-prior rule) | prior rule α chosen on inner OOF | same rule, α chosen on TRAIN100 inner OOF |
| AGT-img triage (EF grade must agree with IMG-C3) | ridge + LightGBM EF regressors, shrinkage | ridge α = 1e-6, LightGBM 4,000 trees, min_child_samples 1, no L2 |

Every model has the same inputs, the same original four-class labels, and the same Phase-2 patient-grouped 5-fold CV × 3
repeats as the original. The networks (layers, hidden units, dropout layers) are unchanged; only the stopping rule and
the regularisation strength change. Training accuracy is measured on all training ECGs of the fold in evaluation mode
(no dropout).

## What is not retrained, and why

* **Image CNNs (ResNet-18/34, EfficientNet, ConvNeXt):** no GPU in this environment; about 20 hours of CPU per network.
* **EchoNext / SUPER ensembles:** EchoNext data is under a PhysioNet data-use agreement and stays on the owner's laptop.
* **Locked test set:** used once, with the owner's authorisation. It is **not** rescored. The 70s-band result stays with
  the original frozen model.

## Reporting

* For each model: mean training accuracy and mean held-out accuracy over the 15 folds, with the corrected
  repeated-CV 95 % interval, next to the original held-out accuracy.
* Aggregates only (means, confusion counts averaged per repeat). Nothing per ECG is committed.
* Whatever the numbers are, they are reported as they come out; no model is re-tuned after seeing them.
