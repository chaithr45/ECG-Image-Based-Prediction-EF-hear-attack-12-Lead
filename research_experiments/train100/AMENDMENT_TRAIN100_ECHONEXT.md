# Amendment TRAIN100-EchoNext: the top 3 all-patient ensembles retrained to (nearly) 100 % training accuracy

Status: **declared before any TRAIN100-EchoNext run.** This file and `train100_echonext.py` are committed and pushed
first; the smoke test, the timed fold and the full run come after. Nothing is re-tuned after seeing results.
Branch: `research/train100-echonext`, cut from `research/train100` (which already holds `research_experiments/public_ef`
and `BEST_RESULTS` from `main`). `main` and every earlier result stay as they are.

## Why this amendment exists

The owner asked for the three best all-patient models of report 45 (Tables 5.2 and 6.1) to be retrained so that every
model inside them reaches (nearly) 100 % training accuracy, with the hospital data and EchoNext used exactly as in the
original pipelines. `AMENDMENT_TRAIN100.md` left these ensembles out because they need EchoNext, which is kept on the
owner's laptop under the PhysioNet data-use agreement. This run happens on that laptop.

**What we expect** (written down before running): training accuracy close to 100 % for the members and for the
ensembles, and held-out accuracy the same as or lower than the original (overfitting). That is a valid result, and it
will be reported as it comes out.

## The three models (all from `research_experiments/public_ef/en_super.py`)

| Model | Original run | Output used | Original held-out accuracy [corrected 95 % CI] |
|---|---|---|---|
| SUPER-TOP10-ORD | `--labels original --select top10` (base pool) | `SUPER-ORD` | 0.587 [0.565, 0.610] |
| STAGED-4, extended pool X | `--labels original --pool X --select top10` | `STAGED-4` | 0.585 [0.557, 0.613] |
| SUPER | `--labels original` (base pool, greedy selection) | `SUPER` | 0.583 [0.540, 0.626] |

The originals come from the committed `public_ef/results/en_super_original_top10.json`, `en_super_original_X_top10.json`
and `en_super_original.json`.

## TRAIN100 rules (as `AMENDMENT_TRAIN100.md`, restated by the owner for this run)

Inputs, labels (the original four classes), architectures, folds, inner splits, seeds and decision rules stay exactly as
in the original. Only the training brake changes:

* **Neural nets** (MLP / linear heads, the pixel CNN): no early stopping, weight decay 0 (L2 0 for the L-BFGS head);
  no augmentation, mixup, label smoothing or EMA; constant learning rate after the original warm-up (none of these
  networks had a warm-up, so the learning rate is constant from the start); train until the training accuracy
  (evaluation mode, all training rows) is >= 99.5 %. Caps: 3,000 epochs for MLP heads, 100 epochs for the image CNN,
  5,000 L-BFGS iterations for the linear head (the cap used in `research/train100`).
* **Logistic regression** (members and meta-learners): C = 10,000. Every C grid and the selection C (`C_SEL`) become
  10,000. Solver iteration limits stay at the original values (3,000 for members, 5,000 for meta-learners).
* **Gradient boosting** (LightGBM, HistGradientBoosting): no early stopping, 2,000 trees / iterations,
  `min_child_samples` (LightGBM) or `min_samples_leaf` (HistGB) 1, L2 0. Tree size, learning rate, row and column
  subsampling and class weights stay as in the original.
* **kNN:** the smallest k of the original grid (15), distance-weighted as in the original.
* **Ridge / EF regressors:** alpha = 1e-6.
* **Kept unchanged** (decision rules, still chosen on inner / meta-CV out-of-fold predictions): the top-10 selection,
  the greedy forward selection (at most 10 members, minimum gain 0.002), the floor rule (class weights so that every
  recall is >= 0.18, else >= 0.15), the STAGED-4 class-weight grid and Mild/Moderate threshold, class weights,
  the band rule (per-ECG Normal limit 52 men / 54 women, 41, 31), candidate order and meta-CV splits.

## Data

* **Hospital:** the 837 development ECGs, original four-class labels, Phase-2 patient-grouped 5-fold CV x 3 repeats
  (`bands.splits`), inner 4-fold patient-grouped splits (random_state 1000 + 10 r + k), meta-CV 4-fold patient-grouped
  splits (random_state 3000 + 10 r + k). Echo EF is used only as a training target (M6, EF-band, the EchoNext ridges and
  the image EF head), as in the originals, never as an input. The **locked hospital test set (210 ECGs) is not loaded**.
* **EchoNext:** the pipeline's cache `en_cache.npz` holds 4,827 EchoNext ECGs with LVEF and sex (all from the EchoNext
  file the pipeline embedded). `en_split.locked` puts 974 of them (20 % of patients, fixed hash) in the pipeline's
  **EchoNext locked test: never used and not scored here.** The other 3,853 are the training pool. The pipeline's
  validation split is the grouped 80/20 `GroupShuffleSplit(random_state=0)` of that pool for the tabular members and
  the grouped 90/10 split (3,467 / 386 images) for the pixel CNN. Validation accuracy is reported for every EchoNext
  member; the locked rows are not scored.

## Members: original setting -> TRAIN100 setting

### Hospital-trained members (inner OOF on the 4 inner folds + one fit on all outer training rows)

| Member | Inputs | Original training | TRAIN100 |
|---|---|---|---|
| ECGF-MLP | ECGFounder f6, 1,024 | Phase-2 MLP head (dropout 0.5, 64 GELU, balanced CE), AdamW lr 1e-3, batch 64, weight decay chosen from {0.01, 0.1, 1}, early stopping (patience 25, <= 200 epochs) on a grouped 20 % validation part; fitted on the other 80 %; seeds 42 / 43 / 44 averaged | same head, optimizer, batch, seeds; weight decay 0, no early stopping, constant lr 1e-3; until >= 99.5 % (cap 3,000 epochs) |
| ECGF-lin | ECGFounder f6 | linear head, CE + distance penalty (lambda 1), L-BFGS (lr 0.5, 300 iterations, strong Wolfe), L2 chosen from {1e-3 ... 10} on the grouped 20 % validation part; seed 42 | same head and L-BFGS settings, L2 0, repeated in blocks of 100 iterations until >= 99.5 % (cap 5,000) |
| M5 | f6 + qrs, st, hrv, freq, wavelet, leadstat, sq, logits150 (1,549) | LR, balanced, C from 1e-4 ... 10 (11 values) by inner log-loss | C = 10,000 |
| M6 | aligned f6 + synthetic-10-s f6 + the same families (2,573) | ridge on EF, alpha from {1, 10, 100, 1e3, 1e4} by inner RMSE -> Gaussian bands (Normal >= 53 / 41 / 31), sd = inner RMSE | alpha = 1e-6 |
| M8 | multi-scale ECGFounder f4, f5, f6max (2,448) | LR, balanced, C grid as M5 | C = 10,000 |
| M9 | PCA-64 of both f6 | kNN, distance-weighted, k from {15, 30, 60} by inner log-loss | k = 15 |
| M10 | families + f6 PCA-32 (557) | HistGB depth 3, lr 0.05, balanced, iterations from {100, 300}, L2 1.0, min_samples_leaf 20 | 2,000 iterations, L2 0, min_samples_leaf 1, early stopping off |
| M12 | as M5, fitted on zone rows only (Mild / Moderate, or Normal with EF < 62) | LR, balanced, C grid as M5 | C = 10,000 |
| H-TAB-LR | ECGFounder logits (aligned + synthetic, 150 + 150) + 467 tabular features (block-standardised, median imputation) | LR, balanced, C from {1e-4 ... 1} by inner log-loss | C = 10,000 |
| H-TAB-GBM | the same 767 columns | LightGBM, 7 leaves, lr 0.03, subsample 0.8, colsample 0.5, balanced, 300 trees, min_child_samples 15 | 2,000 trees, min_child_samples 1, L2 0 |
| H-X-GBM (pool X) | the 767 columns + 73 FEATX features | LightGBM as H-TAB-GBM, 300 trees | as H-TAB-GBM |
| H-MiMo-GBM (pool X) | as H-X-GBM; Mild vs Moderate, fitted on Mild / Moderate rows | LightGBM as H-TAB-GBM, 300 trees | as H-TAB-GBM |
| EF-band | ridge on the M6 inputs (`bands.mu_job`) -> band probabilities with the per-ECG Normal limit; EF / 50 as an extra meta input | alpha from {1 ... 1e4} by inner RMSE | alpha = 1e-6 |

In the original, M6 and EF-band are the same ridge (same inputs, same inner splits, same RMSE criterion; checked on the
cached folds: identical alpha and inner RMSE), so TRAIN100 fits that ridge once per fold and feeds both members.

### EchoNext-trained members (fitted on EchoNext; applied to the hospital ECGs, never trained on them)

| Member | Inputs | Original training | TRAIN100 |
|---|---|---|---|
| EN-cls | standardised aligned + synthetic f6 (2,048) | LR, balanced, C from 1e-4 ... 10 by validation log-loss; refit on all 3,853 rows | C = 10,000; refit on all 3,853 rows |
| EN-band | the same | ridge on EF, alpha from {1 ... 1e4} by validation MSE; refit on all rows | alpha = 1e-6 |
| EN-cls-all-H / -plain | block-standardised f6a, f6s, la, ls, tab (2,815); one model applied to the hospital inputs harmonised (H) and as recorded (plain) | LR, balanced, C from {1e-4 ... 1} by validation log-loss; refit on all rows | C = 10,000 |
| EN-gbm-H / -plain | la, ls, tab (767) | LightGBM as H-TAB-GBM, up to 400 trees with early stopping (50 rounds) on the validation part; refit on all rows with the best number of trees | 2,000 trees, no early stopping, min_child_samples 1, L2 0 |
| EN-band-all-H / -plain | as EN-cls-all | ridge on EF, alpha grid by validation MSE | alpha = 1e-6 |
| PIX-ZS / PIX-ZS-band | EchoNext printout images 320 x 640; ResNet-18 (ImageNet initialisation) with a class head and an EF head | AdamW lr 3e-4, weight decay 1e-4, cosine schedule over 15 epochs, early stopping (patience 3) on validation macro-F1, augmentation (affine: translate 2 %, scale 0.95-1.05; brightness / contrast 0.2), batch 32, mixed precision, loss = balanced CE + 0.5 MSE(EF); trained on 3,467 images | weight decay 0, no augmentation, constant lr 3e-4, no early stopping; until >= 99.5 % (cap 100 epochs); same split, loss, batch, seed 42; **retrained on the laptop GPU (RTX 3060)** |
| EN-cls-all-X (pool X) | as EN-cls-all + the 73 FEATX features in the tabular block | LR, C grid by validation log-loss | C = 10,000 |
| EN-gbm-X (pool X) | la, ls, tab + FEATX (840) | LightGBM, 400 trees + early stopping | as EN-gbm |
| EN-MiMo-LR (pool X) | as EN-cls-all-X; Moderate vs Mild, fitted on Mild / Moderate rows | LR, C grid by validation log-loss | C = 10,000 |
| EN-MiMo-GBM (pool X) | as EN-gbm-X, Mild / Moderate rows | LightGBM, 400 trees + early stopping | as EN-gbm |
| EN-mu-mid (pool X) | as EN-cls-all-X, rows with EF 25-60 | ridge on EF, alpha grid by validation MSE | alpha = 1e-6 |

## Meta-learners (same inputs: log-probabilities of the chosen members' inner OOF, plus EF / 50 of the band members)

| Step | Original | TRAIN100 |
|---|---|---|
| Top-10 selection (SUPER-TOP10-ORD, STAGED-4) | the 10 members with the best single-member meta-CV accuracy (floor rule), selection LR C_SEL = 0.1 | rule unchanged, C_SEL = 10,000 |
| Greedy selection (SUPER) | forward selection, <= 10 members, minimum gain 0.002, C_SEL = 0.1 | rule unchanged, C_SEL = 10,000 |
| SUPER meta-learner | multinomial LR, C from {0.01, 0.03, 0.1, 0.3, 1} by meta-CV log-loss; floor rule on the meta-CV OOF | C = 10,000; floor rule unchanged |
| SUPER-ORD meta-learner | Frank-Hall ordinal (three binary LR for P(y > j)), C grid as above; floor rule | C = 10,000; floor rule unchanged |
| STAGED-4 meta-learner | 3-class LR (Normal / Mild+Moderate / Severe) + binary LR (Moderate vs Mild, on Mild / Moderate rows), C grids as above; class weights and threshold tuned on the meta-CV OOF | C = 10,000; weights and threshold rules unchanged |

The three meta functions are reused from `en_super.py` unchanged: the script takes their source and replaces only the C
grid literal by (10,000,) and binds `C_SEL` = 10,000 (it checks that the grid literal occurs exactly where expected).

## Changes beyond the knobs above (declared here, before running)

1. **Fit rows of ECGF-MLP and ECGF-lin.** The original kept a grouped 20 % of every fit aside, only to stop early and
   to choose the L2 / weight-decay strength. TRAIN100 has neither, so those rows are trained on too (as in
   `research/train100`). The inner 4-fold splits and the outer folds are unchanged.
2. **EchoNext validation fits.** For every EchoNext tabular member, one extra fit on the 80 % part is scored on the
   20 % validation part (the original did this fit to choose C / alpha / the number of trees). The model applied to
   the hospital ECGs is the refit on all 3,853 training rows, as in the original. The pixel CNN is trained on the
   3,467 training images only, as in the original, and scored on the 386 validation images.
3. **Threads.** LightGBM / BLAS / OpenMP thread counts are set explicitly so that outer folds can run in parallel.

## What is measured and reported (aggregates only)

* **Each of the 3 models:** training accuracy = the fitted pipeline (members fitted on all outer training rows, then
  the meta-learner) applied to those same training rows, mean over the 15 folds; held-out accuracy as in the original
  (per repeat over all 837 ECGs, mean of the 3 repeats) with the corrected repeated-CV 95 % CI
  (`phase_5.p5common.corr_ci` over the 15 fold accuracies); recall Normal / Mild / Moderate / Severe; macro-F1; the
  original accuracy and the change in points. Confusion counts averaged per repeat.
* **Each hospital member:** training accuracy (in-sample on its own training rows: outer training rows; zone rows for
  M12; Mild / Moderate rows for H-MiMo-GBM, as a Mild-vs-Moderate accuracy) and held-out accuracy over the 15 folds
  (argmax of the member's probabilities; band probabilities for M6 / EF-band).
* **Each EchoNext member:** training accuracy on its EchoNext training rows, accuracy on the EchoNext validation split
  (RMSE, r and graded four-class accuracy for the EF regressors), and its accuracy on the hospital ECGs (which it was
  never trained on). Selection frequency of every member in each model.
* Files: `research_experiments/train100/results/train100_echonext.json` and
  `research_experiments/train100/TRAIN100_ECHONEXT_REPORT.md`.

## Data protection (unchanged project rules)

* EchoNext rows, per-record predictions, embeddings, features, images, weights and plots never leave the laptop; only
  aggregate numbers are committed.
* Hospital: no patient names or ids, ECG ids, timestamps, per-ECG predictions or embeddings, raw signals, the ECG_ECHO
  sheets, weights or credentials are committed or printed. Echo EF is a target only.
* All intermediate caches, per-fold predictions and the retrained CNN weights go to the new, git-ignored folder
  `research_experiments/public_ef/data/private/train100_echonext/` (smoke test: `.../train100_echonext_smoke/`). The
  script refuses to write anywhere else except the aggregate results JSON; the original caches are only read.
* Before every commit: `phase_5.p5common.privacy_scan` on every staged file and a read of the staged diff.

## Order of work

1. Commit and push this declaration and the script (before any run).
2. Smoke test: fold r0 k0 with tiny budgets, separate private folder.
3. One real fold (r0 k0), timed; time per fold and total estimate reported to the owner.
4. Full run in the background with per-fold caching (resumable).
5. Results JSON + report committed and pushed to `research/train100-echonext`.
