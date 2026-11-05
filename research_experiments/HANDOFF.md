# HANDOFF: ECG-based 4-class LVEF prediction research (state at the end of the Phase-4 audit)

Read this file first, then `DATA_ACCESS.md`, `EXPERIMENT_INVENTORY.md`, `RESUME.md`,
`phase_3/reports/CLAIMS_REGISTER.md` and `reports/ERRATUM_phase1.md`.

**Do not re-run completed experiments.** Every result listed in `EXPERIMENT_INVENTORY.md` exists on disk and in
git; the run functions skip existing outputs.

## 1. Objective and architecture
- **Task:** predict the echocardiographic LVEF class (Normal / Mild / Moderate / Severe) from a 12-lead ECG PDF
  printout.
- **Owner's targets** (aspirational): accuracy ≥ 60 %, balanced accuracy ≥ 50 %, macro-F1 ≥ 0.48, Mild F1 ≥ 0.35,
  QWK ≥ 0.60.
- **Inputs** (all derived from the PDF; nothing else):
  - **header numbers:** age, sex and 14 machine measurements (16 features);
  - **ECGFounder embedding:** frozen public foundation model, Li et al. NEJM AI 2025, MIT licence. It takes the
    12-lead waveform recovered exactly from the PDF vector paths, in the time-aligned "aligned" construction,
    and outputs a 1,024-d mean-pooled last-stage embedding;
  - **clinical features:** 28 features from waveform + header;
  - **image CNN:** ResNet-18 on the grid-filtered page image.
- **Best overall model** (Phase 3): the equal-weight **vote ensemble** of
  - (a) MLP fusion of header numbers ‖ ECGFounder (3 seeds),
  - (b) ECGFounder linear head with distance-penalised CE,
  - (c) logistic regression on header numbers + clinical features.

## 2. Branches and commits (never force-push, never merge into the original branch)
| Branch | State |
|---|---|
| `claude/funny-gauss-udlmv0` | original project, `c2487308`; **untouched** |
| `research/ecg-ef-model-improvements` | phases 1–3; last pushed `39aa39f` (Phase-3 report still a draft there) |
| `research/ecg-ef-phase4` | local only, created from `39aa39f` |
| `research/phase4-handoff` | this branch: `39aa39f` + final Phase-3 results/report + Phase-4 audit + handoff documents |

## 3. Dataset accounting and splits [VALID]
- **Total:** 1,047 ECG PDFs from 993 patients (anonymous hash `data/manifest.csv:patient`). Folder labels:
  Normal 480 / Mild 232 / Moderate 221 / Severe 114. Normal is capped at 120 per year (2020, 2021, 2022, 2024).
- **Original split** (patient-grouped): train 669 / validation 168 / test 210 ECGs (635 / 159 / 199 patients).
  **"669" is the training split, never the total.**
- **Phase 2–4 protocol:**
  - development set **D** = train + validation = **837 ECGs / 794 patients**;
  - **locked test T = 210 ECGs / 199 patients**; 0 patients overlap;
  - T was inspected for ~40 models in Phase 1, so it is **exposed** and has **not** been scored since.
- **Folds:** `phase_2/configs/folds_D.csv`, StratifiedGroupKFold(5) on patients, repeats 0/1/2 (random_state
  2025 + r).
  - Every base model trains on inner-train and selects / early-stops on inner-val (≈ 535 / 134 ECGs per fold).
  - Post-hoc calibration is **fully nested** (an inner CV inside each outer training portion).
- **Integrity:** SHA-256 snapshots in `phase_4/results/integrity_phase4_before.json` (keys are hashed paths, no
  names). Last verification: 0 of 4,648 changed.

## 4. Results (development CV; mean over runs; full tables in `phase_3/reports/PHASE3_REPORT.md`)
| Model | Runs | Accuracy | Balanced acc. | Macro-F1 | Mild F1 | QWK | S3 |
|---|---|---|---|---|---|---|---|
| Header + ResNet-18 (original default) | 1 repeat | 0.495 | 0.444 | 0.389 | 0.193 | 0.491 | 0.442 |
| ECGFounder frozen, linear | 3 | 0.528 | 0.442 | 0.434 | 0.244 | 0.543 | 0.468 |
| 2-way MLP fusion header + ECGFounder | 9 | 0.542 | 0.465 | 0.436 | 0.176 | 0.579 | 0.481 |
| **Vote ensemble (recommended, best overall)** | 3 | **0.547 ± 0.002** | **0.467** | **0.455** | **0.255** | 0.574 | **0.490** |
| MLP fusion + nested protected prior calibration (**best accuracy**) | 3 | **0.568** | 0.463 | 0.442 | 0.159 | 0.591 | – |

- **Best accuracy: 56.8 %** (calibrated MLP fusion). It sacrifices Mild: F1 0.159, and predicts Normal 56 % vs a
  true 46 %.
- **Best overall:** vote ensemble, **accuracy 54.7 %, balanced accuracy 46.7 %, macro-F1 0.455**. Note: 54.7 % is
  accuracy, not balanced accuracy.
- **Statistics:**
  - nothing survives Holm correction;
  - vote vs original default: +5.4 accuracy (p = 0.013), +6.9 macro-F1 (p = 0.038);
  - vote vs MLP fusion: Mild F1 +0.083 (p = 0.014, nominal).
- S3 = (accuracy + balanced accuracy + macro-F1) / 3, the pre-registered selection score.

### Completed but unsuccessful (do not repeat without new evidence)
- gradient boosting on header / clinical / engineered features;
- attention fusion and vector gates;
- stacking (unstable across repeats);
- per-class threshold calibration (collapses Mild);
- ECGFounder fine-tuning (equal to frozen);
- ordinal heads (CORAL / cumulative link): lower accuracy;
- clinical features inside the fusion: Δ S3 +0.002, p = 0.92;
- image CNN added to the ensemble: no gain;
- masking alternatives (per-window, multiview): none beats "aligned".

## 5. Phase 4 progress (owner authorised the spreadsheets and the ECG_ECHO matching files)
**Done** (see `phase_4/reports/PHASE4_DATA_AUDIT.md`, `PHASE4_CLAIMS_REGISTER.md`):
- **4A:** the four authorised sheets are file-copy logs. EF lives in `ECG_ECHO` (git history, read-only), where
  matching is by **fuzzy name only**.
- **4B:**
  - thresholds recovered: Teichholz; Normal > 52 M / ≥ 54 F; Mild ≥ 41; Moderate ≥ 31;
  - 93.5 % folder vs nearest-echo agreement;
  - 22 labels have no consistent echo;
  - 91 ECGs have conflicting echoes;
  - measurement uncertainty is not quantifiable (18 repeat pairs).
- **4C:** cohorts with dev counts:

  | Cohort | A all | B ≤ 7 d | C ≤ 14 d | D ≤ 30 d | E ≤ 60 d | F ≤ 90 d | G high-confidence |
  |---|---|---|---|---|---|---|---|
  | Dev ECGs | 837 | 541 | 550 | 587 | 647 | 688 | 357 |

  The EF target is the **nearest-in-time** echo, chosen label-agnostically (ties → mean). The private table is
  `phase_4/data/private/cohorts.csv` (git-ignored).

**Outstanding:** 4D–4H and reports 2–8 (§9).

## 6. Environment
```bash
pip install -r research_experiments/requirements-research.txt
```

- Python 3.13.18, CPU-only, 4 cores, Linux. Versions pinned: torch 2.15.0, timm 1.0.30, numpy 2.5.3,
  pandas 3.0.5, scikit-learn 1.9.1, scipy 1.18.1, xgboost 3.4.1, opencv 5.0.0, pymupdf 2.1, matplotlib 3.11.2,
  openpyxl 3.1.5, joblib 1.6.0.
- Use `OMP_NUM_THREADS=1` for parallel single-thread jobs. Oversubscription slowed runs up to 4×.

## 7. Artefacts not in git (regenerate; do NOT retrain models)

**Project cache `data/`** (git-ignored by the original project; needed by every script that reads
`data/manifest.csv`, `data/header_features.csv` or `data/signals.npz`). A fresh clone does not have it. Rebuild it
with the **safe** script below, which writes only into `data/` and verifies the recorded SHA-256 values:

```bash
python -m research_experiments.regenerate_data_cache
```

- **Do NOT use** `python -m ecg_ef.preprocess` (rewrites `results/split.csv`) or `python -m ecg_ef.header`
  (re-trains and overwrites `models/header_logreg.joblib`).
- Add `--images` only if you need the filtered page-image cache (image CNN).

**ECGFounder 12-lead weights** (needed for any new waveform feature) → `research_experiments/models/pretrained/`,
sha256 `ee199f3781f4ae1f732973267f003da0a759ea12bddb0dd28a77faa60aca7997`:

```bash
curl -L -o research_experiments/models/pretrained/12_lead_ECGFounder.pth https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
```

**Frozen features:**
- `phase_2/models/ecgf_aligned_features.npz` is **committed** (14.6 MB). Phase-4 work needs only this plus the
  header features.
- Other feature variants and the activation caches: `python -m research_experiments.phase_2.ecgf_features`
  (≈ 5 min).

**Clinical features** (small CSV, regenerable in ≈ 6 min):
```bash
python -m research_experiments.phase_3.clinical_features
```

**Private Phase-4 tables:** `DATA_ACCESS.md` (≈ 1 min).

**Model checkpoints** (`*.pt`, 42–102 MB) are **not** in git. Their predictions (OOF + locked-test
probabilities) **are**. They are needed only for inference on *new* ECGs; the SHA-256 of each is listed in
`EXPERIMENT_INVENTORY.md`.

## 8. Reproduce existing tables (no training; reads saved predictions)
```bash
python -m research_experiments.status
```
```bash
python -m research_experiments.phase_3.p3_robust
```
```bash
python -m research_experiments.phase_3.calibration_summary
```
```bash
python -m research_experiments.phase_3.confusion_collection
```

Experiment → result mapping: `EXPERIMENT_INVENTORY.md`.

## 9. Rules (methodological restrictions and leakage prevention)
1. Same 1,047 ECGs, folder labels and patient-grouped splits. Never modify `LV_ECG_Dataset_*`, `data/`,
   `results/`, `models/` or `ecg_ef/`.
2. Never print or store patient names, dates of birth or identifying values. Mask text when inspecting sheets.
   Committed outputs contain aggregates only.
3. The locked test T is never used for any choice. Score it once, for a pre-registered shortlist, **only with
   the owner's explicit approval**.
4. All preprocessing, feature selection, hyper-parameters, calibration and thresholds are chosen inside the
   training folds. Thresholds come from inner-CV OOF predictions (nested).
5. Statistics:
   - corrected repeated-CV t-test (Bouckaert–Frank) on matched (repeat, fold) cells, plus a patient-cluster
     bootstrap;
   - Holm within metric;
   - report mean ± SD over seeds × repeats, never a single best run.
6. Regression targets use the **label-agnostic nearest echo**, never a label-consistent pick (that would leak
   the folder label).
7. No commit or push without the owner's explicit approval.

## 10. Next steps for the incoming agent (in this order)
1. **Set up the private data and check state.**
   - Install the requirements (§6) and rebuild `data/` with `python -m research_experiments.regenerate_data_cache`
     (§7).
   - Recover the private data (`DATA_ACCESS.md`).
   - Run `python -m research_experiments.phase_3.clinical_features` (the only `[TODO]` that `status` reports in a
     fresh clone).
   - Run `python -m research_experiments.status` and the integrity check:
   ```bash
   python -m research_experiments.phase_4.integrity4 verify phase4_before
   ```
2. **4D EF regression** (`phase_4/regression.py`, to write).
   - **Features:** ECGFounder f6, header numeric, and header ‖ ECGFounder.
   - **Models:** ridge (L2 on inner-val), MLP regression (Huber), 3 seeds.
   - **Train cohorts:** A / C / G. **Evaluate** on the outer folds of A (folder labels) **and** of G.
   - **Metrics:** MAE, RMSE, R², Pearson, Spearman, plus residual plots.
   - **Class conversion:** EF → class with the recovered sex-specific thresholds (sex from header `male`);
     optional inner-val linear recalibration of the predictions.
   - **Regression-assisted classification:** Gaussian class probabilities around the predicted EF, fused with the
     vote ensemble.
3. **4E multitask:** shared MLP on header ‖ ECGFounder; heads EF + 4-class; λ ∈ {0, 0.3, 1, 3} + uncertainty
   weighting; λ selected on inner-val.
4. **4G Mild study:**
   - Mild AUROC: one-vs-rest, vs Normal, vs Moderate;
   - errors by EF distance to 41 / 52 / 54 and by interval.
5. **4H benchmark:** compare against the vote ensemble with identical folds (repeats 0–2) using
   `phase_3/p3_robust.py`-style tables.
6. **Reports + claims:** write the Phase-4 reports 2–8 and keep the claims register.
