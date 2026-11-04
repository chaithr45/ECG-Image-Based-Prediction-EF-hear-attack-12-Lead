# RESUME HERE: hand-off for any agent or person continuing this research

Run `python -m research_experiments.status` first. It prints DONE/TODO for every step and the next
command. Then run `bash research_experiments/resume_pipeline.sh`. Every step skips finished work:
- `cv_features.run`, `p3common.run3` and `exp_p3` return early when `cv_metrics.json` exists;
- `cv_image_cnn` skips folds whose `foldK.npz` exists;
- `masking_audit` skips feature files that already exist.

## Rules (from the project owner; keep them)
- **Dataset freeze:** the same 1,047 ECGs (837 development / 210 locked test), labels and patient
  groups. Never modify `LV_ECG_Dataset_*`, `data/`, `results/`, `models/` or `ecg_ef/`.
- **Spreadsheets:** never open `LV_ECG_Dataset_*/*.xlsx` (restricted: EF values and dates).
- **Locked test** (`phase_2/locked_test/`, `phase_3/locked_test/`): probabilities only, never scored.
  It is scored once, for a pre-registered shortlist, **only with the owner's explicit authorisation**.
- **Selection:** on development CV only (`phase_2/configs/folds_D.csv`, repeat 0 primary, repeats
  1–2 for robustness).
- **Calibration:** thresholds fully nested (`phase_2/nested_calibration.py`).
- **Git:** do not commit or push without the owner's explicit permission. Work on branch
  `research/ecg-ef-model-improvements` only. The last pushed commit is `069c2bf`.
- **Patient names:** never print patient names. `data/manifest.csv:source` contains them; use
  `common.load_meta()`, which drops that column.
- **Verify integrity at the end:** `python -m research_experiments.phase_2.integrity2 verify phase2_before`.

## Where things are
| What | Where |
|---|---|
| Phase-1 report + erratum | `reports/FINAL_REPORT.md`, `reports/ERRATUM_phase1.md` (the phase-1 p < 0.001 CV claim is **retracted**) |
| Claims status | `phase_3/reports/CLAIMS_REGISTER.md` |
| Phase-2 protocol / pre-registration | `phase_2/protocol.py`, `phase_2/configs/preregistration.md` (+ sha256) |
| Phase-2 audits | `phase_2/results/audit_*.json`, `dataset_reconciliation.json`; `phase_2/reports/AUDIT_AND_STATUS.md` |
| Phase-2 CV runs | `phase_2/results/<tag>/{oof.csv, info.json, cv_metrics.json}` |
| Phase-3 code / runs | `phase_3/*.py`; `phase_3/results/<tag>/` |
| Dataset accounting | `phase_3/results/dataset_accounting.json` |

## Key results so far (development CV, 837 ECGs; locked test NOT scored)
| Model | S3 | Accuracy |
|---|---|---|
| Header numeric LR | 0.432 | 0.485 |
| Header numeric + clinical LR | 0.468 | 0.520 |
| ECGFounder frozen, linear | 0.470 | 0.532 |
| 2-way MLP fusion (header + ECGFounder) | 0.479–0.505 over seeds (repeat 0); repeat 1: 0.476 | 0.536–0.559 |
| 3-way MLP fusion (+ clinical) | 0.481 / 0.489 (seeds 42 / 43) | ≈ 0.54 |
| Gated 3-way + distance penalty | 0.482 (seed 42) | 0.536 |
| Gradient boosting / attention fusion / fine-tuning | no gain | – |

## Remaining steps (in order; all in `resume_pipeline.sh`)
1. Image-CNN CV folds 3–4 (about 25 min per fold on 3 threads).
2. Matched 3-way runs and gated-dist runs over seeds × repeats.
3. Masking dose-response CV (`phase_3.masking_cv`).
4. Header + image fusions, including the reference R (`phase_2.run_fusion --stage combos`), once the
   CNN `oof.csv` exists.
5. `phase_3.p3_robust` (tables and paired tests, including the 3-way vs 2-way question).
6. Nested calibration (`phase_2.nested_calibration`) for `fusion_mlp`, repeats 0–2.
7. Complementarity (`phase_2.complementarity --ecgf ecgf_aligned_f6_mlp --fusion ens3__fusion_mlp_r0`).
   Ensembles: build from the OOF files with `phase_2.fusion.run`.
8. Phase-3 report `phase_3/reports/PHASE3_REPORT.md`. Separate MEASURED results from HYPOTHESES and
   estimated gains; compare 3-way vs 2-way on matched folds, seeds and repeats.
9. Final model on all development data (`phase_2.final_model`), then ask the owner before any
   locked-test scoring.

## If the container was lost (only the GitHub branch survives)
Regenerate the git-ignored artefacts. All are deterministic and need no labels:
1. **ECGFounder 12-lead checkpoint:** download
   https://huggingface.co/PKUDigitalHealth/ECGFounder/resolve/d9b1793951b2342f5f7e84f1ac03cd37f8a08724/12_lead_ECGFounder.pth
   into `research_experiments/models/pretrained/` (sha256 `ee199f37…a7997`).
2. **1-lead checkpoint:** same URL with `1_lead_ECGFounder.pth`, into
   `research_experiments/phase_2/models/pretrained/` (sha256 `f863a388…37603`).
3. **Features:**
   - `python -m research_experiments.phase_2.ecgf_features` (≈ 5 min)
   - `python -m research_experiments.phase_3.clinical_features`
   - `python -m research_experiments.phase_3.masking_audit`
4. **CV runs not on GitHub:** rerun `resume_pipeline.sh`. Runs are deterministic given seeds; the
   CNN folds are the slow part, ≈ 2 h.
