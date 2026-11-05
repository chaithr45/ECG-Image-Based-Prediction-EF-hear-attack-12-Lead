# Public-EF study: learning ECG → EF from EchoNext, formatted like the hospital printouts (pre-registration)

**Date:** 2026-11-05. Written before any run. Branch `research/public-ef`, kept separate from `research/phase6` at the
owner's request ("do this separately in a branch").

**Owner's goal:** improve four-class LVEF classification of the hospital ECG images (837 development ECGs, unchanged)
by learning from a larger public ECG–echo dataset. The dataset under study stays the same; the public data only teach
the model.

## Data-use rules (PhysioNet credentialed licence; non-negotiable)
- EchoNext files stay on the owner's laptop in `C:\krithika_major\public_data\echonext\`, outside the repository
  and outside OneDrive.
- No EchoNext row, waveform, identifier or derived per-ECG array is committed, pushed, uploaded or printed. Caches go
  to the git-ignored `research_experiments/public_ef/data/private/`.
- Committed: code, plus aggregate metrics (counts, accuracies, AUROCs).
- Models trained on EchoNext stay private and are not committed.
- The scripts refuse an `ECHONEXT_DIR` inside the repository.

## EchoNext inputs (metadata columns verified from the header only)
- **Metadata:** `echonext_metadata_100k.csv`, with `lvef_value` (continuous EF), `sex`, `patient_key` and `split`.
- **Waveforms:** `EchoNext_<split>_waveforms.npy`, shape N × 1 × 2500 × 12 (10 s at 250 Hz). The lead order is I, II,
  III, aVR, aVL, aVF, V1–V6, in the order of the metadata rows of that split (as in the public EchoNext-Mini
  notebook).
- **Splits used first:** `val` and `test` (about 2.2 GB). `train` (16 GB) is optional, for a later run.
- **Labels:** the project's own rule applied to `lvef_value` and sex:
  - **Normal:** EF > 52 (men) / ≥ 54 (women);
  - **Mild:** ≥ 41;
  - **Moderate:** 31 to < 41;
  - **Severe:** < 31.
- **Excluded:** rows without `lvef_value` or with sex other than male / female.

## Making EchoNext look like the hospital printouts
Per ECG:
1. **Recover millivolts.** If the arrays are the normalised model inputs (per-lead SD ≈ 1), invert the published
   EchoNext-Mini normalisation (z-score → MUSE units → × 4.88 µV). Values the publishers clipped stay clipped.
2. **Resample** 250 → 500 Hz (polyphase).
3. **Cut into the printout layout:** I / II / III from 0–2.5 s, aVR / aVL / aVF from 2.5–5 s, V1–V3 from 5–7.5 s,
   V4–V6 from 7.5–10 s; lead II for the full 10 s as the rhythm strip.
4. **Build the two inputs** used for the hospital ECGs:
   - "aligned" (`ecgfounder.inputs.build`; the PDF gap masking is not applicable);
   - "synth" (`enh_build.build_one`, the beat-synchronous 10-s reconstruction).
5. **Embed** each with the frozen ECGFounder: f6 and the 150 logits.

The hospital ECGs go through the same steps, from their digitised printout traces. Both sources therefore enter the
model in the same format.

## Models trained on EchoNext only
- **EN-cls:** class-balanced multinomial LR on [aligned f6 ‖ synth f6] → four-class probabilities.
- **EN-mu:** ridge regression of EF on the same inputs → predicted EF μ.
- **Hyper-parameters:** C ∈ 10^{−4…1} (step 0.5) and α ∈ {1, 10, 100, 10³, 10⁴}, chosen on an 80 / 20
  patient-grouped split of the EchoNext rows. The models are then refitted on all EchoNext rows.
- **Internal check (aggregate):** accuracy, per-class recall and EF r on the held-out 20 %.

## Evaluation on the 837 hospital development ECGs (Phase-2 folds, 5 × 3; locked test not used)
1. **Zero-shot transfer:** EN-cls argmax on all 837. Also EN-cls with class weights tuned on each outer-training
   fold, using the ULT decision rule: maximise accuracy with every recall ≥ 0.18, fallback 0.15. Also the AUROC
   of μ for echo EF ≤ 45, Normal vs Mild, and Mild vs Moderate.
2. **Added value (primary):** stack-PLUS vs stack-BASE.
   - **BASE members:** M1 IMG-vote, M5 and M6 (ENH definitions), with nested inner OOF.
   - **PLUS members:** BASE plus EN-cls and EN-band. EN-band = N(μ, σ̂²) integrated over the class bands, with σ̂
     the residual SD on the outer-training rows. The EchoNext members are fixed functions, so their inner values
     are simply their predictions on the training rows.
   - **Meta-learner:** multinomial LR on the members' log-probabilities and μ / 50; C chosen by patient-grouped
     4-fold meta-CV.
   - **Decision rule:** as in ULT.
   - **Test:** paired corrected t-test over 15 cells (PLUS vs BASE, PLUS vs C3).
3. **Reported:** accuracy (corrected 95 % CI), per-class recall and precision, admissibility (every recall ≥ 0.15).

**Status:** exploratory. Results are compared with C3 (0.561), ULT-stack (0.571) and ULT2-stack (0.569).

## Amendment 2 (2026-11-06): EchoNext test split only
**Owner:** "for my phase 1 results lets only use test file....val and train file lets do some other day".

**Changed:** the final run of this plan and of Amendment 1 (`AMENDMENT_1_IMAGE_ONLY.md`) uses the EchoNext `test`
split only (5,442 ECGs). `val` and `train` are deferred to a later run.

**Unchanged:** everything else. Interim updates (`en_stream.py`) already use the test split only.

## Amendment 3 (2026-11-06): train / test split inside the EchoNext test file
**Owner:** "save few in test for test also...split accordingly"; "i mean split train test accordingly as u will only be
given test file".

**Split:** the EchoNext `test` file is divided by patient (`en_split.locked`):
- **EchoNext locked test:** 20 % of patients, those with sha256(`patient_key`) mod 5 = 0. The assignment depends only
  on the patient, so it is identical in interim and final runs.
- **EchoNext train:** the other 80 %.

**Use of the locked test:**
- Never used for training, standardisation, early stopping, hyper-parameter choice or class weights.
- It is only scored: accuracy, per-class recall, EF r.
- It is shown at every interim update; nothing is chosen with it.

**Use of the train part:**
- All EchoNext models (the plan's EN-cls / EN-mu and Amendment 1's members) are fitted only on the train part.
- The plan's patient-grouped 80 / 20 selection split is now taken inside the train part.

**Code:** `en_split.py`; `en_train.train_en` and `en_image.fit_en_all` exclude locked patients.

**Unchanged:** the hospital evaluation (837 development ECGs, Phase-2 folds).

## Amendment 4 (2026-11-06): SUPER ensemble of all image-derived models, pruned inside the folds; echo-consistent-label sensitivity analysis
Declared before running. **Code:** `en_super.py`.

**Owner:** "use all models which you think would give the highest accuracy ... compare each result and remove few
models which aren't performing good ... so u have only few 8-10 models for final result"; and, on the 6.6 % of
labels that do not match their own echo-EF band, "do this".

### A. SUPER (headline: original labels, all 837, Phase-2 folds 5 × 3)
**Inputs:** all image-derived; no header, machine measurements, age or sex.

**Candidate members:**
- **Hospital-trained** (nested inner OOF, Phase 5 / 6 protocol):
  - ECGF-MLP, ECGF-lin;
  - M5, M6 (ENH);
  - M8, M9, M10, M12 (ULT);
  - EF-band: N(μ, σ̂²) over the class bands, with μ from BANDS `mu_job` and σ̂ the inner residual SD;
  - H-TAB-LR and H-TAB-GBM: class-balanced LR and LightGBM on the hospital G2–G4 features (logits + CLIN + waveform
    families, plain inputs). Inner 4-fold OOF on the same inner splits. Settings as `en_image` (LR C from the
    EN-cls-all grid by inner log-loss; GBM fixed MILDFEAT settings with 300 rounds).
- **EchoNext-trained, fixed functions:**
  - EN-cls and EN-band (plan);
  - EN-cls-all, EN-gbm and EN-band-all, each in variant H and in plain.
- **Pixel, fixed function:** PIX-ZS, i.e. the IMG_PLAN EchoNext-pre-trained ResNet-18 applied to the hospital images,
  giving class probabilities and an EF band.

**Pruning inside each outer training fold:**
- Greedy forward selection on the members' inner values: at each step, add the member that most improves meta-CV
  accuracy.
  - Meta-learner for selection: multinomial LR, C = 0.1, on the log-probabilities plus μ / 50 for EF-band members.
  - Patient-grouped 4-fold meta-CV (random_state 3000 + 10 r + k).
  - Accuracy after the ULT decision rule.
- Stop when no addition improves it by ≥ 0.002, or at 10 members.
- Then the plan's meta-learner (C chosen by meta-CV log-loss) and the ULT decision rule on the selected members.

**Reported:**
- accuracy (corrected CI), per-class recall and precision, macro-F1, admissibility;
- how often each member is selected (out of 15 folds);
- tests: SUPER vs stack-BASE and SUPER vs C3 (corrected t, Holm with the plan's comparisons).

### B. Echo-consistent labels (sensitivity analysis; not a headline result)
**Labels:** each ECG's label is re-derived from its own recorded echo EF with the project's class rule. 55 of 837
labels change (counts by transition in the report). No label comes from a model.

**Retraining:**
- Every hospital-trained member, the inner OOF caches and the outer OOF of ECGF-MLP / ECGF-lin are retrained on
  these labels, in separate cache folders. The original-label caches are never read.
- EchoNext-trained and pixel members are unchanged (they never see hospital labels).

**Scoring:** SUPER and stack-BASE are re-run and scored against the echo-consistent labels.

**Reporting:** separately, as "labels corrected to their recorded echo EF", next to the original-label result.

## Amendment 5 (2026-11-06): STAGED-4, a staged four-class decision on the SUPER members
Declared before running. **Code:** `en_super.staged_fold`.

**Owner:** "try staging if 2 classes are always mismatching ... or one is affecting the other too much"; "its stricty 4
class using images only ... you can do staging if it comes under proper 4 class".

**The task stays four classes with the original labels.** No class definition changes, and no three-class result is
reported.

**STAGED-4:** uses the members SUPER selected in each fold and the same meta inputs.
- **Stage 1:** a multinomial LR separates Normal / {Mild or Moderate} / Severe.
  - C by meta-CV log-loss.
  - Class weights from the grid {0.25 … 4}² (Normal weight 1), chosen on meta-CV accuracy for this grouping, with
    every recall ≥ 0.18 (fallback 0.15).
- **Stage 2:** a binary LR (Mild vs Moderate), fitted only on training rows whose true label is Mild or Moderate.
  C by meta-CV log-loss.
- **Final four-class output:** Normal or Severe from stage 1; otherwise Mild / Moderate by a threshold on
  P(Moderate). The threshold is chosen on the meta-CV out-of-fold predictions to maximise four-class accuracy with
  every four-class recall ≥ 0.18 (fallback 0.15).

**All choices are made inside the outer training fold** (patient-grouped meta-CV).

**Reported:** like SUPER, scored against the original four-class labels. Test: STAGED-4 vs SUPER (corrected t, Holm
together with Amendment 4).

## Amendment 6 (2026-11-06): echo-trained models scored against the original labels
Declared before running.

**What:** in the echo-consistent-label run (Amendment 4 B), the outer predictions of SUPER, STAGED-4 and stack-BASE
are also scored against the **original** four-class labels.

**Why:** this measures whether cleaning the training labels with echo evidence helps the honest original-label result.
- The test labels are untouched.
- Only the training labels differ, and they come from each ECG's recorded echo EF, never from a model.

**Reporting:** a sensitivity result, labelled "trained on echo-consistent labels, scored on original labels". It is
never the headline.

## Amendment 7 (2026-11-06): SUPER-ORD, an ordinal meta-learner on the SUPER members
Declared before running. **Code:** `en_super.ordinal_fold`.

**Method:** the same members SUPER selected in each fold, and the same meta inputs. The meta-learner is Frank-Hall
ordinal.
- Three binary LRs estimate P(y > j), j = 0, 1, 2 (Normal < Mild < Moderate < Severe).
- Class probabilities are the successive differences, clipped at 10⁻⁶ and renormalised.
- One C, chosen by patient-grouped meta-CV log-loss.
- ULT decision rule on the meta-CV out-of-fold probabilities.

**Scoring:** four classes, original labels. Test: SUPER-ORD vs SUPER (corrected t, Holm with Amendments 4 and 5).

**Also checked and not used:** averaging a patient's ECGs at test time.
- Only 81 of 837 ECGs (9.7 %) belong to patients with more than one ECG.
- Only 34 % of those patients have the same label on all their ECGs.

## Amendment 8 (2026-11-06): FEATX features and Mild / Moderate-targeted members (SUPER-X)
Declared before running. **Code:** `featx.py`, `en_featx.py`, `en_super.py --pool X`.

**Owner:** "extract every possible feature from the dataset (image) ... mainly for mild and moderate".

### G5, new label-free features
73 features, none of them computed by MILDFEAT or the Phase 5 families, all from the same synthesised 12-lead median
beat (`featx.py`). Computed identically for the hospital ECGs (plain and harmonised H) and the EchoNext ECGs.

| Group | Features |
|---|---|
| Heterogeneity | QT dispersion; T-end SD; R-peak-time dispersion; Tpeak–Tend / QT |
| Loop shape (PCA of the 8 independent leads; Kors VCG) | QRS and T complexity; QRS and T planarity; QRS and T non-dipolar residuum; TCRT; TMD (V2–V6); VCG loop roundness and planarity |
| Beat-to-beat (10-s lead II strip) | RT-peak SD and CV; RT-variability index; T-amplitude SD, CV and alternation; QRS-amplitude CV |
| Per-lead QRS shape (all 12 leads) | R/(R+S); Q/R; Q duration; R-peak time. Also T symmetry (II, V2, V5) and scar-type counts |

### New members
**EchoNext-trained, fixed functions; variant H.** Fitted on the EchoNext train part; settings chosen on the plan's
80 / 20 split, as in Amendment 1.
- **EN-cls-all-X / EN-gbm-X:** as EN-cls-all / EN-gbm, with G5 added.
- **EN-MiMo-LR / EN-MiMo-GBM:** binary Mild vs Moderate, trained only on EchoNext Mild and Moderate rows, on
  [G1‖G2‖G3‖G4‖G5] (LR) and [G2‖G3‖G4‖G5] (GBM). Output: P(Mild), P(Moderate).
- **EN-mu-mid:** ridge EF on EchoNext rows with EF 25–60 (a mid-range regressor), used as an EF-band member.

**Hospital-trained, nested inner OOF on the Bd.splits inner folds.**
- **H-X-GBM:** four-class LightGBM on the plain [G2‖G3‖G4‖G5], 300 rounds.
- **H-MiMo-GBM:** binary Mild vs Moderate LightGBM on the same features. Each fit uses only the Mild / Moderate rows
  of its training data, and predicts all rows.

### SUPER-X
- **Pool:** Amendment 4's SUPER pool plus the 7 members above.
- **Method:** the same greedy pruning (at most 10 members, gain ≥ 0.002), meta-learner and decision rule, plus
  STAGED-4-X and SUPER-ORD-X on the same selected members.
- **Scoring:** original four-class labels only.
- **Test:** SUPER-X vs SUPER (corrected t, Holm with Amendments 4–7).
- **Descriptive only (never used for any choice):** univariate Mild-vs-Moderate AUROC of each G5 feature on the
  hospital ECGs.
