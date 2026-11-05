# Phase 6 claims register

**Status labels:** VALID (directly measured) · CORRECTED (a previously stated value revised; C6-x) · RETRACTED ·
HYPOTHESIS (plausible, unverified) · ESTIMATE (projected, not measured).

Earlier registers stay authoritative for their phases (`phase_5/PHASE5_CLAIMS_REGISTER.md`,
`phase_4/reports/PHASE4_CLAIMS_REGISTER.md`, `phase_3/reports/CLAIMS_REGISTER.md`). All Phase 6 numbers are
development-set patient-grouped CV (837 ECGs / 794 keys); the locked test is not used.

| # | Claim | Status | Evidence |
|---|---|---|---|
| P6-1 | All remote branches (`main`, `claude/funny-gauss-udlmv0`, `research/ecg-ef-model-improvements`, `research/phase4-handoff`, `research/phase5`) form one linear history ending at `27cbdbd`; `research/phase6` starts there | VALID | `REPOSITORY_AUDIT_P6.md` §1 |
| P6-2 | In a fresh container the Phase 5 feature families are reproduced exactly (validation summary identical to the committed one) from development ECGs only | VALID | `results/restore_inputs.json` |
| P6-3 | Experiment C (extra unlabelled / matched ECGs) cannot be run: no extra PDFs or waveforms exist | VALID (owner-confirmed) | `PHASE6_PLAN.md` §4.3 |
| P6-4 | In a fresh container the per-ECG predictions of C1, C2, C3 and the C1 members (24 run files, incl. the nested inner OOF they depend on) are regenerated **byte-identically** (SHA-256 equal to the committed Phase 5 aggregates; identical per-fold α): C1 0.547, C2 0.568, C3 0.561 | VALID | `results/baseline_regeneration.json` |
| P6-5 | On this 4-core CPU, fine-tuning ECGFounder stage 6 (A1) or stages 5–6 (A2) under the pre-registered recipe costs about 0.3–1.3 h per 15-fold screen; even full-backbone fine-tuning would fit in about 6–10 h for Stages 1 + 2, so a GPU is not required | ESTIMATE (from measured step times) | `results/timing_A.json`; plan Amendment 1 |
| P6-6 | The frozen same-code reference R0 (joint class + EF head on header ‖ frozen ECGFounder) reproduces Phase 4: seed-combined EF r 0.659 (Phase 4: 0.66), accuracy 0.542 (Phase 4 multitask: 0.546); seed 42 alone r 0.652. ECGFounder activations recomputed from development PDFs reproduce the committed f6 to 1.4 × 10⁻⁶ | VALID | `results/experiments/P6R0-*`, `results/stage1_screen.json`, `results/ecgf_cache_validation.json` |
| P6-7 | Fine-tuning ECGFounder stage 6 (A1) or stages 5–6 (A2) with the pre-registered joint recipe does **not** raise the EF correlation: r 0.650 vs 0.652 for R0 (Δr −0.002; 15-fold corrected p 0.87; cluster-bootstrap 95 % CI on repeat 0 [−0.018, +0.006]); macro-F1 0.436 / 0.435. Both fail the Stage-1 screen (r ≥ 0.69 or macro-F1 ≥ 0.465) | VALID | `results/stage1_screen.json` |
| P6-8 | Under inner-val early stopping the selected epochs were 4–15 (stage 6 trainable for ≤ 10 epochs at lr 1e-4; stage 5 unfrozen at epoch 15 in A2), so A2 is practically identical to A1; inner-val EF r does not rise after unfreezing. Interpretation: with ~535 training ECGs per fold, adapting the backbone overfits before it adds EF information | VALID (selection facts) / HYPOTHESIS (interpretation) | fold histories in the A aggregates |
| P6-9 | Licence gate for alternative encoders: ECG-FM is MIT (eligible); HuBERT-ECG and ST-MEM are CC-BY-NC-4.0 (owner decision); ECG-JEPA and MERL are MIT but their weights are on Google Drive, which the environment's network policy blocks | VALID | `configs/encoders.json` |
