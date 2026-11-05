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
