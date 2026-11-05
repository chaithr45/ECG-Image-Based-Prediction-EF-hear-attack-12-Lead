# Phase 4 claims register (started at handoff; extends `phase_3/reports/CLAIMS_REGISTER.md`)

| # | Claim | Status | Evidence |
|---|---|---|---|
| P4-1 | The four `LV_ECG_Dataset_*/*.xlsx` sheets contain no EF values, no echo dates and no patient IDs; they are file-copy logs | VALID | `PHASE4_DATA_AUDIT.md` §1 |
| P4-2 | Folder labels were produced by fuzzy NAME matching (no patient ID, no date window) with Teichholz EF and sex-specific thresholds 52 / 54 / 41 / 31 | VALID | §4 (recovered notebook code) |
| P4-3 | 93.5 % of folder labels agree with the class of the nearest-in-time linked echo; 22 ECGs have no linked echo consistent with their label | VALID | §5 |
| P4-4 | 91 ECGs are linked to echoes of different EF classes; the median EF spread across multiple links is 17 points | VALID | §5 |
| P4-5 | Within 7 / 14 / 30 days: 541 / 550 / 587 development ECGs (672 / 688 / 735 incl. test) | VALID | §6 |
| P4-6 | Large EF spreads across links reflect wrong-person matches | HYPOTHESIS | not verifiable without patient IDs |
| P4-7 | Label noise limits 4-class accuracy | HYPOTHESIS | to be tested in 4D–4G (cohort-restricted training/evaluation) |
| P4-8 | EF measurement uncertainty can be quantified from these data | **NOT SUPPORTED** | only 18 qualifying repeat pairs |
| — | All phase 1–3 claims keep their status | – | see `phase_3/reports/CLAIMS_REGISTER.md`; the phase-1 CV p < 0.001 stays RETRACTED |
