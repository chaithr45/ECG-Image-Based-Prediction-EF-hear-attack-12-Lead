# Phase 6A claims register

Labels as in `phase_6/PHASE6_CLAIMS_REGISTER.md`. Development-set CV only; the locked test is not used.

| # | Claim | Status | Evidence |
|---|---|---|---|
| P6A-1 | The four-class rule is: Normal M > 52 / F ≥ 54; Mild 41 ≤ EF ≤ 52 (M) / < 54 (F); Moderate 31 ≤ EF < 41; Severe < 31 (Teichholz); male 52.0 and female 53.9 are Mild | VALID | matching-notebook code (`classify_lv`), `ecg_ef/config.py`, Phase 4 audit |
| P6A-2 | The original fixed training split contains 307 Normal ECGs (≥ 270); each CV model is fitted on 245–247 Normal ECGs | VALID | `PHASE6A_PLAN.md` §2 |
| P6A-3 | No continuous EF value is verified: all come from fuzzy name matches without a patient identifier | VALID | Phase 4 audit; `results/label_configurations.json` |
