# Phase 5 claims register

**Status labels:**
- **VALID**: directly measured;
- **CORRECTED**: a previously stated value revised;
- **RETRACTED**: unsupported or incorrect;
- **HYPOTHESIS**: plausible but unverified;
- **ESTIMATE**: projected, not measured.

Earlier phases keep their registers (`phase_4/reports/PHASE4_CLAIMS_REGISTER.md`, `phase_3/reports/CLAIMS_REGISTER.md`).

| # | Claim | Status | Evidence |
|---|---|---|---|
| P5-1 | The 56.8 % benchmark is a four-class development-CV accuracy on the same labels and the same 837 ECGs (not a test-set or three-class result) | VALID | `REPOSITORY_AUDIT.md` §4 |
| P5-2 | The 56.8 % model has Mild recall 0.08–0.16 and is not admissible under the Phase 5 recall-floor rule | VALID | `REPOSITORY_AUDIT.md` §4; `PHASE5_PLAN.md` §3 |
| P5-3 | Validation is grouped by a name-derived key; 29 near-identical name pairs exist under different keys (15 within development, 11 straddling development / locked test) | VALID (counts); same-person status unverifiable | `data_audit/grouping_audit.json` |
| P5-4 | There are no duplicate recordings (0 exact, 0 with r ≥ 0.98 rhythm-strip correlation) | VALID | `data_audit/grouping_audit.json` |
