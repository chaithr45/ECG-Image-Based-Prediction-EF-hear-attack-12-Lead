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
| P5-5 | The calibrated MLP (56.8 %) is reproduced exactly with stored predictions (0.5711 / 0.5687 / 0.5651; identical per-fold α) | VALID | `reports/PHASE5_BASELINE_REPORT.md` |
| P5-6 | Best admissible baseline accuracy is 0.560 (vote + EF-regression member). No baseline is significantly more accurate than the vote ensemble (smallest p 0.06) | VALID | `results/phase5_baseline_results.csv` |
| P5-7 | The accuracy gain of calibration (+0.021 to +0.025) comes with Mild recall 0.09–0.11 (below the floor); under the recall floor it vanishes (0.549) | VALID | 5A report |
| P5-8 | Mild errors go both ways (35 % → Normal, 27 % → Moderate) and are only slightly more boundary-concentrated than correct Mild predictions | VALID | `reports/PHASE5_ERROR_ANALYSIS.md` §1 |
| P5-9 | Missed Severe ECGs cluster near EF 31 (58 % within 5 points vs 33 % of detected) and have narrower QRS | VALID | 5B §8 |
| P5-10 | Patient keys with ≥ 2 ECGs are classified about 13–15 points less accurately; the possible split identities show no accuracy advantage (no evidence of leakage) | VALID | 5B §5 |
| P5-11 | "HF power" in digitised printouts reflects QRS sharpness, not noise (confounded quality indicator) | HYPOTHESIS | 5B §4 |
| P5-12 | Accuracy on the 50 % most confident ECGs is 0.66–0.68 (selective prediction; not the four-class target) | VALID | 5B §7 |
