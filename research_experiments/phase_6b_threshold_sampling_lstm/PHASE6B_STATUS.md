# Phase 6B status

| Date | Step | Outcome |
|---|---|---|
| 2026-11-05 | Feasibility | All Normal sizes feasible: outer-training Normal 307-308 -> 300 / 270 / 240 / 200 by removing 7-8 / 37-38 / 67-68 / 107-108 Normal ECGs (whole patients) from inner-train; T50 moves 27 Mild -> Normal, T55 moves 10 Normal -> Mild, XUNC removes 45 ECGs from training. |
| 2026-11-05 | Pre-registration | PHASE6B_PLAN.md + SHA-256 committed (08d3126) before any 6B run. |
| 2026-11-05 | Handover | No 6B model run yet. All 6B runs (audit report, existing model under T55 / N-sizes, LSTM family, fusions, comparison, report) continue on the owner's GPU laptop session. |

## Already available from Phase 6 / 6A (re-used, not repeated)
- Existing model under ORIG (= 6A B1, reproduces C1 / C3 exactly), T50 (= B2), XUNC (= B3), A270X (= A270): no accuracy gain on the original labels (Holm p 1.00); B2 / B3 reduce C3 Mild recall.
- Matching / data check (6A `results/matching_data_check.json`), 80 % feasibility (6A `results/feasibility_80.json`).
- Phase 6: fine-tuning (A1 / A2), ECG-FM and ECGFounder + ECG-FM, input-fidelity D1 / D2 and the name-pair sensitivity all failed to improve; see `phase_6/PHASE6_CLAIMS_REGISTER.md`.
