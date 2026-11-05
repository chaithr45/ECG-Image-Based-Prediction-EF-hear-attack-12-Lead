# Phase 6B claims register

Labels as in `phase_6/PHASE6_CLAIMS_REGISTER.md`. Development-set CV only; the locked test is not used. Accuracy on the
original labels unless stated; CIs are corrected repeated-CV 95 % intervals.

| # | Claim | Status | Evidence |
|---|---|---|---|
| P6B-1 | 30 folder-Mild ECGs lie within 3 EF points of the sex-specific Normal/Mild boundary: 15 with matching uncertainty, 10 with a potentially unreliable EF, 5 high-confidence, < 5 genuinely difficult (descriptive); 10 further Mild ECGs have a nearest echo far above the boundary (link inconsistency) | VALID | `PHASE6B_AUDIT.md`, `results/audit6b.json` |
| P6B-2 | Removing 7-108 Normal ECGs from training (N300-N200) does not improve accuracy (C3 0.562 -> 0.553; Holm p 1.00); it shifts predictions towards the reduced-EF classes (C3 Mild recall 0.178 -> 0.220, Normal recall 0.846 -> 0.802) with unchanged macro-F1 | VALID | `PHASE6B_MODEL_COMPARISON.csv`, `plots/normal_size_curve.png` |
| P6B-3 | The T50 boundary (EF >= 50 % Normal) raises scores only through a definition effect (+0.007 to +0.019 for unchanged models); training on T50 adds −0.003 to +0.006 (Holm p 1.00) and lowers C3 Mild recall on the original labels (0.178 -> 0.142) | VALID | `results/phase6b_summary.json` (decomposition) |
| P6B-4 | The T55 boundary (EF >= 55 % Normal) has no material effect: definition effect −0.007 to 0.000, training effect −0.004 to +0.006 (Holm p 1.00) | VALID | same |
| P6B-5 | No LSTM-family model (LSTM-sel, LSTM-fixed, CNN-LSTM, CNN-LSTM + header, early fusion F1 / F2, late fusion F3 / F4) improves on C1 or C3; the standalone LSTMs are significantly worse (accuracy −0.09 vs C3, Holm p 0.016 / 0.045; macro-F1 −0.10 to −0.12, Holm p <= 0.011) | VALID | `results/phase6b_comparisons.csv` |
| P6B-6 | Adding header + ECGFounder f6 to an LSTM encoder (F1 / F2) recovers the existing performance (0.540 / 0.551 vs C1 0.547) but does not exceed it | VALID | same |
| P6B-7 | No configuration or definition gives a genuine improvement (Holm p < 0.05, same sign in all repeats, admissible, eligible) in any of the 224 tests | VALID | same |
| P6B-8 | Conclusion 1 (original labels): the best admissible eligible model is C3 (Phase 5 selected), 0.561 [0.520, 0.603]; 60 % is not reached | VALID | `PHASE6B_FINAL_REPORT.md` |
| P6B-9 | Conclusion 2 (alternative definition, own labels, not comparable with P6B-8): the C3 recipe trained on T55 reaches 0.567 [0.535, 0.598]; under T50 the C3 and MLP-fusion recipes miss the 0.15 Mild-recall floor (0.1498) and are not admissible; 60 % is not reached | VALID | same |
| P6B-10 | Training cost on an RTX 3060 laptop GPU: 0.26-0.74 min per 5-fold run for every LSTM-family network (17,668-198,820 parameters), inference 0.03-0.10 ms per ECG | VALID | `PHASE6B_MODEL_COMPARISON.csv` |
