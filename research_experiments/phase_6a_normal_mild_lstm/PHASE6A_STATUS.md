# Phase 6A status (Normal/Mild boundary + LSTM study)

| Date | Step | Outcome |
|---|---|---|
| 2026-11-05 | Label rule and cohort facts | Exact rule recovered from the matching notebook's code; class, patient and fold distributions documented (`PHASE6A_PLAN.md` §1–2, `results/label_configurations.json`). |
| 2026-11-05 | Experiment A design | 270 is below the original training split's 307 Normal ECGs but above the 245–247 Normal ECGs each CV model is fitted on; A270 raises inner-train Normal to exactly 270 in every fold by moving whole Normal patients from inner-val (23–25 ECGs per fold). |
| 2026-11-05 | Label configurations | B2 (EF ≥ 50 % → Normal, consistent links only): 27 Mild ECGs / 27 patients relabelled. B3: 45 uncertain Normal/Mild ECGs (44 patients) excluded from inner-train only. |
| 2026-11-05 | Matching / data check (owner request) | Uploaded sheets identical to the ones used; linking, EF values and labels verified on development ECGs; found and quantified a day/month swap in the 2021 sheet's ECG-date column (C6A-1): it changes interval-based counts and cohort G, not labels, EF targets or the 6A configurations. |
| 2026-11-05 | Existing model under B1 / A270 / B2 / B3 | Done; B1 reproduces the stored C1 / C3 exactly. |
| 2026-11-05 | Existing model: A270 / B2 / B3 results | No configuration improves accuracy on the original labels (all Holm p 1.00); B2's apparent gain is a definition effect; B2 / B3 reduce Mild recall for C3 (interim comparison committed; final comparison after the LSTM runs). |
| 2026-11-05 | 80 % feasibility (owner request) | Four-class 80 % needs r ≈ 0.975 (have 0.66): not attainable with these data; selective prediction tops out at ≈ 0.70. Binary Normal-vs-reduced EF reaches 0.775 (AUROC 0.83) from existing models. |
| 2026-11-05 | Handover | Cloud LSTM grid stopped (L4 29/45, L1 1/45 folds; not committed). Open work (LSTM L0-L4 grid, nested selection, LSTM-fixed under A270 / B2 / B3, conditional CNN-LSTM / fusion, final comparison and report) continues on the owner's GPU laptop. |
| 2026-11-05 | Pre-registration | `PHASE6A_PLAN.md` + SHA-256 written before any Phase 6A model run. |

## Expected compute
| Stage | Estimate |
|---|---|
| Existing best model under B1 / B2 / B3 / A270 (4 parallel single-thread processes; nested inner OOF re-fitted for B2 / B3) | ≈ 20–30 min |
| LSTM L0–L4, 3 seeds × 3 repeats (45 fold fits each; 4 parallel jobs) | ≈ 3–4 h in total (timed: 2–12 s per epoch) |
| LSTM-fixed under A270 / B2 / B3 | ≈ 0.5–1.5 h each |
| CNN-LSTM, fusion (conditional) | ≈ 1–2 h |
