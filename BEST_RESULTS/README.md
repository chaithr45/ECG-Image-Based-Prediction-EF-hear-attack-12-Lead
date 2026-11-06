# BEST RESULTS: where to find the best model for each accuracy level

Every number below is from **image-derived inputs only** (the ECG drawing on the PDF; no printed machine text, age or
header) and uses the **original four-class labels** (Normal, Mild, Moderate, Severe). All files hold aggregates only.

| Folder | Best result | Accuracy | On which patients | Confusion matrix |
|---|---|---|---|---|
| [`best_for_50s/`](best_for_50s/) | Image-only four-class classifier **IMG-C3** | **55.3 %** (locked test) · 55.8 % (cross-validation) | **every** patient gets a class | yes (both) |
| [`best_for_50s/`](best_for_50s/) | Highest score: ensemble **SUPER-TOP10-ORD** | **58.7 %** [56.5, 61.0] (cross-validation) | every patient | not stored (recall / precision only) |
| [`best_for_60s/`](best_for_60s/) | **Triage model**, cross-validation | **69.4 %** [65.5, 73.6] | the **56 %** of patients it answers | yes (mean of 2 of 3 repeats, recomputed) |
| [`best_for_70s/`](best_for_70s/) | **Same frozen triage model**, locked test (scored once) | **74.7 %** [66.0, 83.7] | the **49.7 %** of unseen patients it answers | yes |

**How to read the levels**
- **50s** = accuracy on *all* patients. This is the honest four-class number; the ceiling found everywhere is about 0.59.
- **60s / 70s** = accuracy on the patients the triage model *answers*; the others are referred to echocardiography.
  The 60s and 70s are the **same model**: 69.4 % in cross-validation on the 837 development ECGs, then 74.7 % when
  scored once on 199 unseen locked-test ECGs.

**Each folder contains**
- `README.md`: what the model is, its numbers, what each file does, what to say if asked;
- `best_for_XXs_confusion_matrix.png / .pdf`;
- `best_for_XXs_code_N_<step>.py`: copies of the exact code, numbered in pipeline order (each starts with a note
  naming the original file to run);
- `best_for_XXs_results*.json`: the stored result files.

**Full reports**
- All branches, all models: [`research_experiments/final_report/main.pdf`](../research_experiments/final_report/main.pdf)
- Phase I report in the NITW template (PDF, Word, LaTeX): [`research_experiments/report45_nitw_format/`](../research_experiments/report45_nitw_format/)
- Phase 7 (triage, test kit): [`research_experiments/phase_7/PHASE7_FINAL_REPORT.md`](../research_experiments/phase_7/PHASE7_FINAL_REPORT.md)
- EchoNext / ensembles: [`research_experiments/public_ef/PUBLIC_EF_REPORT.md`](../research_experiments/public_ef/PUBLIC_EF_REPORT.md)
