# BEST RESULTS: the best model for each accuracy level

Every model was trained until its training accuracy reached (nearly) 100 %, and is scored on held-out patients in
development cross-validation: patient-grouped 5 folds × 3 repeats on the 837 development ECGs. The inputs are image
derived (the ECG printout), and the labels are the four classes Normal, Mild, Moderate and Severe. All files hold
aggregates only. The rows are sorted by training accuracy. These are the headline results of the final report.

| Folder | Best result | Training acc. | Held-out acc. [95 % CI] | On which patients |
|---|---|---|---|---|
| [`best_for_60s/`](best_for_60s/) | **Image-only triage model** | 1.000 (on the 58 % it answers) | **0.696** [0.630, 0.762] | the **48 %** of patients it answers |
| [`best_for_50s/`](best_for_50s/) | E2 LightGBM, best single model | 1.000 | 0.557 [0.517, 0.597] | every patient |
| [`best_for_50s/`](best_for_50s/) | ENH-vote, best without printed measurements | 1.000 | 0.549 [0.513, 0.585] | every patient |
| [`best_for_50s/`](best_for_50s/) | IMG-C3, the main image-only classifier | 1.000 | 0.532 [0.492, 0.572] | every patient |
| [`best_for_60s/`](best_for_60s/) | NFS, C3 recipe on clean labels | 0.985 | 0.694 | 264 clean, clear-cut ECGs |
| [`best_for_50s/`](best_for_50s/) | **SUPER-TOP10-ORD, highest all-patient score** | 0.748 | **0.558** [0.530, 0.587] | every patient |
| [`best_for_70s/`](best_for_70s/) | the triage model **frozen for new ECGs** (test kit) | 1.000 | to be measured on a new test set | the patients it answers |

**How to read the levels**
- **50s** = accuracy on *all* patients. This is the honest four-class number.
- **60s** = accuracy on the patients the triage model *answers*; the others are referred to echocardiography.
- **70s** = the same triage model, frozen and ready to be scored on new ECGs with the test kit.

**Each folder contains**
- `README.md`: what the model is, its numbers, what each file does, and what to say if asked;
- `best_for_XXs_confusion_matrix.png / .pdf` (50s and 60s);
- `best_for_XXs_code_N_<step>.py`: copies of the code, numbered in pipeline order. Each starts with a note naming the
  original file to run;
- `best_for_XXs_results.json` or the model card: the result files.

**Full report:** [`research_experiments/report_final_nitw/`](../research_experiments/report_final_nitw/) (PDF, Word,
Overleaf). The code that trains every model to 100 % is in
[`research_experiments/train100/`](../research_experiments/train100/).
