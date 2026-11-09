# Final report: what changed and why

This is the team's Overleaf version of report 45 (57 pages, uploaded 8 November 2026) merged with all later results.
The output is 66 pages (LaTeX/PDF), with an editable Word version (67 pages) and the Overleaf project.
Every result in it comes from the models trained to (nearly) 100 % training accuracy. There are no earlier results and no comparison with them (see Section 3).

## 1. Your edits: kept

- **Title.** Kept, with one grammar fix: "… for Early Heart Failure Detection **using** Deep Learning" (it read "… **and** Deep Learning").
- **People.** Students and roll numbers (Krithika Reddy Jakkula 23ECB0A21, Cheruku Vikas Goud 23ECB0A32, Monika Varathur 23ECB0F02); guide Prof. L. Anjaneyulu, Professor.
- **Dates.** Approval page 19/11/2026; certificate 9th November 2026.
- **Front-matter removals.** No co-supervisor line, no letter-head note, no index terms.
- **Data-preparation work:**
  - objective 1 and the new contribution bullet;
  - Sections 3.2.1–3.2.6 (collection, Excel cleaning, RapidFuzz matching, LVEF labels, class folders, partitioning);
  - the "class-wise folders" label in Figure 3.1.

  The facts are unchanged; the wording is tightened.

## 2. Corrections: please read these

1. **EchoNext "Mild only" (Chapter 5 and abstract) was not correct, so it was reverted.** Your version said the EchoNext models were trained only on Mild cases. In fact they were trained on all 4,827 EchoNext ECG–echo pairs, from every EF class. A model that only ever sees Mild could not give the results in the report: four-class accuracy of up to 0.539 on the hospital ECGs (Table 5.1) and a Mild-against-Moderate model (0.586). If your guide wants a Mild-only EchoNext study, it has to be run as a new experiment on the laptop, and the text can then describe that.
2. **Literature review (Section 2.3).** The sentences citing ECGFounder, ECG-FM, EchoNext and the earlier 1-D networks ([21]–[26]) had been deleted, but those references were still in the list. A reference that is never cited is an examiner red flag, so the sentences are restored.
3. **Text fixes.**
   - A stray "“‘latex" before Chapter 5 is removed.
   - "categorey" is corrected.
   - The research-gap line "signals for mild ,we need more dataset ." is rewritten as a full sentence.
   - Four words split across lines by the PDF conversion are rejoined: augmentation, corrected, configuration, feasible.
   - The ConvNeXt-Atto row of Table 3.3 is repaired: it showed "(10 )" with the "−3" on a stray row of its own.
4. **AI-assistance declaration (item f) is kept.** Your version deleted it. Large parts of the code, experiments and text were produced with an AI assistant. Item d of your own declaration says all sources are credited, and most institutes, NIT Warangal included, treat undisclosed AI-generated text as an academic-integrity matter. Removing the line would make the declaration inaccurate. If your guide or the department gives a different instruction, that is their call, but please check with them before deleting it.
5. **Section 1.5 (organisation of the report) is restored.** It describes the seven chapters of the report.
6. **Running headers** ("Chapter N. Title" at the top of each page) follow the NITW template. Your compiled PDF had none. If that was intentional, remove the `\fancyhead` lines in `nitw-report.sty`.

## 3. Results: only the models trained to 100 % training accuracy (this version)

At your request, the report now contains **only** the results of the models trained, for as many epochs as they needed, until their training accuracy reached (nearly) 100 %.
- The earlier early-stopped results are gone, and so is any comparison with them. The former comparison chapter is removed, so the report has **seven chapters**: Chapter 7 is now Summary and Conclusions.
- **Sorting.** Every results table is sorted by **training accuracy, highest first**. Models with the same training accuracy are sorted by held-out (test) accuracy. Every results table has a training-accuracy column.

**Headline results now (Table 6.1, sorted by training accuracy):**

| Result | Model | Training acc. | Held-out acc. [95 % CI] |
|---|---|---|---|
| Best in the 60s | Image-only triage (48 % of patients answered) | 1.000 | 0.696 [0.630, 0.762] |
| Best single model | E2 LightGBM | 1.000 | 0.557 [0.517, 0.597] |
| Best image-only model | ENH-vote | 1.000 | 0.549 [0.513, 0.585] |
| Best on clean labels | NFS (264 clean ECGs) | 0.985 | 0.694 |
| Best on all patients | SUPER-TOP10-ORD (with EchoNext members) | 0.748 | 0.558 [0.530, 0.587] |
| Best staged model | STAGED-4 (with EchoNext members) | 0.722 | 0.548 [0.516, 0.581] |

**What changed, chapter by chapter:**
- **Abstract and Chapter 1 (objectives, contributions, organisation):** current numbers only. Table 2.1's "This work" row also has the current numbers.
- **Chapter 3:**
  - The training settings now describe the train-to-100 % rules.
  - Table 3.3 and Figure 3.2 show the current training against validation accuracy, and the training curves.
  - Table 3.4 (transfer learning) has current numbers.
  - New Table 3.5 shows how the main models are built, and new Table 3.6 gives the training settings of every model family.
  - Table 3.2 no longer lists the networks that were not retrained.
- **Chapter 4:**
  - Tables 4.1–4.7 are current.
  - Fixed-split networks: Table 4.2.
  - Main models: Table 4.3, with macro-F1, Mild F1, balanced accuracy, heart-failure screen AUC and recall.
  - Phase 5: Table 4.4.
  - EF regression and fine-tuning: Table 4.5.
  - Label studies: Table 4.6.
  - Image-only models: Table 4.7.
  - Figure 4.1 is new.
- **Chapter 5:**
  - EchoNext-trained models: Table 5.1.
  - Ensembles: Table 5.2.
  - Pixel models: Table 5.3.
  - Figure 5.1: all ensembles and variants with their confidence intervals.
- **Chapter 6:**
  - New headline table (Table 6.1).
  - All 79 classification models sorted by training accuracy (Tables 6.2–6.4).
  - Every figure (6.1–6.8) is rebuilt from the current models: confusion matrices, training against held-out accuracy, screen AUC, results by population, accuracy against coverage, Mild recall against accuracy, the EF-precision simulation, and error rate against distance from the class limits.
- **Chapter 7 (Summary and Conclusions):** current numbers only.

**Removed:**
- the models that were not trained to 100 %: the LSTM family, ConvNeXt-Tiny / EfficientNet-B2 and the other EchoNext-pre-trained pixel CNNs;
- the earlier analyses that were not repeated (ESC 2021 class limits, learning curve, ECG-FM, layer choice);
- the locked-test result, because the locked test set is not used in this report.

**New computation for this version:** the screen AUC, the confidence curve and the error-by-distance figure needed per-patient held-out probabilities, which the first run had not kept. The cross-validation models were therefore rerun with the same seeds and folds (`train100/train100_scores.py`). The rerun reproduced every held-out confusion matrix exactly, and only aggregates were written to the repository.

## 4. Plagiarism

**Done in this version:**
- **Literature review rewritten.** The prose of Chapter 2 (Sections 2.1–2.4) is in new wording, so no sentence follows a paper's abstract.
- **Citations kept and restored.** Every number taken from the literature keeps its citation, and the deleted citations are back.
- **No unattributed quotations.** There are no verbatim quotations, and Table 2.1 states that its numbers are as printed in each cited abstract.
- **Methods and results are the project's own work and description.**

**Not possible here:** running Turnitin or iThenticate. Please run the institute's similarity check before submission. Expect the reference list, the Table 2.1 data and standard method names to match other papers; that is normal and not plagiarism.

## Files

All of the following are in this folder:
- `Report_Final_NITW.pdf`
- `word/Report_Final_NITW.docx`
- `Report_Final_NITW_overleaf.zip`, which contains `latex/`
- this note
