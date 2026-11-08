# Final report: what changed and why

This is the team's Overleaf version of report 45 (57 pages, uploaded 8 November 2026) merged with all later results.
The output is 83 pages (LaTeX/PDF), with an editable Word version (83 pages) and the Overleaf project.
The results book (`TRAIN100_results_book.pdf`) is now fully included in Chapter 7 (see Section 3 below).

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

1. **EchoNext "Mild only" (Chapter 5 and abstract) was not correct, so it was reverted.** Your version said the EchoNext models were trained only on Mild cases. They were trained on all 4,827 EchoNext ECG–echo pairs, from every EF class. A model that only ever sees Mild could not produce the results the report gives: EF ≤ 45 % detection (AUROC 0.846), Normal against Mild (0.768) and an EF estimate (r = 0.62). The report must describe what was actually done. If your guide wants a Mild-only EchoNext study, it has to be run as a new experiment on the laptop, and the text can then describe that.
2. **Literature review (Section 2.3).** The sentences citing ECGFounder, ECG-FM, EchoNext and the earlier 1-D networks ([21]–[26]) had been deleted, but those references were still in the list. A reference that is never cited is an examiner red flag, so the sentences are restored.
3. **Text fixes.**
   - A stray "“‘latex" before Chapter 5 is removed.
   - "categorey" is corrected.
   - The research-gap line "signals for mild ,we need more dataset ." is rewritten as a full sentence.
   - Four words split across lines by the PDF conversion are rejoined: augmentation, corrected, configuration, feasible.
   - The ConvNeXt-Atto row of Table 3.3 is repaired: it showed "(10 )" with the "−3" on a stray row of its own.
4. **AI-assistance declaration (item f) is kept.** Your version deleted it. Large parts of the code, experiments and text were produced with an AI assistant. Item d of your own declaration says all sources are credited, and most institutes, NIT Warangal included, treat undisclosed AI-generated text as an academic-integrity matter. Removing the line would make the declaration inaccurate. If your guide or the department gives a different instruction, that is their call, but please check with them before deleting it.
5. **Section 1.5 (organisation of the report) is restored.** The report now has eight chapters, and the reader needs the new Chapter 7 signposted.
6. **Running headers** ("Chapter N. Title" at the top of each page) follow the NITW template. Your compiled PDF had none. If that was intentional, remove the `\fancyhead` lines in `nitw-report.sty`.

## 3. New results added (all models trained to ~100 % training accuracy)

- **New Chapter 7, "Training Every Model to 100 % Training Accuracy"** (Tables 7.1–7.15, Figures 7.1–7.9):
  - the rules of the experiment;
  - how each main model is built: inputs, layers, parameters, epochs;
  - training against held-out accuracy for every model;
  - macro-F1 and Mild-F1 before and after;
  - the confusion matrices of all twelve main models;
  - the image and signal networks;
  - the variants of every phase;
  - the label studies;
  - the three best ensembles retrained with EchoNext on the laptop;
  - what was not retrained;
  - where to stop training;
  - **Section 7.9 "Detailed results"**, which brings in every remaining table and chart of the results book:
    - Table 7.11: all twelve main models ranked by held-out accuracy, with macro-F1, Mild F1, balanced accuracy and the recall of each class;
    - Figures 7.5–7.7: accuracy / macro-F1 / Mild F1 side by side, every model against its original, and the recall of each class;
    - Figures 7.8–7.9: the fixed-split networks and the variants of every phase, original against trained to 100 %;
    - Table 7.12: the members of the three EchoNext ensembles from the laptop run;
    - Tables 7.13–7.15: the training accuracy and epochs of every model.
  - Two earlier tables gained columns from the results book: Table 7.5 (fixed split) now lists layers, parameters and the best validation accuracy reached at any epoch; Table 7.6 (variants) now lists macro-F1 and Mild F1.
- **The report's main results are unchanged.** Chapters 4–6 and Table 6.1 still report the original, early-stopped models, because on new patients they are as good or better in almost every case (the two small gains, LGBM-FEAT and variant E2, are inside the 95 % confidence interval). The trained-to-100 % numbers are in Chapter 7 and in one extra column of Table 6.1.
- **Elsewhere in the report:**
  - Table 6.1 has a new column for each headline model trained to 100 %.
  - The abstract has one new paragraph.
  - The objectives, the contributions and Section 3.5.3 each gain a sentence about the experiment.
  - The conclusions gain an item about it.
- **How test (held-out) accuracy changed:**

  | Model | Original | Trained to ~100 % |
  |---|---|---|
  | Best ensemble, SUPER-TOP10-ORD | 0.587 | 0.558 |
  | IMG-C3 accuracy (macro-F1) | 0.558 (0.441) | 0.532 (0.416) |
  | C3 | 0.561 | 0.533 |
  | Triage | 0.694 on 56 % of patients | 0.696 on 48 % |
  | ResNet-18 (validation) | 0.524 | 0.423 |

  The locked test was not rescored.

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
