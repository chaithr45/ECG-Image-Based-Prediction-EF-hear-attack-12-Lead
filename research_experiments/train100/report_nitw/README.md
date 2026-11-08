# Report 45, TRAIN100 edition (NIT Warangal template)

This is report 45 with one new chapter: **Chapter 7, Training Every Model to 100 % Training Accuracy**. The old Chapter 7,
Summary and Conclusions, is now Chapter 8.

## Files

| File | What it is |
|---|---|
| `Report_TRAIN100_NITW.pdf` | Compiled report, 69 pages. |
| `word/Report_TRAIN100_NITW.docx` | Editable Word version in the same layout. Install the fonts in `word/LatinModern_fonts` to see it exactly as the PDF. |
| `latex/` | Overleaf / LaTeX project. Compile `main.tex` with pdfLaTeX (`latexmk -pdf main.tex`). |
| `make_tables.py` | Writes Chapter 7 tables 7.2–7.6 from the aggregate results in `../results`. |
| `tools/word/` | The Word build (parse LaTeX → docx-js → native equations → layout calibration against the PDF). Paths inside point to the build machine. |

## Changes against report 45

**New Chapter 7:**
- method;
- development-CV models;
- fixed-split networks;
- variants of every phase;
- label studies;
- models not retrained;
- where to stop training.

It adds Tables 7.1–7.7 and Figures 7.1–7.3.

**Other changes:**
- Table 6.1 has a new column with each headline model trained to (nearly) 100 %.
- The abstract has one new paragraph.
- The chapter overview (Section 1.5) and the conclusions (Section 8.1) each mention the new chapter.
- The title says "TRAIN100 Edition".

**Fixed from report 45:**
- Table 3.3: "ConvNeXt-Atto (10$^{-3}$)" and its stray row.
- Words split by the PDF conversion: augmentation, corrected, configuration, feasible.

The rules of the experiment are in `../AMENDMENT_TRAIN100.md`, and a short summary is in `../TRAIN100_REPORT.md`.
