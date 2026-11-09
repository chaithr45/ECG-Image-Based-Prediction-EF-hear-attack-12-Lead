# Word version of the final report

These scripts build `../../word/Report_Final_NITW.docx` from the LaTeX sources in `../../latex`. The Word file follows the
layout of the compiled PDF: same text, tables, figures, equations, captions and contents.

**Requirements:**
- Python 3 with `pymupdf`;
- Node.js with the `docx` package (`npm install docx`);
- LibreOffice, needed only for the optional calibration step.

## Build

1. Compile the LaTeX in a separate folder, for example `/tmp/build`. Copy `../../latex` there and run `latexmk -pdf main.tex`.
2. In a work folder holding the files of this folder, run:

```bash
export LATEX_DIR=/tmp/build
```

```bash
python3 parse.py $LATEX_DIR .
```

```bash
node build.js
```

```bash
python3 post.py report_final_raw.docx report_final.docx
```

The steps produce, in order:
- `parse.py`: `doc.json` (text, tables and captions) and `media/` (figure images);
- `build.js`: `report_final_raw.docx`;
- `post.py`: `report_final.docx`, with the Word equations and page numbers inserted.

## Optional: re-calibrate after large text changes

```bash
SOFFICE_CMD=soffice python3 calibrate.py 2
```

```bash
python3 pagecache.py
```

Then run `node build.js` and `post.py` again.
- `calibrate.py` renders the docx with LibreOffice and adjusts the spacing in `cal.json` so that page breaks follow the PDF.
- `pagecache.py` refreshes the page numbers in `pagecache.json` from that render, for the contents and the lists of figures and tables. Word also recomputes them when the document is opened.

## Files

| File | Purpose |
|---|---|
| `parse.py`, `build.js`, `post.py` | the build |
| `calibrate.py`, `pagecache.py` | optional layout calibration |
| `cal.json` | spacing calibration for the current report |
| `pagecache.json` | cached page numbers for the current report |
| `equations_omml.xml` | the report's equations in Word (OMML) format |
