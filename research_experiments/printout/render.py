"""Draw 12-lead ECG signals as printouts in the same layout as the dataset's ECG PDFs (the inverse of ecg_ef.signals).

The geometry was measured from the vector drawings of development PDFs:
  * page 2296 x 1728 pt; pink grid from y = 547 to 1697: minor lines (1, .75, .75) every ~8.21 pt (1 mm), major lines
    (1, .5, .5) every ~41.1 pt (5 mm), border width 2 (line positions identical on all PDFs, reproduced exactly);
  * 25 mm/s and 10 mm/mV: 205.33 pt per second (4 x 2.5 s = 10 s from x = 75 to 2128) and 82 pt per mV;
  * rows I/II/III, aVR/aVL/aVF, V1-V3, V4-V6 in four consecutive 2.5-s columns (baselines y = 707 / 970 / 1233),
    10-s lead-II rhythm strip on baseline y = 1496; 1-mV calibration pulse at the right end of each row, column ticks;
  * every trace is one black polyline (width 2) through the 500 Hz samples, each rounded to whole points, with
    repeated points dropped -- exactly how the machine drew the originals (integer x and y, <= 3 points per x).

Because the trace is drawn the same way, `ecg_ef.signals.extract_signals` reads a rendered PDF back like an original.
The header carries no patient identity; it says the page was rendered from a digital signal and, when given, prints
age, sex and machine intervals in the layout `ecg_ef.header.parse_header` reads.

    python -m research_experiments.printout.render --npy WAVES.npy --fs 250 --units mV \
        [--meta META.csv --split train] [--format pdf,png] [--n 20] [--out data/private/printouts]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pymupdf

LEADS = ["I", "II", "III", "aVR", "aVL", "aVF", "V1", "V2", "V3", "V4", "V5", "V6"]
GRID = [["I", "aVR", "V1", "V4"], ["II", "aVL", "V2", "V5"], ["III", "aVF", "V3", "V6"]]
PAGE_W, PAGE_H = 2296.0, 1728.0
GRID_TOP, GRID_BOTTOM = 547.0, 1697.0
PINK_MINOR, PINK_MAJOR = (1.0, 0.75, 0.75), (1.0, 0.5, 0.5)
BLACK = (0.0, 0.0, 0.0)
X0 = 75.0
PT_PER_S = 616.0 / 3.0                                 # 205.33 pt/s = 25 mm/s
PT_PER_MV = 82.0                                       # 10 mm/mV
ROW_BASE = (707.0, 970.0, 1233.0, 1496.0)
COL_TICKS = (588.0, 1102.0, 1615.0, 2129.0)
FS = 500
SECONDS = 10.0
FONT, FONT_SIZE = "cobo", 24                           # Courier bold, as in the originals
ROOT = Path(__file__).resolve().parents[2]


# ---------------------------------------------------------------------------------------------------- signal handling
def to_500hz(sig: np.ndarray, fs: float) -> np.ndarray:
    """[12, T] at fs -> [12, 5000] at 500 Hz (first 10 s)."""
    from scipy.signal import resample_poly
    sig = np.asarray(sig, np.float64)
    if fs != FS:
        from fractions import Fraction
        f = Fraction(FS / fs).limit_denominator(1000)
        sig = resample_poly(sig, f.numerator, f.denominator, axis=1)
    n = int(SECONDS * FS)
    if sig.shape[1] < n:
        raise ValueError(f"need 10 s of signal, got {sig.shape[1] / FS:.2f} s")
    return sig[:, :n]


def _trace(v: np.ndarray, t0: float, base: float) -> list[tuple[float, float]]:
    t = t0 + np.arange(len(v)) / FS
    p = np.stack([np.round(X0 + t * PT_PER_S), np.round(base - v * PT_PER_MV)], 1)
    p[:, 1] = np.clip(p[:, 1], 0, PAGE_H)
    keep = np.r_[True, np.any(np.diff(p, axis=0) != 0, axis=1)]
    return [tuple(q) for q in p[keep]]


def _lines(shape, segs, color, width):
    for a, b in segs:
        shape.draw_line(a, b)
    shape.finish(color=color, width=width, lineCap=0, lineJoin=2, closePath=False)


# ---------------------------------------------------------------------------------------------------------- drawing
def _grid(page):
    xs_minor, ys_minor = np.ceil(-0.99 + np.arange(280) * 8.21755), np.ceil(546.34 + np.arange(141) * 8.21715)
    xs_major, ys_major = np.ceil(-0.95 + np.arange(56) * 41.086), np.ceil(546.45 + np.arange(29) * 41.07775)
    for xs, ys, col in ((xs_minor, ys_minor, PINK_MINOR), (xs_major, ys_major, PINK_MAJOR)):
        sh = page.new_shape()
        _lines(sh, [((x, GRID_TOP), (x, GRID_BOTTOM)) for x in xs] + [((0, y), (PAGE_W, y)) for y in ys], col, 1.0)
        sh.commit()
    sh = page.new_shape()
    sh.draw_rect(pymupdf.Rect(0, GRID_TOP, PAGE_W, GRID_BOTTOM))
    sh.finish(color=PINK_MAJOR, width=2.0)
    sh.commit()


def _marks(page):
    for r, b in enumerate(ROW_BASE):                    # one drawing per tick, as in the originals (a combined
        for x in (COL_TICKS if r < 3 else COL_TICKS[-1:]):   # path of > 20 segments would be read as a trace)
            for seg in (((x, b - 41), (x, b - 16)), ((x, b + 16), (x, b + 41))):
                sh = page.new_shape()
                _lines(sh, [seg], BLACK, 2.0)
                sh.commit()
    for b in ROW_BASE:                                  # 1-mV calibration pulse (82 pt), as drawn by the machine
        sh = page.new_shape()
        sh.draw_polyline([(2129, b), (2137, b), (2137, b - 82), (2158, b - 82), (2178, b - 82), (2178, b), (2186, b)])
        sh.finish(color=BLACK, width=2.0, lineCap=0, lineJoin=2, closePath=False)
        sh.commit()


def _text(page, x, y_top, s):
    page.insert_text((x, y_top + 24.5), s, fontname=FONT, fontsize=FONT_SIZE, color=BLACK)


def _header(page, info: dict | None, filter_note: str):
    sh = page.new_shape()
    _lines(sh, [((75, 176), (2272, 176))], BLACK, 2.0)
    sh.commit()
    info = info or {}
    _text(page, 75, 40, f"ECG {info.get('id', '')}".strip())
    if info.get("age") is not None and np.isfinite(info["age"]):
        _text(page, 75, 80, f"{int(round(info['age']))} Years")
    if info.get("sex") in ("Male", "Female"):
        _text(page, 75, 120, info["sex"])
    _text(page, 1300, 40, "Rendered from digital signal")
    _text(page, 1300, 80, "(not a machine printout)")
    y = 200                                             # label line, value line: the layout parse_header reads
    for col, keys in enumerate((("Rate", "RR", "PR"), ("QRSD", "QT", "QTcB", "QTcF"))):
        yy = y
        for k in keys:
            v = info.get(k)
            if v is None or not np.isfinite(v):
                continue
            _text(page, 75 + 330 * col, yy, k)
            _text(page, 75 + 330 * col + 150, yy, f"{int(round(v))}")
            yy += 40
    _text(page, 384, 1635, "Speed: 25 mm/sec")
    _text(page, 716, 1635, "Limb: 10 mm/mV")
    _text(page, 996, 1635, "Chest: 10.0 mm/mV")
    if filter_note:
        _text(page, 1587, 1635, filter_note)
        sh = page.new_shape()
        sh.draw_rect(pymupdf.Rect(1579, 1615, max(1872, 1595 + 14.4 * len(filter_note)), 1677))
        sh.finish(color=BLACK, width=2.0)
        sh.commit()


def render(sig500: np.ndarray, info: dict | None = None, filter_note: str = "") -> pymupdf.Document:
    """sig500: [12, 5000] in mV at 500 Hz, leads in LEADS order. Returns a one-page PDF document."""
    if sig500.shape != (12, int(SECONDS * FS)):
        raise ValueError(f"expected [12, {int(SECONDS * FS)}], got {sig500.shape}")
    doc = pymupdf.open()
    page = doc.new_page(width=PAGE_W, height=PAGE_H)
    _grid(page)
    _header(page, info, filter_note)
    _marks(page)
    n = int(2.5 * FS)
    for r in range(3):
        for c in range(4):
            lead = GRID[r][c]
            _text(page, COL_TICKS[c - 1] if c else X0, ROW_BASE[r] - 104, lead)
            sh = page.new_shape()
            sh.draw_polyline(_trace(sig500[LEADS.index(lead), c * n:(c + 1) * n], c * 2.5, ROW_BASE[r]))
            sh.finish(color=BLACK, width=2.0, lineCap=0, lineJoin=2, closePath=False)
            sh.commit()
    _text(page, X0, ROW_BASE[3] - 104, "II")
    sh = page.new_shape()
    sh.draw_polyline(_trace(sig500[1], 0.0, ROW_BASE[3]))
    sh.finish(color=BLACK, width=2.0, lineCap=0, lineJoin=2, closePath=False)
    sh.commit()
    return doc


def save(doc: pymupdf.Document, stem: Path, formats=("pdf",), dpi: int = 72):
    stem.parent.mkdir(parents=True, exist_ok=True)
    if "pdf" in formats:
        doc.save(str(stem.with_suffix(".pdf")), garbage=3, deflate=True)
    if "png" in formats:
        doc[0].get_pixmap(dpi=dpi).save(str(stem.with_suffix(".png")))


# ------------------------------------------------------------------------------------------------------------- input
def load_waveforms(path: Path, n_leads: int = 12) -> np.ndarray:
    """.npy of shape [N, T, 12], [N, 12, T] or [N, 1, T, 12] -> [N, 12, T]."""
    a = np.load(path, mmap_mode="r")
    a = a.reshape(a.shape[0], *[s for s in a.shape[1:] if s != 1])
    if a.ndim != 3 or n_leads not in a.shape[1:]:
        raise ValueError(f"cannot find a {n_leads}-lead axis in shape {a.shape}")
    return a if a.shape[1] == n_leads else np.swapaxes(a, 1, 2)


def looks_normalised(w: np.ndarray, k: int = 200) -> bool:
    """True when the leads look z-scored rather than in mV (every lead's SD close to 1)."""
    sd = np.asarray(w[:k], np.float64).std(axis=2)
    return bool(np.median(np.abs(sd - 1.0)) < 0.05)


def meta_info(row) -> dict:
    g = lambda k: float(row[k]) if k in row and row[k] == row[k] else None          # noqa: E731
    rate = g("ventricular_rate")
    return dict(id=str(row.get("ecg_key", "")), age=g("age_at_ecg"),
                sex={"male": "Male", "female": "Female"}.get(str(row.get("sex", "")).lower()),
                Rate=rate, RR=(60000.0 / rate if rate else None), PR=g("pr_interval"), QRSD=g("qrs_duration"),
                QTcB=g("qt_corrected"))


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--npy", type=Path, required=True)
    ap.add_argument("--fs", type=float, required=True, help="sampling rate of the file (Hz)")
    ap.add_argument("--units", choices=["mV", "uV"], default="mV")
    ap.add_argument("--leads", default=",".join(LEADS), help="lead order in the file")
    ap.add_argument("--meta", type=Path, help="CSV with one row per waveform (e.g. ecg_key, age_at_ecg, sex, ...)")
    ap.add_argument("--split", help="keep meta rows with this value in the 'split' column (file order)")
    ap.add_argument("--filter-note", default="", help="filter text for the footer box, e.g. 'F 0.5-40 Hz'")
    ap.add_argument("--format", default="pdf", help="pdf, png or pdf,png")
    ap.add_argument("--dpi", type=int, default=72, help="PNG resolution (72 = 1 px per pt = 2296 x 1728)")
    ap.add_argument("--n", type=int, help="render only the first n ECGs")
    ap.add_argument("--out", type=Path, default=ROOT / "data" / "private" / "printouts")
    ap.add_argument("--allow-normalised", action="store_true")
    a = ap.parse_args()

    w = load_waveforms(a.npy)
    order = [x.strip() for x in a.leads.split(",")]
    if sorted(order) != sorted(LEADS):
        raise SystemExit(f"--leads must name the 12 standard leads, got {order}")
    idx = [order.index(ld) for ld in LEADS]
    if looks_normalised(w) and not a.allow_normalised:
        raise SystemExit("every lead has SD ~ 1: the signals look normalised, not in mV, so the 10 mm/mV scale would "
                         "be wrong. Pass --allow-normalised to draw them anyway.")
    rows = None
    if a.meta is not None:
        import pandas as pd
        m = pd.read_csv(a.meta, index_col=0)
        if a.split:
            m = m[m["split"] == a.split]
        if len(m) != len(w):
            raise SystemExit(f"{len(m)} meta rows but {len(w)} waveforms: cannot pair them by order")
        rows = m.reset_index(drop=True)
    n = len(w) if a.n is None else min(a.n, len(w))
    fmts = tuple(f.strip() for f in a.format.split(","))
    scale = 1e-3 if a.units == "uV" else 1.0
    for i in range(n):
        info = meta_info(rows.iloc[i]) if rows is not None else {"id": str(i)}
        sig = to_500hz(np.asarray(w[i], np.float64)[idx] * scale, a.fs)
        save(render(sig, info, a.filter_note), a.out / f"ecg_{info['id'] or i}", fmts, a.dpi)
        if (i + 1) % 500 == 0:
            print(f"{i + 1}/{n}", flush=True)
    print(f"rendered {n} ECGs -> {a.out}")


if __name__ == "__main__":
    main()
