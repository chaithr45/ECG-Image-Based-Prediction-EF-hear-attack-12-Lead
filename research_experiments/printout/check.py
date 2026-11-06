"""Check the renderer against development PDFs (never the locked test set).

For n development ECGs: digitised signals (data/signals.npz) -> render -> compare with the original PDF's drawing
(grid, ticks, pulses, label positions, trace points) and read the rendered PDF back with ecg_ef.signals.
Prints aggregates only; temporary PDFs go to data/private/printout_check/ and are deleted.

    python -m research_experiments.printout.check [--n 100]
"""
from __future__ import annotations

import argparse
import tempfile

import numpy as np
import pandas as pd
import pymupdf

from ecg_ef.header import parse_header
from ecg_ef.signals import extract_signals
from research_experiments.printout.render import GRID, LEADS, ROOT, render

TMP = ROOT / "data" / "private" / "printout_check"


def _drawn(pdf):
    """Return (trace polylines by (row, col), other black items, pink line coordinates, lead-label boxes)."""
    pg = pdf[0]
    tr, other, pink = {}, [], {"minor": [], "major": []}
    for d in pg.get_drawings():
        c = d.get("color")
        if not c:
            continue
        c = tuple(round(x, 2) for x in c)
        if max(c) <= .3 and len(d["items"]) > 20:
            it = d["items"]
            p = np.array([(it[0][1].x, it[0][1].y)] + [(i[2].x, i[2].y) for i in it])
            row = int(np.argmin(np.abs(np.array([707, 970, 1233, 1496]) - np.median(p[:, 1]))))
            col = -1 if row == 3 else int(np.clip(np.searchsorted([75, 588, 1102, 1615], np.median(p[:, 0]), "right") - 1, 0, 3))
            tr[(row, col)] = p
        elif max(c) <= .3 and 547 < d["rect"].y0 < 1600:              # ticks + pulses (not the footer box)
            for i in d["items"]:
                if i[0] == "l":
                    other.append((round(i[1].x), round(i[1].y), round(i[2].x), round(i[2].y)))
        elif c in ((1.0, .75, .75), (1.0, .5, .5)) and d.get("width") == 1.0:
            for i in d["items"]:
                if i[0] == "l":
                    key = "minor" if c == (1.0, .75, .75) else "major"
                    pink[key] += [("v", i[1].x)] if abs(i[1].x - i[2].x) < .01 else [("h", i[1].y)]
    names = set(LEADS)
    labels = sorted((s["text"].strip(), round(s["bbox"][0]), round(s["bbox"][1]))
                    for b in pg.get_text("dict", clip=pymupdf.Rect(0, 560, 2296, 1600))["blocks"]
                    for ln in b.get("lines", []) for s in ln["spans"] if s["text"].strip() in names)
    return tr, sorted(set(other)), pink, labels


def _on_grid(p):
    """Trace as y on every integer x (mean of the points drawn at that x)."""
    xs = p[:, 0].astype(int)
    u, inv = np.unique(xs, return_inverse=True)
    return u, np.bincount(inv, p[:, 1]) / np.bincount(inv)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=100)
    a = ap.parse_args()
    m = pd.read_csv(ROOT / "data" / "manifest.csv", usecols=["ecg_id", "source", "split"])
    dev = m[m.split != "test"].reset_index(drop=True)                     # development ECGs only
    z = np.load(ROOT / "data" / "signals.npz", allow_pickle=True)
    pos = pd.Series(np.arange(len(z["ecg_id"])), index=z["ecg_id"])
    rng = np.random.default_rng(0)
    pick = dev.iloc[np.sort(rng.choice(len(dev), min(a.n, len(dev)), replace=False))]
    TMP.mkdir(parents=True, exist_ok=True)

    err_l, err_r, cor, dy_med, dy_p95, same_pts, ppx_o, ppx_r = [], [], [], [], [], [], [], []
    geo_other = geo_labels = 0
    grid_dev = []
    ii_same = []
    for _, row in pick.iterrows():
        i = pos[row.ecg_id]
        leads, rhythm = z["leads"][i].astype(np.float64), z["rhythm"][i].astype(np.float64)
        full = np.zeros((12, 5000))
        for r in range(3):
            for c in range(4):
                full[LEADS.index(GRID[r][c]), c * 1250:(c + 1) * 1250] = leads[LEADS.index(GRID[r][c])]
        ii_same.append(float(np.max(np.abs(full[1, :1250] - rhythm[:1250]))))
        full[1] = rhythm
        doc = render(full)
        with tempfile.NamedTemporaryFile(suffix=".pdf", dir=TMP, delete=False) as f:
            tmp = f.name
        doc.save(tmp)
        L2, R2 = extract_signals(tmp)
        err_l.append(np.abs(L2 - leads).mean()); err_r.append(np.abs(R2 - rhythm).mean())
        cor.append(np.mean([np.corrcoef(L2[k], leads[k])[0, 1] for k in range(12) if leads[k].std() > 0]))
        with pymupdf.open(str(ROOT / row.source)) as orig:
            to, oo, po, lo = _drawn(orig)
        tn, on, pn, ln = _drawn(pymupdf.open(tmp))
        geo_other += oo == on
        geo_labels += [t for t, *_ in lo] == [t for t, *_ in ln] and \
            max(abs(p[1] - q[1]) + abs(p[2] - q[2]) for p, q in zip(lo, ln)) <= 2
        for kind in ("minor", "major"):
            for ax in ("v", "h"):
                o = np.array(sorted({v for a_, v in po[kind] if a_ == ax})); r_ = np.array(sorted({v for a_, v in pn[kind] if a_ == ax}))
                grid_dev.append(max(np.abs(o[:, None] - r_[None]).min(1).max(), np.abs(r_[:, None] - o[None]).min(1).max()))
        for key in to:
            if key not in tn:
                continue
            uo, yo = _on_grid(to[key]); un, yn = _on_grid(tn[key])
            common, io, inn = np.intersect1d(uo, un, return_indices=True)
            d = np.abs(yo[io] - yn[inn]); dy_med.append(np.median(d)); dy_p95.append(np.percentile(d, 95))
            so = set(map(tuple, to[key].astype(int))); sn = set(map(tuple, tn[key].astype(int)))
            same_pts.append(len(so & sn) / len(so))
            ppx_o.append(len(to[key]) / len(uo)); ppx_r.append(len(tn[key]) / len(un))
        doc.close()
        pymupdf.open(tmp).close()
        try:
            __import__("os").remove(tmp)
        except OSError:
            pass

    n = len(pick)
    print(f"development ECGs checked: {n}")
    print(f"lead II: column-1 cell vs first 2.5 s of rhythm strip, max |diff| median {np.median(ii_same):.4f} mV")
    print("GEOMETRY (rendered vs original PDF)")
    print(f"  ticks + calibration pulses identical: {geo_other}/{n}")
    print(f"  lead labels same text and position (<= 2 pt): {geo_labels}/{n}")
    print(f"  pink grid lines: max distance to the nearest original line {max(grid_dev):.1f} pt")
    print("TRACES (rendered vs original drawing, per trace)")
    print(f"  |dy| median of medians {np.median(dy_med):.2f} pt, 95th pct (median over traces) {np.median(dy_p95):.2f} pt"
          f"  (1 pt = {1 / 82 * 1000:.0f} uV)")
    print(f"  original points reproduced exactly: median {np.median(same_pts):.3f}")
    print(f"  points per x: original {np.mean(ppx_o):.3f}, rendered {np.mean(ppx_r):.3f}")
    print("READ BACK by ecg_ef.signals.extract_signals (vs data/signals.npz)")
    print(f"  12 leads: mean |error| {np.mean(err_l) * 1000:.1f} uV; rhythm: {np.mean(err_r) * 1000:.1f} uV; "
          f"mean correlation {np.mean(cor):.4f}")
    info = dict(id="demo", age=63, sex="Female", Rate=81, RR=741, PR=154, QRSD=90, QTcB=449)
    doc = render(np.zeros((12, 5000)), info, "F 0.5-40 Hz")
    tmp = TMP / "header_check.pdf"
    doc.save(str(tmp))
    h = parse_header(tmp)
    print("HEADER read by ecg_ef.header.parse_header:", {k: h.get(k) for k in ("age", "male", "Rate", "RR", "PR", "QRSD", "QT", "QTcB")})
    tmp.unlink()


if __name__ == "__main__":
    main()
