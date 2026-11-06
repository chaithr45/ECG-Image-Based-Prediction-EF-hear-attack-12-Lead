"""FEATX (PUBLIC_EF_PLAN Amendment 8): further label-free ECG-image features, aimed at Mild vs Moderate.

Built on the MILDFEAT synthesised simultaneous 12-lead median beat (same helpers, validated settings: QRS threshold
0.1, T end median, P threshold 0.5). Inputs: printout-format traces (12 leads x 2.5 s + 10-s lead II, mV, 500 Hz),
identical for hospital and EchoNext ECGs. Groups (none of them is computed by MILDFEAT):
  A heterogeneity: QT dispersion, T-end SD, R-peak-time dispersion, Tpeak-Tend / QT
  B loop / PCA (8 independent leads I, II, V1-V6; Kors VCG): QRS and T complexity (lambda2/lambda1), planarity
    (lambda3/lambda1), T-wave non-dipolar residuum (sum lambda4..8 / sum), TCRT, TMD (V2-V6), VCG loop roundness
  C beat-to-beat on the 10-s lead II strip (dominant beats): RT-peak SD / CV, QT-variability-type index
    log10[(var RT / mean RT^2) / (var RR / mean RR^2)], T-amplitude SD / CV / alternation, QRS-amplitude CV
  D per-lead QRS shape in all 12 leads: R/(R+S), Q/R, Q duration, R-peak time; T-wave symmetry (II, V2, V5);
    scar-type counts (Q >= 30 ms; R/(R+S) < 0.2 in I, aVL, V3-V6)
"""
import warnings

import numpy as np

from research_experiments.phase_6b_threshold_sampling_lstm import mildfeat as MF

MF.TEMODE, MF.PFRAC = "median", 0.5
FRAC = 0.1
LEADS, FS, R0 = MF.LEADS, MF.FS, MF.R0
ms = MF.ms
IND8 = ["I", "II", "V1", "V2", "V3", "V4", "V5", "V6"]


def _eig(M):
    M = M - M.mean(1, keepdims=True)
    U, s, _ = np.linalg.svd(M, full_matrices=False)
    return s ** 2, U, s


def _cos(a, b):
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return float(np.dot(a, b) / (na * nb)) if na > 0 and nb > 0 else np.nan


def featx(leads, rh):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        try:
            return _featx(np.asarray(leads, np.float32), np.asarray(rh, np.float32))
        except Exception:                                                # noqa: BLE001  (row stays missing)
            return {}


def _featx(leads, rh):
    tp = MF.templates(leads, rh)
    if tp is None:
        return {}
    T, peaks, dom = tp["T"], tp["peaks"], tp["dom"]
    on, off = MF.qrs_bounds(T, FRAC)
    if not (np.isfinite(on) and np.isfinite(off)):
        return {}
    on, off = int(on), int(off)
    qrsd = (off - on) / FS * 1000
    base = {n: MF.base_of(T[n], on) for n in LEADS}
    Y = {n: T[n] - base[n] for n in LEADS}
    Ld = {n: MF.qrs_lead(Y[n][on:off + 1], qrsd) for n in LEADS}
    rrs = int(np.median(np.diff(peaks)))
    tps, tes, tam = {}, {}, {}
    for n in LEADS:
        tps[n], tes[n], tam[n] = MF.t_end_lead(T[n], off, base[n], rrs)
    F = {}
    # ---- A heterogeneity
    te = np.array([tes[n] for n in LEADS if np.isfinite(tes[n])])
    if len(te) >= 6:
        F["qt_dispersion"] = float((te.max() - te.min()) / FS * 1000); F["tend_sd"] = float(te.std() / FS * 1000)
    tend = int(round(np.median(te))) if len(te) else min(off + ms(320), len(Y["I"]) - 1)
    rpt = np.array([Ld[n]["RPT"] for n in LEADS if np.isfinite(Ld[n]["RPT"])])
    if len(rpt) >= 6:
        F["rpt_dispersion"] = float(rpt.max() - rpt.min())
    tpe = [(tes[n] - tps[n]) / FS * 1000 for n in ("V2", "V3", "V4", "V5", "V6") if np.isfinite(tes[n])]
    if tpe and len(te):
        F["tpe_qt_ratio"] = float(np.median(tpe) / ((np.median(te) - on) / FS * 1000))
    # ---- B loop / PCA
    K = np.stack([Y[n] for n in IND8])
    if np.isfinite(K[:, on:tend + 1]).all() and tend - off > ms(80):
        lq, _, _ = _eig(K[:, on:off + 1]); lt, Ut, st = _eig(K[:, off:tend + 1])
        F.update(qrs_complexity=float(lq[1] / lq[0]), qrs_planarity=float(lq[2] / lq[0]),
                 t_complexity=float(lt[1] / lt[0]), t_planarity=float(lt[2] / lt[0]),
                 t_nondipolar=float(lt[3:].sum() / lt.sum()), qrs_nondipolar=float(lq[3:].sum() / lq.sum()))
        W = Ut[:, :2] * st[:2]                                          # lead vectors in the T-wave PCA plane
        pre = [IND8.index(n) for n in ("V2", "V3", "V4", "V5", "V6")]
        ang = [np.degrees(np.arccos(np.clip(_cos(W[i], W[j]), -1, 1))) for a_, i in enumerate(pre) for j in pre[a_ + 1:]]
        F["tmd_v2_v6"] = float(np.nanmean(ang))
        V = MF.KORS @ np.stack([Y[n] for n in MF.KORS_LEADS])           # Kors VCG (X, Y, Z)
        vq, vt = V[:, on:off + 1], V[:, off:tend + 1]
        mq, mt = np.linalg.norm(vq, axis=0), np.linalg.norm(vt, axis=0)
        eT = vt[:, int(np.argmax(mt))]
        big = mq >= 0.7 * mq.max()
        F["tcrt"] = float(np.nanmean([_cos(vq[:, i], eT) for i in np.flatnonzero(big)]))
        lqv, _, _ = _eig(vq); ltv, _, _ = _eig(vt)
        F.update(vcg_qrs_roundness=float(lqv[1] / lqv[0]), vcg_t_roundness=float(ltv[1] / ltv[0]),
                 vcg_qrs_planarity=float(lqv[2] / lqv[0]), vcg_t_planarity=float(ltv[2] / ltv[0]))
    # ---- C beat-to-beat on the rhythm strip
    rhf = MF.filt(rh)
    M = MF.beat_matrix(rhf, peaks)
    rr = np.diff(peaks) / FS * 1000
    rt, ta, qa = [], [], []
    for i in np.flatnonzero(dom):
        b = M[i]
        a_, z_ = R0 + ms(120), R0 + min(ms(480), int(0.7 * rrs))
        if z_ - a_ < ms(40) or not np.isfinite(b[R0 - ms(60):z_]).all():
            continue
        bl = np.nanmedian(b[R0 - ms(120):R0 - ms(60)]) if np.isfinite(b[R0 - ms(120):R0 - ms(60)]).all() else 0.0
        seg = b[a_:z_] - bl
        k = int(np.argmax(np.abs(seg)))
        rt.append((a_ + k - R0) / FS * 1000); ta.append(float(seg[k]))
        qa.append(float(np.ptp(b[R0 - ms(60):R0 + ms(60)])))
    if len(rt) >= 4:
        rt, ta, qa = np.array(rt), np.array(ta), np.array(qa)
        F.update(rt_sd=float(rt.std()), rt_cv=float(rt.std() / rt.mean()), t_amp_sd=float(ta.std()),
                 t_amp_cv=float(ta.std() / (abs(ta.mean()) + 1e-3)), t_alternation=float(np.mean(np.abs(np.diff(ta)))),
                 qrs_amp_cv=float(qa.std() / (qa.mean() + 1e-3)))
        if len(rr) >= 3 and rr.std() > 0 and rt.std() > 0:
            F["rt_variability_index"] = float(np.log10((rt.var() / rt.mean() ** 2) / (rr.var() / rr.mean() ** 2)))
    # ---- D per-lead QRS shape, T symmetry, scar-type counts
    for n in LEADS:
        L = Ld[n]
        if np.isfinite(L["R"]) and np.isfinite(L["S"]) and L["R"] + L["S"] > 0:
            F[f"rs_{n}"] = float(L["R"] / (L["R"] + L["S"]))
        if np.isfinite(L["Q"]):
            F[f"qr_{n}"] = float(L["Q"] / max(L["R"], 0.05)); F[f"qdur_{n}"] = float(L["Qdur"])
        if n not in ("V1", "V5", "V6") and np.isfinite(L["RPT"]):
            F[f"rpt_{n}"] = float(L["RPT"])
    for n in ("II", "V2", "V5"):
        if np.isfinite(tps[n]) and np.isfinite(tes[n]) and tes[n] > tps[n]:
            F[f"t_symmetry_{n}"] = float((tps[n] - (off + ms(40))) / (tes[n] - tps[n]))
    F["q30_count"] = float(sum(1 for n in LEADS if n != "aVR" and np.isfinite(Ld[n]["Qdur"]) and Ld[n]["Qdur"] >= 30))
    F["low_r_count"] = float(sum(1 for n in ("I", "aVL", "V3", "V4", "V5", "V6") if f"rs_{n}" in F and F[f"rs_{n}"] < 0.2))
    return {f"x:{k}": v for k, v in F.items()}
