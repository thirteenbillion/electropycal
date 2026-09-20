#!/usr/bin/env python3
"""Measure EIS / FSCV reference statistics from real in-vitro PSTrace sessions.

These are the numbers baked into ``electropycal.data.synthetic.REAL_INVITRO_STATS``, which the
synthetic demo generator uses so the demo trees look like real neurostring data instead of a
schematic. Re-run this if the reference sessions change, and update that dict with the output.

``--raw`` is required and takes a root holding ``<date>_<devicetype>_<testtype>/`` session
folders. There is deliberately no default: pointing it at the synthetic demo tree would be
circular, since that tree is generated from the very statistics this script measures. Give
it your own PSTrace export.

Usage:
    python scripts/measure_invitro_stats.py --raw /path/to/your/pstrace/export
"""

from __future__ import annotations

import argparse
import collections
import glob
import os
import re

import numpy as np


def _pct(v, p):
    return float(np.nanpercentile(np.asarray(v, float), p))


def measure_eis(root: str) -> dict:
    from electropycal.data.pstrace import read_pstrace

    n_freq, f_lo, f_hi, rs, rct, f_apex = [], [], [], [], [], []
    for p in sorted(glob.glob(os.path.join(root, "*", "*_eis_*.csv"))):
        for _key, d in read_pstrace(p).eis.items():
            f = np.asarray(d["freq"], float)
            a = np.asarray(d["z_real"], float)
            b = np.asarray(d["z_imag"], float)
            o = np.argsort(f)
            f, a, b = f[o], a[o], b[o]
            zm = np.hypot(a, b)
            if len(f) < 10 or not np.all(np.isfinite(zm)):
                continue
            n_freq.append(len(f)); f_lo.append(f[0]); f_hi.append(f[-1])
            rs.append(zm[-1]); rct.append(zm[0] - zm[-1]); f_apex.append(f[np.argmax(-b)])
    rct = [x for x in rct if x > 0]
    out = {"n_spectra": len(rs), "n_freq": int(np.median(n_freq)), "f_lo": _pct(f_lo, 50),
           "f_hi": _pct(f_hi, 50), "Rs": _pct(rs, 50), "Rct": _pct(rct, 50),
           "f_apex": _pct(f_apex, 50)}
    out["Cdl"] = 1.0 / (2 * np.pi * out["f_apex"] * out["Rct"])
    return out


def measure_fscv(root: str) -> dict:
    from electropycal.data.pstrace import read_pstrace

    n_sweep, v_lo, v_hi, slope, offset, loop = [], [], [], [], [], []
    sessions: dict = collections.defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(root, "*", "*_fscv_*.csv"))):
        if "stabilization" in p:
            continue
        m = re.search(r"([\d-]+)_fscv_(\d+)nM\.csv$", os.path.basename(p), re.I)
        ex = read_pstrace(p)
        per = collections.defaultdict(list)
        for (ch, _rep), d in ex.fscv.items():
            v = np.asarray(d["voltage"], float)
            i = np.asarray(d["current"], float)
            if len(v) < 300:
                continue
            n_sweep.append(len(v)); v_lo.append(v.min()); v_hi.append(v.max())
            per[ch].append((v, i))
            if m and int(m.group(2)) == 0:                       # 0 nM background shape
                h = len(v) // 2
                va, ia, vc, ic = v[:h], i[:h], v[h:], i[h:]
                A = np.polyfit(va, ia, 1)
                slope.append(A[0]); offset.append(A[1])
                loop.append(np.mean(np.abs(ia - np.interp(va, vc[::-1], ic[::-1]))))
        if m:
            sessions[(os.path.dirname(p), m.group(1))][int(m.group(2))] = {
                ch: (np.mean([a[0] for a in L], 0), np.mean([a[1] for a in L], 0))
                for ch, L in per.items()}

    v_ox, amp, fwhm = [], [], []
    for _key, cd in sessions.items():
        if 0 not in cd:
            continue
        for conc in sorted(c for c in cd if c > 0):
            for ch, (v, i) in cd[conc].items():
                if ch not in cd[0]:
                    continue
                v0, i0 = cd[0][ch]
                if len(v) != len(v0):
                    continue
                d = i - i0
                h = len(v) // 2
                va, da = v[:h], d[:h]
                w = (va > 0.4) & (va < 1.1)
                if w.sum() < 5:
                    continue
                k = int(np.argmax(da[w]))
                if da[w][k] <= 0:
                    continue
                v_ox.append(va[w][k]); amp.append(da[w][k])
                above = va[w][da[w] >= da[w][k] / 2]
                if above.size > 1:
                    fwhm.append(above.max() - above.min())
    return {"n_sweeps": len(n_sweep), "n_sweep": int(np.median(n_sweep)), "v_lo": _pct(v_lo, 50),
            "v_hi": _pct(v_hi, 50), "bg_slope": _pct(slope, 50), "bg_offset": _pct(offset, 50),
            "loop_uA": _pct(loop, 50), "n_peaks": len(v_ox), "v_ox": _pct(v_ox, 50),
            "peak_fwhm": _pct(fwhm, 50), "peak_sigma": _pct(fwhm, 50) / 2.355,
            "peak_uA_med": _pct(amp, 50), "peak_uA_p95": _pct(amp, 95)}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--raw", required=True,
                    help="root holding <date>_<devicetype>_<testtype>/ session folders")
    args = ap.parse_args(argv)
    print(f"reference root: {args.raw}\n")
    for name, fn in (("EIS", measure_eis), ("FSCV", measure_fscv)):
        print(f"--- {name} ---")
        for k, v in fn(args.raw).items():
            print(f"  {k:12s} {v:.6g}" if isinstance(v, float) else f"  {k:12s} {v}")
        print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
