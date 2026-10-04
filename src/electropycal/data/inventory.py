"""Raw-dataset inventory for dashboards / QC.

Walks a raw PSTrace directory and returns tidy tables: what was measured
(``index_raw``), channel survival from channeltests (``channel_survival``), and
EIS quality per channel (folded into ``index_raw``). Timepoints are elapsed days
since each device's first *signal* session (matching ``features.extract``), so
channeltests preceding the first signal get negative timepoints.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .pstrace import parse_filename, parse_folder, read_pstrace
from .quality import eis_quality, inductive_onset


def _devicetype_filter(devicetype):
    """Normalize a ``devicetype`` argument to a set of names, or ``None`` for "any".

    Accepts a single name or a collection, so callers that restrict extraction to several
    device types can compute the band over exactly those types rather than over the whole
    corpus (which would include the types they excluded).
    """
    if devicetype is None:
        return None
    if isinstance(devicetype, str):
        return {devicetype}
    return {str(t) for t in devicetype}


def recommended_band(root: str | Path, lo: float = 10.0, default_hi: float = 100_000.0,
                     devicetype=None,
                     onset_percentile: float = 10.0) -> tuple[float, float]:
    """Data-driven EIS analysis band ``(lo, hi)`` from the raw signal folders.

    ``hi`` is the highest measured frequency **strictly below** the
    ``onset_percentile``-th percentile of the inductive onsets (lowest ``f`` with
    ``Im(Z) >= 0``) across every replicate-averaged EIS spectrum. With the default
    ``onset_percentile=10``, the lowest-onset ~10% of channels have their onset
    **inside** the band and therefore **fail EIS.1** as genuinely poor electrodes,
    instead of a few early-onset outliers dragging the band down for everyone. Set
    ``onset_percentile=0`` for the strict minimum (every inductive channel stays
    capacitive in-band, the most conservative band). Falls back to
    ``(lo, default_hi)`` when no spectrum goes inductive in range. Pass
    ``devicetype`` (e.g. ``"neurostring"``, or a collection of names) to restrict which
    device types the onsets are taken over.

    Set ``extract_dataset(..., band=recommended_band(root))`` so discovery does not
    silently starve when the default 100 kHz upper bound sits above real onsets.
    """
    from ..features.extract import _avg_eis
    root = Path(root)
    _types = _devicetype_filter(devicetype)
    onsets: list[float] = []
    freqs: set[float] = set()
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        if _types is not None and fmeta["devicetype"] not in _types:
            continue
        for csv in sorted(folder.glob("*.csv")):
            nm = parse_filename(csv.name)
            if not nm or nm["signaltype"] != "eis":
                continue
            exp = read_pstrace(csv)
            for ch in exp.eis_channels:
                specs = [v for (c, _), v in exp.eis.items() if c == ch]
                fr, _zr, zi = _avg_eis(specs)
                freqs.update(float(x) for x in fr)
                on = inductive_onset(fr, zi)
                if np.isfinite(on):
                    onsets.append(on)
    if not onsets:
        return (float(lo), float(default_hi))
    pct = float(np.clip(onset_percentile, 0.0, 100.0))
    threshold = float(min(onsets)) if pct <= 0.0 else float(np.percentile(onsets, pct))
    grid = np.array(sorted(freqs))
    below = grid[grid < threshold]
    hi = float(below.max()) if below.size else float(threshold)
    return (float(lo), hi)


def band_retention_curve(root: str | Path, uppers=None, lo: float = 10.0,
                         devicetype=None) -> "pd.DataFrame":
    """How many channel-timepoints stay capacitive (pass EIS.1) at each candidate upper band edge.

    Quantifies the **retention** side of the band tradeoff (feature bandwidth vs data kept). A
    channel-timepoint whose inductive onset (lowest ``f`` with ``Im(Z) >= 0``) is ``on`` passes
    EIS.1 at upper edge ``U`` iff ``on > U`` (or it never goes inductive in range). So raising ``U``
    can only drop channel-timepoints, never add them. Pair this with the drift-alignment-vs-frequency
    view (does the high-frequency band carry *aligned* signal worth those dropped channels?).

    Returns a frame ``[upper_hz, kept, dropped_vs_lowest, kept_frac]`` over ``uppers`` (default: the
    measured frequency grid above ``lo``). Cheap: one EIS scan, no waveform/FSCV parsing.
    """
    from ..features.extract import _avg_eis
    root = Path(root)
    _types = _devicetype_filter(devicetype)
    onsets: list[float] = []
    freqs: set[float] = set()
    n_ct = 0
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        if _types is not None and fmeta["devicetype"] not in _types:
            continue
        for csv in sorted(folder.glob("*.csv")):
            nm = parse_filename(csv.name)
            if not nm or nm["signaltype"] != "eis":
                continue
            exp = read_pstrace(csv)
            for ch in exp.eis_channels:
                specs = [v for (c, _), v in exp.eis.items() if c == ch]
                fr, _zr, zi = _avg_eis(specs)
                freqs.update(float(x) for x in fr)
                n_ct += 1
                on = inductive_onset(fr, zi)
                if np.isfinite(on):
                    onsets.append(on)
    onset_arr = np.asarray(onsets, float)
    if uppers is None:
        grid = np.array(sorted(freqs))
        uppers = [float(f) for f in grid if f > lo]
    base = n_ct - int((onset_arr <= min(uppers)).sum()) if len(uppers) and n_ct else n_ct
    rows = []
    for U in uppers:
        dropped = int((onset_arr <= U).sum())                 # onset in-band -> fails EIS.1
        kept = n_ct - dropped
        rows.append({"upper_hz": float(U), "kept": kept,
                     "dropped_vs_lowest": base - kept,
                     "kept_frac": round(kept / n_ct, 3) if n_ct else float("nan")})
    return pd.DataFrame(rows)


def index_raw(root: str | Path, band: tuple[float, float] = (10.0, 100_000.0)) -> pd.DataFrame:
    """One row per (device, date, testtype, channel, signaltype, dose).

    Columns: ``device, date, timepoint, testtype, channel, signaltype, dose,
    n_replicates, eis_valid`` plus the per-check booleans ``eis_A`` (capacitive),
    ``eis_B`` (monotonic |Z|), ``eis_C`` (environment), all NaN for FSCV rows.
    """
    root = Path(root)
    parsed = []
    signal_dates: dict[str, set] = defaultdict(set)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta:
            continue
        for csv in sorted(folder.glob("*.csv")):
            nm = parse_filename(csv.name)
            if not nm:
                continue
            exp = read_pstrace(csv)
            blocks = exp.eis if nm["signaltype"] == "eis" else exp.fscv
            reps: dict[int, int] = defaultdict(int)
            for (ch, _) in blocks:
                reps[ch] += 1
            parsed.append((fmeta, nm, exp, dict(reps), str(csv)))
            if fmeta["testtype"] == "signal":
                signal_dates[nm["deviceid"]].add(fmeta["date"])

    d0 = {dev: min(dates) for dev, dates in signal_dates.items()}
    rows = []
    for fmeta, nm, exp, reps, path in parsed:
        dev, date = nm["deviceid"], fmeta["date"]
        tp = (date - d0.get(dev, date)).days
        for ch, nrep in reps.items():
            valid = a = b = c = np.nan
            if nm["signaltype"] in ("eis", "paired") and (ch, 0) in exp.eis:
                s = exp.eis[(ch, 0)]
                q = eis_quality(s["freq"], s["z_real"], s["z_imag"], band=band)
                valid, a, b, c = (bool(q["valid"]), bool(q["A_capacitive"]),
                                  bool(q["B_monotonic"]), bool(q["C_environment"]))
            rows.append({"device": dev, "date": date, "timepoint": tp,
                         "testtype": fmeta["testtype"], "channel": ch,
                         "signaltype": nm["signaltype"], "dose": nm["dose"],
                         "n_replicates": nrep, "eis_valid": valid,
                         "eis_A": a, "eis_B": b, "eis_C": c, "path": path})
    return pd.DataFrame(rows)


#: the checks that GATE ``overall_valid`` (EIS.1-3 + FSCV.1), in check order; this is the
#: exact gate ``features.extract.extract_dataset(acceptance="monotonic")`` applies.
QUALITY_CHECKS = ("eis_A", "eis_B", "eis_C", "fscv_monotonic")
#: all checks shown on the dashboard, including the informational per-concentration noise
#: floor (FSCV.2) and the peak-in-window diagnostic; neither gates discovery under the default
#: ``acceptance="monotonic"`` (the noise floor gates only under ``"monotonic+snr"``).
DISPLAY_CHECKS = ("eis_A", "eis_B", "eis_C", "fscv_monotonic", "fscv_noise", "fscv_peak_inwindow")
CHECK_CODE = {"eis_A": "E.1", "eis_B": "E.2", "eis_C": "E.3",
              "fscv_monotonic": "F.1", "fscv_noise": "F.2", "fscv_peak_inwindow": "F.pk"}
CHECK_NAME = {"eis_A": "EIS.1 capacitive", "eis_B": "EIS.2 |Z|-monotone",
              "eis_C": "EIS.3 environment", "fscv_monotonic": "FSCV.1 dose-monotonic",
              "fscv_noise": "FSCV.2 noise-floor", "fscv_peak_inwindow": "FSCV peak in-window"}
CHECK_FORMULA = {"eis_A": "Im(Z) < 0 at every in-band f",
                 "eis_B": "|Z| non-increasing with f (tol mono_tol)",
                 "eis_C": "Z' > 0 across band",
                 "fscv_monotonic": "r(log c, NormIpeak) >= monotonic_r_min",
                 "fscv_noise": "repeatability_snr = NormIpeak/std(reps) >= min_norm_snr (every c)",
                 "fscv_peak_inwindow": "V_peak inside PEAK_WINDOW (not edge-pinned) for majority of c"}
CHECK_LABEL = CHECK_NAME             # backward-compatible alias


def _qc_session(dev, date, entry, *, d0, band, P):
    """QC one (device, date) session -> (ch_rows, dose_rows). Independent of every other
    session (keeps all channels, no shared ref_grid), so it parallelizes with no determinism
    caveat: concatenating in sorted-session order reproduces the serial tables exactly."""
    from ..features.extract import _avg_eis, _avg_fscv
    from ..features.fscv import (PEAK_WINDOW, dose_response_corr, mean_vpeak, noise_floor,
                                 norm_ipeak, peak_at_edge, repeatability_snr, smooth_current)
    sw, sp = P["sw"], P["sp"]
    def _sm(cur):
        return smooth_current(cur, sw, sp) if sw else np.asarray(cur, float)
    ch_rows: list[dict] = []
    dose_rows: list[dict] = []
    tp = (date - d0[dev]).days
    eis_exp = read_pstrace(entry["eis"]) if "eis" in entry else None
    bg_exp = read_pstrace(entry["fscv"][0.0]) if 0.0 in entry["fscv"] else None
    dose_exps = {d: read_pstrace(p) for d, p in entry["fscv"].items()
                 if isinstance(d, (int, float)) and d > 0}
    eis_chs = set(eis_exp.eis_channels) if eis_exp is not None else set()
    bg_chs = set(bg_exp.fscv_channels) if bg_exp is not None else set()
    for ch in sorted(eis_chs | bg_chs):
        has_eis, has_fscv = ch in eis_chs, ch in bg_chs
        a = b = c = eis_valid = False
        z_rise = float("nan")
        if has_eis:
            specs = [v for (cc, _), v in eis_exp.eis.items() if cc == ch]
            fr, zr, zi = _avg_eis(specs)                       # replicate-average first
            q = eis_quality(fr, zr, zi, band=band, mono_tol=P["mono_tol"])
            a, b, c, eis_valid = (q["A_capacitive"], q["B_monotonic"],
                                  q["C_environment"], q["valid"])
            z_rise = float(q.get("z_rise_max", float("nan")))   # EIS.2 statistic (max frac |Z| rise)
        # FSCV dose-response on replicate-averaged cycles
        nips, conc_passes, r = [], [], float("nan")
        n_edge = 0                                   # edge-pinned doses among those used
        bg = _avg_fscv(bg_exp, ch, max_reps=P["max_reps"]) if has_fscv else None
        if bg is not None:
            bo = np.argsort(bg["voltage"])
            bg_cur = _sm(bg["current"])              # smooth the background once, before subtraction
            for dose, fexp in sorted(dose_exps.items()):
                reps = [v for (cc, _), v in sorted(fexp.fscv.items()) if cc == ch]
                if P["max_reps"] is not None:
                    reps = reps[:P["max_reps"]]
                sig = _avg_fscv(fexp, ch, max_reps=P["max_reps"])
                if sig is None:
                    continue
                v = sig["voltage"]
                sig_cur = _sm(sig["current"])
                bg_i = np.interp(v, bg["voltage"][bo], bg_cur[bo])
                nip = norm_ipeak(sig_cur, bg_i, v, method=P["peak_method"],
                                 require_interior=P["require_interior_peak"])
                nf = noise_floor(sig_cur, bg_i, v, window=P["nfw"])
                if not np.isfinite(nip):
                    continue
                vpk = mean_vpeak(sig_cur, bg_i, v, method=P["peak_method"])
                edge = peak_at_edge(vpk, PEAK_WINDOW, P["etol"])
                negative = bool(nip < 0)
                rep_nips = [norm_ipeak(_sm(rc["current"]), np.interp(rc["voltage"], bg["voltage"][bo],
                            bg_cur[bo]), rc["voltage"], method=P["peak_method"],
                            require_interior=P["require_interior_peak"]) for rc in reps]
                rep_nips = [x for x in rep_nips if np.isfinite(x)]
                rsnr = repeatability_snr(nip, rep_nips)          # NormIpeak / std across cycles
                # per-dose noise check is the REPRODUCIBILITY SNR (>= min_norm_snr); the RMS
                # floor (nip/nf) is kept as `rms_snr` for reference but no longer the pass rule.
                cpass = bool(np.isfinite(rsnr) and rsnr >= P["min_norm_snr"])
                dose_rows.append({"device": dev, "date": date, "timepoint": tp,
                                  "channel": int(ch), "concentration": float(dose),
                                  "NormIpeak_mean": float(np.mean(rep_nips)) if rep_nips else float(nip),
                                  "NormIpeak_std": float(np.std(rep_nips)) if len(rep_nips) > 1 else 0.0,
                                  "NormIpeak_avg": float(nip), "noise_floor": float(nf),
                                  "repeatability_snr": float(rsnr), "snr": float(rsnr),
                                  "rms_snr": float(nip / nf) if nf else float("nan"),
                                  "v_peak": float(vpk), "peak_at_edge": edge, "negative": negative,
                                  "conc_pass": cpass})
                # a negative NormIpeak is non-physical; drop it from the monotonic fit + edge
                # tally by default so this inventory tracks extract_dataset(drop_negative=True).
                if P["drop_negative"] and negative:
                    continue
                nips.append((float(dose), float(nip)))
                conc_passes.append(cpass)
                n_edge += int(edge)
        n_conc = len(nips)
        dr_range = float("nan")
        if n_conc >= 3:
            d = np.array([x[0] for x in nips]); y = np.array([x[1] for x in nips])
            r = dose_response_corr(d, y, method=P["mono_method"])
            dr_range = float(y.max() - y.min())          # NormIpeak dynamic range (amplitude diagnostic)
        fscv_monotonic = bool(np.isfinite(r) and r >= P["monotonic_r_min"])
        # F.2 (informational): every measured concentration clears the noise floor
        fscv_noise = bool(n_conc > 0 and all(conc_passes))
        # peak-window diagnostic (non-gating): True = V_peak resolved INSIDE the window for a
        # majority of doses; False = mostly edge-pinned (window likely wrong for this channel).
        fscv_peak_inwindow = bool(n_conc > 0 and n_edge / n_conc < 0.5)
        gating = {"eis_A": bool(a), "eis_B": bool(b), "eis_C": bool(c),
                  "fscv_monotonic": fscv_monotonic}
        complete = has_eis and has_fscv
        overall = complete and all(gating.values())     # noise floor + peak-edge do not gate
        reasons = ([] if complete else ["incomplete (missing " +
                   ("EIS" if not has_eis else "FSCV") + ")"])
        reasons += [CHECK_NAME[k] for k, ok in gating.items() if not ok]
        ch_rows.append({"device": dev, "date": date, "timepoint": tp, "channel": int(ch),
                        "has_eis": has_eis, "has_fscv": has_fscv, **gating,
                        "fscv_noise": fscv_noise, "fscv_peak_inwindow": fscv_peak_inwindow,
                        "n_peak_edge": n_edge, "eis_valid": bool(eis_valid),
                        "eis_z_rise": z_rise,
                        "n_conc": n_conc, "dose_response_r": r,
                        "dose_response_range": dr_range,     # amplitude: max-min NormIpeak (flags near-dead monotone channels)
                        "overall_valid": overall,
                        "fail_reasons": "; ".join(reasons) if reasons else ""})
    return ch_rows, dose_rows


def channel_quality_report(root: str | Path, band: tuple[float, float] | str = "auto",
                           devices: list[str] | None = None,
                           mono_tol: float = 0.10, peak_method: str = "direct",
                           max_reps: int | None = 3, min_norm_snr: float = 3.0,
                           monotonic_r_min: float = 0.6, mono_method: str = "pearson",
                           smooth_window: int | None = None, smooth_poly: int | None = None,
                           drop_negative: bool = True,
                           nonfaradaic_window: tuple[float, float] | None = None,
                           peak_edge_tol: float | None = None,
                           require_interior_peak: bool = False,
                           n_jobs: int = 1,
                           progress: bool = False):
    """Per ``(device, date, timepoint, channel)`` quality verdict + per-concentration doses.

    Re-runs the *same* primitives the feature extractor uses (:func:`data.quality.eis_quality`
    on replicate-averaged EIS, :func:`features.fscv.norm_ipeak` / ``noise_floor`` on
    replicate-averaged FSCV) but keeps **every** channel that was measured, including the
    ones that fail, so a dashboard can show what dropped and why. Only ``signal`` folders
    are considered; the channel roster for a device-timepoint is the union of its EIS
    channels and its ``0nM`` FSCV background channels.

    Like :func:`features.extract.extract_dataset`, the replicate-averaged signal and background
    cycles are Savitzky-Golay **smoothed** (``smooth_window`` / ``smooth_poly``; ``None`` = the
    electropycal defaults, smoothing **on**) *before* background subtraction, so ``overall_valid``
    here matches the set ``extract_dataset(acceptance="monotonic")`` keeps. Pass ``smooth_window=0``
    to reproduce the un-smoothed signal.

    **Averaging happens before every check**: EIS replicates are averaged
    (:func:`features.extract._avg_eis`) and each concentration's FSCV replicate cycles are
    averaged (``_avg_fscv``, first ``max_reps`` cycles) *before* quality is evaluated; the
    per-replicate NormIpeak spread is reported only as ``NormIpeak_std`` for display.

    **A channel-timepoint is ``overall_valid`` iff all of** EIS A/B/C **and** FSCV
    dose-monotonicity (log-concentration Pearson ``r ≥ monotonic_r_min``, ≥3 concentrations)
    **pass**: the exact gate ``features.extract.extract_dataset(acceptance="monotonic")``
    applies before discovery. The per-concentration reliability check (``conc_pass``) is the
    **reproducibility SNR** ``repeatability_snr = NormIpeak / std(NormIpeak across cycles) ≥
    min_norm_snr``, robust where the RMS ``noise_floor`` (residual-dominated, kept as ``rms_snr``)
    reads ~1× even for clear peaks. It is *informational* here (it only gates discovery under
    ``acceptance="monotonic+snr"``). Monotonicity ``r`` is computed on **all** doses' averaged
    NormIpeak, i.e. **before / independent of** the reliability check.

    **Non-paired data does not raise:** a channel with EIS but no FSCV (or vice-versa) gets
    ``has_eis``/``has_fscv`` flags and ``overall_valid=False`` with an "incomplete" reason.

    ``band`` is the EIS analysis window ``(lo, hi)`` in Hz, or ``"auto"`` (**default**) to set it
    from data via :func:`recommended_band`, the same data-driven default ``extract_dataset`` uses,
    so this inventory's EIS.1 verdicts match the featureset's. ``devices`` restricts the scan to those
    device ids (``None`` = every device at ``root``), so the heavy QC only runs on the selection.

    ``progress=True`` prints timestamped ``[HH:MM:SS]`` lines (one per device-timepoint session
    with a running elapsed/ETA) so the full parse + QC can be tracked (same style as
    ``discovery.runner.run_condition``). Default off; logging doesn't change the result.

    ``n_jobs`` (default **1**, serial) parses + QCs the device-timepoint sessions in parallel worker
    processes (joblib). The sessions are independent (every channel is kept; there is no shared EIS
    ``ref_grid`` as in :func:`~electropycal.features.extract.extract_dataset`), and rows are
    concatenated in the same sorted order, so **the tables are identical to the serial run**, only
    faster. ``-1`` uses all cores. The per-session live ETA prints only when serial.

    Returns ``(channel_tbl, dose_tbl)`` DataFrames.
    """
    if isinstance(band, str):
        if band != "auto":
            raise ValueError(f"band must be a (lo, hi) tuple or 'auto', got {band!r}")
        band = recommended_band(Path(root))
    # the heavy per-session primitives are imported inside _qc_session (the parallel worker);
    # channel_quality_report itself only needs the smoothing/window defaults to build P.
    from ..features.fscv import (FSCV_SMOOTH_POLY, FSCV_SMOOTH_WINDOW, NONFARADAIC_WINDOW,
                                 PEAK_EDGE_TOL)
    nfw = nonfaradaic_window or NONFARADAIC_WINDOW
    etol = PEAK_EDGE_TOL if peak_edge_tol is None else peak_edge_tol
    # Match extract_dataset's default raw-signal smoothing so this inventory mirrors exactly what the
    # feature extractor gates on. Passing None uses electropycal's defaults (smoothing ON);
    # pass smooth_window=0 to disable and reproduce the un-smoothed signal.
    sw = FSCV_SMOOTH_WINDOW if smooth_window is None else smooth_window
    sp = FSCV_SMOOTH_POLY if smooth_poly is None else smooth_poly
    root = Path(root)
    keep_devices = set(devices) if devices is not None else None   # None = every device at root

    # index signal files by (device, date): eis path + fscv {dose: path}
    index: dict[tuple[str, object], dict] = defaultdict(lambda: {"fscv": {}})
    signal_dates: dict[str, set] = defaultdict(set)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        for csv in sorted(folder.glob("*.csv")):
            nm = parse_filename(csv.name)
            if not nm:
                continue
            dev, date = nm["deviceid"], fmeta["date"]
            if keep_devices is not None and dev not in keep_devices:
                continue                                           # only process selected devices
            signal_dates[dev].add(date)
            if nm["signaltype"] == "eis":
                index[(dev, date)]["eis"] = csv
            elif nm["signaltype"] == "fscv":
                index[(dev, date)]["fscv"][nm["dose"]] = csv

    d0 = {dev: min(dates) for dev, dates in signal_dates.items()}
    ch_rows: list[dict] = []
    dose_rows: list[dict] = []
    import time as _time
    from datetime import datetime as _dt
    _items = sorted(index.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
    _t0 = _time.time()
    if progress:
        _pw = f" on {n_jobs} worker process(es)" if (n_jobs not in (0, 1) and len(_items) > 1) else ""
        print(f"[{_dt.now():%H:%M:%S}] channel_quality_report: parsing + QC over "
              f"{len(_items)} device-timepoint session(s){_pw}...", flush=True)
    P = {"mono_tol": mono_tol, "peak_method": peak_method, "max_reps": max_reps,
         "min_norm_snr": min_norm_snr, "monotonic_r_min": monotonic_r_min, "mono_method": mono_method,
         "drop_negative": drop_negative, "require_interior_peak": require_interior_peak,
         "sw": sw, "sp": sp, "nfw": nfw, "etol": etol}
    from .._parallel import map_sessions, should_parallelize
    _parallel = should_parallelize(n_jobs, len(_items))
    if _parallel:
        from joblib import delayed                  # loky processes (BLAS-capped); sessions independent
        _results = map_sessions(n_jobs, (
            delayed(_qc_session)(dev, date, entry, d0=d0, band=band, P=P)
            for (dev, date), entry in _items))       # joblib preserves order -> deterministic
        for _cr, _dr in _results:
            ch_rows.extend(_cr); dose_rows.extend(_dr)
    else:
        for _i, ((dev, date), entry) in enumerate(_items, 1):
            _cr, _dr = _qc_session(dev, date, entry, d0=d0, band=band, P=P)
            ch_rows.extend(_cr); dose_rows.extend(_dr)
            if progress:
                _el = _time.time() - _t0
                _eta = _el / _i * (len(_items) - _i)
                print(f"[{_dt.now():%H:%M:%S}]   {_i}/{len(_items)}  {dev} D{(date - d0[dev]).days}: "
                      f"{len(_cr)} channels  ({_el:.0f}s elapsed, ETA {_eta:.0f}s)", flush=True)
    if progress:
        print(f"[{_dt.now():%H:%M:%S}] channel_quality_report done in {_time.time() - _t0:.0f}s "
              f"({len(ch_rows)} channel-timepoints).", flush=True)
    channel_tbl = pd.DataFrame(ch_rows)
    dose_tbl = pd.DataFrame(dose_rows)
    return channel_tbl, dose_tbl


def channel_survival(index_df: pd.DataFrame, total_channels: int = 16) -> pd.DataFrame:
    """Per (device, timepoint) channel survival from channeltest FSCV blocks."""
    ct = index_df[(index_df.testtype == "channeltest") & (index_df.signaltype == "fscv")]
    surv = (ct.groupby(["device", "timepoint"])["channel"].nunique()
            .reset_index(name="n_survived"))
    surv["total"] = total_channels
    surv["fraction"] = surv["n_survived"] / total_channels
    return surv


def quality_filtering(index_df: pd.DataFrame) -> pd.DataFrame:
    """Per (device, timepoint) EIS channels before vs after quality checks (signal)."""
    eis = index_df[(index_df.testtype == "signal") & (index_df.signaltype == "eis")]
    g = eis.groupby(["device", "timepoint"])
    out = g.agg(n_channels=("channel", "nunique"),
                n_valid=("eis_valid", "sum")).reset_index()
    out["n_valid"] = out["n_valid"].astype(int)
    out["n_rejected"] = out["n_channels"] - out["n_valid"]
    return out
