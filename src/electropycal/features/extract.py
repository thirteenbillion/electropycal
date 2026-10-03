"""Featureset assembly from a raw PSTrace export directory.

Walks ``<root>/<YYYYMMDD>_<devicetype>_signal/`` sessions, derives each device's
timepoint as **elapsed days since its first signal session** (handles staggered
series), and for every valid ``(device, channel, timepoint, concentration)``
assembles a feature row: EIS features (replicate-averaged, band-limited,
concentration-invariant → shared across that channel-timepoint's concentrations)
+ FSCV predictor features + ``NormIpeak`` (per concentration, background = the
``0nM`` file). Sensor identity is ``(device, channel)``.

Quality-gated: EIS checks A–C (``data.quality``) and the FSCV noise-floor check
(``NormIpeak ≥ 3·noise_floor``). Output columns are consumable by
``discovery.config.RunData.from_frame``.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.pstrace import PSTraceExport, parse_filename, parse_folder, read_pstrace
from ..data.quality import eis_quality
from .eis import eis_features, eis_global_features
from .fscv import (FSCV_SMOOTH_POLY, FSCV_SMOOTH_WINDOW, NONFARADAIC_WINDOW,
                   PEAK_EDGE_TOL, PEAK_WINDOW, background_features, dose_response_corr,
                   mean_ibg, mean_vpeak, noise_floor, norm_ipeak, peak_at_edge, peak_shape,
                   repeatability_snr, smooth_current)


# Bump whenever the emitted feature/target columns change (new predictors,
# renamed columns, changed math). Notebook caches include this in their key so a
# schema change auto-invalidates stale ``*_featureset__<hash>.parquet`` files
# instead of silently reloading a parquet that lacks the new columns.
FEATURE_SCHEMA_VERSION = 6


class EmptySessionWarning(UserWarning):
    """A ``(device, timepoint)`` session yielded no rows.

    Its own category so callers can filter, escalate
    (``warnings.simplefilter("error", EmptySessionWarning)``) or collect these without
    touching unrelated warnings.
    """


class EmptySessionError(RuntimeError):
    """A session yielded no rows and ``on_empty_session="raise"`` was set."""


def _report_empty_session(dev, date, n_eis_channels: int, mode: str) -> None:
    """Report a session that produced zero rows, distinguishing *why*.

    A session absent from the featureset is otherwise indistinguishable from one that was
    never measured, which is how three sessions sat at zero rows unnoticed. The two causes
    need different responses, so they are reported differently: missing files mean an
    incomplete fetch, while channels that all failed mean the data is there but unusable.
    """
    if mode == "ignore":
        return
    where = f"{dev}@{date.isoformat() if hasattr(date, 'isoformat') else date}"
    why = (f"all {n_eis_channels} EIS channel(s) were dropped by a gate — EIS quality, a "
           f"frequency-grid mismatch against this device type's ref_grid, a missing FSCV "
           f"background, or the FSCV acceptance test"
           if n_eis_channels else
           "no EIS file, or no 0 nM background FSCV file (an incomplete fetch looks like this)")
    msg = (f"session {where} produced 0 rows: {why}. It will be absent from the featureset, "
           f"indistinguishable from a session that was never measured.")
    if mode == "raise":
        raise EmptySessionError(msg)
    import warnings
    warnings.warn(msg, EmptySessionWarning, stacklevel=3)


def _avg_eis(specs: list[dict]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Replicate-average EIS spectra (assumes a shared fixed frequency grid)."""
    fr = specs[0]["freq"]
    zr = np.mean([s["z_real"] for s in specs], axis=0)
    zi = np.mean([s["z_imag"] for s in specs], axis=0)
    return fr, zr, zi


def _avg_fscv(exp: PSTraceExport, ch: int, max_reps: int | None = 3) -> dict | None:
    """Replicate-average FSCV cycles for a channel, in acquisition (rep) order.

    Replicates share the sweep but real exports occasionally differ by a sample;
    truncate to the common length. ``max_reps`` keeps only the first N replicate
    cycles — use it to drop erroneously-appended extra rounds (the intended protocol
    is the first ``[0]…[N-1]``); ``None`` averages all cycles present.
    """
    cycles = [v for (c, _), v in sorted(exp.fscv.items()) if c == ch]
    if max_reps is not None:
        cycles = cycles[:max_reps]
    if not cycles:
        return None
    m = min(len(c["current"]) for c in cycles)
    if m == 0:
        return None
    return {"voltage": np.asarray(cycles[0]["voltage"])[:m],
            "current": np.mean([c["current"][:m] for c in cycles], axis=0)}


def _first_ref_grid_invivo(items, band, mono_tol) -> dict:
    """In-vivo counterpart of :func:`_first_ref_grid`: one grid per device type.

    Same rule — the in-band sorted grid of the first EIS channel of that device type which
    passes EIS quality, in sorted ``items`` order — but reading the in-vivo index shape
    (``baseline``/``live`` slots) rather than ``{dose: path}``. Kept separate rather than
    generalised because the two walks locate their EIS files differently; the *policy* they
    share is the quality gate and the per-type anchor.
    """
    grids: dict = {}
    for (dev, date), entry in items:
        dtype = entry.get("devicetype")
        if dtype in grids:
            continue
        eis_src = entry["live"].get("eis") or entry["baseline"].get("eis")
        if eis_src is None:
            continue
        exp = read_pstrace(eis_src)
        for ch in exp.eis_channels:
            specs = [v for (c, _), v in sorted(exp.eis.items()) if c == ch]
            if not specs:
                continue
            fr, zr, zi = _avg_eis(specs)
            if not eis_quality(fr, zr, zi, band=band, mono_tol=mono_tol)["valid"]:
                continue
            m = (fr >= band[0]) & (fr <= band[1])
            order = np.argsort(fr[m])
            grids[dtype] = fr[m][order]
            break
    return grids


def extract_invivo(root: str | Path, band: tuple[float, float] = (10.0, 100_000.0),
                   peak_method: str = "direct", detrend: bool = False,
                   baseline_window: tuple[float, float] = (-0.1, 0.2),
                   smooth_window: int = FSCV_SMOOTH_WINDOW,
                   smooth_poly: int = FSCV_SMOOTH_POLY,
                   mono_tol: float = 0.10,
                   ref_grid_hz: dict | None = None) -> pd.DataFrame:
    """Build an in-vivo featureset from a raw directory (see DESIGN §9).

    In vivo, concentration is **unlabeled** and FSCV is measured continuously, so
    each row is one *time sample* per ``(device, channel)`` rather than a dose. Files
    follow ``<deviceid>_<signaltype>_<period>.csv`` with ``period ∈ {baseline, live}``
    and ``signaltype ∈ {eis, fscv, paired}`` (``paired`` = time-sequential EIS+FSCV
    per channel in one file). The ``baseline`` period supplies the background FSCV
    cycle (and a reference EIS if no live EIS); the ``live`` period supplies the
    time-sequential EIS + FSCV. Feature columns match the in-vitro featureset so a
    frozen model applies directly. ``timepoint`` = days since the device's first
    in-vivo session; ``time_index`` orders samples within a session. No
    ``concentration`` column. Assumes the same internal CSV format as in-vitro.

    The EIS reference grid is fixed **per device type** by a quality-gated pre-pass, as
    in-vitro. ``ref_grid_hz`` (``{devicetype: grid}``, e.g. from an extraction pin) supplies it
    instead — pass the in-vitro grids when preparing data for a frozen model, so in-vivo
    features land on the same frequency grid the model was trained on rather than on whichever
    channel this directory happened to present first.
    """
    root = Path(root)
    index: dict[tuple[str, object], dict] = defaultdict(
        lambda: {"baseline": {}, "live": {}})
    device_dates: dict[str, set] = defaultdict(set)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        for csv in folder.glob("*.csv"):
            nm = parse_filename(csv.name)
            if not nm or nm["dose"] not in ("baseline", "live"):
                continue
            dev, date, period = nm["deviceid"], fmeta["date"], nm["dose"]
            device_dates[dev].add(date)
            index[(dev, date)]["devicetype"] = fmeta["devicetype"]
            slot = index[(dev, date)][period]
            if nm["signaltype"] == "paired":
                slot["eis"] = slot["fscv"] = csv
            else:
                slot[nm["signaltype"]] = csv

    device_d0 = {dev: min(dates) for dev, dates in device_dates.items()}
    rows: list[dict] = []
    _items = sorted(index.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
    # One reference grid per device type, from the first EIS channel of that type that passes
    # EIS quality. Previously this was a single mutable `ref_grid` seeded by the very first
    # channel encountered with NO quality gate at all -- so a bad first channel silently became
    # the grid every other channel had to match, and one device type's grid was imposed on all.
    _ref_grids = ({str(t): np.asarray(g, float) for t, g in ref_grid_hz.items()}
                  if ref_grid_hz is not None else _first_ref_grid_invivo(_items, band, mono_tol))
    for (dev, date), entry in _items:
        base, live = entry["baseline"], entry["live"]
        if "fscv" not in live or "fscv" not in base:
            continue                                         # need live signal + a baseline background
        timepoint = float((date - device_d0[dev]).days)
        bg_exp = read_pstrace(base["fscv"])
        live_fscv = read_pstrace(live["fscv"])
        eis_src = live.get("eis") or base.get("eis")
        if eis_src is None:
            continue
        live_eis = read_pstrace(eis_src)

        for ch in sorted(set(live_fscv.fscv_channels) & set(live_eis.eis_channels)):
            bg = _avg_fscv(bg_exp, ch)
            if bg is None:
                continue
            bg["current"] = smooth_current(bg["current"], smooth_window, smooth_poly)
            cycles = [v for (c, _), v in sorted(live_fscv.fscv.items()) if c == ch]
            specs = [v for (c, _), v in sorted(live_eis.eis.items()) if c == ch]
            if not specs:
                continue
            bo = np.argsort(bg["voltage"])
            for ti in range(len(cycles)):
                spec = specs[ti] if ti < len(specs) else specs[-1]   # reuse a static reference EIS
                fr, zr, zi = spec["freq"], spec["z_real"], spec["z_imag"]
                m = (fr >= band[0]) & (fr <= band[1])
                order = np.argsort(fr[m])
                fb, zrb, zib = fr[m][order], zr[m][order], zi[m][order]
                ref_grid = _ref_grids.get(entry.get("devicetype"))
                if ref_grid is None or fb.shape != ref_grid.shape \
                        or not np.allclose(fb, ref_grid, rtol=1e-3):
                    continue                            # inconsistent frequency grid -> skip
                cyc = cycles[ti]
                v = cyc["voltage"]
                cur = smooth_current(cyc["current"], smooth_window, smooth_poly)
                bg_i = np.interp(v, bg["voltage"][bo], bg["current"][bo])
                nip = norm_ipeak(cur, bg_i, v, method=peak_method,
                                 detrend=detrend, baseline_window=baseline_window)
                row = {"device": dev, "channel": int(ch), "timepoint": timepoint,
                       "time_index": ti, "NormIpeak": float(nip),
                       "mean_Vpeak": mean_vpeak(cur, bg_i, v, method=peak_method,
                                                detrend=detrend, baseline_window=baseline_window),
                       "mean_Ibg": mean_ibg(bg_i, cur, v, method=peak_method,
                                            detrend=detrend, baseline_window=baseline_window)}
                for ftype, arr in eis_features(fb, zrb, zib).items():
                    for i, val in enumerate(arr):
                        row[f"{ftype}_f{i:02d}"] = float(val)
                row.update({k: float(val) for k, val in
                            eis_global_features(fb, zrb, zib, full_spectrum=(fr, zr, zi)).items()})
                rows.append(row)
    return pd.DataFrame(rows)


def _accept(candidates: list[tuple[float, float, float, dict]], acceptance: str,
            min_norm_snr: float, monotonic_r_min: float,
            mono_method: str = "pearson",
            min_dose_response_range: float | None = None) -> list[dict]:
    """Apply the FSCV acceptance test to one channel-timepoint's dose rows.

    SNR-based modes threshold the **reproducibility** SNR (``repeatability_snr`` = NormIpeak / std
    across replicate cycles) — the robust metric — not the residual-dominated RMS-floor SNR.

    ``min_dose_response_range`` (default ``None`` = off) adds an **amplitude floor** to the monotonic
    test: the correlation ``dose_response_corr`` is scale-invariant, so a clean but *tiny* monotone
    response (a near-dead electrode whose NormIpeak creeps 0.01→0.05) passes with ``r≈1``. Requiring
    the NormIpeak dynamic range (max − min over doses) to reach this threshold drops those
    functionally-dead channel-timepoints. Off by default so the featureset is unchanged unless asked.
    """
    def _snr_ok(row):
        rs = row.get("repeatability_snr", float("nan"))
        return np.isfinite(rs) and rs >= min_norm_snr

    if not candidates:
        return []
    if acceptance == "snr":
        return [row for *_, row in candidates if _snr_ok(row)]
    if acceptance == "none":
        return [row for *_, row in candidates]
    # monotonic (optionally + a lenient per-row SNR): keep the channel iff its
    # NormIpeak rises with log-concentration, then (for 'monotonic+snr') drop
    # individual dose rows below the reproducibility-SNR floor.
    doses = np.array([d for d, *_ in candidates], float)
    nips = np.array([nip for _, nip, _, _ in candidates], float)
    if doses.size < 3 or np.std(nips) == 0:
        return []
    r = dose_response_corr(doses, nips, method=mono_method)
    if not np.isfinite(r) or r < monotonic_r_min:
        return []
    if min_dose_response_range is not None and float(nips.max() - nips.min()) < min_dose_response_range:
        return []                                        # monotone but functionally flat (near-dead)
    if acceptance == "monotonic+snr":
        return [row for *_, row in candidates if _snr_ok(row)]
    return [row for *_, row in candidates]


def _first_ref_grid(items, band, mono_tol) -> dict:
    """Deterministic pre-pass: one reference EIS frequency grid **per device type**.

    Each device type's grid is the in-band sorted frequency grid of the first EIS channel (in
    sorted ``items`` order) of that type which passes EIS quality — computed up front so
    parallel workers can share it. Reads only EIS (cheap) and stops at the first accepted
    channel per type. Returns ``{devicetype: grid}``; a type with no qualifying channel is
    absent.

    Per device type, not one grid for the whole corpus: the anchor used to be the globally
    first-sorted passing channel, which made an unrelated device's measurement geometry the
    grid every other device had to match to ``rtol=1e-3`` or be skipped. In this study that
    anchor was the sole **cfme** electrode setting the grid for all 69 neurostring sensors —
    a coupling that came from sort order, not from anything physical.
    """
    grids: dict = {}
    for (dev, date), entry in items:
        dtype = entry.get("devicetype")
        if dtype in grids or "eis" not in entry or 0.0 not in entry["fscv"]:
            continue
        eis_exp = read_pstrace(entry["eis"])
        for ch in eis_exp.eis_channels:
            specs = [v for (c, _), v in eis_exp.eis.items() if c == ch]
            fr, zr, zi = _avg_eis(specs)
            if not eis_quality(fr, zr, zi, band=band, mono_tol=mono_tol)["valid"]:
                continue
            m = (fr >= band[0]) & (fr <= band[1])
            order = np.argsort(fr[m])
            grids[dtype] = fr[m][order]
            break
    return grids


def _extract_session(dev, date, entry, *, device_d0, band, ref_grid, P):
    """Feature-extract ONE ``(device, timepoint)`` session into its dose rows.

    A pure function of its inputs — ``ref_grid`` is fixed by the caller's deterministic pre-pass
    (:func:`_first_ref_grid`), so sessions carry no shared state and parallelize cleanly. ``P``
    bundles the scalar extraction params. Returns ``(rows, n_eis_channels)``; concatenating the
    ``rows`` in sorted-session order reproduces the serial output exactly.
    """
    if "eis" not in entry or 0.0 not in entry["fscv"]:
        return [], 0                                    # need EIS + a 0nM background
    timepoint = float((date - device_d0[dev]).days)
    eis_exp = read_pstrace(entry["eis"])
    bg_exp = read_pstrace(entry["fscv"][0.0])
    # only numeric positive doses are concentrations; skip protocol tokens
    # (e.g. a ``stabilization`` FSCV file, which is not a dose-response point).
    fscv_exp = {d: read_pstrace(p) for d, p in entry["fscv"].items()
                if isinstance(d, (int, float)) and d > 0}
    rows: list[dict] = []
    for ch in eis_exp.eis_channels:
        specs = [v for (c, _), v in eis_exp.eis.items() if c == ch]
        fr, zr, zi = _avg_eis(specs)
        if not eis_quality(fr, zr, zi, band=band, mono_tol=P["mono_tol"])["valid"]:
            continue
        m = (fr >= band[0]) & (fr <= band[1])
        order = np.argsort(fr[m])
        fb, zrb, zib = fr[m][order], zr[m][order], zi[m][order]
        # ref_grid is pre-fixed: the first-accepted channel's grid passes (allclose to itself);
        # any channel on a different grid is skipped, exactly as the serial loop did.
        if ref_grid is None or fb.shape != ref_grid.shape or not np.allclose(fb, ref_grid, rtol=1e-3):
            continue                                    # inconsistent frequency grid → skip
        efeat = eis_features(fb, zrb, zib)
        eglob = eis_global_features(fb, zrb, zib, full_spectrum=(fr, zr, zi))

        bg = _avg_fscv(bg_exp, ch, max_reps=P["max_reps"])
        if bg is None:
            continue
        # smooth the raw background once (removes hardware noise), before any
        # subtraction — so the background trace and every dose share the same de-noising.
        bg["current"] = smooth_current(bg["current"], P["smooth_window"], P["smooth_poly"])
        # DA-independent background-CV predictors (dose-invariant; from the 0 nM cycle only).
        bgfeat = background_features(bg["voltage"], bg["current"])
        # compute every dose's NormIpeak first, then apply the acceptance test
        # (needed for the across-concentration monotonicity criterion).
        bo = np.argsort(bg["voltage"])
        candidates: list[tuple[float, float, float, dict]] = []  # (dose, nip, nf, row)
        for dose, fexp in sorted(fscv_exp.items()):
            sig = _avg_fscv(fexp, ch, max_reps=P["max_reps"])
            if sig is None:
                continue
            sig["current"] = smooth_current(sig["current"], P["smooth_window"], P["smooth_poly"])
            v = sig["voltage"]
            bg_i = np.interp(v, bg["voltage"][bo], bg["current"][bo])  # align to sig grid
            nip = norm_ipeak(sig["current"], bg_i, v, method=P["peak_method"],
                             detrend=P["detrend"], baseline_window=P["baseline_window"],
                             require_interior=P["require_interior_peak"])
            nf = noise_floor(sig["current"], bg_i, v, detrend=P["detrend"],
                             baseline_window=P["baseline_window"],
                             window=P["nonfaradaic_window"] or NONFARADAIC_WINDOW)
            vpk = mean_vpeak(sig["current"], bg_i, v, method=P["peak_method"],
                             detrend=P["detrend"], baseline_window=P["baseline_window"])
            # per-replicate NormIpeak (each cycle smoothed like the average, background
            # interpolated onto that cycle's grid) -> reproducibility SNR: NormIpeak / std across
            # cycles, robust where the RMS floor (residual-dominated) is not. Averaged NormIpeak
            # stays the numerator.
            rep_cycles = [rc for (c, _), rc in sorted(fexp.fscv.items()) if c == ch][:P["max_reps"]]
            rep_nips = [norm_ipeak(smooth_current(rc["current"], P["smooth_window"], P["smooth_poly"]),
                        np.interp(rc["voltage"], bg["voltage"][bo], bg["current"][bo]),
                        rc["voltage"], method=P["peak_method"], detrend=P["detrend"],
                        baseline_window=P["baseline_window"], require_interior=P["require_interior_peak"])
                        for rc in rep_cycles]
            _fin = [x for x in rep_nips if np.isfinite(x)]
            rep_sd = float(np.std(_fin)) if len(_fin) >= 2 else float("nan")
            rsnr = repeatability_snr(nip, rep_nips)
            # a negative NormIpeak means signal < background at V_ox — non-physical for an
            # oxidation peak (imperfect background matching); drop by default so it neither
            # feeds the monotonic fit nor reaches discovery.
            if not np.isfinite(nip) or (P["drop_negative"] and nip < 0):
                continue
            row = {"device": dev, "channel": int(ch), "timepoint": timepoint,
                   "concentration": float(dose), "NormIpeak": float(nip),
                   "noise_floor": float(nf), "rep_std": rep_sd,
                   "repeatability_snr": float(rsnr), "snr": float(rsnr),  # snr alias = default (repeatability)
                   "rms_snr": float(nip / nf) if nf else float("nan"),
                   "mean_Vpeak": vpk,
                   "peak_at_edge": peak_at_edge(vpk, PEAK_WINDOW, P["peak_edge_tol"]),
                   "mean_Ibg": mean_ibg(bg_i, sig["current"], v, method=P["peak_method"],
                                        detrend=P["detrend"], baseline_window=P["baseline_window"])}
            # faradaic peak-SHAPE (deformation-mode characterization; RESERVED — target/diagnostic,
            # NOT a predictor: height/area are the NormIpeak numerator and would leak).
            _ps = peak_shape(sig["current"], bg_i, v, method=P["peak_method"], detrend=P["detrend"],
                             baseline_window=P["baseline_window"])
            row["peak_height"] = _ps["height"]
            row["peak_area"] = _ps["area"]
            row["peak_fwhm"] = _ps["fwhm"]
            # QC flag: the faradaic lobe is truncated by the sweep/window edge, so peak_area is an
            # under-estimate (common at low dose; a hard limit of the voltage sweep span, not filtering).
            row["peak_area_clipped"] = bool(_ps["area_clipped"])
            row.update(bgfeat)                            # bg_charge / bg_cap / bg_switch (predictors)
            for ftype, arr in efeat.items():
                for i, val in enumerate(arr):
                    row[f"{ftype}_f{i:02d}"] = float(val)
            row.update({k: float(val) for k, val in eglob.items()})
            candidates.append((float(dose), float(nip), float(nf), row))

        # per-channel-timepoint dose-response correlation (QC metadata on every row)
        if len(candidates) >= 3:
            d = np.array([c[0] for c in candidates])
            y = np.array([c[1] for c in candidates])
            r = dose_response_corr(d, y, method=P["mono_method"])
            for c in candidates:
                c[3]["dose_response_r"] = r
        rows.extend(_accept(candidates, P["acceptance"], P["min_norm_snr"],
                            P["monotonic_r_min"], P["mono_method"],
                            P.get("min_dose_response_range")))
    return rows, len(eis_exp.eis_channels)


def _reliability_session(dev, date, entry, *, device_d0, band, ref_grid, keep, P):
    """Per-replicate feature values for ONE session → list of ``(feature, kind, obs_key, value)``.

    ``obs_key`` is the observation's identity tuple (``(dev, ch, tp)`` for EIS — dose-invariant —
    and ``(dev, ch, tp, dose)`` for FSCV), stable across sessions, so parallel workers need no shared
    id counter; the caller maps keys → integer ids in sorted-session order to reproduce the serial
    grouping exactly. ``ref_grid`` is fixed by the caller's pre-pass. ``[]`` if the session lacks an
    EIS + 0 nM background."""
    if "eis" not in entry or 0.0 not in entry["fscv"]:
        return []
    timepoint = float((date - device_d0[dev]).days)
    eis_exp = read_pstrace(entry["eis"])
    bg_exp = read_pstrace(entry["fscv"][0.0])
    fscv_exp = {d: read_pstrace(p) for d, p in entry["fscv"].items()
                if isinstance(d, (int, float)) and d > 0}
    recs: list[tuple] = []
    for ch in eis_exp.eis_channels:
        if keep is not None and (dev, int(ch), timepoint) not in keep:
            continue
        specs = [v for (c, _), v in eis_exp.eis.items() if c == ch]
        fr, zr, zi = _avg_eis(specs)
        if not eis_quality(fr, zr, zi, band=band, mono_tol=P["mono_tol"])["valid"]:
            continue
        m = (fr >= band[0]) & (fr <= band[1])
        order = np.argsort(fr[m])
        grid = fr[m][order]
        if ref_grid is None or grid.shape != ref_grid.shape or not np.allclose(grid, ref_grid, rtol=1e-3):
            continue
        ekey = (dev, int(ch), timepoint)
        # per-replicate EIS features (each spectrum on the same in-band grid)
        for spec in specs:
            sf = np.asarray(spec["freq"], float); so = np.argsort(sf[m])
            fb = sf[m][so]
            if fb.shape != ref_grid.shape or not np.allclose(fb, ref_grid, rtol=1e-3):
                continue
            zrb = np.asarray(spec["z_real"], float)[m][so]
            zib = np.asarray(spec["z_imag"], float)[m][so]
            for ftype, arr in eis_features(fb, zrb, zib).items():
                for j, val in enumerate(arr):
                    recs.append((f"{ftype}_f{j:02d}", "eis", ekey, float(val)))
            for name, val in eis_global_features(fb, zrb, zib, full_spectrum=(fr, zr, zi)).items():
                recs.append((name, "eis", ekey, float(val)))

        bg = _avg_fscv(bg_exp, ch, max_reps=P["max_reps"])
        if bg is None:
            continue
        bg["current"] = smooth_current(bg["current"], P["smooth_window"], P["smooth_poly"])
        bo = np.argsort(bg["voltage"])
        for dose, fexp in sorted(fscv_exp.items()):
            cyc = [rc for (c, _), rc in sorted(fexp.fscv.items()) if c == ch][:P["max_reps"]]
            fkey = (dev, int(ch), timepoint, float(dose))
            for rc in cyc:
                v = rc["voltage"]
                cur = smooth_current(rc["current"], P["smooth_window"], P["smooth_poly"])
                bg_i = np.interp(v, bg["voltage"][bo], bg["current"][bo])
                kw = dict(method=P["peak_method"], detrend=P["detrend"], baseline_window=P["baseline_window"])
                _shp = peak_shape(cur, bg_i, v, **kw)
                for name, val in (("mean_Vpeak", mean_vpeak(cur, bg_i, v, **kw)),
                                  ("mean_Ibg", mean_ibg(bg_i, cur, v, **kw)),
                                  ("NormIpeak", norm_ipeak(cur, bg_i, v, **kw)),
                                  ("peak_height", _shp["height"]), ("peak_area", _shp["area"]),
                                  ("peak_fwhm", _shp["fwhm"])):
                    if np.isfinite(val):
                        recs.append((name, "fscv", fkey, float(val)))
    return recs


def extract_dataset(root: str | Path, band: tuple[float, float] | str | None = None,
                    min_norm_snr: float = 3.0, mono_tol: float = 0.10,
                    peak_method: str = "direct", detrend: bool = False,
                    baseline_window: tuple[float, float] = (-0.1, 0.2),
                    acceptance: str = "monotonic", monotonic_r_min: float = 0.6,
                    min_dose_response_range: float | None = None,
                    mono_method: str = "pearson", max_reps: int | None = 3,
                    smooth_window: int = FSCV_SMOOTH_WINDOW,
                    smooth_poly: int = FSCV_SMOOTH_POLY,
                    drop_negative: bool = True,
                    nonfaradaic_window: tuple[float, float] | None = None,
                    peak_edge_tol: float = PEAK_EDGE_TOL,
                    require_interior_peak: bool = False,
                    d0_normalize: bool = True,
                    device_types=None,
                    on_empty_session: str = "warn",
                    pin=None,
                    pin_mode: str = "reproduce",
                    pin_out: str | Path | None = None,
                    n_jobs: int = 1,
                    progress: bool = False) -> pd.DataFrame:
    """Build ``featureset_extracted`` from a raw PSTrace directory.

    FSCV peak definition (see :func:`electropycal.features.fscv.norm_ipeak`):
    ``peak_method`` is ``"direct"`` (current at V_ox, the **default** — robust to the
    broad DA peaks in real data) or ``"chord"`` (the chord-baseline height,
    for sharp peaks); ``detrend`` optionally removes a residual baseline slope using
    ``baseline_window`` (opt-in — can eat broad-onset signal).

    Raw-signal smoothing: ``smooth_window`` (samples, odd; default
    :data:`~electropycal.features.fscv.FSCV_SMOOTH_WINDOW`, **on**) Savitzky-Golay smooths the
    replicate-averaged signal **and** background cycles to remove hardware noise, *before*
    background subtraction — so subtraction, NormIpeak, the noise floor, and the predictor
    features all see the same de-noised trace. Set ``smooth_window=0`` to disable.

    Quality gates (calibrate against known good/broken channels):
    - ``band`` — EIS analysis band ``(lo, hi)`` in Hz, **required**: pass a tuple, or ``"auto"``
      to set it from data via :func:`electropycal.data.inventory.recommended_band` (upper bound
      just below the 10th-percentile inductive onset, so the lowest-onset ~10% of channels fall
      in-band and fail EIS.1 as poor electrodes rather than shrinking the band for all). A fixed
      100 kHz band would sit above real onsets and drop most channels.
      ``"auto"`` is **opt-in, not the default**: it is a percentile over every EIS file in the
      corpus, so it makes the featureset a function of which sessions happen to be present — the
      same band over a staged subset is a different band, silently. There is no default because
      neither choice is safe to assume: a fixed band can starve a new corpus, and an automatic one
      cannot be reproduced over a subset. Pass ``pin=`` to reuse a previous run's band instead.
      The study default is ``(2.0, 2000.0)``
      (:class:`~electropycal.analysis_config.AnalysisConfig`).
    - ``mono_tol`` — EIS ``|Z|``-monotonicity tolerance (check B). Default relaxed to
      0.10, relaxed from a stricter 0.05: real capacitive channels show ~5%
      noise-driven ``|Z|`` bumps that 0.05 wrongly rejects, while genuine DC drift is
      far larger and still caught.
    - ``max_reps`` — keep only the first N replicate cycles per FSCV file (in
      acquisition order); default **3** (the protocol's intended ``[0]…[2]`` rounds),
      which drops erroneously-appended extra rounds. ``None`` averages all cycles.
    - ``acceptance`` — FSCV channel acceptance test at a timepoint:
      ``"monotonic"`` (**default**) keeps **all** of a channel's dose rows iff its
      ``NormIpeak`` is monotone-increasing in log-concentration (dose-response
      ``r ≥ monotonic_r_min``; ``mono_method`` selects ``"pearson"`` linear-in-log
      or ``"spearman"`` rank/monotonic, the latter robust to saturating responses) —
      a meaningful early-timepoint criterion that a
      single-cycle SNR (hostage to imperfect background matching) is not;
      ``"snr"`` keeps each dose row whose **reproducibility** SNR
      (``repeatability_snr`` = NormIpeak / std across replicate cycles) ``≥ min_norm_snr``;
      ``"none"`` keeps every finite row.
      (The reproducibility SNR replaced the old ``NormIpeak / noise_floor`` cut: the RMS floor is
      dominated by the background-subtraction residual, so it read ~1× even for obvious peaks.)
    - ``min_dose_response_range`` (default **None**, off) — amplitude floor complementing the
      scale-invariant monotonicity ``r``: a channel-timepoint is kept only if its NormIpeak dynamic
      range (max − min over doses) reaches this value. Excludes functionally-dead electrodes that are
      cleanly monotone but *tiny* (``r≈1`` yet flat). Off by default (featureset unchanged); set e.g.
      ``0.05`` to drop near-flat channels from the modeling dataset.
    - ``drop_negative`` (**default True**) — drop any dose whose ``NormIpeak < 0`` (signal below
      background at V_ox, non-physical for an oxidation peak) *before* the acceptance test, so a
      spurious negative neither feeds the monotonic fit nor reaches discovery.
    - ``nonfaradaic_window`` — voltage window the noise floor is measured over
      (default :data:`~electropycal.features.fscv.NONFARADAIC_WINDOW`, which sits below the
      peak-search window so DA signal is not counted as noise).
    - ``peak_edge_tol`` — a dose whose ``mean_Vpeak`` lands within this of either ``PEAK_WINDOW``
      bound is flagged ``peak_at_edge=True`` (the peak pinned to the boundary; likely unresolved).
      Reported per row; non-gating.
    - ``require_interior_peak`` (default **off**) — require the NormIpeak peak to be an *interior*
      local maximum; when the in-window argmax pins to a boundary sample, ``norm_ipeak`` returns
      NaN and the dose is dropped (like a missing peak) rather than reading the boundary. Off by
      default (edge-pinning is instead reported non-gating via ``peak_at_edge`` / ``F.pk``).
    - ``d0_normalize`` (default **True**) — D0-normalize each feature per sensor
      against its own earliest timepoint (additive shift for bounded/phase/log-slope features,
      division for magnitude/area features), so the model learns drift-from-baseline rather than
      absolute per-sensor fabrication differences. The target (``NormIpeak``) and QC columns are not
      touched. This is the same transform deployment applies to in-vivo data, so the frozen model
      sees a consistent representation at train and deploy time. Set ``False`` for the raw features.
    - ``n_jobs`` (default **1**, serial) — number of worker processes to feature-extract the
      device-timepoint sessions in parallel (joblib). Sessions are independent once the shared
      EIS frequency ``ref_grid`` is fixed by a deterministic pre-pass, and results are concatenated
      in the same sorted order, so **the output is bit-identical to the serial run** — only faster.
      ``-1`` uses all cores. The per-session live ETA is shown only when serial (``n_jobs=1``);
      parallel prints a single elapsed/rows summary (workers finish out of order). Extraction is a
      one-time, cached pass — parallelize it to shorten the first pass, then reuse the parquet.
    - ``progress`` (default **off**) — emit timestamped ``[HH:MM:SS]`` per-session lines with an
      ETA (like :func:`~electropycal.data.inventory.channel_quality_report`) so CLI/script runs
      show live progress; default silent so notebooks and tests stay quiet.

    - ``device_types`` (default **None** = no filter) — restrict extraction to these device
      types (the ``<devicetype>`` in each session folder name), e.g. ``["neurostring"]``.
      Recorded in the pin. Filtering is *not* a late row filter: it changes which EIS channels
      feed the ``band`` percentile and which device types get a ``ref_grid`` anchor, so
      selecting types after extraction is not the same as extracting only those types. That is
      why this is pinned rather than left to the caller — the notebook's
      ``DEVICE_TYPES=["neurostring"]`` discards cfme rows *after* the fact, while the cfme
      device still influenced extraction.

    - ``on_empty_session`` — what to do when a ``(device, timepoint)`` session yields **no
      rows**: ``"warn"`` (**default**) emits an :class:`EmptySessionWarning` naming the session
      and whether the cause was missing files or every channel failing; ``"raise"`` raises
      :class:`EmptySessionError`; ``"ignore"`` restores the old silence. Default is a warning
      rather than an error because a few sessions legitimately produce zero rows, and raising
      would block the featureset build — but silence is what let three such sessions sit
      unnoticed, since an absent ``(device, timepoint)`` is otherwise indistinguishable from
      one that was never measured.

    Reproducibility (``pin`` / ``pin_out``)
    ---------------------------------------
    Extraction derives five parameters run-wide from *whatever input set is present* —
    ``band`` (when ``"auto"``), ``device_d0``, the EIS ``ref_grid``, each sensor's ``d0_row``,
    and the emitted feature column set. All five are functions of the corpus, so extracting
    over a subset silently re-anchors the result rather than failing.

    - ``pin_out`` — write the parameters this run derived to a JSON record
      (:mod:`electropycal.features.pin`, schema 2). The same record is always attached to the
      returned frame as ``df.attrs["extraction_pin"]``, so writing a file is optional.
    - ``pin`` — a path to that record (or the parsed dict, or a ``run_config.json`` containing
      it). When given, ``band``, ``device_d0``, ``ref_grid`` and the feature column set are
      taken **verbatim and never recomputed**, and the run raises
      :class:`~electropycal.features.pin.PinMismatch` on a device the pin has not seen, a
      session predating its pinned ``device_d0``, a feature-column mismatch, or a per-session
      row count that differs from the pin. That last check is what catches a partial fetch:
      a session short its EIS or 0 nM background yields zero rows without raising anywhere
      else, so a staged extraction would otherwise just be quietly smaller.

    - ``pin_mode`` — ``"reproduce"`` (default) is the behaviour above: the input set must be
      a subset of what the pin recorded, and a session the pin has never seen is an error.
      ``"extend"`` keeps every pinned *anchor* (band, ``device_d0``, ``ref_grid``, the
      per-sensor ``d0_row`` baselines, the feature columns) but permits **new** sessions, so
      a corpus that has grown since the pin was written extracts without re-anchoring the
      sessions that were already there. Pinned sessions are still row-count checked, and a
      pinned session that produces no rows at all still raises, so extend does not weaken
      the partial-fetch guarantee for data the pin knows about. Use it for an incremental
      rebuild; use ``"reproduce"`` to verify a staged subset.

    With a pin, re-extracting a two-session subset reproduces exactly those sessions' rows
    from the full run — the property that makes analysis over staged data valid.
    """
    if peak_method not in ("chord", "direct"):
        raise ValueError(f"peak_method must be 'chord' or 'direct', got {peak_method!r}")
    if acceptance not in ("snr", "monotonic", "monotonic+snr", "none"):
        raise ValueError("acceptance must be 'monotonic', 'monotonic+snr', 'snr', or "
                         f"'none', got {acceptance!r}")
    if on_empty_session not in ("warn", "raise", "ignore"):
        raise ValueError("on_empty_session must be 'warn', 'raise', or 'ignore', got "
                         f"{on_empty_session!r}")
    # Before any inference over the corpus: the band percentile, the reference grid and
    # device_d0 all iterate this directory, so an unvalidated path fails from inside one of
    # them rather than at the argument the user actually got wrong.
    from ..data.paths import validate_raw_root
    root = validate_raw_root(root, "root")
    from .pin import (check_feature_columns, check_row_counts, check_sessions_against_pin,
                      load_pin, pinned_band, pinned_d0_rows, pinned_device_d0, ref_grid_for,
                      session_key)
    pin_block = load_pin(pin) if pin is not None else None

    # Resolve the device-type filter before the walk: it decides which EIS channels feed the
    # band percentile and which types get a ref_grid anchor, so it cannot be applied later.
    if pin_mode not in ("reproduce", "extend"):
        raise ValueError(f"pin_mode must be 'reproduce' or 'extend', got {pin_mode!r}")
    if pin_block is None and pin_mode != "reproduce":
        raise ValueError("pin_mode is only meaningful with pin=; pass a pin or drop pin_mode.")

    if pin_block is not None:
        _pinned_types = pin_block.get("device_types") or None
        if device_types is None:
            device_types = _pinned_types                  # reproduce the pinned selection
        elif sorted(str(t) for t in device_types) != _pinned_types:
            from .pin import PinMismatch
            raise PinMismatch(
                f"device_types={sorted(str(t) for t in device_types)} conflicts with the pinned "
                f"selection {_pinned_types}. Extracting a different set of device types changes "
                f"the band percentile and the ref_grid anchors, so the rows would not be "
                f"comparable. Drop the argument to use the pin, or re-pin.")
    _types = None if device_types is None else {str(t) for t in device_types}

    # index files by (device, date): eis path + fscv {dose: path} + the folder's device type
    index: dict[tuple[str, object], dict] = defaultdict(lambda: {"fscv": {}})
    device_dates: dict[str, set] = defaultdict(set)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        if _types is not None and fmeta["devicetype"] not in _types:
            continue
        for csv in folder.glob("*.csv"):
            nm = parse_filename(csv.name)
            if not nm:
                continue
            dev, date = nm["deviceid"], fmeta["date"]
            device_dates[dev].add(date)
            index[(dev, date)]["devicetype"] = fmeta["devicetype"]
            if nm["signaltype"] == "eis":
                index[(dev, date)]["eis"] = csv
            else:
                index[(dev, date)]["fscv"][nm["dose"]] = csv

    # Resolve the run-wide derived parameters. With a pin they are read back verbatim and
    # NOTHING is recomputed -- in particular band="auto" must not trigger its full-corpus
    # pre-pass, whose percentile would be taken over the subset rather than the pinned corpus.
    if pin_block is not None:
        _pinned = pinned_band(pin_block)
        if band is not None and not isinstance(band, str) \
                and (float(band[0]), float(band[1])) != _pinned:
            from .pin import PinMismatch
            raise PinMismatch(
                f"explicit band={(float(band[0]), float(band[1]))} conflicts with the pinned "
                f"band={_pinned}. Drop the band argument to use the pin, or re-pin.")
        band, band_source = _pinned, str(pin_block.get("band_source", "explicit"))
        check_sessions_against_pin(pin_block, list(index))
        device_d0 = pinned_device_d0(pin_block)
    else:
        if band is None:
            raise ValueError(
                "band is required: pass (lo, hi) in Hz, or band='auto' to derive it from this "
                "corpus. 'auto' is a percentile over every EIS file present, so it makes the "
                "featureset depend on which sessions were staged -- there is deliberately no "
                "default. The study band is (2.0, 2000.0); pass pin=<run_config> to reuse a "
                "previous run's band instead.")
        if isinstance(band, str):
            if band != "auto":
                raise ValueError(f"band must be a (lo, hi) tuple or 'auto', got {band!r}")
            from ..data.inventory import recommended_band      # lazy: avoids import cycle
            band = recommended_band(root, devicetype=device_types)   # below every onset in scope
            band_source = "auto"
            if progress:
                print(f"[extract_dataset] data-driven BAND = ({band[0]:.0f}, {band[1]:.0f}) Hz",
                      flush=True)
        else:
            band, band_source = (float(band[0]), float(band[1])), "explicit"
        device_d0 = {dev: min(dates) for dev, dates in device_dates.items()}

    import time as _time
    from datetime import datetime as _dt
    _items = sorted(index.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
    _t0 = _time.time()
    # Fix the shared EIS ref_grid ONCE, deterministically, so sessions are independent and the
    # parallel output matches serial byte-for-byte (see _first_ref_grid / _extract_session).
    if pin_block is not None:
        def _grid_for(dtype):
            return ref_grid_for(pin_block, dtype)
    else:
        _ref_grids = _first_ref_grid(_items, band, mono_tol)          # one grid per device type

        def _grid_for(dtype):
            return _ref_grids.get(dtype)
    P = {"max_reps": max_reps, "smooth_window": smooth_window, "smooth_poly": smooth_poly,
         "peak_method": peak_method, "detrend": detrend, "baseline_window": baseline_window,
         "mono_tol": mono_tol, "nonfaradaic_window": nonfaradaic_window,
         "require_interior_peak": require_interior_peak, "peak_edge_tol": peak_edge_tol,
         "drop_negative": drop_negative, "acceptance": acceptance, "min_norm_snr": min_norm_snr,
         "monotonic_r_min": monotonic_r_min, "mono_method": mono_method,
         "min_dose_response_range": min_dose_response_range}
    from .._parallel import map_sessions, should_parallelize
    _parallel = should_parallelize(n_jobs, len(_items))
    if progress:
        print(f"[{_dt.now():%H:%M:%S}] extract_dataset: feature extraction over "
              f"{len(_items)} device-timepoint session(s)"
              f"{f' on {n_jobs} worker process(es)' if _parallel else ''}...", flush=True)

    rows: list[dict] = []
    # per-session row counts -> the pin's partial-fetch check (a session short its EIS or its
    # 0 nM background yields zero rows without raising anywhere else).
    session_rows: dict[str, dict] = {}

    def _record(dev, date, srows, n_ch):
        session_rows[session_key(dev, date)] = {
            "timepoint": float((date - device_d0[dev]).days), "n_rows": len(srows)}
        if not srows:
            _report_empty_session(dev, date, n_ch, on_empty_session)

    if _parallel:
        from joblib import delayed                  # loky processes (BLAS-capped); sessions independent
        results = map_sessions(n_jobs, (
            delayed(_extract_session)(dev, date, entry, device_d0=device_d0, band=band,
                                      ref_grid=_grid_for(entry.get("devicetype")), P=P)
            for (dev, date), entry in _items))       # joblib preserves input order -> deterministic
        for ((dev, date), _entry), (_srows, _nch) in zip(_items, results):
            rows.extend(_srows)
            _record(dev, date, _srows, _nch)
        if progress:
            print(f"[{_dt.now():%H:%M:%S}]   done: {len(rows)} dose rows from {len(_items)} "
                  f"session(s) in {_time.time() - _t0:.0f}s ({n_jobs} workers)", flush=True)
    else:
        for _i, ((dev, date), entry) in enumerate(_items, 1):
            _srows, _nch = _extract_session(dev, date, entry, device_d0=device_d0, band=band,
                                            ref_grid=_grid_for(entry.get("devicetype")), P=P)
            rows.extend(_srows)
            _record(dev, date, _srows, _nch)
            if progress:
                _el = _time.time() - _t0
                _eta = _el / _i * (len(_items) - _i)
                print(f"[{_dt.now():%H:%M:%S}]   {_i}/{len(_items)}  {dev}: "
                      f"{_nch} EIS channels -> {len(_srows)} dose rows "
                      f"({_el:.0f}s elapsed, ETA {_eta:.0f}s)", flush=True)

    if pin_block is not None:
        check_row_counts(pin_block, session_rows, allow_new=(pin_mode == "extend"))

    out = pd.DataFrame(rows)
    # The emitted column set is itself input-dependent (a feature type that produced no values
    # on this subset simply never appears), so it is pinned and enforced like the rest. Reindex
    # to the pinned order so a subset extraction is column-for-column comparable.
    if pin_block is not None and not out.empty:
        out = out[check_feature_columns(pin_block, out.columns)]
    if d0_normalize and not out.empty:
        # D0-normalize each feature PER SENSOR against its own earliest timepoint
        # (additive shift for bounded/phase/log-slope types, division for magnitude/area types) so the
        # model learns drift-from-baseline, not absolute per-sensor fabrication differences. CV-safe:
        # D0 is the earliest timepoint, always in the training side of a forward-chained fold. This is
        # also what deployment applies to in-vivo data (data.normalize.d0_normalize_frame), so the
        # frozen model sees the same representation at train and deploy.
        from ..data.schema import RESERVED_COLUMNS
        from .normalize import d0_normalize_frame
        feat_cols = [c for c in out.columns if c not in set(RESERVED_COLUMNS) and c != "time_index"]
        # A sensor's d0_row is its earliest timepoint *in this input set* -- so on a subset that
        # drops a sensor's true D0 session, the baseline silently moves to a later timepoint.
        # Pinned rows keep the original baseline; otherwise they are computed and recorded.
        out, d0_rows = d0_normalize_frame(
            out, feat_cols, d0_rows=(pinned_d0_rows(pin_block) if pin_block is not None else None),
            return_d0_rows=True)
    else:
        feat_cols, d0_rows = [], None

    from .pin import build_pin
    record = build_pin(
        band=band, band_source=band_source, device_d0=device_d0,
        ref_grid_hz=({t: g for t, g in _ref_grids.items()} if pin_block is None
                     else (pin_block.get("ref_grid_hz") or {})),
        feature_columns=list(out.columns), sessions=session_rows, device_types=device_types,
        d0_rows=(d0_rows if d0_normalize else None),
        params={"peak_method": peak_method, "acceptance": acceptance, "mono_tol": mono_tol,
                "min_norm_snr": min_norm_snr, "monotonic_r_min": monotonic_r_min,
                "mono_method": mono_method, "max_reps": max_reps, "detrend": detrend,
                "smooth_window": smooth_window, "smooth_poly": smooth_poly,
                "drop_negative": drop_negative, "require_interior_peak": require_interior_peak,
                "min_dose_response_range": min_dose_response_range,
                "d0_normalize": bool(d0_normalize),
                "feature_schema_version": FEATURE_SCHEMA_VERSION})
    out.attrs["extraction_pin"] = record
    if pin_out is not None:
        from ..data.io import write_json
        write_json(pin_out, record)
    return out


def replicate_feature_reliability(
    root: str | Path, band: tuple[float, float] | str = "auto",
    keep: set[tuple[str, int, float]] | None = None,
    peak_method: str = "direct", detrend: bool = False,
    baseline_window: tuple[float, float] = (-0.1, 0.2),
    mono_tol: float = 0.10, max_reps: int | None = 3,
    smooth_window: int = FSCV_SMOOTH_WINDOW, smooth_poly: int = FSCV_SMOOTH_POLY,
    n_jobs: int = 1, progress: bool = False) -> pd.DataFrame:
    """Per-feature measurement (replicate) variance ``σ²_meas`` for the drift-reliability bound.

    :func:`extract_dataset` stores ``rep_std`` only for the response, so per-*feature* reliability
    (Assumption 5 for the predictors) needs each feature recomputed **per replicate**. This walk does
    that: EIS features are computed on each **replicate spectrum** (observation = ``(device, channel,
    timepoint)``, since EIS is dose-invariant); the FSCV predictors ``mean_Vpeak`` / ``mean_Ibg`` and
    ``NormIpeak`` on each **cycle** (observation = ``(device, channel, timepoint, concentration)``).
    Each feature's ``σ²_meas`` is the pooled within-observation variance
    (:func:`~electropycal.diagnostics.variance.measurement_reliability`). Pair it with the drift variance
    from the averaged featureset via
    :func:`~electropycal.diagnostics.variance.drift_reliability`.

    ``keep`` (optional) restricts to a set of ``(device, channel, timepoint)`` — pass the averaged
    featureset's kept sensors so ``σ²_meas`` is measured on the same observations. Uses the same band /
    smoothing / peak conventions as ``extract_dataset``. Returns one row per feature with
    ``sigma2_meas``, ``n_rep`` (mean replicates per observation), ``n_obs``, and ``kind`` (``eis`` |
    ``fscv``).

    ``n_jobs`` (default **1**, serial) walks the device-timepoint sessions in parallel worker
    processes (joblib). The shared EIS ``ref_grid`` is fixed by a deterministic pre-pass and the
    observation ids are assigned in the same sorted-session order afterwards, so **the result is
    byte-identical to serial** — only faster. ``-1`` uses all cores.
    """
    from ..diagnostics.variance import measurement_reliability
    root = Path(root)
    if isinstance(band, str):
        if band != "auto":
            raise ValueError(f"band must be a (lo, hi) tuple or 'auto', got {band!r}")
        from ..data.inventory import recommended_band
        band = recommended_band(root)

    index: dict[tuple[str, object], dict] = defaultdict(lambda: {"fscv": {}})
    device_dates: dict[str, set] = defaultdict(set)
    for folder in sorted(p for p in root.iterdir() if p.is_dir()):
        fmeta = parse_folder(folder.name)
        if not fmeta or fmeta["testtype"] != "signal":
            continue
        for csv in folder.glob("*.csv"):
            nm = parse_filename(csv.name)
            if not nm:
                continue
            dev, date = nm["deviceid"], fmeta["date"]
            device_dates[dev].add(date)
            index[(dev, date)]["devicetype"] = fmeta["devicetype"]
            if nm["signaltype"] == "eis":
                index[(dev, date)]["eis"] = csv
            else:
                index[(dev, date)]["fscv"][nm["dose"]] = csv
    device_d0 = {dev: min(dates) for dev, dates in device_dates.items()}

    _items = sorted(index.items(), key=lambda kv: (kv[0][0], str(kv[0][1])))
    # Fix the EIS ref_grid ONCE per device type (deterministic pre-pass) so sessions are
    # independent and the parallel walk matches serial; observation ids are assigned below in the
    # same _items order. Shares _first_ref_grid with extract_dataset rather than re-deriving the
    # anchor here, so reliability statistics describe the same grid the featureset was built on.
    _ref_grids = _first_ref_grid(_items, band, mono_tol)
    P = {"peak_method": peak_method, "detrend": detrend, "baseline_window": baseline_window,
         "mono_tol": mono_tol, "max_reps": max_reps,
         "smooth_window": smooth_window, "smooth_poly": smooth_poly}
    from .._parallel import map_sessions, should_parallelize
    if should_parallelize(n_jobs, len(_items)):
        from joblib import delayed                  # loky processes (BLAS-capped); sessions independent
        results = map_sessions(n_jobs, (
            delayed(_reliability_session)(dev, date, entry, device_d0=device_d0, band=band,
                                          ref_grid=_ref_grids.get(entry.get("devicetype")),
                                          keep=keep, P=P)
            for (dev, date), entry in _items))       # joblib preserves order -> deterministic ids
    else:
        results = []
        for _i, ((dev, date), entry) in enumerate(_items, 1):
            results.append(_reliability_session(
                dev, date, entry, device_d0=device_d0, band=band,
                ref_grid=_ref_grids.get(entry.get("devicetype")), keep=keep, P=P))
            if progress:
                print(f"  {_i}/{len(_items)}  {dev}: replicate features accumulated", flush=True)

    # merge in _items order, assigning integer observation ids by first appearance (matches serial)
    vals: dict[str, list[float]] = defaultdict(list)
    obs: dict[str, list[int]] = defaultdict(list)
    kind: dict[str, str] = {}
    _oid: dict[tuple, int] = {}
    for _recs in results:
        for name, knd, okey, val in _recs:
            oid = _oid.setdefault(okey, len(_oid))
            vals[name].append(val); obs[name].append(oid); kind[name] = knd

    out_rows = []
    for name, v in vals.items():
        v = np.asarray(v, float)
        o = np.asarray(obs[name])
        _fin = np.isfinite(v)
        if _fin.sum() < 2:
            continue
        mr = measurement_reliability(v[_fin], o[_fin])
        _, counts = np.unique(o[_fin], return_counts=True)
        out_rows.append({"feature": name, "kind": kind[name], "sigma2_meas": mr["sigma2_meas"],
                         "n_rep": float(counts.mean()), "n_obs": int(counts.size)})
    return pd.DataFrame(out_rows).set_index("feature") if out_rows else pd.DataFrame(
        columns=["kind", "sigma2_meas", "n_rep", "n_obs"]).rename_axis("feature")
