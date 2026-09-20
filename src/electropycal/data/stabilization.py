"""FSCV stabilization convergence check — optional.

The interface is "stabilized" when the background FSCV cycle stops changing
cycle-to-cycle. This module provides an objective plateau criterion for that, as
an alternative to manual review. It works two ways:

- **live logging (recommended):** the acquisition script logs a per-cycle scalar
  drift metric (tiny, avoids saving all cycles); pass that series to
  :func:`check_converged`.
- **offline:** if a subset of raw background cycles was saved, compute the drift
  series first with :func:`cycle_drift`.

Not wired into the mandatory pipeline — stabilization is judged during measurement.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def i_at_v_target(voltage: np.ndarray, current: np.ndarray, v_target: float = 0.7) -> float:
    """Interpolate the FSCV current at ``v_target`` on the anodic (rising) sweep.

    ``v_target`` is the dopamine oxidation potential by default; this scalar is the
    quantity whose cycle-to-cycle stabilization we track.
    """
    from ..features.fscv import anodic_sweep
    v, i = anodic_sweep(voltage, current)
    if v.size == 0:
        return float("nan")
    return float(np.interp(v_target, v, i))


def stabilization_traces(export, v_target: float = 0.7, cycles_per_round: int = 20):
    """Tidy per-cycle ``I(v_target)`` table from a parsed stabilization export.

    ``export`` is a :class:`~electropycal.data.pstrace.PSTraceExport` whose FSCV blocks
    are the stabilization cycles, one block per cycle. The round comes from each
    block's trailing ``[n]`` bracket when present (real ``Channel N Scan k [round]``
    labels); otherwise it falls back to ``rep // cycles_per_round``.

    **Cycle order is temporal:** cycles are sorted by ``(round, numeric scan)`` when
    scans are present — real PSTrace exports lay the columns out in *lexicographic*
    scan order (1, 10, 11, …, 19, 2, 20, 3, …), which is **not** acquisition order, so
    ordering by column appearance would make consecutive cycles non-adjacent in time
    and spuriously inflate the cycle-to-cycle drift. Without scans it falls back to
    column-appearance order (``rep``).

    Returns a DataFrame with columns ``channel``, ``cycle`` (0-based per channel),
    ``round`` (0-based), ``cycle_in_round``, ``scan`` (PSTrace scan id or NaN), and
    ``i_target``. Also carries the full-cycle ``_voltage`` / ``_current`` arrays so the
    notebook can plot the raw I–V loop and compute a cycle-to-cycle drift metric.
    """
    import pandas as pd
    recs = []
    for (ch, rep), cyc in export.fscv.items():
        scan = cyc.get("scan")
        rnd = int(cyc["bracket"]) if scan is not None else int(rep) // cycles_per_round
        recs.append({
            "channel": int(ch),
            "rep": int(rep),
            "round": rnd,
            "scan": scan,
            "i_target": i_at_v_target(cyc["voltage"], cyc["current"], v_target),
            "_voltage": np.asarray(cyc["voltage"], float),
            "_current": np.asarray(cyc["current"], float),
        })
    df = pd.DataFrame(recs)
    # temporal order: by (round, numeric scan) if scans exist, else column-appearance.
    order = ["channel", "round", "scan"] if df["scan"].notna().any() else ["channel", "rep"]
    df = df.sort_values(order).reset_index(drop=True)
    df["cycle"] = df.groupby("channel").cumcount()
    df["cycle_in_round"] = df.groupby(["channel", "round"]).cumcount()
    return df[["channel", "cycle", "round", "cycle_in_round", "scan",
               "i_target", "_voltage", "_current"]]


def cycle_drift(cycles: np.ndarray, voltage: np.ndarray | None = None,
                region: str = "anodic") -> np.ndarray:
    """Per-cycle normalized drift ``‖cₖ − cₖ₋₁‖ / ‖cₖ‖`` for a run of FSCV cycles
    ``(n_cycles, n_samples)``. Returns length ``n_cycles − 1``.

    ``region="anodic"`` (default) restricts the comparison to the rising (anodic)
    sweep — the faradaic half, excluding the switching-potential spike — when
    ``voltage`` (the shared 1-D sweep grid) is given; ``region="full"`` (or no
    ``voltage``) uses the whole cycle.
    """
    c = np.asarray(cycles, float)
    if region == "anodic" and voltage is not None:
        v = np.asarray(voltage, float)
        if v.size == c.shape[1]:
            imin, imax = int(np.argmin(v)), int(np.argmax(v))
            c = c[:, min(imin, imax):max(imin, imax) + 1]
    diffs = np.linalg.norm(np.diff(c, axis=0), axis=1)
    norms = np.linalg.norm(c[1:], axis=1)
    return diffs / np.where(norms == 0, np.nan, norms)


def drift_series(traces_ch, voltage: np.ndarray | None = None,
                 region: str = "anodic", drop_round_boundaries: bool = True) -> np.ndarray:
    """Boundary-filtered per-cycle drift for one channel's :func:`stabilization_traces` rows.

    Sorts the rows by ``cycle``, stacks the full-cycle ``_current`` waveforms, and calls
    :func:`cycle_drift`. When ``drop_round_boundaries`` is set (default), the two diffs
    adjacent to every **round-start cycle** (``cycle_in_round == 0``) are dropped.

    This is what removes the periodic drift spikes seen on real PSTrace data: the first
    cycle of each round (scan 1) is a *cold-start* whose ``I(v_target)`` sits well off the
    settled value, so both entering it (prev round's last scan → scan 1) and leaving it
    (scan 1 → scan 2) show up as large, non-physical cycle-to-cycle jumps. Keying on
    ``cycle_in_round == 0`` catches the first round's cold-start too (unlike a plain
    ``cycle % cycles_per_round == 0`` test that special-cases cycle 0).

    Pass a channel-subset DataFrame (``traces[traces.channel == ch]``). ``voltage`` is
    taken from the rows' ``_voltage`` when present; the explicit argument is a fallback.
    """
    t = traces_ch.sort_values("cycle")
    cur = np.vstack(t["_current"].to_numpy())
    v = t["_voltage"].iloc[0] if "_voltage" in t.columns else voltage
    dr = cycle_drift(cur, voltage=v, region=region)
    if drop_round_boundaries and "cycle_in_round" in t.columns and dr.size:
        is_start = t["cycle_in_round"].to_numpy() == 0
        keep = ~(is_start[:-1] | is_start[1:])
        dr = dr[keep]
    return dr


def round_average_cycles(traces_ch, drop_coldstart: bool = True):
    """Per-round mean FSCV waveform for one channel's :func:`stabilization_traces` rows.

    The acquisition measures every channel for ``cycles_per_round`` consecutive cycles
    (a *round*), then repeats all channels in the next round. Averaging the cycles within
    each round collapses within-round measurement noise and the per-round cold-start,
    leaving one clean waveform per round.

    Returns ``(rounds, avg_current, voltage)``: the sorted round ids, an
    ``(n_rounds, n_samples)`` array of per-round mean current, and the shared voltage grid.
    With ``drop_coldstart`` (default), the first cycle of each round
    (``cycle_in_round == 0``) — a cold-start transient that sits off the settled value —
    is excluded from its round mean.
    """
    t = traces_ch
    if drop_coldstart and "cycle_in_round" in t.columns:
        t = t[t["cycle_in_round"] != 0]
    rounds, avg = [], []
    for rnd, g in t.sort_values("cycle").groupby("round"):
        rounds.append(int(rnd))
        avg.append(np.vstack(g["_current"].to_numpy()).mean(axis=0))
    voltage = np.asarray(traces_ch["_voltage"].iloc[0], float)
    return np.asarray(rounds), np.asarray(avg, float), voltage


def round_drift_series(traces_ch, region: str = "anodic",
                       drop_coldstart: bool = True) -> np.ndarray:
    """Round-to-round normalized drift for one channel: average each round's cycles
    (:func:`round_average_cycles`), then :func:`cycle_drift` across the round means.

    This is the robust convergence signal for many-round stabilization runs — within-round
    noise, the per-round cold-start, and occasional glitch cycles are averaged out, so the
    series tracks the genuine round-over-round settling of the electrode rather than
    per-cycle jitter. Length ``n_rounds - 1`` (empty if fewer than two rounds).
    """
    _rounds, avg, voltage = round_average_cycles(traces_ch, drop_coldstart=drop_coldstart)
    if avg.shape[0] < 2:
        return np.array([])
    return cycle_drift(avg, voltage=voltage, region=region)


def _smooth(x: np.ndarray, window: int) -> np.ndarray:
    if window <= 1 or x.size < window:
        return x
    return np.convolve(x, np.ones(window) / window, mode="same")


@dataclass
class ConvergenceResult:
    converged: bool
    converged_at: int | None    # index in the drift series where the stable run began
    n_stable: int               # trailing consecutive below-tolerance points
    final_drift: float
    tol: float
    smoothed: np.ndarray


def check_converged(drift: np.ndarray, tol: float = 0.01, patience: int = 10,
                    smooth_window: int = 5) -> ConvergenceResult:
    """Plateau detection: converged if the most recent ``patience`` smoothed drift
    values are all ≤ ``tol`` (the same safety-valve pattern as CARS/Q²-tracking).

    ``converged_at`` marks the cycle at which the trailing stable run began.
    """
    d = np.asarray(drift, float)
    sm = _smooth(d, smooth_window)
    below = sm <= tol
    n_stable = 0
    for b in below[::-1]:
        if b:
            n_stable += 1
        else:
            break
    converged = n_stable >= patience
    converged_at = (len(sm) - n_stable) if converged else None
    return ConvergenceResult(converged, converged_at, int(n_stable),
                             float(sm[-1]) if sm.size else float("nan"), tol, sm)


def estimate_settle_tau(drift: np.ndarray) -> float:
    """Time constant (in cycles) of an exponential fit ``A·e^{-t/τ}+c`` to the
    drift envelope — a summary of how fast the interface settles. NaN if the fit
    fails or the series is too short."""
    d = np.asarray(drift, float)
    d = d[np.isfinite(d)]
    if d.size < 5:
        return float("nan")
    t = np.arange(d.size)
    try:
        from scipy.optimize import curve_fit
        (a, tau, c), _ = curve_fit(lambda t, a, tau, c: a * np.exp(-t / tau) + c,
                                   t, d, p0=[d[0] - d[-1], max(d.size / 3, 1.0), d[-1]],
                                   maxfev=5000, bounds=([-np.inf, 1e-3, -np.inf], np.inf))
        return float(tau)
    except Exception:
        return float("nan")


def stabilization_table(drift: np.ndarray, tol: float = 0.01, smooth_window: int = 5):
    """Tidy DataFrame (cycle, drift, smoothed, below_tol) for plotting the curve."""
    import pandas as pd
    d = np.asarray(drift, float)
    sm = _smooth(d, smooth_window)
    return pd.DataFrame({"cycle": np.arange(1, d.size + 1), "drift": d,
                         "smoothed": sm, "below_tol": sm <= tol})
