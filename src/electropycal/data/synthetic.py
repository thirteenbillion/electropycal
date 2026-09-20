"""Synthetic dataset matching the featureset schema (DESIGN §3), for running the
pipeline and notebooks end-to-end before real PSTrace data is wired in.

Generates a post-extraction featureset (X: N×157, y: NormIpeak) with realistic
structure — between-channel offsets, temporal drift of sensitivity, a dose
response, a handful of *informative* EIS features that track sensitivity, plus
replicate-level values so the measurement-reliability diagnostic runs.
Swap this out for real feature extraction once PSTrace CSV parsing lands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .schema import EIS_GLOBAL_FEATURES, FSCV_FEATURES


def _feature_names() -> list[str]:
    """157 feature names: 2 FSCV + 143 freq-EIS + 12 global (DESIGN §3)."""
    names = list(FSCV_FEATURES)
    types = ["R_s", "R_p", "C_s", "C_p", "ideality_C", "tau", "local_n"]
    counts = [21, 21, 21, 20, 20, 20, 20]  # 63 + 80 = 143
    for t, c in zip(types, counts):
        names += [f"{t}_f{i:02d}" for i in range(c)]
    names += list(EIS_GLOBAL_FEATURES)
    return names  # 2 + 143 + 12 = 157


@dataclass
class SyntheticDataset:
    X: np.ndarray            # (N, 157)
    y: np.ndarray            # (N,) NormIpeak
    channel: np.ndarray      # (N,)
    timepoint: np.ndarray    # (N,) days: 0, 1, 7, 20
    concentration: np.ndarray  # (N,) nM
    feature_names: list[str]
    replicates: np.ndarray   # (N, 157, R) per-replicate feature values
    informative: np.ndarray  # indices of features that carry sensitivity signal

    @property
    def n_features(self) -> int:
        return self.X.shape[1]


def make_dataset(
    n_channels: int = 7,
    timepoints: tuple[int, ...] = (0, 1, 7, 20),
    concentrations: tuple[float, ...] = (100, 250, 500, 1000, 5000),
    n_informative: int = 8,
    n_replicates: int = 3,
    drop_fraction: float = 0.36,   # ~89/160 valid, the QC yield measured on real data
    meas_noise: float = 0.02,
    random_state: int = 0,
) -> SyntheticDataset:
    """Build a synthetic featureset with a learnable sensitivity signal."""
    rng = np.random.default_rng(random_state)
    names = _feature_names()
    p = len(names)

    # full grid of (channel, timepoint, concentration)
    grid = [(c, t, k) for c in range(n_channels) for t in timepoints for k in concentrations]
    ch = np.array([g[0] for g in grid])
    tp = np.array([g[1] for g in grid], dtype=float)
    conc = np.array([g[2] for g in grid], dtype=float)
    n_full = len(grid)

    # latent sensitivity s(channel, timepoint): channel offset + downward drift
    ch_offset = rng.normal(0, 0.3, size=n_channels)
    s = 1.0 + ch_offset[ch] - 0.02 * tp + rng.normal(0, 0.03, size=n_full)

    # base features: channel structure + mild drift + noise (uninformative)
    X = (rng.normal(0, 1, size=(n_full, p))
         + ch_offset[ch][:, None] * rng.normal(0, 0.2, size=(1, p))
         - 0.005 * tp[:, None])

    # a few informative features track sensitivity s
    informative = rng.choice(p, size=n_informative, replace=False)
    alpha = rng.normal(0, 1, size=n_informative)
    X[:, informative] = alpha[None, :] * s[:, None] + rng.normal(0, 0.1, size=(n_full, n_informative))

    # response: sensitivity part (learnable from X) + dose response (mostly not) + noise
    conc_scaled = np.log10(conc) / np.log10(concentrations[-1])
    beta = rng.normal(0, 1, size=n_informative)
    y = X[:, informative] @ beta + 1.5 * conc_scaled * s + rng.normal(0, 0.05, size=n_full)
    y = y - y.min() + 0.05      # NormIpeak is positive (enables log_plsr)

    # replicate-level measurement noise (feeds the reliability diagnostic)
    reps = X[:, :, None] + rng.normal(0, meas_noise, size=(n_full, p, n_replicates))
    reps[:, informative, :] += rng.normal(0, meas_noise * 0.5,
                                          size=(n_full, n_informative, n_replicates))

    # drop rows to mimic data gaps (keep all D0 so baselines exist)
    keep = rng.random(n_full) >= drop_fraction
    keep |= (tp == 0)  # never drop the D0 baseline
    idx = np.where(keep)[0]

    return SyntheticDataset(
        X=X[idx], y=y[idx], channel=ch[idx], timepoint=tp[idx],
        concentration=conc[idx], feature_names=names,
        replicates=reps[idx], informative=informative,
    )


def make_invivo_drift(
    ds: SyntheticDataset,
    timepoints: tuple[int, ...] = (0, 7, 21, 42),
    drift_rate: float = 0.15,
    random_state: int = 1,
) -> dict[int, np.ndarray]:
    """Progressively drifted feature matrices per in-vivo timepoint (for CORAL demo).

    Applies a growing mean shift + covariance inflation to the in-vitro X, so
    CORAL distance increases with time — the drift path deployment tracks.
    """
    rng = np.random.default_rng(random_state)
    base = ds.X
    direction = rng.normal(0, 1, size=base.shape[1])
    out: dict[int, np.ndarray] = {}
    for t in timepoints:
        scale = 1.0 + drift_rate * t / max(timepoints)
        shift = drift_rate * (t / max(timepoints)) * direction
        out[t] = base * scale + shift + rng.normal(0, 0.05 * t / max(timepoints) + 1e-6, size=base.shape)
    return out


# --- synthetic raw PSTrace directory (for the raw-data dashboard / ingestion tests) ---

import datetime as _dt  # noqa: E402
from pathlib import Path as _Path  # noqa: E402


# ---------------------------------------------------------------------------------------------
# Reference statistics measured from real in-vitro neurostring PSTrace sessions
# (4 sessions, 69 PSTrace exports: 114 EIS spectra and 1,026 FSCV sweeps). The synthetic writers
# below reproduce these so the demo trees are dimensionally and distributionally realistic rather
# than schematic. Re-measure with ``scripts/measure_invitro_stats.py`` if the reference data changes.
#
#   EIS   37 log-spaced points, 1 Hz - 1 MHz. |Z| 1.25e4 Ohm (high f) -> 2.1e6 Ohm (low f).
#         Randles asymptotes give Rs ~ 1.25e4, Rct ~ 2.08e6; -Z'' still rising at the 1 Hz floor,
#         so the semicircle apex sits at/below 1 Hz => Cdl ~ 7.6e-8 F.
#   FSCV  340 samples per triangular sweep, V -0.400 -> +1.299 -> -0.400 V.
#         0 nM background: anodic dI/dV ~ 18.4 uA/V, offset ~ 6.1 uA, anodic/cathodic capacitive
#         loop ~ 22 uA wide; currents span roughly +-50 uA and are centred near 0.
#         Dopamine oxidation peak: V_ox ~ 0.819 V, FWHM ~ 0.44 V (sigma ~ 0.187 V),
#         peak dI median 0.28 uA (p25 0.14, p75 0.63, p95 2.82).
REAL_INVITRO_STATS = {
    "eis": {"n_freq": 37, "f_lo": 1.0, "f_hi": 1e6, "Rs": 1.251e4, "Rct": 2.083e6, "Cdl": 7.6e-8},
    "fscv": {"n_sweep": 340, "v_lo": -0.400, "v_hi": 1.299, "bg_slope": 18.39, "bg_offset": 6.12,
             "loop_uA": 22.29, "v_ox": 0.819, "peak_sigma": 0.187,
             # dose response: NormIpeak = normipeak_at_1uM * (C/1uM)**peak_beta. The exponent is the
             # measured log-log slope of median NormIpeak vs concentration (Freundlich, sub-linear).
             "normipeak_at_1uM": 0.02405, "peak_beta": 0.394,
             # measurement noise. ``noise_uA`` is the amplitude of a *correlated* (low-pass) current
             # noise with correlation length ``noise_tau`` samples — electrochemical noise is not
             # white, and white noise of the same power would manufacture spurious sharp peaks.
             # ``rep_cv`` is the replicate-to-replicate scatter of the peak height.
             #
             # Calibrated to reproduce the real repeatability_snr (~5.6) and dose_response_r (~0.95)
             # while keeping QC deterministic: the only channel that fails quality in the demo is the
             # one the generator deliberately breaks. Turning the noise up far enough to also match
             # the real noise_floor (~0.017) starts failing the FSCV.1 monotonicity gate on healthy
             # early channels, which would make the demo's QC dropout random rather than explainable,
             # so the extracted noise_floor here (~0.005) is deliberately quieter than real.
             "noise_uA": 0.15, "noise_tau": 9, "rep_cv": 0.12,
             # the max-based peak estimator is biased upward by noise, on real and synthetic data
             # alike; scale the injected amplitude so the *extracted* NormIpeak matches the measured
             # ``normipeak_at_1uM`` instead of overshooting it.
             "peak_bias_correction": 0.88},
}
_EIS = REAL_INVITRO_STATS["eis"]
_FSCV = REAL_INVITRO_STATS["fscv"]

#: voltage separation between the two branches of the triangular sweep (see ``_sweep``). Far below
#: the instrument's own float32 resolution, so it is physically invisible, but it keeps anodic and
#: cathodic samples from colliding exactly.
BRANCH_EPS = 1e-8


def _fv(x) -> str:
    """Format a voltage. Needs enough significant digits to preserve ``BRANCH_EPS`` (~1e-8 on a
    ~1 V sweep) through the CSV round-trip — otherwise the two branches collide again."""
    return f"{x:.12g}"


def _fi(x) -> str:
    """Format a current / impedance. Instrument exports carry ~7 significant digits; writing full
    float64 repr instead would roughly double the size of the demo tree for no added fidelity."""
    return f"{x:.7g}"


def _sweep():
    """The real triangular FSCV sweep: ``n_sweep`` samples, ``v_lo`` -> ``v_hi`` -> ``v_lo``.

    The descending branch is the ascending one shifted up by one step and reversed, so the apex is
    its first sample — the real instrument's geometry (``desc[::-1] ≈ asc + step``, to within the
    ~1e-8 V of its float32 export).

    This is load-bearing, not cosmetic. A perfect mirror (``asc`` then ``asc[::-1]``) places anodic
    and cathodic samples at *bit-identical* voltages, and the background alignment in
    ``extract_dataset`` interpolates the signal grid onto the **sorted** background voltages — on an
    exact tie ``np.interp`` returns the last match, i.e. the cathodic sample. Every anodic point
    would then be subtracted against the other branch and ``NormIpeak`` would report the ~22 uA
    capacitive loop instead of the ~0.3 uA faradaic peak. Real exports are safe only because their
    two branches never land on bit-identical voltages.
    """
    half = _FSCV["n_sweep"] // 2
    step = (_FSCV["v_hi"] - _FSCV["v_lo"]) / (half - 1)
    asc = _FSCV["v_lo"] + step * np.arange(half)
    # BRANCH_EPS keeps every descending sample off its ascending twin. Plain float64 arithmetic
    # already separates most of them, but ~26 of 170 land on bit-identical values by coincidence,
    # and each of those becomes a spurious ~25 uA spike in the background-subtracted trace. The
    # real instrument's branches sit ~3.7e-8 V apart; this is the same idea, made deterministic.
    desc = (asc + step + BRANCH_EPS)[::-1][: _FSCV["n_sweep"] - half]
    return np.concatenate([asc, desc])


def _background(v, ch=0):
    """0 nM capacitive background: a sloped charging current with an anodic/cathodic loop.

    The forward (anodic) half sits ``loop_uA/2`` above the mean ramp and the reverse half the same
    distance below, reproducing the measured hysteresis loop instead of a single-valued line. The
    per-channel scaling is centred on 1.0 so the *median* channel reproduces the measured background
    rather than inflating it.
    """
    half = len(v) // 2
    ramp = _FSCV["bg_offset"] + _FSCV["bg_slope"] * v
    side = np.concatenate([np.full(half, 0.5), np.full(len(v) - half, -0.5)]) * _FSCV["loop_uA"]
    return (ramp + side) * (1.0 + 0.05 * (ch - 5))    # mild per-channel area scaling


def _background_at_vox(ch=0):
    """The background level at ``v_ox`` — the denominator of ``NormIpeak``."""
    return ((_FSCV["bg_offset"] + _FSCV["bg_slope"] * _FSCV["v_ox"] + _FSCV["loop_uA"] / 2)
            * (1.0 + 0.05 * (ch - 5)))


def _peak(v, amp_uA):
    """Dopamine oxidation peak at the measured ``v_ox`` / ``peak_sigma``."""
    return amp_uA * np.exp(-((v - _FSCV["v_ox"]) / _FSCV["peak_sigma"]) ** 2 / 2.0)


def _peak_uA(conc_nM, ch=0):
    """Peak height in uA for a concentration.

    Chosen so the resulting ``NormIpeak`` (= peak / background at ``v_ox``) follows the measured
    Freundlich dose response ``normipeak_at_1uM * (C/1uM)**peak_beta``.
    """
    if conc_nM <= 0:
        return 0.0
    nip = _FSCV["normipeak_at_1uM"] * (conc_nM / 1000.0) ** _FSCV["peak_beta"]
    return nip * _background_at_vox(ch) * _FSCV["peak_bias_correction"]


def _current_noise(rng, n):
    """Correlated (low-pass) current noise — see ``noise_uA`` / ``noise_tau`` in the stats block."""
    tau = int(_FSCV["noise_tau"])
    kernel = np.exp(-np.arange(4 * tau) / tau)
    kernel /= np.linalg.norm(kernel)
    white = rng.standard_normal(n + 4 * tau)
    return np.convolve(white, kernel, mode="same")[:n] * _FSCV["noise_uA"]


def _write_eis_file(path, channels, bad=(), Rs=_EIS["Rs"], Rct=_EIS["Rct"], Cdl=_EIS["Cdl"],
                    reps=3, random_state=0):
    """Write an EIS export for a **Randles cell** (``Rs`` in series with a ``Rct‖Cdl``
    parallel combination): ``Z = Rs + Rct/(1 + jωRctCdl)``. This traces the standard
    charge-transfer **semicircle** in the Nyquist plane (``Z'`` sweeps ``Rs+Rct`` → ``Rs``
    with frequency), unlike a pure ``Rs+C`` series whose constant ``Z'`` would draw a
    vertical line. Still passes EIS checks A–C (capacitive, ``|Z|``-monotone, ``Z' > 0``).

    ``reps`` replicate spectra per channel are written (each a separate ``CH N`` block with
    small multiplicative noise), so replicate-spread views have something to show."""
    rng = np.random.default_rng(random_state)
    freqs = np.logspace(np.log10(_EIS["f_lo"]), np.log10(_EIS["f_hi"]), _EIS["n_freq"])
    w = 2 * np.pi * freqs
    L = ["Date and time:,2025-01-01 00:00:00", "Notes:",
         "Measurement:,Impedance Spectroscopy", "Notes:,", "Date and time:,2025-01-01"]
    for ch in channels:
        cdl = Cdl * (1 + 0.1 * ch)                     # per-channel double-layer capacitance
        x = w * Rct * cdl
        zr0 = Rs + Rct / (1 + x**2)                    # Nyquist semicircle: Rs+Rct (low f) -> Rs (high f)
        im0 = -w * Rct**2 * cdl / (1 + x**2)           # capacitive (Im(Z) < 0)
        if ch in bad:                                  # amplifier-saturation-like failure (C check)
            zr0 = -zr0
        for _r in range(reps):
            zr = zr0 * (1 + 0.02 * rng.standard_normal(zr0.shape))   # small per-replicate noise
            im = im0 * (1 + 0.02 * rng.standard_normal(im0.shape))
            zmag = np.hypot(zr, im)
            stored = -im
            negph = -np.degrees(np.arctan2(im, zr))
            L.append(f"CH {ch}: Fixed at {len(freqs)} freqs")
            L.append("freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F")
            L += [f"{_fi(freqs[k])},{_fi(negph[k])},1,{_fi(zmag[k])},{_fi(zr[k])},{_fi(stored[k])},1e-9"
                  for k in range(len(freqs) - 1, -1, -1)]
    _Path(path).write_text("\n".join(L), encoding="utf-16")


def _write_fscv_file(path, channels, conc_nM=0.0, reps=3, random_state=0, sens=1.0):
    """Write an FSCV export for one dose.

    ``conc_nM`` sets the oxidation-peak height through the measured Freundlich dose response
    (:func:`_peak_uA`); ``sens`` scales it, so a drifting sensor can be written by passing
    ``sens < 1``. Background, sweep geometry, replicate scatter (``rep_cv``) and the additive
    current noise (``noise_uA``) all come from the measured real statistics, so the extracted
    ``noise_floor`` and ``repeatability_snr`` land in the real range instead of being noiseless.
    """
    rng = np.random.default_rng(random_state)
    v = _sweep()                                       # full triangular sweep (anodic + cathodic)
    hdr, un = [], []
    for ch in channels:
        for r in range(reps):
            hdr += [f"Fast Cyclic Voltammetry: FCV i vs E Channel {ch}" + ("" if r == 0 else f" [{r}]"), ""]
            un += ["V", "µA"]
    L = ["Date and time:,2025-01-01", "Notes:", ",,,,", ",".join(hdr),
         ",".join(["dt"] * len(hdr)), ",".join(un)]
    bg = {ch: _background(v, ch) for ch in channels}
    # per (channel, replicate) trace: peak scatter around the dose response + additive noise
    trace = {}
    for ch in channels:
        amp = _peak_uA(conc_nM, ch) * sens
        for r in range(reps):
            a = amp * (1.0 + _FSCV["rep_cv"] * rng.standard_normal()) if amp > 0 else 0.0
            trace[(ch, r)] = bg[ch] + _peak(v, a) + _current_noise(rng, len(v))
    for i in range(len(v)):
        cells = []
        for ch in channels:
            for r in range(reps):
                cells += [_fv(v[i]), _fi(trace[(ch, r)][i])]
        L.append(",".join(cells))
    _Path(path).write_text("\n".join(L), encoding="utf-16")


def _write_fscv_stabilization_file(path, channels, rounds=3, cycles_per_round=20,
                                   v_target=0.7, random_state=0, settled=True):
    """Write a ``<deviceid>_fscv_stabilization-{start,end}.csv`` file: the first or
    last ``rounds`` rounds of FSCV stabilization, ``cycles_per_round`` consecutive
    cycles per channel per round, stored as consecutive ``Channel N Scan k [round]``
    blocks per channel.

    ``I(v_target)`` drifts down and plateaus across cycles (an exponential settle).
    With ``settled=False`` (a *stabilization-start* file) the interface is still
    drifting — a long time-constant and large residual — so start-vs-end comparison
    shows the improvement; ``settled=True`` (*stabilization-end*) has converged.
    """
    rng = np.random.default_rng(random_state)
    v = _sweep()                                       # full triangular sweep
    total = rounds * cycles_per_round
    hdr, un = [], []
    for ch in channels:
        for k in range(total):
            rnd, scan = divmod(k, cycles_per_round)     # round 0/1/2, scan 0..19
            br = "" if rnd == 0 else f" [{rnd}]"          # trailing bracket = round
            hdr += [f"Fast Cyclic Voltammetry: FCV i vs E Channel {ch} Scan {scan + 1}{br}", ""]
            un += ["V", "µA"]
    # per-(channel, cycle) peak height at v_target: converge to a plateau
    peak = {}
    for ci, ch in enumerate(channels):
        # end file: short time-constant, small noise (settled); start file: long
        # time-constant + larger residual + more noise (still equilibrating)
        tau = (8.0 + 3.0 * ci) if settled else (45.0 + 8.0 * ci)
        # amplitudes in uA, on the scale of the measured capacitive background drift
        amp = 3.0 if settled else 7.0
        noise = 0.08 if settled else 0.24
        k = np.arange(total)
        env = amp * np.exp(-k / tau)                   # decaying drift envelope
        peak[ch] = 4.0 + env + rng.normal(0, noise, size=total)
    L = ["Date and time:,2025-01-01", "Notes:", ",,,,", ",".join(hdr),
         ",".join(["dt"] * len(hdr)), ",".join(un)]
    bgs = {ch: _background(v, ch) for ch in channels}
    for i in range(len(v)):
        cells = []
        for ch in channels:
            for k in range(total):
                pk = peak[ch][k] * np.exp(
                    -((v[i] - v_target) / _FSCV["peak_sigma"]) ** 2 / 2.0)
                cells += [_fv(v[i]), _fi(bgs[ch][i] + pk)]
        L.append(",".join(cells))
    _Path(path).write_text("\n".join(L), encoding="utf-16")


def _write_paired_file(path, channels, n_time, sens=1.0, drift=0.0,
                       Rs=_EIS["Rs"], C=_EIS["Cdl"], random_state=0):
    """Write an in-vivo ``paired`` export: per channel, ``n_time`` time-sequential
    FSCV cycles (horizontal blocks) followed by ``n_time`` EIS spectra (vertical
    blocks), in the same internal PSTrace format as in-vitro. ``sens`` scales the
    FSCV oxidation peak; ``drift`` (0..1) grows the EIS capacitance and shrinks the
    peak across time, so CORAL domain distance and recalibrated NormIpeak evolve.
    """
    rng = np.random.default_rng(random_state)
    v = _sweep()
    # --- FSCV section (horizontal blocks: channel x time) ---
    hdr, un = [], []
    for ch in channels:
        for t in range(n_time):
            tag = "" if t == 0 else f" [{t}]"
            hdr += [f"Fast Cyclic Voltammetry: FCV i vs E Channel {ch}{tag}", ""]
            un += ["V", "µA"]
    L = ["Date and time:,2025-01-01", "Notes:", ",,,,", ",".join(hdr),
         ",".join(["dt"] * len(hdr)), ",".join(un)]
    bgs = {ch: _background(v, ch) for ch in channels}
    for i in range(len(v)):
        cells = []
        for ch in channels:
            for t in range(n_time):
                s = sens * (1.0 - drift * t / max(n_time - 1, 1))     # sensitivity fades with drift
                pk = _peak(v[i], _peak_uA(1000.0) * s)
                cells += [_fv(v[i]), _fi(bgs[ch][i] + pk + rng.normal(0, 0.01))]
        L.append(",".join(cells))
    # --- EIS section (vertical blocks: channel x time) ---
    freqs = np.logspace(np.log10(_EIS["f_lo"]), np.log10(_EIS["f_hi"]), _EIS["n_freq"])
    w = 2 * np.pi * freqs
    L.append("Measurement:,Impedance Spectroscopy")
    for ch in channels:
        for t in range(n_time):
            Ct = C * (1 + 0.1 * ch) * (1 + drift * t / max(n_time - 1, 1))  # capacitance grows
            im = -1.0 / (w * Ct)
            zr = np.full_like(freqs, Rs)
            zmag = np.hypot(zr, im)
            negph = -np.degrees(np.arctan2(im, zr))
            L.append(f"CH {ch}: Fixed at {len(freqs)} freqs")
            L.append("freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F")
            L += [f"{_fi(freqs[k])},{_fi(negph[k])},1,{_fi(zmag[k])},{_fi(zr[k])},{_fi(-im[k])},1e-9"
                  for k in range(len(freqs) - 1, -1, -1)]
    _Path(path).write_text("\n".join(L), encoding="utf-16")


def write_synthetic_invivo_dir(root, devices=("3-2", "4-2"), sessions=(0, 7, 21),
                               channels=(3, 5, 6), n_live=8, random_state=0):
    """Create a raw in-vivo tree: ``<date>_neurostring_signal/<dev>_paired_{baseline,
    live}.csv``. ``baseline`` is a short stable reference; ``live`` has ``n_live``
    time samples with growing drift per later session (rising CORAL distance).
    Returns the root path.
    """
    root = _Path(root)
    base = _dt.date(2026, 8, 1)
    for di, dev in enumerate(devices):
        for si, day in enumerate(sessions):
            date = base + _dt.timedelta(days=day)
            sig = root / f"{date.strftime('%Y%m%d')}_neurostring_signal"
            sig.mkdir(parents=True, exist_ok=True)
            drift = 0.15 * si                                    # more drift each later session
            _write_paired_file(sig / f"{dev}_paired_baseline.csv", channels, n_time=3,
                               sens=1.0, drift=0.0, random_state=di * 10 + si)
            _write_paired_file(sig / f"{dev}_paired_live.csv", channels, n_time=n_live,
                               sens=1.0, drift=drift, random_state=di * 10 + si + 100)
    return root


#: Every stabilization spelling this generator can emit, richest last. ``-full`` is what the
#: current acquisition protocols produce; ``-start``/``-end`` are an older convention, kept so
#: the two-phase branch of ``stabreview.chosen_phases`` stays exercisable.
STAB_PHASES = ("start", "end", "full")


def write_synthetic_pstrace_dir(root, devices=("2-2", "2-3"), timepoints=(0, 1, 7, 20),
                                channels=(3, 5, 6, 7), concentrations=(0, 100, 500, 1000, 5000),
                                total_channels=16, stagger_days=1, random_state=0,
                                stab_phases=STAB_PHASES, stab_cycles_per_round=20):
    """Create a full raw PSTrace tree (channeltest + signal folders) for the dashboard.

    Models channel dropout (one channel lost per later timepoint) and one EIS
    quality failure, so survival/quality views have something to show. Returns the
    root path.

    ``stab_phases`` selects which stabilization spellings to emit (see :data:`STAB_PHASES`);
    ``stab_cycles_per_round`` sets how many consecutive cycles each round holds. Both default
    to the full, 20-cycle tree so library callers and the test-suite fixtures are unchanged.
    The shipped demo overrides them — stabilization CSVs dominate its size, and file size is
    linear in ``len(stab_phases) x rounds x stab_cycles_per_round``.

    Lowering ``stab_cycles_per_round`` is safe for the review path: rounds come from each
    block's ``[n]`` bracket, and ``stabilization_traces`` derives the global cycle index by
    counting, so nothing infers the value back. Do not go below 3 — ``round_average_cycles``
    drops each round's cold-start cycle, so 2 would leave a single cycle to "average".
    """
    unknown = set(stab_phases) - set(STAB_PHASES)
    if unknown:
        raise ValueError(f"unknown stabilization phase(s) {sorted(unknown)}; "
                         f"expected a subset of {list(STAB_PHASES)}")
    if stab_phases and stab_cycles_per_round < 3:
        raise ValueError(
            f"stab_cycles_per_round={stab_cycles_per_round} is too small: round_average_cycles "
            f"drops each round's cold-start cycle, so fewer than 3 leaves at most one cycle per "
            f"round to average and the round-drift series stops meaning anything.")
    root = _Path(root)
    base = _dt.date(2026, 7, 15)
    for di, dev in enumerate(devices):
        for tp in timepoints:
            date = base + _dt.timedelta(days=tp + di * stagger_days)
            ds = date.strftime("%Y%m%d")
            surv = list(channels[: len(channels) - (tp // 7)])  # lose a channel after weeks
            if not surv:
                continue
            # sensitivity fades as the electrode ages, so the demo has a drift trajectory to model
            sens = 1.0 - 0.015 * tp
            ct = root / f"{ds}_neurostring_channeltest"
            ct.mkdir(parents=True, exist_ok=True)
            _write_fscv_file(ct / f"{dev}_fscv_0nm.csv", surv, conc_nM=0.0,
                             random_state=di * 1000 + tp)
            sig = root / f"{ds}_neurostring_signal"
            sig.mkdir(parents=True, exist_ok=True)
            bad = (surv[-1],) if tp == max(timepoints) else ()   # one late quality failure
            _write_eis_file(sig / f"{dev}_eis_0nm.csv", surv, bad=bad)
            _stab = {  # (rounds, random_state offset, settled)
                "start": (3, 0, False),
                "end": (3, 50, True),
                "full": (6, 70, True),
            }
            for ph in stab_phases:
                rounds, off, settled = _stab[ph]
                _write_fscv_stabilization_file(
                    sig / f"{dev}_fscv_stabilization-{ph}.csv", surv, rounds=rounds,
                    cycles_per_round=stab_cycles_per_round,
                    random_state=di * 100 + tp + off, settled=settled)
            for c in concentrations:
                _write_fscv_file(sig / f"{dev}_fscv_{c}nm.csv", surv, conc_nM=c, sens=sens,
                                 random_state=di * 1000 + tp * 10 + int(c) % 7)
    return root
