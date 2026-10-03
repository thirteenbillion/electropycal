"""Raw-spectra review: data access + plots as a library API (folds the notebook's inline code).

The ``raw_spectra_review`` notebook used to carry ~500 lines of loaders and plotting inline. That logic
lives here instead, so the notebook is thin function calls and the *same* figures/tables are reproducible
from a script, the CLI, or another session without re-deriving anything.

Two pieces:

- :class:`RawSpectraIndex` — a filename-only index of a data ``ROOT`` (device / date / timepoint /
  signaltype / dose / path, no waveform reads up front) plus cached, replicate-averaged trace loaders
  (EIS spectra, FSCV raw / background-subtracted / normalized cycles, per-replicate peak stats). All
  processing knobs (replicates, peak method, Savitzky-Golay smoothing, band) come from the
  :class:`~electropycal.analysis_config.AnalysisConfig`, so a review matches what extraction computes.
- ``plot_*`` / ``*_table`` functions take an index and a device and draw one figure (channels as panels,
  timepoints overlaid) or return one tidy table. FSCV loop/peak plots surface the **edge-clip** flag
  (:func:`~electropycal.features.fscv.peak_edge_clipped`): a peak whose lobe is truncated by the sweep
  edge is marked, because its ``peak_area`` is an under-estimate.

Example::

    from electropycal.viz import set_pub_style
    from electropycal.rawspectra import RawSpectraIndex, plot_eis, plot_fscv_loops
    set_pub_style()
    idx = RawSpectraIndex("/path/to/data")
    for dev in idx.devices(device_types=["neurostring"]):
        plot_eis(idx, dev, "phase", band=idx.band)
        plot_fscv_loops(idx, dev, "bgsub", anodic_only=True)
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_config import AnalysisConfig, load_analysis_config, resolve_band
from .data.pstrace import parse_filename, parse_folder, read_pstrace
from .features.extract import _avg_fscv
from .features.fscv import (PEAK_WINDOW, DA_WINDOW, anodic_sweep, _locate_peak, norm_ipeak,
                            peak_at_edge, peak_edge_clipped, repeatability_snr, smooth_current)
from .viz import channel_color, concentration_color, panel_grid, timepoint_label, tp_alpha
from . import viz

_CHRE = re.compile(r"Channel\s+(\d+)")
_NAVY = "#1f2d7a"


def _fscv_channels_fast(path) -> list[int]:
    """Channels in an FSCV export from its header line only (no waveform parse)."""
    if path is None:
        return []
    for enc in ("utf-16", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                for _ in range(30):
                    ln = f.readline()
                    if not ln:
                        break
                    if "Fast Cyclic" in ln:
                        return sorted({int(x) for x in _CHRE.findall(ln)})
            return []
        except (UnicodeError, UnicodeDecodeError):
            continue
    return []


class NoSignalSessions(FileNotFoundError):
    """Raised when a tree holds signal files but none that a session table can be built from.

    :meth:`RawSpectraIndex.available_table` is built from **0 nM FSCV** exports only, since
    that background is what defines a session for this review. A tree can therefore hold
    plenty of EIS and dosed FSCV and still yield no rows, which is a different situation
    from an empty tree and needs a different message. Left unguarded it surfaced as
    ``KeyError: 'device'`` from inside a ``sort_values`` on a column-less frame, naming
    neither the tree nor what was missing from it.
    """


class RawSpectraIndex:
    """Filename index + cached trace loaders for a data ``root`` (see module docstring)."""

    def __init__(self, root, *, cfg: AnalysisConfig | None = None, replicates: int | None = None,
                 peak_method: str | None = None, smooth_window: int | None = None,
                 smooth_poly: int | None = None, band=None, total_channels: int = 16,
                 testtype: str = "signal"):
        self.root = str(root)
        self.cfg = cfg or load_analysis_config(root)
        self.replicates = self.cfg.max_reps if replicates is None else replicates
        self.peak_method = peak_method or self.cfg.peak_method
        from .features.fscv import FSCV_SMOOTH_WINDOW, FSCV_SMOOTH_POLY
        self.smooth_window = FSCV_SMOOTH_WINDOW if smooth_window is None else smooth_window
        self.smooth_poly = FSCV_SMOOTH_POLY if smooth_poly is None else smooth_poly
        self.total_channels = total_channels
        self._eis_cache: dict = {}
        self._fscv_cache: dict = {}
        self._ch_at: dict = {}

        recs, dates = [], defaultdict(set)
        from .data.paths import validate_raw_root
        root = validate_raw_root(root, 'root')
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            fm = parse_folder(folder.name)
            if not fm or fm.get("testtype") != testtype:
                continue
            for csv in sorted(folder.glob("*.csv")):
                m = parse_filename(csv.name)
                if not m:
                    continue
                dates[m["deviceid"]].add(fm["date"])
                recs.append({"device": m["deviceid"], "devicetype": fm.get("devicetype", "?"),
                             "date": fm["date"], "signaltype": m["signaltype"],
                             "dose": m["dose"], "path": str(csv)})
        self._d0 = {dev: min(ds) for dev, ds in dates.items()}
        self.signal = pd.DataFrame(recs)
        if len(self.signal):
            self.signal["timepoint"] = [(d - self._d0[dev]).days
                                        for dev, d in zip(self.signal.device, self.signal.date)]
        self.dtype_of = {r.device: r.devicetype for r in self.signal.itertuples()} if len(self.signal) else {}
        self.date_of = {(r.device, r.timepoint): r.date for r in self.signal.itertuples()} \
            if len(self.signal) else {}
        self.band = resolve_band(root, self.cfg) if band is None else band

    # ---- selection / identity ------------------------------------------------------------------
    def devices(self, device_types=None, devices=None) -> list[str]:
        devs = sorted(self.signal.device.unique()) if len(self.signal) else []
        if devices is not None:
            devs = [d for d in devices if d in devs]
        if device_types is not None:
            devs = [d for d in devs if self.dtype_of.get(d) in device_types]
        return devs

    def title(self, dev) -> str:
        return f"{self.dtype_of.get(dev, '?')} {dev}"

    def label_devices(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepend a ``devicetype`` column and rename ``device`` → ``deviceid`` (display copy)."""
        if "device" not in df.columns:
            return df
        out = df.copy()
        out.insert(0, "devicetype", out["device"].map(self.dtype_of))
        return out.rename(columns={"device": "deviceid"})

    def available_table(self) -> pd.DataFrame:
        """One row per signal session (from 0 nM FSCV headers only): device, date, timepoint, channels."""
        if not len(self.signal):
            return pd.DataFrame(columns=["device", "date", "timepoint", "n_channels", "channels"])
        rows = [{"device": r.device, "date": r.date, "timepoint": timepoint_label(r.timepoint),
                 "n_channels": len(chs), "channels": chs}
                for r in self.signal[(self.signal.signaltype == "fscv") & (self.signal.dose == 0.0)].itertuples()
                for chs in [_fscv_channels_fast(r.path)]]
        if not rows:
            # The guard above only asks whether ANY signal file was found. This table is
            # built from 0 nM FSCV alone, so a tree can clear that guard and still produce
            # nothing, and `pd.DataFrame([])` has no columns for `sort_values` to sort on.
            by_type = self.signal.signaltype.value_counts().to_dict()
            fscv = self.signal[self.signal.signaltype == "fscv"]
            fscv_doses = sorted({d for d in fscv.dose if isinstance(d, float)})
            raise NoSignalSessions(
                f"{self.root}: found {len(self.signal)} signal file(s) but none is a 0 nM FSCV "
                f"background, which is what a session row is built from.\n"
                f"  found by type : {by_type}\n"
                f"  FSCV doses    : {fscv_doses if fscv_doses else 'none'} nM "
                f"(0 nM is the one needed; a 0 nM EIS file does not substitute)\n"
                f"  needs         : a file per session named like "
                f"'<deviceid>_fscv_0nM.csv' beside the EIS and dosed FSCV exports.\n"
                f"If the tree was copied selectively, the 0 nM backgrounds are the ones to "
                f"add: every other table here is keyed off them.")
        return pd.DataFrame(rows).sort_values(["device", "date"]).reset_index(drop=True)

    def concs(self, dev=None, include_zero=True) -> list[float]:
        """Sorted numeric concentrations measured (globally, or by ``dev`` from its own files)."""
        sig = self.signal[self.signal.signaltype == "fscv"]
        if dev is not None:
            sig = sig[sig.device == dev]
        have = {float(x) for x in sig.dose.unique() if isinstance(x, (int, float))}
        return [c for c in sorted(have) if include_zero or c > 0]

    # ---- lazy channel / timepoint discovery ----------------------------------------------------
    def _channels_at(self, dev, tp, signaltype) -> list[int]:
        key = (dev, tp, signaltype)
        if key not in self._ch_at:
            if signaltype == "fscv":
                self._ch_at[key] = _fscv_channels_fast(self._fscv_path(dev, tp, 0.0))
            else:
                p = self._eis_path(dev, tp)
                self._ch_at[key] = self._read(p, self._eis_cache).eis_channels if p is not None else []
        return self._ch_at[key]

    def _device_tps(self, dev, signaltype) -> list[float]:
        dose0 = (self.signal.dose == 0.0) if signaltype == "fscv" else True
        return sorted(self.signal[(self.signal.device == dev) & (self.signal.signaltype == signaltype)
                                  & dose0].timepoint.unique())

    def eis_channels(self, dev) -> list[int]:
        return sorted({c for tp in self._device_tps(dev, "eis") for c in self._channels_at(dev, tp, "eis")})

    def fscv_channels(self, dev) -> list[int]:
        return sorted({c for tp in self._device_tps(dev, "fscv") for c in self._channels_at(dev, tp, "fscv")})

    def timepoints_with(self, dev, ch, signaltype, timepoints=None) -> list[float]:
        tps = [tp for tp in self._device_tps(dev, signaltype) if ch in self._channels_at(dev, tp, signaltype)]
        return [t for t in tps if timepoints is None or int(t) in {int(x) for x in timepoints}]

    # ---- raw loaders ---------------------------------------------------------------------------
    def _read(self, path, cache):
        if path not in cache:
            cache[path] = read_pstrace(path)
        return cache[path]

    def _eis_path(self, dev, tp):
        r = self.signal[(self.signal.device == dev) & (self.signal.timepoint == tp)
                        & (self.signal.signaltype == "eis")]
        return r.path.iloc[0] if len(r) else None

    def _fscv_path(self, dev, tp, dose):
        r = self.signal[(self.signal.device == dev) & (self.signal.timepoint == tp)
                        & (self.signal.signaltype == "fscv") & (self.signal.dose == dose)]
        return r.path.iloc[0] if len(r) else None

    def _sm(self, y, window):
        return smooth_current(y, window, self.smooth_poly) if window else np.asarray(y, float)

    def eis_full(self, dev, tp, ch):
        """Replicate-averaged EIS spectrum, ascending frequency: ``(freq, z_real, z_imag)`` or ``None``."""
        p = self._eis_path(dev, tp)
        if p is None:
            return None
        sp = [v for (c, _), v in self._read(p, self._eis_cache).eis.items() if c == ch]
        if not sp:
            return None
        fr = sp[0]["freq"]; o = np.argsort(fr)
        zr = np.mean([s["z_real"] for s in sp], axis=0); zi = np.mean([s["z_imag"] for s in sp], axis=0)
        return fr[o], zr[o], zi[o]

    def eis_reps(self, dev, tp, ch):
        """Per-replicate EIS spectra ``(freq, z_real, z_imag)`` (sorted) for the replicate-spread view."""
        p = self._eis_path(dev, tp)
        if p is None:
            return []
        out = []
        for s in [v for (c, _), v in sorted(self._read(p, self._eis_cache).eis.items()) if c == ch]:
            f = np.asarray(s["freq"], float); o = np.argsort(f)
            out.append((f[o], np.asarray(s["z_real"], float)[o], np.asarray(s["z_imag"], float)[o]))
        return out

    def fscv_raw(self, dev, tp, ch, dose):
        """Replicate-averaged raw FSCV cycle (full loop): ``(voltage, current)`` or ``None``."""
        p = self._fscv_path(dev, tp, dose)
        if p is None:
            return None
        a = _avg_fscv(self._read(p, self._fscv_cache), ch, max_reps=self.replicates)
        return (a["voltage"], a["current"]) if a is not None else None

    def _bg_interp(self, v, dev, tp, ch, smooth_window):
        bg = self.fscv_raw(dev, tp, ch, 0.0)
        if bg is None:
            return None
        bo = np.argsort(bg[0])
        return np.interp(v, bg[0][bo], self._sm(bg[1], smooth_window)[bo])

    def fscv_bgsub(self, dev, tp, ch, dose, smooth_window=0, anodic_only=False):
        """Background-subtracted cycle ``i − i_bg`` (bg = 0 nM); optional smoothing / anodic-only."""
        sig = self.fscv_raw(dev, tp, ch, dose)
        if sig is None:
            return None
        bgi = self._bg_interp(sig[0], dev, tp, ch, smooth_window)
        if bgi is None:
            return None
        y = self._sm(sig[1], smooth_window) - bgi
        return anodic_sweep(sig[0], y) if anodic_only else (sig[0], y)

    def fscv_norm(self, dev, tp, ch, dose, smooth_window=0, anodic_only=False):
        """Background-normalized cycle ``(i − i_bg) / i_bg`` — the NormIpeak-normalized loop."""
        sig = self.fscv_raw(dev, tp, ch, dose)
        if sig is None:
            return None
        bgi = self._bg_interp(sig[0], dev, tp, ch, smooth_window)
        if bgi is None:
            return None
        y = (self._sm(sig[1], smooth_window) - bgi) / np.where(bgi == 0, np.nan, bgi)
        return anodic_sweep(sig[0], y) if anodic_only else (sig[0], y)

    def peak_of(self, dev, tp, ch, dose, smooth_window=0):
        """Peak on the replicate-averaged smoothed anodic cycle: dict ``{v, ipeak, norm, clipped}`` or
        ``None``. ``clipped`` = the lobe is truncated by the sweep/window edge (``peak_area`` under-est.)."""
        sig = self.fscv_raw(dev, tp, ch, dose)
        if sig is None:
            return None
        bgi = self._bg_interp(sig[0], dev, tp, ch, smooth_window)
        if bgi is None:
            return None
        va, sa, ba = anodic_sweep(sig[0], self._sm(sig[1], smooth_window), bgi)
        bg_sub = sa - ba
        pk, v_ox, height = _locate_peak(bg_sub, va, PEAK_WINDOW, method=self.peak_method)
        if pk < 0 or ba[pk] == 0:
            return None
        clip = peak_edge_clipped(bg_sub, va, height, PEAK_WINDOW)
        return {"v": float(v_ox), "ipeak": float(bg_sub[pk]), "norm": float(height / ba[pk]),
                "clipped": bool(clip["clipped"])}

    def normipeak_reps(self, dev, tp, ch, dose, smooth_window=0) -> list[float]:
        """Per-replicate NormIpeak scalars (each replicate + the averaged 0 nM bg smoothed like extraction)."""
        ps, pb = self._fscv_path(dev, tp, dose), self._fscv_path(dev, tp, 0.0)
        if ps is None or pb is None:
            return []
        bg = _avg_fscv(self._read(pb, self._fscv_cache), ch, max_reps=self.replicates)
        if bg is None:
            return []
        bo = np.argsort(bg["voltage"]); bgc = self._sm(bg["current"], smooth_window)
        reps = [v for (c, _), v in sorted(self._read(ps, self._fscv_cache).fscv.items()) if c == ch][:self.replicates]
        out = []
        for rc in reps:
            v = rc["voltage"]; bgi = np.interp(v, bg["voltage"][bo], bgc[bo])
            val = norm_ipeak(self._sm(rc["current"], smooth_window), bgi, v, method=self.peak_method)
            if np.isfinite(val):
                out.append(float(val))
        return out

    def repeatability_snr_of(self, dev, tp, ch, dose, smooth_window=0) -> float:
        """Reproducibility SNR = averaged NormIpeak / std(per-replicate NormIpeak)."""
        reps = self.normipeak_reps(dev, tp, ch, dose, smooth_window=smooth_window)
        pk = self.peak_of(dev, tp, ch, dose, smooth_window)
        return float(repeatability_snr(pk["norm"] if pk else float("nan"), reps))

    def peak_stats_reps(self, dev, tp, ch, dose, smooth_window=0) -> dict:
        """Per-replicate peak stats: lists ``v_peak, raw_ipeak, bgsub_ipeak, normipeak``."""
        ps, pb = self._fscv_path(dev, tp, dose), self._fscv_path(dev, tp, 0.0)
        if ps is None or pb is None:
            return {}
        bg = _avg_fscv(self._read(pb, self._fscv_cache), ch, max_reps=self.replicates)
        if bg is None:
            return {}
        bo = np.argsort(bg["voltage"]); bgc = self._sm(bg["current"], smooth_window)
        reps = [v for (c, _), v in sorted(self._read(ps, self._fscv_cache).fscv.items()) if c == ch][:self.replicates]
        out = {"v_peak": [], "raw_ipeak": [], "bgsub_ipeak": [], "normipeak": []}
        for rc in reps:
            v = rc["voltage"]; bgi = np.interp(v, bg["voltage"][bo], bgc[bo])
            va, sa, ba = anodic_sweep(v, self._sm(rc["current"], smooth_window), bgi)
            bg_sub = sa - ba
            pk, v_ox, height = _locate_peak(bg_sub, va, PEAK_WINDOW, method=self.peak_method)
            if pk < 0 or ba[pk] == 0:
                continue
            out["v_peak"].append(float(v_ox)); out["raw_ipeak"].append(float(sa[pk]))
            out["bgsub_ipeak"].append(float(bg_sub[pk])); out["normipeak"].append(float(height / ba[pk]))
        return out

    def describe(self) -> str:
        """One block: the data ROOT, the resolved default settings, and how to customize them."""
        L = ["RawSpectraIndex settings (defaults from the shared AnalysisConfig):",
             f"  ROOT            = {self.root}",
             f"  band            = {self.band} Hz",
             f"  peak_method     = {self.peak_method!r}   (NormIpeak height: 'direct' | 'chord')",
             f"  smooth_window   = {self.smooth_window} samples (poly {self.smooth_poly}); 0 = off",
             f"  replicates      = {self.replicates}   (first N FSCV cycles averaged)",
             f"  total_channels  = {self.total_channels}   (color scale 1..N)",
             f"  devices found   = {len(self.devices())}  |  concentrations = {self.concs()}",
             "customize by rebuilding, e.g.:",
             "  IDX = RawSpectraIndex(ROOT, peak_method='chord', smooth_window=0, band=(2, 2000),",
             "                        replicates=3)   # any omitted knob falls back to the saved config"]
        return "\n".join(L)

    def print_settings(self) -> None:
        print(self.describe(), flush=True)

    @staticmethod
    def inductive_onset(freq, z_imag) -> float:
        """Lowest frequency where Im(Z) ≥ 0 (capacitive→inductive crossover); NaN if always capacitive."""
        f = np.asarray(freq, float); zi = np.asarray(z_imag, float); o = np.argsort(f)
        up = np.where(zi[o] >= 0)[0]
        return float(f[o][up[0]]) if up.size else float("nan")


# ==================================================================================================
# Plot functions (each draws one device figure: channels as panels, timepoints overlaid)
# ==================================================================================================

def _plt():
    import matplotlib.pyplot as plt
    return plt


def plot_eis(idx: RawSpectraIndex, dev, kind="phase", band=None, nyq_top=15, timepoints=None):
    """EIS figure: ``kind ∈ {'phase','mag','nyquist'}``. ``band`` restricts freqs; ``nyq_top`` keeps
    the N highest freqs on the Nyquist. Navy dashed = inductive onset (Bode); red points = inductive
    (fail EIS.1) on the Nyquist."""
    viz.ensure_style()
    from matplotlib.lines import Line2D
    plt = _plt()
    chs = idx.eis_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no EIS"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "eis", timepoints); col = channel_color(ch, idx.total_channels)
        for tp in tps:
            e = idx.eis_full(dev, tp, ch)
            if e is None:
                continue
            f, zr, zi = e
            if band is not None:
                m = (f >= band[0]) & (f <= band[1]); f, zr, zi = f[m], zr[m], zi[m]
                if f.size == 0:
                    continue
            al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
            onset = idx.inductive_onset(f, zi)
            if kind == "phase":
                a.semilogx(f, -np.degrees(np.arctan2(zi, zr)), "-", color=col, alpha=al, lw=lw,
                           label=timepoint_label(tp))
                if np.isfinite(onset):
                    a.axvline(onset, color=_NAVY, ls="--", lw=0.9, alpha=al)
            elif kind == "mag":
                a.loglog(f, np.hypot(zr, zi), "-", color=col, alpha=al, lw=lw, label=timepoint_label(tp))
                if np.isfinite(onset):
                    a.axvline(onset, color=_NAVY, ls="--", lw=0.9, alpha=al)
            else:
                idxs = np.argsort(f)
                if nyq_top:
                    idxs = idxs[-nyq_top:]
                ff, zrr, zii = f[idxs], zr[idxs], zi[idxs]
                a.plot(zrr, -zii, "-", color=col, alpha=al, lw=lw, label=timepoint_label(tp))
                if np.isfinite(onset):
                    mk = ff >= onset
                    a.plot(zrr[mk], -zii[mk], "o", color="red", ms=3.5, alpha=al)
        if kind == "phase":
            a.axhline(0, color="red", ls="--", lw=0.8); a.set_xlabel("freq (Hz)"); a.set_ylabel("-phase (deg)")
        elif kind == "mag":
            a.set_xlabel("freq (Hz)"); a.set_ylabel("|Z| (Ohm)")
        else:
            a.axhline(0, color="k", ls="--", lw=0.7); a.axvline(0, color="k", ls="--", lw=0.7)
            a.set_xlabel("Z' (Ohm)"); a.set_ylabel("-Z'' (Ohm)")
        a.set_title(f"ch {ch}", fontsize=8)
        h = a.get_legend_handles_labels()[0]
        if kind in ("phase", "mag"):
            extra = [Line2D([0], [0], color=_NAVY, ls="--", label="inductive onset")]
        else:
            extra = [Line2D([0], [0], color="red", marker="o", ls="", label="inductive (fail E.1)")]
        a.legend(handles=h + extra, fontsize=5.5, loc="best")
    name = {"phase": "Bode -phase", "mag": "Bode |Z|", "nyquist": "Nyquist"}[kind]
    suff = f" in BAND {tuple(int(b) for b in band)}" if band is not None else ""
    if kind == "nyquist" and nyq_top:
        suff += f" (top {nyq_top} freqs)"
    fig.suptitle(f"{idx.title(dev)} - EIS {name}{suff}", y=1.02)
    fig.tight_layout(); viz.emit("raw_eis")


def plot_eis_replicate_spread(idx: RawSpectraIndex, dev, comp="zr", timepoints=None):
    """EIS replicate spread (mean ± s.d. over replicate cycles). ``comp ∈ {'zr','zi'}`` (Z′ / Z″)."""
    viz.ensure_style()
    plt = _plt()
    j = 1 if comp == "zr" else 2
    lab = "Z' (Ohm)" if comp == "zr" else "Z'' = Im(Z) (Ohm)"
    tname = "Z'" if comp == "zr" else "Z''"
    chs = idx.eis_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no EIS"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "eis", timepoints); col = channel_color(ch, idx.total_channels)
        for tp in tps:
            reps = idx.eis_reps(dev, tp, ch)
            if not reps:
                continue
            al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
            f = reps[0][0]; M = np.vstack([r[j] for r in reps])
            mean, sd = M.mean(axis=0), M.std(axis=0)
            for r in reps:
                a.semilogx(f, r[j], "-", color=col, alpha=al * 0.30, lw=0.6)
            a.semilogx(f, mean, "-", color=col, alpha=al, lw=lw, label=timepoint_label(tp))
            a.fill_between(f, mean - sd, mean + sd, color=col, alpha=al * 0.18)
        a.set_xlabel("freq (Hz)"); a.set_ylabel(lab); a.set_title(f"ch {ch}", fontsize=8)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=5.5, loc="best")
    fig.suptitle(f"{idx.title(dev)} - EIS {tname} replicate spread (mean +/- s.d.)", y=1.02)
    fig.tight_layout(); viz.emit("raw_eis_replicate_spread")


def plot_inductive_onset_vs_time(idx: RawSpectraIndex, dev, band=None, timepoints=None):
    """Inductive-onset frequency vs timepoint (true day spacing, log Hz). ``band`` draws the upper
    bound + the EIS.1-fail zone."""
    viz.ensure_style()
    plt = _plt()
    chs = idx.eis_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no EIS"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        days, ons = [], []
        for tp in idx.timepoints_with(dev, ch, "eis", timepoints):
            e = idx.eis_full(dev, tp, ch)
            if e is None:
                continue
            days.append(int(tp)); ons.append(idx.inductive_onset(e[0], e[2]))
        a.plot(days, ons, "-o", color=channel_color(ch, idx.total_channels), ms=4, lw=1.4)
        a.set_yscale("log"); a.set_title(f"ch {ch}", fontsize=8)
        a.set_xlabel("timepoint (days)"); a.set_ylabel("onset (Hz)")
        if days:
            a.set_xticks(sorted(set(days)))
        if band is not None:
            a.axhline(band[1], color="red", ls="--", lw=0.9)
            top = a.get_ylim()[1]
            a.axhspan(band[1], max(top, band[1] * 1.05), color="red", alpha=0.06)
    fig.suptitle(f"{idx.title(dev)} - inductive onset frequency vs timepoint"
                 + (f" (BAND upper {int(band[1])} Hz)" if band is not None else ""), y=1.02)
    fig.tight_layout(); viz.emit("raw_inductive_onset_vs_time")


def plot_fscv_loops(idx: RawSpectraIndex, dev, transform="bgsub", concs=None, show_window=(0.0, 1.0),
                    scale_window=(0.5, 0.9), anodic_only=True, shared_y=False, smooth_window=None,
                    mark_clipped=False, timepoints=None):
    """FSCV cycles per concentration: channels as panels, timepoints overlaid.
    ``transform ∈ {'raw','bgsub','norm'}``. The located peak is dotted.

    - ``show_window`` — x-range drawn (default full ``(0.0, 1.0)`` V). ``scale_window`` (a sub-window) is
      what the y-axis autoscales to, so context can be shown past the peak while the y-scale frames it.
    - ``anodic_only`` (default True, bgsub/norm) draws only the rising sweep. Set ``anodic_only=False``
      to draw the **full loop (anodic + cathodic together)** — then widen ``show_window`` to see all
      currents (raw loops always show the full cycle).
    - ``mark_clipped`` (default **off** — too busy at scale) marks an **edge-clipped** peak (lobe
      truncated by the sweep edge → ``peak_area`` under-estimate) with a red ✕. The clip is always
      recorded on the extracted ``peak_area_clipped`` column regardless of this flag."""
    viz.ensure_style()
    from matplotlib.lines import Line2D
    plt = _plt()
    sw = idx.smooth_window if smooth_window is None else smooth_window
    ylab = {"raw": "i (uA)", "bgsub": "i - i_bg (uA)", "norm": "NormIpeak (i-i_bg)/i_bg"}[transform]

    def load(tp, ch, cc):
        if transform == "raw":
            return idx.fscv_raw(dev, tp, ch, cc)
        if transform == "bgsub":
            return idx.fscv_bgsub(dev, tp, ch, cc, smooth_window=sw, anodic_only=anodic_only)
        return idx.fscv_norm(dev, tp, ch, cc, smooth_window=sw, anodic_only=anodic_only)

    chs = idx.fscv_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no FSCV"); return
    have = set(idx.concs(dev))
    want = idx.concs(dev) if concs is None else [c for c in concs if c in have]
    any_clipped = False
    for cc in want:
        fig, axes = panel_grid(len(chs))
        panel_y = {}
        for a, ch in zip(axes, chs):
            tps = idx.timepoints_with(dev, ch, "fscv", timepoints); col = channel_color(ch, idx.total_channels)
            yscale = []
            for tp in tps:
                r = load(tp, ch, cc)
                if r is None:
                    continue
                x, y = r
                if show_window:
                    m = (x >= show_window[0]) & (x <= show_window[1]); x, y = x[m], y[m]
                if x.size == 0:
                    continue
                al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
                a.plot(x, y, "-", color=col, alpha=al, lw=lw, label=timepoint_label(tp))
                if transform != "raw":
                    pk = idx.peak_of(dev, tp, ch, cc, sw)
                    if pk is not None:
                        yv = pk["ipeak"] if transform == "bgsub" else pk["norm"]
                        a.plot(pk["v"], yv, "o", color=col, alpha=al, ms=5, mec="k", mew=0.4, zorder=3)
                        if mark_clipped and pk["clipped"]:
                            a.plot(pk["v"], yv, "x", color="red", ms=7, mew=1.4, zorder=4)
                            any_clipped = True
                        yscale.append(np.array([yv]))
                if scale_window is not None:
                    ms = (x >= scale_window[0]) & (x <= scale_window[1])
                    yscale.append(y[ms])
            a.axvspan(DA_WINDOW[0], DA_WINDOW[1], color="0.85", alpha=0.5)
            if transform != "raw":
                a.axhline(0, color="k", ls="--", lw=0.7, zorder=1); yscale.append(np.array([0.0]))
            if show_window:
                a.set_xlim(*show_window)
            panel_y[a] = yscale
            if scale_window is not None and yscale and not shared_y:
                v = np.concatenate([s for s in yscale if s.size]); v = v[np.isfinite(v)]
                if v.size:
                    lo, hi = min(float(v.min()), 0.0), max(float(v.max()), 0.0)
                    pad = 0.08 * (hi - lo or 1.0); a.set_ylim(lo - pad, hi + pad)
            a.set_xlabel("E (V)"); a.set_ylabel(ylab); a.set_title(f"ch {ch}", fontsize=8)
            h = a.get_legend_handles_labels()
            extra = [Line2D([0], [0], color="red", marker="x", ls="", label="edge-clipped area")] \
                if (mark_clipped and any_clipped) else []
            if h[1] or extra:
                a.legend(handles=h[0] + extra, fontsize=5.5, loc="best")
        if scale_window is not None and shared_y:
            allv = np.concatenate([s for ys in panel_y.values() for s in ys if s.size] or [np.array([0.0])])
            allv = allv[np.isfinite(allv)]
            if allv.size:
                lo, hi = min(float(allv.min()), 0.0), max(float(allv.max()), 0.0)
                pad = 0.08 * (hi - lo or 1.0)
                for a in axes[:len(chs)]:
                    a.set_ylim(lo - pad, hi + pad)
        fig.suptitle(f"{idx.title(dev)} - FSCV {transform} @ {int(cc)} nM", y=1.02)
        fig.tight_layout(); viz.emit("raw_fscv_loops")


def plot_dose_response(idx: RawSpectraIndex, dev, timepoints=None):
    """NormIpeak vs concentration (per-replicate points + mean ± s.d. line), channels as panels."""
    viz.ensure_style()
    plt = _plt()
    chs = idx.fscv_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no FSCV"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "fscv", timepoints); col = channel_color(ch, idx.total_channels)
        for tp in tps:
            al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
            cvals, means, stds = [], [], []
            for cc in idx.concs(dev, include_zero=False):
                reps = idx.normipeak_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                if not reps:
                    continue
                a.plot([cc] * len(reps), reps, "o", color=col, alpha=al * 0.55, ms=3)
                cvals.append(cc); means.append(np.mean(reps)); stds.append(np.std(reps))
            if cvals:
                a.errorbar(cvals, means, yerr=stds, fmt="-", color=col, alpha=al, lw=lw, capsize=2,
                           label=timepoint_label(tp))
        a.axhline(0, color="k", ls="--", lw=0.7, zorder=0)
        a.set_ylim(bottom=min(0.0, a.get_ylim()[0]))
        a.set_xscale("log"); a.set_xlabel("concentration (nM)"); a.set_ylabel("NormIpeak")
        a.set_title(f"ch {ch}", fontsize=8)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=5.5, loc="best")
    fig.suptitle(f"{idx.title(dev)} - NormIpeak vs concentration ({idx.peak_method})", y=1.02)
    fig.tight_layout(); viz.emit("raw_dose_response")


def plot_snr_vs_conc(idx: RawSpectraIndex, dev, timepoints=None):
    """Reproducibility SNR (mean/σ over replicate cycles) vs concentration; red dashed = MIN_NORM_SNR cutoff."""
    viz.ensure_style()
    from matplotlib.lines import Line2D
    plt = _plt()
    cutoff = idx.cfg.min_norm_snr
    chs = idx.fscv_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no FSCV"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "fscv", timepoints); col = channel_color(ch, idx.total_channels)
        for tp in tps:
            al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
            cvals, means, stds = [], [], []
            for cc in idx.concs(dev, include_zero=False):
                reps = idx.normipeak_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                if len(reps) < 2:
                    continue
                sd = float(np.std(reps))
                if sd <= 0:
                    continue
                pts = np.asarray(reps, float) / sd
                a.plot([cc] * len(pts), pts, "o", color=col, alpha=al * 0.5, ms=3)
                cvals.append(cc); means.append(float(np.mean(reps)) / sd); stds.append(float(np.std(pts)))
            if cvals:
                a.errorbar(cvals, means, yerr=stds, fmt="-", color=col, alpha=al, lw=lw, capsize=2,
                           label=timepoint_label(tp))
        a.axhline(cutoff, color="red", ls="--", lw=1.1, zorder=2)
        a.axhline(0, color="k", ls="--", lw=0.7, zorder=1)
        lo = min(0.0, a.get_ylim()[0]); hi = max(cutoff, a.get_ylim()[1])
        pad = 0.06 * (hi - lo or 1.0); a.set_ylim(lo - pad, hi + pad)
        a.set_xscale("log"); a.set_xlabel("concentration (nM)")
        a.set_ylabel("NormIpeak / std(reps)"); a.set_title(f"ch {ch}", fontsize=8)
        h = a.get_legend_handles_labels()[0]
        extra = [Line2D([0], [0], color="red", ls="--", label=f"cutoff ({cutoff:g}x)"),
                 Line2D([0], [0], marker="o", color="0.5", ls="", label="replicates / sigma")]
        a.legend(handles=h + extra, fontsize=5.5, loc="best")
    fig.suptitle(f"{idx.title(dev)} - reproducibility SNR vs concentration (red = cutoff {cutoff:g}x)", y=1.02)
    fig.tight_layout(); viz.emit("raw_snr_vs_conc")


def plot_vpeak_vs_conc(idx: RawSpectraIndex, dev, timepoints=None):
    """V_peak vs concentration; gray = ideal DA window, crimson bands = edge-pinned zones."""
    viz.ensure_style()
    from .features.fscv import PEAK_EDGE_TOL
    plt = _plt()
    chs = idx.fscv_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no FSCV"); return
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "fscv", timepoints); col = channel_color(ch, idx.total_channels)
        a.axhspan(DA_WINDOW[0], DA_WINDOW[1], color="0.85", alpha=0.6, zorder=0)
        a.axhspan(PEAK_WINDOW[0], PEAK_WINDOW[0] + PEAK_EDGE_TOL, color="crimson", alpha=0.12, zorder=0)
        a.axhspan(PEAK_WINDOW[1] - PEAK_EDGE_TOL, PEAK_WINDOW[1], color="crimson", alpha=0.12, zorder=0)
        for tp in tps:
            al = tp_alpha(tp, tps); lw = 1.4 if al == 1.0 else 1.0
            cvals, means, stds = [], [], []
            for cc in idx.concs(dev, include_zero=False):
                st = idx.peak_stats_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                if not st or not st["v_peak"]:
                    continue
                a.plot([cc] * len(st["v_peak"]), st["v_peak"], "o", color=col, alpha=al * 0.55, ms=3)
                cvals.append(cc); means.append(np.mean(st["v_peak"])); stds.append(np.std(st["v_peak"]))
            if cvals:
                a.errorbar(cvals, means, yerr=stds, fmt="-", color=col, alpha=al, lw=lw, capsize=2,
                           label=timepoint_label(tp))
        a.set_xscale("log"); a.set_ylim(*PEAK_WINDOW)
        a.set_xlabel("concentration (nM)"); a.set_ylabel("V_peak (V)"); a.set_title(f"ch {ch}", fontsize=8)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=5.5, loc="best")
    fig.suptitle(f"{idx.title(dev)} - V_peak vs concentration ({idx.peak_method}); gray = ideal DA window", y=1.02)
    fig.tight_layout(); viz.emit("raw_vpeak_vs_conc")


def onset_table(idx: RawSpectraIndex, devices, timepoints=None) -> pd.DataFrame:
    """Inductive-onset frequency per (device, channel, timepoint) — with a ``devicetype`` column."""
    rows = []
    for dev in devices:
        for ch in idx.eis_channels(dev):
            for tp in idx.timepoints_with(dev, ch, "eis", timepoints):
                e = idx.eis_full(dev, tp, ch)
                if e is None:
                    continue
                rows.append({"devicetype": idx.dtype_of.get(dev, "?"), "device": dev,
                             "date": idx.date_of.get((dev, tp)), "timepoint": timepoint_label(tp),
                             "channel": int(ch), "inductive_onset_frequency_Hz": idx.inductive_onset(e[0], e[2])})
    return pd.DataFrame(rows)


def eis_spread_table(idx: RawSpectraIndex, devices, timepoints=None, wide=False) -> pd.DataFrame:
    """EIS replicate mean/σ of Z′/Z″ per (device, channel, timepoint, frequency). ``wide`` pivots
    frequency across columns (one row per device·channel·timepoint)."""
    rows = []
    for dev in devices:
        for ch in idx.eis_channels(dev):
            for tp in idx.timepoints_with(dev, ch, "eis", timepoints):
                reps = idx.eis_reps(dev, tp, ch)
                if not reps:
                    continue
                f = reps[0][0]; ZR = np.vstack([r[1] for r in reps]); ZI = np.vstack([r[2] for r in reps])
                for k in range(len(f)):
                    rows.append({"device": dev, "date": idx.date_of.get((dev, tp)),
                                 "timepoint": timepoint_label(tp), "channel": int(ch),
                                 "frequency_Hz": round(float(f[k]), 3),
                                 "Zr_mean": round(float(ZR[:, k].mean()), 2), "Zr_std": round(float(ZR[:, k].std()), 3),
                                 "Zi_mean": round(float(ZI[:, k].mean()), 2), "Zi_std": round(float(ZI[:, k].std()), 3)})
    tbl = pd.DataFrame(rows)
    if wide and len(tbl):
        return (tbl.pivot_table(index=["device", "date", "timepoint", "channel"], columns="frequency_Hz",
                                values=["Zr_mean", "Zr_std", "Zi_mean", "Zi_std"])
                .swaplevel(axis=1).sort_index(axis=1, level=0))
    return tbl


def plot_vs_time(idx: RawSpectraIndex, dev, kind="norm", timepoints=None):
    """Temporal drift: channels as panels, one plasma-colored trace per concentration, true day
    spacing on x. ``kind ∈ {'norm','vpeak','snr'}``."""
    viz.ensure_style()
    plt = _plt()
    chs = idx.fscv_channels(dev)
    if not chs:
        print(f"{idx.title(dev)}: no FSCV"); return
    concs = idx.concs(dev, include_zero=False)
    fig, axes = panel_grid(len(chs))
    for a, ch in zip(axes, chs):
        tps = idx.timepoints_with(dev, ch, "fscv", timepoints)
        if kind == "vpeak":
            a.axhspan(DA_WINDOW[0], DA_WINDOW[1], color="0.85", alpha=0.6, zorder=0)
        for cc in concs:
            col = concentration_color(cc, concs)
            if kind == "snr":
                days, snrs = [], []
                for tp in tps:
                    vals = [v for v in idx.normipeak_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                            if np.isfinite(v)]
                    if len(vals) < 2:
                        continue
                    sd = float(np.std(vals))
                    if sd <= 0:
                        continue
                    days.append(int(tp)); snrs.append(float(np.mean(vals)) / sd)
                if days:
                    a.plot(days, snrs, "-o", color=col, ms=4, lw=1.4, label=f"{int(cc)} nM")
                continue
            days, means, stds = [], [], []
            for tp in tps:
                if kind == "norm":
                    vals = idx.normipeak_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                else:
                    st = idx.peak_stats_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                    vals = st.get("v_peak", []) if st else []
                if not vals:
                    continue
                a.plot([tp] * len(vals), vals, "o", color=col, alpha=0.5, ms=3)
                days.append(int(tp)); means.append(float(np.mean(vals))); stds.append(float(np.std(vals)))
            if days:
                a.errorbar(days, means, yerr=stds, fmt="-o", color=col, ms=4, lw=1.4, capsize=2,
                           label=f"{int(cc)} nM")
        if kind == "norm":
            a.axhline(0, color="k", ls="--", lw=0.7, zorder=1)
            a.set_ylim(bottom=min(0.0, a.get_ylim()[0])); a.set_ylabel("NormIpeak")
        elif kind == "snr":
            a.axhline(idx.cfg.min_norm_snr, color="crimson", ls="--", lw=0.9, zorder=1)
            a.set_ylim(bottom=0); a.set_ylabel("repeatability SNR")
        else:
            a.set_ylim(*PEAK_WINDOW); a.set_ylabel("V_peak (V)")
        t = sorted({int(x) for x in tps})
        if t:
            a.set_xticks(t); a.set_xticklabels([timepoint_label(x) for x in t])
            a.set_xlim(min(t) - 0.5, max(t) + 0.5)
        a.set_xlabel("timepoint (days)"); a.set_title(f"ch {ch}", fontsize=8)
        if a.get_legend_handles_labels()[1]:
            a.legend(fontsize=5.5, loc="best")
    ttl = {"norm": "NormIpeak vs time", "vpeak": "V_peak vs time", "snr": "repeatability SNR vs time"}[kind]
    fig.suptitle(f"{idx.title(dev)} - {ttl} ({idx.peak_method}; color = concentration [plasma])", y=1.02)
    fig.tight_layout(); viz.emit("raw_vs_time")


def dose_stats_table(idx: RawSpectraIndex, devices, timepoints=None) -> pd.DataFrame:
    """Per (device, channel, timepoint, concentration) replicate mean/std of V_peak, raw/bgsub/Norm Ipeak."""
    def ms(vals, nd):
        a = np.asarray(vals, float)
        return round(float(a.mean()), nd), round(float(a.std()), nd)
    rows = []
    for dev in devices:
        for ch in idx.fscv_channels(dev):
            for tp in idx.timepoints_with(dev, ch, "fscv", timepoints):
                for cc in idx.concs(dev, include_zero=False):
                    st = idx.peak_stats_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                    if not st or not st["normipeak"]:
                        continue
                    vpk = ms(st["v_peak"], 3); rip = ms(st["raw_ipeak"], 3)
                    bip = ms(st["bgsub_ipeak"], 3); nip = ms(st["normipeak"], 4)
                    rows.append({"device": dev, "date": idx.date_of.get((dev, tp)),
                                 "timepoint": timepoint_label(tp), "channel": int(ch), "concentration": cc,
                                 "Vpeak_mean": vpk[0], "Vpeak_std": vpk[1], "rawIpeak_mean": rip[0],
                                 "rawIpeak_std": rip[1], "bgsubIpeak_mean": bip[0], "bgsubIpeak_std": bip[1],
                                 "NormIpeak_mean": nip[0], "NormIpeak_std": nip[1]})
    return pd.DataFrame(rows)


def edge_pinning_table(idx: RawSpectraIndex, devices, timepoints=None) -> pd.DataFrame:
    """Per-concentration tally of V_peak pinned at a PEAK_WINDOW bound (low-conc clipping diagnostic)."""
    from .features.fscv import PEAK_EDGE_TOL
    lo_b, hi_b = PEAK_WINDOW
    tally = defaultdict(lambda: {"n": 0, "lower": [], "upper": [], "any": []})
    for dev in devices:
        for ch in idx.fscv_channels(dev):
            for tp in idx.timepoints_with(dev, ch, "fscv", timepoints):
                for cc in idx.concs(dev, include_zero=False):
                    st = idx.peak_stats_reps(dev, tp, ch, cc, smooth_window=idx.smooth_window)
                    if not st or not st["v_peak"]:
                        continue
                    vp = float(np.mean(st["v_peak"])); t = tally[cc]; t["n"] += 1
                    lab = f"{dev} ch{ch} {timepoint_label(tp)}"
                    at_lo, at_hi = vp <= lo_b + PEAK_EDGE_TOL, vp >= hi_b - PEAK_EDGE_TOL
                    if at_lo:
                        t["lower"].append(lab)
                    if at_hi:
                        t["upper"].append(lab)
                    if at_lo or at_hi:
                        t["any"].append(lab)

    def pct(k, n):
        return round(100 * k / n, 1) if n else float("nan")
    return pd.DataFrame([
        {"concentration": cc, "n": t["n"], "n_lower": len(t["lower"]), "lower_edge_%": pct(len(t["lower"]), t["n"]),
         "n_upper": len(t["upper"]), "upper_edge_%": pct(len(t["upper"]), t["n"]),
         "n_any": len(t["any"]), "any_edge_%": pct(len(t["any"]), t["n"]), "any_chan_tps": t["any"]}
        for cc, t in sorted(tally.items())])
