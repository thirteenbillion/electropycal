"""Quality-filtering dashboard: QC computation + plots as a library API (folds quality_filtering_dashboard).

The notebook used to compute the channel-timepoint quality table, the per-concentration check registry,
the gating logic, and a dozen plots/sweeps inline — and its QC-stats artifact was assembled from a web of
notebook globals. :class:`QCDashboard` owns all of that: it runs
:func:`~electropycal.data.inventory.channel_quality_report`, expands it with the per-dose SNR checks and
the gating decision, exposes one method per figure/table, and assembles+saves the QC-stats companion from
its own attributes (no globals). The §6.1 finalized-dataset overview reuses
:mod:`electropycal.overview`.
"""

from __future__ import annotations

import datetime as _dt
import re
from collections import defaultdict
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis_config import AnalysisConfig, load_analysis_config, resolve_band, save_qc_stats
from .data.inventory import channel_quality_report
from .data.pstrace import parse_filename, parse_folder
from .features.fscv import PEAK_EDGE_TOL
from .viz import channel_color, timepoint_label

_CHRE = re.compile(r"Channel\s+(\d+)")


def _fscv_channels_fast(path) -> list[int]:
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


class QCDashboard:
    """Runs QC over a data ``ROOT`` and holds every table + plot the dashboard needs (see module docstring).

    Parameters mirror the notebook's shared config + local selection: ``cfg`` (defaults from
    ``<ROOT>/electropycal_analysis_config.json``), ``devices`` / ``device_types`` / ``timepoints``
    selection, ``require_interior_peak``, ``n_jobs``. After construction, ``CH_TBL`` / ``DOSE_TBL`` and
    the check registry are populated; call the ``plot_*`` / ``*_table`` methods, then ``save_stats()``.
    """

    def __init__(self, root, *, cfg: AnalysisConfig | None = None, devices=None, device_types=None,
                 timepoints=None, require_interior_peak: bool = False, n_jobs: int = 4,
                 total_channels: int = 16, progress: bool = True):
        self.root = str(root)
        self.cfg = cfg or load_analysis_config(root)
        self.total_channels = total_channels
        self.band = resolve_band(root, self.cfg)

        # available index (0 nM FSCV headers only) + device-type map
        recs, dates, self.dtype_of = [], defaultdict(set), {}
        for folder in sorted(p for p in Path(root).iterdir() if p.is_dir()):
            fm = parse_folder(folder.name)
            if not fm or fm.get("testtype") != "signal":
                continue
            for csv in sorted(folder.glob("*.csv")):
                m = parse_filename(csv.name)
                if m and m["signaltype"] == "fscv" and m["dose"] == 0.0:
                    dates[m["deviceid"]].add(fm["date"]); recs.append((m["deviceid"], fm["date"], str(csv)))
                    self.dtype_of[m["deviceid"]] = fm.get("devicetype", "?")
        self._d0 = {dev: min(ds) for dev, ds in dates.items()}
        self._avail = pd.DataFrame([
            {"device": dev, "date": date, "timepoint": timepoint_label((date - self._d0[dev]).days),
             "n_channels": len(chs), "channels": chs}
            for dev, date, path in recs for chs in [_fscv_channels_fast(path)]])
        if len(self._avail):
            self._avail = self._avail.sort_values(["device", "date"]).reset_index(drop=True)

        all_devices = sorted(self._avail.device.unique()) if len(self._avail) else []
        self.devices = list(devices) if devices is not None else all_devices
        if device_types is not None:
            self.devices = [d for d in self.devices if self.dtype_of.get(d) in device_types]

        # run QC on the selected devices
        self.CH_TBL, self.DOSE_TBL = channel_quality_report(
            root, band=self.band, devices=self.devices, mono_tol=self.cfg.mono_tol,
            peak_method=self.cfg.peak_method, max_reps=self.cfg.max_reps, min_norm_snr=self.cfg.min_norm_snr,
            monotonic_r_min=self.cfg.monotonic_r_min, mono_method=self.cfg.mono_method,
            require_interior_peak=require_interior_peak, n_jobs=n_jobs, progress=progress)
        if timepoints is not None:
            self.CH_TBL = self.CH_TBL[self.CH_TBL.timepoint.isin(timepoints)].reset_index(drop=True)
            self.DOSE_TBL = self.DOSE_TBL[self.DOSE_TBL.timepoint.isin(timepoints)].reset_index(drop=True)
        self._build_registry_and_gates()
        # sweep stats, cached by the plot methods for save_stats()
        self._paired = self.CH_TBL[self.CH_TBL.has_eis & self.CH_TBL.has_fscv]
        self._imp = self._r = self._rs = self._nmed = self._cmp = self._snr = self._zr = self._summary = None

    # ---- registry + gating (notebook cell 7 tail) ----------------------------------------------
    def _build_registry_and_gates(self):
        self.concs_sorted = sorted(self.DOSE_TBL.concentration.unique())
        n = len(self.concs_sorted)
        key = ["device", "date", "timepoint", "channel"]
        self.conc_checks = []
        for i, cc in enumerate(self.concs_sorted):
            col = f"fscv_snr_{int(cc)}"
            sub = (self.DOSE_TBL[self.DOSE_TBL.concentration == cc][key + ["conc_pass"]]
                   .rename(columns={"conc_pass": col}))
            self.CH_TBL = self.CH_TBL.merge(sub, on=key, how="left")
            self.CH_TBL[col] = self.CH_TBL[col].fillna(False).astype(bool)
            self.conc_checks.append((col, f"F.{1 + i}", cc))
        self.dchecks = (["eis_A", "eis_B", "eis_C"] + [c[0] for c in self.conc_checks]
                        + ["fscv_monotonic", "fscv_peak_inwindow"])
        m = self.cfg.min_norm_snr
        self.code = {"eis_A": "E.1", "eis_B": "E.2", "eis_C": "E.3", **{c[0]: c[1] for c in self.conc_checks},
                     "fscv_monotonic": f"F.{n + 1}", "fscv_peak_inwindow": f"F.{n + 2}"}
        self.cname = {"eis_A": "EIS.1 capacitive", "eis_B": "EIS.2 |Z|-monotone", "eis_C": "EIS.3 environment",
                      **{c[0]: f"FSCV reproducibility SNR @ {int(c[2])}nM" for c in self.conc_checks},
                      "fscv_monotonic": "FSCV dose-monotonic", "fscv_peak_inwindow": "FSCV V_peak in-window"}
        self.cexpl = {"eis_A": "sign of Im(Z) at every in-band frequency",
                      "eis_B": "is |Z| non-increasing with f within MONO_TOL",
                      "eis_C": "sign of Z' across the band",
                      **{c[0]: f"repeatability_snr at {int(c[2])}nM vs MIN_NORM_SNR" for c in self.conc_checks},
                      "fscv_monotonic": "Pearson/Spearman r of NormIpeak vs log10(conc), >=3 doses",
                      "fscv_peak_inwindow": "fraction of doses with V_peak inside PEAK_WINDOW"}
        self.cfail = {"eis_A": "Im(Z) >= 0 at any in-band f", "eis_B": "|Z| rises with f beyond MONO_TOL",
                      "eis_C": "Z' < 0 somewhere in band",
                      **{c[0]: f"repeatability_snr < {m:g} at {int(c[2])}nM" for c in self.conc_checks},
                      "fscv_monotonic": f"r < {self.cfg.monotonic_r_min:g} (or < 3 concentrations)",
                      "fscv_peak_inwindow": f"V_peak within {PEAK_EDGE_TOL}V of a PEAK_WINDOW bound for >= half the doses"}
        self.gating_cols = [k for k in self.dchecks if self._is_gating(k)]
        paired = self.CH_TBL.has_eis & self.CH_TBL.has_fscv
        gate = paired.copy()
        for k in self.gating_cols:
            gate &= self.CH_TBL[k].astype(bool)
        self.CH_TBL["overall_valid"] = gate

        def reasons(r):
            out = [] if (r.has_eis and r.has_fscv) else \
                ["incomplete (missing " + ("EIS" if not r.has_eis else "FSCV") + ")"]
            out += [self.code[k] for k in self.gating_cols if not bool(r[k])]
            return "; ".join(out)
        self.CH_TBL["fail_reasons"] = self.CH_TBL.apply(reasons, axis=1)

    def _is_gating(self, k) -> bool:
        g = self.cfg.gate_on
        if k in ("eis_A", "eis_B", "eis_C"):
            return g["eis"]
        if k == "fscv_monotonic":
            return g["monotonic"]
        if k == "fscv_peak_inwindow":
            return g["peak_in_window"]
        return g["snr_all"]

    # ---- identity / tables ---------------------------------------------------------------------
    def title(self, dev) -> str:
        return f"{self.dtype_of.get(dev, '?')} {dev}"

    def label_devices(self, df: pd.DataFrame) -> pd.DataFrame:
        if "device" not in df.columns:
            return df
        out = df.copy(); out.insert(0, "devicetype", out["device"].map(self.dtype_of))
        return out.rename(columns={"device": "deviceid"})

    def available_table(self) -> pd.DataFrame:
        return self._avail

    def describe(self) -> str:
        nvalid, ntot = int(self.CH_TBL.overall_valid.sum()), len(self.CH_TBL)
        return "\n".join([
            "QCDashboard settings (shared AnalysisConfig + local selection):",
            f"  ROOT          = {self.root}",
            f"  band          = {self.band} Hz   peak_method = {self.cfg.peak_method!r}",
            f"  mono_tol(EIS.2) = {self.cfg.mono_tol}   min_norm_snr = {self.cfg.min_norm_snr}   "
            f"monotonic_r_min = {self.cfg.monotonic_r_min} ({self.cfg.mono_method})",
            f"  gate_on       = {self.cfg.gate_on}   ->  gating checks: {[self.code[k] for k in self.gating_cols]}",
            f"  devices       = {self.devices}",
            f"  valid sensors = {nvalid} / {ntot} channel-timepoints ({100 * nvalid / max(ntot, 1):.1f}%)",
            "customize by rebuilding, e.g.:",
            "  QC = QCDashboard(ROOT, device_types=['neurostring'], timepoints=[0,4,6], n_jobs=4)",
        ])

    def print_settings(self) -> None:
        print(self.describe(), flush=True)

    def qc_definitions(self) -> pd.DataFrame:
        return pd.DataFrame([
            {"code": self.code[k], "check": self.cname[k], "gating": self._is_gating(k),
             "explanation": self.cexpl[k], "fail_condition": self.cfail[k]} for k in self.dchecks])

    def quality_table(self) -> pd.DataFrame:
        qtbl = self.CH_TBL[["device", "date", "timepoint", "channel", "has_eis", "has_fscv",
                            *self.dchecks, "overall_valid", "fail_reasons"]].copy()
        qtbl["timepoint"] = qtbl["timepoint"].map(timepoint_label)
        return qtbl.rename(columns={c: self.code[c] for c in self.dchecks})

    def valid_sensor_summary(self) -> pd.DataFrame:
        rows = []
        for (dev, date, tp), g in self.CH_TBL.groupby(["device", "date", "timepoint"]):
            valid = sorted(int(c) for c in g[g.overall_valid].channel)
            invalid = sorted(int(c) for c in g[~g.overall_valid].channel)
            nch = int(g.channel.nunique())
            rows.append({"device": dev, "date": date, "timepoint": timepoint_label(tp), "n_channels": nch,
                         "n_valid_channels": len(valid), "valid_channels": valid,
                         "n_invalid_channels": len(invalid), "invalid_channels": invalid,
                         "valid_%": round(100 * len(valid) / max(nch, 1), 1)})
        self._summary = pd.DataFrame(rows).sort_values(["device", "date"]).reset_index(drop=True)
        return self._summary

    # ---- plots ---------------------------------------------------------------------------------
    def _chan_id_axis(self, ax):
        ax.set_ylim(0.5, self.total_channels + 0.5); ax.set_yticks(range(0, self.total_channels + 1))

    def _chan_count_axis(self, ax):
        ax.set_ylim(0, self.total_channels); ax.set_yticks(range(0, self.total_channels + 1))

    def plot_schedule(self, show: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        from matplotlib.patches import Rectangle
        for dev in self.devices:
            d = self.CH_TBL[self.CH_TBL.device == dev]
            if not len(d):
                continue
            tps = sorted(d.timepoint.unique()); day_min, day_max = min(tps), max(tps); ndays = day_max - day_min + 1
            fig, ax = plt.subplots(figsize=(min(2.6 + 0.34 * ndays, 14), 5), dpi=140); ax.set_axisbelow(True)
            for x in np.arange(day_min - 0.5, day_max + 1.0):
                ax.axvline(x, color="0.9", lw=0.5, zorder=0)
            for y in np.arange(0.5, self.total_channels + 1.0):
                ax.axhline(y, color="0.9", lw=0.5, zorder=0)
            for r in d.itertuples():
                ax.add_patch(Rectangle((r.timepoint - 0.5, r.channel - 0.5), 1, 1,
                                       facecolor=channel_color(r.channel, self.total_channels),
                                       edgecolor="white", lw=0.5, zorder=2))
                if not r.overall_valid:
                    ax.plot(r.timepoint, r.channel, "x", color="crimson", ms=8, mew=1.8, zorder=3)
            ax.set_xticks(tps); ax.set_xticklabels([timepoint_label(t) for t in tps])
            ax.set_xlim(day_min - 0.5, day_max + 0.5); self._chan_id_axis(ax)
            ax.set_xlabel("timepoint (days)"); ax.set_ylabel("channel"); ax.set_title(f"{self.title(dev)} - measurement schedule")
            ax.legend(handles=[Line2D([0], [0], marker="s", color="0.6", ls="", label="measured"),
                               Line2D([0], [0], marker="x", color="crimson", ls="", mew=2, label="failed quality")],
                      fontsize=7, loc="center left", bbox_to_anchor=(1.01, 0.5))
            fig.tight_layout()
            if show:
                plt.show()

    def plot_dropout_by_check(self, show: bool = True):
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        eisk = ["eis_A", "eis_B", "eis_C"]; fscvk = [c for c in self.dchecks if c not in eisk]
        blues, oranges = plt.get_cmap("Blues"), plt.get_cmap("Oranges")
        qccol = {k: blues(0.45 + 0.5 * (i / max(len(eisk) - 1, 1))) for i, k in enumerate(eisk)}
        qccol.update({k: oranges(0.40 + 0.5 * (i / max(len(fscvk) - 1, 1))) for i, k in enumerate(fscvk)})
        grey = "0.82"
        for dev in self.devices:
            d = self.CH_TBL[self.CH_TBL.device == dev]
            if not len(d):
                continue
            tps = sorted(d.timepoint.unique()); t = np.array(tps, float)
            n_meas = [int((d.timepoint == tp).sum()) for tp in tps]
            n_pass = [int(d[d.timepoint == tp].overall_valid.sum()) for tp in tps]
            fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), dpi=140)
            ax[0].plot(t, n_meas, "o-", color="0.4", lw=1.4, label="measured")
            ax[0].plot(t, n_pass, "o-", color="#2CA02C", lw=1.4, label="passed (gating)")
            self._chan_count_axis(ax[0]); ax[0].set_xticks(tps); ax[0].set_xticklabels([timepoint_label(x) for x in tps])
            ax[0].set_xlabel("timepoint (days)"); ax[0].set_ylabel("channels")
            ax[0].set_title("channels measured vs passed"); ax[0].legend(fontsize=7)
            min_gap = float(np.min(np.diff(t))) if len(t) > 1 else 1.0
            group_w = 0.85 * min_gap; slot = group_w / len(self.dchecks); bar_w = slot * 0.7
            for j, chk in enumerate(self.dchecks):
                npass = np.array([int(d[(d.timepoint == tp)][chk].astype(bool).sum()) for tp in tps])
                nfail = np.array([int((~d[(d.timepoint == tp)][chk].astype(bool)).sum()) for tp in tps])
                xj = t + (j - (len(self.dchecks) - 1) / 2) * slot
                ax[1].bar(xj, npass, bar_w, color=qccol[chk]); ax[1].bar(xj, nfail, bar_w, bottom=npass, color=grey)
            self._chan_count_axis(ax[1]); ax[1].set_xlim(t.min() - group_w, t.max() + group_w)
            ax[1].set_xticks(tps); ax[1].set_xticklabels([timepoint_label(x) for x in tps])
            ax[1].set_xlabel("timepoint (days)"); ax[1].set_ylabel("channels"); ax[1].set_title("per-check pass / fail")
            handles = [Line2D([0], [0], marker="s", color=qccol[c], ls="", label=f"{self.code[c]}  {self.cname[c]}")
                       for c in self.dchecks] + [Line2D([0], [0], marker="s", color=grey, ls="", label="failed (grey)")]
            ax[1].legend(handles=handles, fontsize=5.5, loc="center left", bbox_to_anchor=(1.01, 0.5))
            fig.suptitle(f"{self.title(dev)} - quality dropout by check", y=1.03); fig.tight_layout()
            if show:
                plt.show()

    def gate_impact(self, show: bool = True) -> pd.DataFrame:
        import matplotlib.pyplot as plt
        paired = self._paired
        total = len(paired); retained = int(paired.overall_valid.sum()); failed = total - retained
        print(f"total EIS-FSCV paired raw data: {total} channel-timepoints")
        print(f"total failed QC:   {failed} ({100 * failed / max(total, 1):.1f}%)")
        print(f"total retained:    {retained} ({100 * retained / max(total, 1):.1f}%)")
        gates = [("EIS.1 capacitive", "eis_A"), ("EIS.2 |Z|-monotone", "eis_B"),
                 ("EIS.3 environment", "eis_C"), ("FSCV.1 dose-monotonic", "fscv_monotonic")]
        imp = pd.DataFrame([{"gate": g, "n_channeltimepoints_failed": int((~paired[c]).sum()),
                             "failed_%": round(100 * (~paired[c]).mean(), 1)} for g, c in gates])
        fig, ax = plt.subplots(figsize=(7, 2.8), dpi=140)
        ax.barh(imp.gate, imp.n_channeltimepoints_failed, color="#E45756"); ax.invert_yaxis()
        ax.set_xlabel("channel-timepoints failed"); ax.set_title("QC-gate impact (paired; per-gate failures overlap)")
        plt.tight_layout()
        if show:
            plt.show()
        self._imp = imp
        return imp

    def plot_dose_monotonicity_sweep(self, rstar_n=None, show: bool = True):
        import matplotlib.pyplot as plt
        from scipy.stats import t as tdist
        from scipy.special import beta as betafn
        from .features.fscv import dose_response_corr
        paired = self._paired

        def r_star(nn, alpha):
            if nn < 3:
                return float("nan")
            tc = float(tdist.ppf(1 - alpha, nn - 2)); return tc / np.sqrt((nn - 2) + tc ** 2)
        percurve = paired["n_conc"].dropna(); percurve = percurve[percurve >= 3]
        nmed = int(rstar_n) if rstar_n else (int(percurve.mode().iloc[0]) if len(percurve) else 3)
        rs = r_star(nmed, 0.05)
        r = paired["dose_response_r"].to_numpy(float); r = r[np.isfinite(r)]
        rmin = self.cfg.monotonic_r_min
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.6), dpi=140)
        grid = np.linspace(0, 0.95, 20)
        ax[0].plot(grid, [int((r >= t).sum()) for t in grid], "-o", ms=3, color="#4C78A8")
        ax[0].axvline(rmin, color="crimson", ls="--", lw=1.3, label=f"current r_min = {rmin:g}")
        if np.isfinite(rs):
            ax[0].axvline(rs, color="0.5", ls="--", lw=1.3, label=f"r* (n={nmed}, p<0.05) = {rs:.2f}")
        ax[0].set_xlabel("MONOTONIC_R_MIN"); ax[0].set_ylabel("channel-timepoints kept")
        ax[0].set_title("dose-monotonicity stringency sweep"); ax[0].legend(fontsize=7)
        ax[1].hist(r, bins=25, range=(-1, 1), density=True, color="#4C78A8", alpha=0.7, edgecolor="white",
                   label=f"observed r (n={len(r)})")
        if nmed > 2:
            rr = np.linspace(-0.999, 0.999, 400); pdf = (1 - rr ** 2) ** ((nmed - 4) / 2) / betafn(0.5, (nmed - 2) / 2)
            ax[1].plot(rr, pdf, "k-", lw=1.4, label=f"null (n={nmed}, rho=0)")
        if np.isfinite(rs):
            ax[1].axvline(rs, color="0.5", ls="--", lw=1.3, label=f"r*={rs:.2f}")
        ax[1].set_xlim(-1, 1); ax[1].set_xlabel("dose-response r"); ax[1].set_ylabel("density")
        ax[1].set_title("observed r vs null"); ax[1].legend(fontsize=6.5)
        plt.tight_layout()
        if show:
            plt.show()
        cmp = pd.DataFrame([
            {"pearson": dose_response_corr(g["concentration"].to_numpy(float), g["NormIpeak_avg"].to_numpy(float), "pearson"),
             "spearman": dose_response_corr(g["concentration"].to_numpy(float), g["NormIpeak_avg"].to_numpy(float), "spearman")}
            for _, g in self.DOSE_TBL.groupby(["device", "channel", "timepoint"])]).dropna()
        self._r, self._rs, self._nmed, self._cmp = r, rs, nmed, cmp
        return {"r_star": rs, "n_doses": nmed, "kept_at_r_min": int((r >= rmin).sum()) if r.size else 0}

    def plot_z_monotonicity_sweep(self, show: bool = True):
        import matplotlib.pyplot as plt
        zr = self._paired["eis_z_rise"].to_numpy(float); zr = zr[np.isfinite(zr)]
        tol = self.cfg.mono_tol
        if not zr.size:
            print("no in-band EIS to sweep MONO_TOL."); self._zr = zr; return
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4), dpi=140)
        hi = max(tol * 1.5, float(np.nanpercentile(zr, 95)))
        ax[0].hist(np.clip(zr, None, hi), bins=30, color="#54A24B", alpha=0.8, edgecolor="white")
        ax[0].axvline(tol, color="crimson", ls="--", lw=1.3, label=f"MONO_TOL = {tol:g}")
        ax[0].set_xlabel("max fractional |Z| rise (EIS.2)"); ax[0].set_ylabel("# channel-timepoints")
        ax[0].set_title(f"|Z|-rise (n={zr.size})"); ax[0].legend(fontsize=7)
        tg = np.linspace(0, max(0.2, float(np.nanpercentile(zr, 95))), 25)
        ax[1].plot(tg, [int((zr <= t).sum()) for t in tg], "-o", ms=3, color="#4C78A8")
        ax[1].axvline(tol, color="crimson", ls="--", lw=1.3, label=f"current = {tol:g}")
        ax[1].set_xlabel("MONO_TOL"); ax[1].set_ylabel("channel-timepoints kept (EIS.2)")
        ax[1].set_title("|Z|-monotonicity stringency sweep"); ax[1].legend(fontsize=7)
        plt.tight_layout()
        if show:
            plt.show()
        self._zr = zr

    def plot_snr_sweep(self, show: bool = True):
        import matplotlib.pyplot as plt
        snr = self.DOSE_TBL["repeatability_snr"].to_numpy(float); snr = snr[np.isfinite(snr)]
        m = self.cfg.min_norm_snr
        if not snr.size:
            print("no finite reproducibility SNR to sweep."); self._snr = snr; return
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.4), dpi=140)
        shi = max(m * 3, float(np.nanpercentile(snr, 97.5))); nover = int((snr > shi).sum())
        ax[0].hist(np.clip(snr, None, shi), bins=30, color="#F58518", alpha=0.8, edgecolor="white"); ax[0].set_xlim(0, shi)
        ax[0].axvline(m, color="crimson", ls="--", lw=1.3, label=f"MIN_NORM_SNR = {m:g}")
        ax[0].set_xlabel(f"per-dose reproducibility SNR (clipped at {shi:.0f}; {nover} above)"); ax[0].set_ylabel("# doses")
        ax[0].set_title("SNR distribution"); ax[0].legend(fontsize=7)
        sg = np.linspace(0, shi, 25)
        ax[1].plot(sg, [int((snr >= t).sum()) for t in sg], "-o", ms=3, color="#4C78A8")
        ax[1].axvline(m, color="crimson", ls="--", lw=1.3, label=f"current = {m:g}")
        ax[1].set_xlabel("min_norm_snr"); ax[1].set_ylabel("doses kept"); ax[1].set_title("SNR stringency sweep (per dose)")
        ax[1].legend(fontsize=7); plt.tight_layout()
        if show:
            plt.show()
        self._snr = snr

    # ---- finalized-dataset overview (reuses electropycal.overview) --------------------------------
    def finalized_featureset(self) -> pd.DataFrame:
        """Valid dose rows for the overview: DOSE_TBL restricted to overall-valid channel-timepoints,
        negatives dropped, ``NormIpeak_avg`` exposed as ``NormIpeak`` (so electropycal.overview plots apply)."""
        vk = self.CH_TBL.loc[self.CH_TBL.overall_valid, ["device", "channel", "timepoint"]]
        ov = self.DOSE_TBL.merge(vk, on=["device", "channel", "timepoint"], how="inner").copy()
        if "negative" in ov.columns:
            ov = ov[~ov["negative"].astype(bool)]
        if "NormIpeak_avg" in ov.columns and "NormIpeak" not in ov.columns:
            ov = ov.rename(columns={"NormIpeak_avg": "NormIpeak"})
        return ov

    def plot_finalized_overview(self, show: bool = True):
        from .overview import plot_qc_grid, plot_normipeak_per_sensor, plot_normipeak_per_device
        ov = self.finalized_featureset()
        print(f"finalized dataset (post-QC): {ov.groupby(['device','channel','timepoint','concentration']).ngroups} "
              f"dose rows | {int(self.CH_TBL.overall_valid.sum())} valid channel-timepoints")
        plot_qc_grid(ov, show=show)
        plot_normipeak_per_sensor(ov, show=show)
        plot_normipeak_per_device(ov, show=show)

    # ---- QC-stats artifact (notebook cell 33) --------------------------------------------------
    def save_stats(self) -> Path:
        """Assemble the QC-stats companion from this dashboard's own tables (no globals) and save it
        next to the shared config. Sweeps not yet run are recorded as null."""
        def num(x):
            try:
                return round(float(x), 4)
            except Exception:
                return None
        paired, r, cmp = self._paired, self._r, self._cmp
        rmin, m, tol = self.cfg.monotonic_r_min, self.cfg.min_norm_snr, self.cfg.mono_tol
        stats = {
            "generated_utc": _dt.datetime.now(_dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "root": self.root, "config": asdict(self.cfg),
            "band_hz": [float(self.band[0]), float(self.band[1])],
            "gating_columns": [self.code[k] for k in self.gating_cols],
            "totals": {"channel_timepoints_paired": int(len(paired)),
                       "overall_valid": int(paired.overall_valid.sum()),
                       "overall_valid_pct": num(100 * paired.overall_valid.mean()) if len(paired) else None},
            "gate_impact": self._imp.to_dict("records") if self._imp is not None else None,
            "dose_monotonicity": ({
                "metric": self.cfg.mono_method, "r_min": float(rmin), "n_doses_median": self._nmed,
                "r_star_p05": num(self._rs), "kept_at_r_min": int((r >= rmin).sum()),
                "below_r_star": int((r < self._rs).sum()) if self._rs is not None and np.isfinite(self._rs) else None,
                "r_quartiles": {k: num(v) for k, v in pd.Series(r).describe()[["min", "25%", "50%", "75%", "max"]].items()}
                if r is not None and len(r) else None} if r is not None else None),
            "mono_method_comparison": ({
                "r_min": float(rmin), "n_compared": int(len(cmp)),
                "pearson_kept": int((cmp["pearson"] >= rmin).sum()),
                "spearman_kept": int((cmp["spearman"] >= rmin).sum()),
                "spearman_rescues_pearson_drops": int(((cmp["spearman"] >= rmin) & (cmp["pearson"] < rmin)).sum()),
                "pearson_keeps_spearman_drops": int(((cmp["pearson"] >= rmin) & (cmp["spearman"] < rmin)).sum()),
            } if cmp is not None and len(cmp) else None),
            "snr": ({"min_norm_snr": float(m), "n_doses": int(self._snr.size),
                     "median": num(np.median(self._snr)), "below_cutoff": int((self._snr < m).sum())}
                    if self._snr is not None and self._snr.size else None),
            "eis2_z_rise": ({"mono_tol": float(tol), "n": int(self._zr.size),
                             "median": num(np.median(self._zr)), "exceed_tol": int((self._zr > tol).sum())}
                            if self._zr is not None and self._zr.size else None),
            # `date` is carried through: `timepoint` here is timepoint_label(tp), a device-local
            # D-label, and two devices' D35 are different calendar dates. Without the date the
            # records cannot be joined back to a session. valid_sensor_summary() has it already.
            "per_timepoint_valid": (self._summary[["device", "date", "timepoint", "n_channels",
                                                   "n_valid_channels", "valid_%"]]
                                    .to_dict("records") if self._summary is not None else None),
        }
        p = save_qc_stats(self.root, stats)
        t = stats["totals"]
        print(f"saved QC stats -> {p}")
        print(f"  overall valid: {t['overall_valid']} / {t['channel_timepoints_paired']} "
              f"({t['overall_valid_pct']}%)  |  band {stats['band_hz']} Hz  |  mono={self.cfg.mono_method} r_min={rmin}")
        return p
