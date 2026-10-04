"""Diagnostics review: variance / reliability / drift-alignment analysis as a library API.

Folds ``diagnostics_review`` (the notebook that asks *is recalibration well-posed here?*) into one
:class:`DiagnosticsReview` object. Construction takes the **raw** (non-D0) featureset and the shared
config, and computes the feature-type taxonomy + per-feature variance decomposition once; each ``plot_*``
method draws one section's figure(s) and returns/stashes the tables later sections (and ``verdict``)
depend on. This keeps the notebook to thin calls and makes every diagnostic reproducible from a script.

Sections: **1** variance hierarchy (distributions, drift-variance breakdown, decomposition, 3D cloud);
**2** reliability (response + feature measurement/drift vs replicate-noise floor); **3** drift alignment
(does any feature's drift track the response?) + the frequency-bandwidth tradeoff + absolute-vs-drift
sanity checks; **4** the well-posedness verdict; **5** the D0-normalization effect + its invariance;
**6** response/deformation modes.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

from .diagnostics.variance import drift_alignment, drift_reliability, variance_hierarchy
from .data.schema import RESERVED_COLUMNS
from . import viz

_FRE = re.compile(r"^(.+)_f(\d+)$")
_FREQDEP_TYPES = ["R_s", "R_p", "C_s", "C_p", "ideality_C", "tau", "local_n"]
_FEATURE_UNITS = {"R_s": "Ω", "R_p": "Ω", "C_s": "F", "C_p": "F", "ideality_C": "", "tau": "s",
                  "local_n": "", "R_s_integral": "Ω·dec", "R_p_integral": "Ω·dec", "C_s_integral": "F·dec",
                  "C_p_integral": "F·dec", "f_ideality_crossover": "Hz", "tau_ratio": "", "min_neg_phase": "°",
                  "inductive_onset_hz": "Hz", "mean_Vpeak": "V", "mean_Ibg": "A", "bg_charge": "A·V",
                  "bg_cap": "A", "bg_switch": "A", "NormIpeak": ""}
RESP = "NormIpeak"
RESP_COLOR = "#D62728"
COMP = [("between_sensor", "between-sensor", "#8C8C8C"), ("within_time", "within-time", "#4C78A8"),
        ("within_dose", "within-dose", "#7B3FA0")]


def feat_type(col: str) -> str:
    m = _FRE.match(col); return m.group(1) if m else col


def feat_fidx(col: str):
    m = _FRE.match(col); return int(m.group(2)) if m else None


class DiagnosticsReview:
    """Variance/reliability/alignment diagnostics over a raw featureset (see module docstring)."""

    def __init__(self, feat: pd.DataFrame, *, cfg, root=None, band=None, n_jobs: int = 4):
        self.FEAT = feat.reset_index(drop=True)
        self.cfg = cfg
        self.root = None if root is None else str(root)
        self.band = band
        self.n_jobs = n_jobs
        if "sensor" not in self.FEAT.columns:
            self.FEAT["sensor"] = self.FEAT.device.astype(str) + ":" + self.FEAT.channel.astype(str)
        self.STATE = [c for c in self.FEAT.columns if c not in set(RESERVED_COLUMNS)
                      and c not in ("sensor", "time_index", "devicetype")]
        # taxonomy
        self.freqdep_cols = {t: sorted([c for c in self.STATE if feat_type(c) == t], key=feat_fidx)
                             for t in _FREQDEP_TYPES}
        self.freqdep_cols = {t: v for t, v in self.freqdep_cols.items() if v}
        self.freqindep_types = [c for c in self.STATE if feat_fidx(c) is None]
        self.predictor_types = list(self.freqdep_cols) + self.freqindep_types
        import matplotlib.pyplot as plt
        # turbo is a rainbow and not colour-blind safe. These are feature TYPES, an
        # unordered set, so a categorical palette is the honest encoding; the ramp is
        # kept only because more types exist than Okabe-Ito has colours.
        cmap = plt.get_cmap(viz.SEQUENTIAL)
        self.type_color = {t: cmap(0.05 + 0.90 * i / max(len(self.predictor_types) - 1, 1))
                           for i, t in enumerate(self.predictor_types)}
        self.freqs = self._inband_freqs()
        # per-feature variance decomposition (once)
        self.VAR = pd.DataFrame({c: self._vh(self.FEAT, c) for c in self.STATE}).T[
            ["between_sensor", "within_time", "within_dose"]]
        # sample-unit dedup for distributions (EIS = per ch-timepoint; FSCV/response = per sample)
        gct = self.FEAT.groupby(["device", "channel", "timepoint"])
        self.ct_level = {c for c in self.STATE if int(gct[c].nunique(dropna=False).max()) <= 1}
        self.FEAT_CT = self.FEAT.drop_duplicates(["device", "channel", "timepoint"]).reset_index(drop=True)
        # stashes populated by the plot methods (verdict + later sections read them)
        self.R = self.sigma2_meas = self.n_rep = None
        self.REL = self.ALIGN = self.DRIFT = self._valid = self._maxa = None

    # ---- helpers -------------------------------------------------------------------------------
    def feat_unit(self, col: str) -> str:
        t = feat_type(col)
        if t in _FEATURE_UNITS:
            return _FEATURE_UNITS[t]
        if t.startswith("ideality_C_band") or t.startswith("n_band"):
            return ""
        return _FEATURE_UNITS.get(col, "")

    def _vh(self, df, col) -> dict:
        h = variance_hierarchy(df[col].to_numpy(float), df.sensor.to_numpy(),
                               df.timepoint.to_numpy(), df.concentration.to_numpy())
        return {"between_sensor": h["between_channel"], "within_time": h["within_temporal"],
                "within_dose": h["within_concentration"]}

    def _col_samples(self, col):
        if col in self.ct_level:
            return self.FEAT_CT[col].to_numpy(float), "n_chtimepoints"
        return self.FEAT[col].to_numpy(float), "n_samples"

    def _annot_hist(self, ax, x, color, bins=40, rng=None, nlabel="n"):
        from matplotlib.lines import Line2D
        from matplotlib.patches import Patch
        x = np.asarray(x, float); x = x[np.isfinite(x)]
        if x.size == 0:
            return [Line2D([], [], ls="none", label=f"{nlabel} = 0")]
        mu, med, sd = float(x.mean()), float(np.median(x)), float(x.std())
        ax.hist(x, bins=bins, range=rng, color=color, alpha=0.8, edgecolor="white", linewidth=0.3)
        ax.axvspan(mu - sd, mu + sd, color="0.5", alpha=0.15, zorder=0)
        ax.axvline(mu, color="tab:blue", ls="--", lw=1.4); ax.axvline(med, color="tab:red", ls="--", lw=1.4)
        return [Line2D([], [], ls="none", label=f"{nlabel} = {x.size}"),
                Line2D([], [], color="tab:blue", ls="--", lw=1.4, label=f"mean = {mu:.3g}"),
                Line2D([], [], color="tab:red", ls="--", lw=1.4, label=f"median = {med:.3g}"),
                Patch(facecolor="0.5", alpha=0.35, label=f"s.d. = {sd:.3g}")]

    def _type_legend_handles(self, types=None):
        from matplotlib.lines import Line2D
        types = list(self.predictor_types) if types is None else types
        return [Line2D([], [], marker="o", ls="none", color=self.type_color[t], ms=5, label=t) for t in types]

    def _panels(self, n, ncols=4, size=(3.0, 2.4)):
        import math
        import matplotlib.pyplot as plt
        nr = math.ceil(n / ncols) if n else 1
        fig, ax = plt.subplots(nr, ncols, figsize=(size[0] * ncols, size[1] * nr), dpi=140, squeeze=False)
        fl = ax.ravel()
        for a in fl[n:]:
            a.axis("off")
        return fig, fl

    def _device_baselines(self) -> dict:
        from .data.pstrace import parse_folder, parse_filename
        from collections import defaultdict
        dates = defaultdict(set)
        for folder in sorted(p for p in Path(self.root).iterdir() if p.is_dir()):
            fm = parse_folder(folder.name)
            if not fm or fm.get("testtype") != "signal":
                continue
            for csv in folder.glob("*.csv"):
                nm = parse_filename(csv.name)
                if nm:
                    dates[nm["deviceid"]].add(fm["date"])
        return {dev: min(ds) for dev, ds in dates.items()}

    def _inband_freqs(self):
        if self.root is None or self.band is None:
            return None
        from .data.pstrace import parse_folder, parse_filename, read_pstrace
        for folder in sorted(p for p in Path(self.root).iterdir() if p.is_dir()):
            fm = parse_folder(folder.name)
            if not fm or fm.get("testtype") != "signal":
                continue
            for csv in sorted(folder.glob("*.csv")):
                nm = parse_filename(csv.name)
                if nm and nm["signaltype"] == "eis":
                    try:
                        exp = read_pstrace(csv)
                        for (_c, _), s in sorted(exp.eis.items()):
                            f = np.asarray(s["freq"], float); m = (f >= self.band[0]) & (f <= self.band[1])
                            return np.sort(f[m])
                    except Exception:
                        return None
        return None

    def _ensure_reliability(self):
        if self.REL is not None:
            return self.REL
        if self.root is None:
            print("replicate reliability needs the raw ROOT (not available); skipping reliability sections.")
            return None
        from .features.extract import replicate_feature_reliability
        keep = {(r.device, int(r.channel), float(r.timepoint)) for r in self.FEAT.itertuples()}
        self.REL = replicate_feature_reliability(
            self.root, band=self.band, keep=keep, peak_method=self.cfg.peak_method,
            mono_tol=self.cfg.mono_tol, max_reps=self.cfg.max_reps, n_jobs=self.n_jobs, progress=True)
        return self.REL

    def print_settings(self) -> None:
        b = f"({self.band[0]:.0f}, {self.band[1]:.0f}) Hz" if self.band is not None else "n/a"
        print("\n".join([
            "DiagnosticsReview settings:",
            f"  ROOT          = {self.root}",
            f"  band          = {b}   peak_method = {self.cfg.peak_method!r}   mono_method = {self.cfg.mono_method!r}",
            f"  featureset    = {len(self.FEAT)} rows | {self.FEAT.sensor.nunique()} sensors | "
            f"{self.FEAT.timepoint.nunique()} timepoints | {len(self.STATE)} predictor features",
            f"  featuretypes  = {len(self.freqdep_cols)} freq-dependent + {len(self.freqindep_types)} "
            f"freq-independent | in-band frequencies: {len(self.freqs) if self.freqs is not None else '?'}",
            "customize by rebuilding: DiagnosticsReview(FEAT, cfg=CFG, root=ROOT, band=BAND, n_jobs=4)",
        ]), flush=True)

    # ---- Section 0: feature dictionary ---------------------------------------------------------
    def feature_dictionary(self) -> pd.DataFrame:
        from . import catalog_frame
        fd = catalog_frame()
        present = {feat_type(col) for col in self.FEAT.columns} | {
            ("ideality_C_band_LF/MF/HF" if t.startswith("ideality_C_band") else
             "n_band_LF/MF/HF" if t.startswith("n_band") else t)
            for t in (feat_type(c) for c in self.FEAT.columns)}
        return fd.assign(in_this_featureset=[t in present for t in fd.index])

    # ---- Section 1: variance hierarchy ---------------------------------------------------------
    def plot_featuretype_distributions(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(5.4, 3.4), dpi=140)
        h = self._annot_hist(ax, self.FEAT[RESP].to_numpy(float), RESP_COLOR, bins=40, nlabel="n_samples")
        ru = self.feat_unit(RESP); ax.set_title(f"response: {RESP}")
        ax.set_xlabel(f"{RESP} ({ru})" if ru else RESP); ax.set_ylabel("count")
        ax.legend(handles=h, fontsize=8, handlelength=1.4, labelspacing=0.25); plt.tight_layout()
        if show:
            viz.emit("diag_featuretype_distributions_1")
        fig, axes = self._panels(len(self.freqindep_types))
        for a, t in zip(axes, self.freqindep_types):
            vals, nlab = self._col_samples(t)
            h = self._annot_hist(a, vals, self.type_color[t], bins=30, nlabel=nlab)
            a.set_title(t, fontsize=8); a.set_xlabel(self.feat_unit(t), fontsize=6)
            a.legend(handles=h, fontsize=4.6, loc="upper right", handlelength=0.9, labelspacing=0.15, borderpad=0.2)
        fig.suptitle("Frequency-independent featuretype distributions", y=1.01); plt.tight_layout()
        if show:
            viz.emit("diag_featuretype_distributions_2")
        nf = 6
        fig, axes = self._panels(len(self.freqdep_cols))
        for a, (t, cols) in zip(axes, self.freqdep_cols.items()):
            idxs = sorted(set(np.linspace(0, len(cols) - 1, min(nf, len(cols))).round().astype(int)))
            valid = []
            for k in idxs:
                d = self.FEAT_CT[cols[k]].to_numpy(float); d = d[np.isfinite(d)]
                if d.size >= 2 and np.unique(d).size > 1:
                    lab = f"{self.freqs[k]:.0f}" if (self.freqs is not None and len(self.freqs) == len(cols)) else f"f{k:02d}"
                    valid.append((lab, d))
            if not valid:
                a.axis("off"); a.set_title(t, fontsize=8); continue
            parts = a.violinplot([v[1] for v in valid], positions=range(len(valid)), widths=0.85,
                                 showmedians=True, showextrema=False)
            for b in parts["bodies"]:
                b.set_facecolor(self.type_color[t]); b.set_alpha(0.6); b.set_edgecolor("0.3")
            parts["cmedians"].set_color("tab:red")
            a.set_xticks(range(len(valid))); a.set_xticklabels([v[0] for v in valid], rotation=45)
            a.set_title(t, fontsize=8); a.set_xlabel("frequency (Hz)", fontsize=6)
            a.set_ylabel(self.feat_unit(t), fontsize=6); a.tick_params(labelsize=6)
        fig.suptitle("Frequency-dependent featuretype distributions: density per in-band frequency "
                     "(median = red)", y=1.01); plt.tight_layout()
        if show:
            viz.emit("diag_featuretype_distributions_3")

    def plot_variance_breakdown(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt

        def typevar(t):
            cols = self.freqdep_cols[t] if t in self.freqdep_cols else [t]
            return self.VAR.loc[[c for c in cols if c in self.VAR.index]].mean()
        typevar_df = pd.DataFrame({t: typevar(t) for t in self.predictor_types}).T[
            ["between_sensor", "within_time", "within_dose"]]
        rh = variance_hierarchy(self.FEAT[RESP].to_numpy(float), self.FEAT.sensor.to_numpy(),
                                self.FEAT.timepoint.to_numpy(), self.FEAT.concentration.to_numpy())
        resp = {"between_sensor": rh["between_channel"], "within_time": rh["within_temporal"],
                "within_dose": rh["within_concentration"]}
        med_wt = float(typevar_df["within_time"].median())
        top = typevar_df[typevar_df["within_time"] > med_wt].sort_values("within_time", ascending=False)
        bars = pd.concat([pd.DataFrame([resp], index=[RESP]), top])
        xpos = np.arange(len(bars))
        stack = [COMP[1], COMP[0], COMP[2]]
        fig, ax = plt.subplots(figsize=(max(7, 0.55 * len(bars) + 2), 4.4), dpi=140)
        lefts = np.zeros(len(bars))
        for key, lab, col in stack:
            ax.bar(xpos, bars[key], bottom=lefts, width=0.82, color=col, label=lab); lefts += bars[key].to_numpy()
        ax.bar([0], [1.0], width=0.82, color="none", edgecolor=RESP_COLOR, lw=2.4, zorder=6)
        ax.set_xticks(xpos); ax.set_xticklabels(list(bars.index), rotation=45, ha="right", fontsize=7)
        ax.get_xticklabels()[0].set_color(RESP_COLOR); ax.get_xticklabels()[0].set_fontweight("bold")
        ax.set_ylim(0, 1); ax.set_ylabel("fraction of variance")
        ax.set_xlabel("featuretype (ordered by within-time variance); within-time at base")
        ax.set_title("Variance breakdown: response + above-median within-time (drift) featuretypes")
        ax.legend(fontsize=7, ncol=3, loc="upper right"); plt.tight_layout()
        if show:
            viz.emit("diag_variance_breakdown")
        return (typevar_df.sort_values("within_time", ascending=False).head(10)
                .rename(columns={"between_sensor": "between_channel_var", "within_time": "within_time_var",
                                 "within_dose": "within_dose_var"}))

    def plot_variance_decomposition(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt
        fig, axes = plt.subplots(1, 3, figsize=(13, 3.8), dpi=140)
        for ax, (key, lab, col) in zip(axes, COMP):
            h = self._annot_hist(ax, self.VAR[key].to_numpy(float), col, bins=40, rng=(0, 1), nlabel="n_features")
            ax.set_title(f"{lab} variance share", fontsize=9); ax.set_xlabel("fraction of variance")
            ax.set_ylabel("# features"); ax.set_xlim(0, 1)
            ax.legend(handles=h, fontsize=7, loc="upper center", handlelength=1.4, labelspacing=0.25, borderpad=0.3)
        fig.suptitle(f"Feature variance decomposition over {len(self.STATE)} predictor features "
                     "(response excluded)", y=1.03); plt.tight_layout()
        if show:
            viz.emit("diag_variance_decomposition")

        def stats(x):
            x = x[np.isfinite(x)]
            return {"n": int(x.size), "mean": x.mean(), "median": float(np.median(x)), "std": x.std(),
                    "min": x.min(), "25%": np.percentile(x, 25), "75%": np.percentile(x, 75), "max": x.max()}
        var_stats = pd.DataFrame({lab: stats(self.VAR[key].to_numpy(float)) for key, lab, _ in COMP}).T
        perfreq = [c for c in self.STATE if re.search(r"_f\d+$", c)]
        types = sorted({re.sub(r"_f\d+$", "", c) for c in perfreq})
        nfreq = len({int(re.search(r"_f(\d+)$", c).group(1)) for c in perfreq}) if perfreq else 0
        bg = [c for c in self.STATE if c.startswith("bg_")]
        fscv = [c for c in self.STATE if c in ("mean_Vpeak", "mean_Ibg")]
        glob = [c for c in self.STATE if c not in perfreq and c not in bg and c not in fscv]
        print(f"predictor features (STATE) = {len(self.STATE)}:")
        print(f"  per-frequency EIS : {len(perfreq):3d} = {len(types)} types x {nfreq} in-band frequencies")
        print(f"  EIS band/global   : {len(glob):3d}   background(0nM): {len(bg)}   FSCV predictors: {len(fscv)}")
        return var_stats

    def plot_feature_3d(self, feature: str = "NormIpeak", color_by_channel: bool = False, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        import matplotlib.colors as mcolors
        x = self.FEAT["channel"].to_numpy(float); y = self.FEAT["timepoint"].to_numpy(float)
        z = self.FEAT["concentration"].to_numpy(float); c = self.FEAT[feature].to_numpy(float)
        ttype = RESP if feature == RESP else feat_type(feature)
        base = RESP_COLOR if feature == RESP else self.type_color.get(ttype, "#4C78A8")
        hexc = mcolors.to_hex(base)
        r, g, bb, _ = mcolors.to_rgba(base)
        light = mcolors.to_hex((r * 0.18 + 0.82, g * 0.18 + 0.82, bb * 0.18 + 0.82))
        try:
            import plotly.graph_objects as go
            fig = go.Figure(go.Scatter3d(x=x, y=y, z=z, mode="markers",
                marker=dict(size=4, color=c, colorscale=[[0, light], [1, hexc]], opacity=0.85,
                            colorbar=dict(title=feature)),
                text=[f"{s} | {feature}={v:.3g}" for s, v in zip(self.FEAT.sensor, c)],
                hovertemplate="ch %{x}<br>day %{y}<br>%{z:.0f} nM<br>%{text}<extra></extra>"))
            fig.update_layout(height=620, margin=dict(l=0, r=0, t=40, b=0),
                              title=f"{feature} across channel × timepoint × concentration",
                              scene=dict(xaxis_title="channel", yaxis_title="timepoint (days)",
                                         zaxis=dict(title="concentration (nM)", type="log")))
            if show:
                fig.show()
            return fig
        except ImportError:
            from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
            zl = np.log10(np.where(z > 0, z, np.nan))
            fig = plt.figure(figsize=(7.5, 6), dpi=140); ax = fig.add_subplot(111, projection="3d")
            cmap = mcolors.LinearSegmentedColormap.from_list("grad", [light, hexc])
            p = ax.scatter(x, y, zl, c=c, cmap=cmap, s=22, depthshade=False); fig.colorbar(p, label=feature, shrink=0.6)
            ax.set_xlabel("channel"); ax.set_ylabel("timepoint (days)"); ax.set_zlabel("log10 concentration (nM)")
            ax.set_title(f"{feature} (static; install plotly for interactive)"); plt.tight_layout()
            if show:
                viz.emit("diag_feature_3d")

    # ---- Section 2: reliability ----------------------------------------------------------------
    def plot_response_reliability(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
        n_rep = self.cfg.max_reps or 3
        if "rep_std" not in self.FEAT.columns or not self.FEAT["rep_std"].notna().any():
            print("no replicate spread (rep_std) in the featureset; cannot bound reliability")
            self.R, self.sigma2_meas, self.n_rep = float("nan"), float("nan"), n_rep
            return
        rs = self.FEAT["rep_std"].to_numpy(float)
        sigma2_meas = float(np.nanmean(rs[np.isfinite(rs)] ** 2))
        sigma2_obs = float(np.var(self.FEAT["NormIpeak"].to_numpy(float), ddof=1))
        noise = sigma2_meas / n_rep
        sigma2_signal = max(0.0, sigma2_obs - noise)
        R = sigma2_signal / sigma2_obs if sigma2_obs > 0 else float("nan")
        self.R, self.sigma2_meas, self.n_rep = R, sigma2_meas, n_rep
        print(f"NormIpeak MEASUREMENT reliability:  R = {R:.3f}   (σ²_obs dose-dominated)")
        fig, ax = plt.subplots(1, 2, figsize=(11, 3.2), dpi=140)
        ax[0].barh([0], [sigma2_signal], color="#2CA02C", label=f"signal σ²_signal  (R = {R:.2f})")
        ax[0].barh([0], [noise], left=[sigma2_signal], color="0.6", label="replicate noise  σ²_meas/n_rep")
        ax[0].set_yticks([0]); ax[0].set_yticklabels(["σ²_obs"]); ax[0].set_xlabel("variance")
        ax[0].set_title("response measurement: signal vs noise"); ax[0].legend(fontsize=6.5, loc="lower right")
        v = rs[np.isfinite(rs)] ** 2
        h = self._annot_hist(ax[1], v, "#1B9E77", bins=40, nlabel="n_obs")
        ax[1].axvline(sigma2_obs, color="crimson", ls="--", lw=1.4)
        h.append(Line2D([], [], color="crimson", ls="--", label=f"σ²_obs = {sigma2_obs:.2g}"))
        ax[1].set_xlabel("per-observation replicate variance rep_std²"); ax[1].set_ylabel("# observations")
        ax[1].set_title("measurement noise vs total spread"); ax[1].legend(handles=h, fontsize=5.5, labelspacing=0.2)
        plt.tight_layout()
        if show:
            viz.emit("diag_response_reliability")

    def plot_response_drift_reliability(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        if self.sigma2_meas is None:
            self.plot_response_reliability(show=False)
        if not np.isfinite(self.sigma2_meas):
            print("no replicate spread; cannot bound response drift reliability"); return
        rd = drift_reliability(self.FEAT[RESP].to_numpy(float), self.FEAT.sensor.to_numpy(),
                               self.FEAT.timepoint.to_numpy(), self.FEAT.concentration.to_numpy(),
                               self.sigma2_meas, self.n_rep)
        hh = variance_hierarchy(self.FEAT[RESP].to_numpy(float), self.FEAT.sensor.to_numpy(),
                                self.FEAT.timepoint.to_numpy(), self.FEAT.concentration.to_numpy())
        tot = float(np.var(self.FEAT[RESP].to_numpy(float)))
        comp = [("between-sensor", hh["between_channel"] * tot, "#8C8C8C"),
                ("within-time (drift)", hh["within_temporal"] * tot, "#4C78A8"),
                ("within-dose", hh["within_concentration"] * tot, "#7B3FA0")]
        print(f"NormIpeak DRIFT reliability:  R_drift = {rd['R_drift']:.3f}  drift_snr = {rd['drift_snr']:.2f}")
        fig, ax = plt.subplots(figsize=(6.6, 3.2), dpi=140)
        ax.bar([c[0] for c in comp], [c[1] for c in comp], color=[c[2] for c in comp], edgecolor="0.3")
        ax.axhline(rd["noise_floor"], color="crimson", ls="--", lw=1.5, label=f"noise floor = {rd['noise_floor']:.1e}")
        ax.set_ylabel("variance (absolute)"); ax.set_title("response variance decomposition vs noise floor")
        ax.legend(fontsize=7); ax.tick_params(axis="x", labelsize=8); plt.tight_layout()
        if show:
            viz.emit("diag_response_drift_reliability")

    def plot_feature_measurement_reliability(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt
        REL = self._ensure_reliability()
        if REL is None:
            return pd.DataFrame()

        def row(col):
            vals, _ = self._col_samples(col); s2obs = float(np.nanvar(vals, ddof=1))
            nrep = float(REL.loc[col, "n_rep"]); s2meas = float(REL.loc[col, "sigma2_meas"])
            noise = s2meas / nrep if nrep else float("nan")
            Rm = max(0.0, s2obs - noise) / s2obs if s2obs > 0 else float("nan")
            return {"feature": col, "type": feat_type(col), "sigma2_obs": s2obs, "noise_floor": noise, "R_meas": Rm}
        MEAS = pd.DataFrame([row(c) for c in self.STATE if c in REL.index]).set_index("feature")
        mv = MEAS.dropna(subset=["R_meas"])
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), dpi=140)
        h = self._annot_hist(ax[0], mv["R_meas"].to_numpy(float), "#E45756", bins=40, rng=(0, 1), nlabel="n_features")
        ax[0].set_xlabel("R_meas (feature resolved above noise)"); ax[0].set_ylabel("# features")
        ax[0].set_title("per-feature measurement reliability"); ax[0].legend(handles=h, fontsize=6.5)
        ob = mv["sigma2_obs"].to_numpy(float); nf = mv["noise_floor"].to_numpy(float)
        m = (ob > 0) & (nf > 0) & np.isfinite(nf)
        ax[1].scatter(nf[m], ob[m], s=14, alpha=0.7,
                      c=[self.type_color.get(t, "#888888") for t in mv["type"].to_numpy()[m]])
        lim = [min(nf[m].min(), ob[m].min()), max(nf[m].max(), ob[m].max())] if m.any() else [1e-6, 1]
        ax[1].plot(lim, lim, "k--", lw=1, label="obs = noise"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("noise floor σ²_meas/n_rep"); ax[1].set_ylabel("observed variance σ²_obs")
        ax[1].set_title("feature signal vs measurement noise")
        ax[1].legend(handles=self._type_legend_handles(), fontsize=4.3, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        if show:
            viz.emit("diag_feature_measurement_reliability")
        print(f"features measured above noise (R_meas > 0.5): {(mv['R_meas'] > 0.5).mean():.0%} of {len(mv)}")
        return MEAS

    def plot_feature_drift_reliability(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt
        REL = self._ensure_reliability()
        if REL is None:
            return pd.DataFrame()

        def row(col):
            dr = drift_reliability(self.FEAT[col].to_numpy(float), self.FEAT.sensor.to_numpy(),
                                   self.FEAT.timepoint.to_numpy(), self.FEAT.concentration.to_numpy(),
                                   REL.loc[col, "sigma2_meas"], REL.loc[col, "n_rep"])
            return {"feature": col, "type": feat_type(col), **dr}
        DRIFT = pd.DataFrame([row(c) for c in self.STATE if c in REL.index]).set_index("feature")
        valid = DRIFT.dropna(subset=["R_drift"])
        self.DRIFT, self._valid = DRIFT, valid
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), dpi=140)
        h = self._annot_hist(ax[0], valid["R_drift"].to_numpy(float), "#1B9E77", bins=40, rng=(0, 1), nlabel="n_features")
        ax[0].set_xlabel("R_drift (drift above noise)"); ax[0].set_ylabel("# features")
        ax[0].set_title("per-feature drift reliability"); ax[0].legend(handles=h, fontsize=6.5)
        sd = valid["sigma2_drift"].to_numpy(float); nf = valid["noise_floor"].to_numpy(float)
        m = (sd > 0) & (nf > 0) & np.isfinite(nf)
        ax[1].scatter(nf[m], sd[m], s=14, alpha=0.7,
                      c=[self.type_color.get(t, "#888888") for t in valid["type"].to_numpy()[m]])
        lim = [min(nf[m].min(), sd[m].min()), max(nf[m].max(), sd[m].max())] if m.any() else [1e-6, 1]
        ax[1].plot(lim, lim, "k--", lw=1, label="drift = noise"); ax[1].set_xscale("log"); ax[1].set_yscale("log")
        ax[1].set_xlabel("noise floor σ²_meas/n_rep"); ax[1].set_ylabel("drift variance σ²_drift")
        ax[1].set_title("drift vs measurement noise")
        ax[1].legend(handles=self._type_legend_handles(), fontsize=4.3, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        if show:
            viz.emit("diag_feature_drift_reliability")
        print(f"resolvable drift: R_drift>0.5 for {(valid['R_drift']>0.5).mean():.0%}  |  "
              f"drift_snr>1 for {(valid['drift_snr']>1).mean():.0%}")
        return (DRIFT.groupby("type").agg(n=("R_drift", "size"), median_R_drift=("R_drift", "median"),
                median_drift_snr=("drift_snr", "median")).sort_values("median_R_drift", ascending=False))

    # ---- Section 3: drift alignment ------------------------------------------------------------
    def plot_drift_alignment(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt

        def row(col):
            a = drift_alignment(self.FEAT[col].to_numpy(float), self.FEAT[RESP].to_numpy(float),
                                self.FEAT.sensor.to_numpy(), self.FEAT.timepoint.to_numpy(),
                                self.FEAT.concentration.to_numpy(), method=self.cfg.mono_method)
            return {"feature": col, "type": feat_type(col), **a}
        ALIGN = pd.DataFrame([row(c) for c in self.STATE]).set_index("feature").dropna(subset=["alignment_r"])
        self.ALIGN = ALIGN
        fig, ax = plt.subplots(1, 2, figsize=(13, 3.8), dpi=140)
        h = self._annot_hist(ax[0], ALIGN["alignment_r"].to_numpy(float), "#E6A817", bins=40, rng=(-1, 1),
                             nlabel="n_features")
        ax[0].axvline(0, color="0.4", lw=0.8); ax[0].set_xlabel("drift alignment r with response (signed)")
        ax[0].set_ylabel("# features"); ax[0].set_title("predictor-response drift alignment")
        ax[0].legend(handles=h, fontsize=6.5)
        if self.DRIFT is not None:
            J = self.DRIFT.join(ALIGN[["alignment_r"]], how="inner").dropna(subset=["R_drift", "alignment_r"])
            wt = np.array([float(self.VAR.loc[f, "within_time"]) if f in self.VAR.index else 0.0 for f in J.index])
            sizes = 10 + 260 * (wt / (np.nanmax(wt) or 1.0))
            ax[1].scatter(J["R_drift"], J["alignment_r"], s=sizes, alpha=0.55,
                          c=[self.type_color.get(t, "#888888") for t in J["type"].to_numpy()],
                          edgecolors="0.3", linewidths=0.3)
            ax[1].axhline(0, color="0.4", lw=0.8); ax[1].axhline(0.5, color="0.7", ls=":")
            ax[1].axhline(-0.5, color="0.7", ls=":"); ax[1].axvline(0.5, color="0.7", ls=":")
            ax[1].set_xlim(0, 1); ax[1].set_ylim(-1, 1)
            ax[1].set_xlabel("R_drift (is the drift real?)"); ax[1].set_ylabel("alignment r (tracks response)")
            ax[1].set_title("feature-quality map (size ∝ within-time variance)")
            ax[1].legend(handles=self._type_legend_handles(), fontsize=4.3, loc="center left", bbox_to_anchor=(1.02, 0.5))
        else:
            ax[1].axis("off"); ax[1].text(0.5, 0.5, "run plot_feature_drift_reliability()\nfor the quality map",
                                          ha="center", va="center", fontsize=8)
        plt.tight_layout()
        if show:
            viz.emit("diag_drift_alignment")
        self._maxa = float(ALIGN["alignment_r"].abs().max())
        print(f"max |alignment| = {self._maxa:.2f}  ->  "
              f"{'a feature tracks the response drift (good)' if self._maxa > 0.5 else 'NO feature tracks the response drift'}")
        return (ALIGN.reindex(ALIGN["alignment_r"].abs().sort_values(ascending=False).index)
                .head(10)[["type", "alignment_r", "n_cells"]])

    def plot_frequency_bandwidth_tradeoff(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        if self.root is None:
            print("frequency-bandwidth tradeoff needs the raw ROOT; skipping."); return
        from .data.pstrace import parse_folder, parse_filename, read_pstrace
        from .features.eis import eis_features
        from .features.extract import _avg_eis
        d0 = self._device_baselines()
        fqd = [t for t in _FREQDEP_TYPES if t in self.freqdep_cols]
        valid_ct = {(d, int(c), float(t)) for d, c, t in
                    self.FEAT[["device", "channel", "timepoint"]].drop_duplicates().itertuples(index=False, name=None)}
        ref = None; onsets = []; nct = 0; ftab = {}
        for folder in sorted(p for p in Path(self.root).iterdir() if p.is_dir()):
            fm = parse_folder(folder.name)
            if not fm or fm.get("testtype") != "signal":
                continue
            for csv in sorted(folder.glob("*.csv")):
                nm = parse_filename(csv.name)
                if not nm or nm["signaltype"] != "eis" or nm["deviceid"] not in d0:
                    continue
                tp = float((fm["date"] - d0[nm["deviceid"]]).days)
                try:
                    exp = read_pstrace(csv)
                except Exception:
                    continue
                for ch in exp.eis_channels:
                    specs = [v for (c, _), v in exp.eis.items() if c == ch]
                    fr, zr, zi = _avg_eis(specs); o = np.argsort(fr); fr, zr, zi = fr[o], zr[o], zi[o]
                    nct += 1
                    ind = np.flatnonzero(zi >= 0); onsets.append(float(fr[ind[0]]) if ind.size else np.nan)
                    key = (nm["deviceid"], int(ch), tp)
                    if key not in valid_ct:
                        continue
                    if ref is None:
                        ref = fr
                    if fr.shape == ref.shape and np.allclose(fr, ref, rtol=1e-3):
                        ftab[key] = eis_features(fr, zr, zi)
        gN = len(ref) if ref is not None else 0
        onset_arr = np.asarray([o for o in onsets if np.isfinite(o)], float)
        keys = list(zip(self.FEAT["device"], self.FEAT["channel"].astype(int), self.FEAT["timepoint"].astype(float)))
        resp = self.FEAT[RESP].to_numpy(float); sen = self.FEAT["sensor"].to_numpy()
        tp = self.FEAT["timepoint"].to_numpy(); cc = self.FEAT["concentration"].to_numpy(float)
        align = {t: np.full(gN, np.nan) for t in fqd}
        for t in fqd:
            for i in range(gN):
                col = np.array([ftab[k][t][i] if k in ftab else np.nan for k in keys])
                m = np.isfinite(col) & np.isfinite(resp)
                if m.sum() >= 4:
                    align[t][i] = drift_alignment(col[m], resp[m], sen[m], tp[m], cc[m],
                                                  method=self.cfg.mono_method)["alignment_r"]
        fig, ax = plt.subplots(1, 2, figsize=(13, 4.0), dpi=140)
        if ref is not None and nct:
            uppers = np.array(sorted(set(ref)))
            kept = [nct - int((onset_arr <= U).sum()) for U in uppers]
            ax[0].plot(uppers, kept, "-o", ms=3, color="#4C78A8")
        ax[0].axvspan(self.band[0], self.band[1], color="0.8", alpha=0.35)
        ax[0].axvline(self.band[0], color="green", ls="--", lw=1.3, label=f"lower edge = {self.band[0]:g} Hz")
        ax[0].axvline(self.band[1], color="crimson", ls="--", lw=1.3, label=f"upper edge = {self.band[1]:g} Hz")
        ax[0].set_xscale("log"); ax[0].set_xlabel("upper band edge (Hz)")
        ax[0].set_ylabel("channel-timepoints kept (EIS.1)")
        ax[0].set_title("retention vs band ceiling"); ax[0].legend(fontsize=7)
        for t in fqd:
            if np.isfinite(align[t]).any():
                ax[1].plot(ref, align[t], "-o", ms=2.5, lw=1.0, color=self.type_color.get(t, "#888888"), label=t)
        ax[1].axhline(0, color="0.5", lw=0.8); ax[1].axvspan(self.band[0], self.band[1], color="0.8", alpha=0.35)
        ax[1].axvline(self.band[0], color="green", ls="--", lw=1.3); ax[1].axvline(self.band[1], color="crimson", ls="--", lw=1.3)
        ax[1].set_xscale("log"); ax[1].set_ylim(-1, 1); ax[1].set_xlabel("frequency (Hz)")
        ax[1].set_ylabel("drift alignment r (signed)")
        ax[1].set_title("drift alignment across full measured range (shaded = current band)")
        ax[1].legend(fontsize=5, ncol=2, loc="lower left"); plt.tight_layout()
        if show:
            viz.emit("diag_frequency_bandwidth_tradeoff")
        inb = (ref >= self.band[0]) & (ref <= self.band[1]) if ref is not None else np.array([], bool)
        ab = (ref > self.band[1]) if ref is not None else np.array([], bool)

        def peak(mask):
            vals = [np.nanmax(np.abs(align[t][mask])) for t in fqd if np.isfinite(align[t][mask]).any()]
            return max(vals) if vals else float("nan")
        print(f"max |aligned drift| in-band = {peak(inb):.2f}   vs above ceiling = {peak(ab):.2f}")

    def plot_alignment_sanity_checks(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        if self.ALIGN is None:
            self.plot_drift_alignment(show=False)
        ALIGN = self.ALIGN

        def abs_align(col):
            zs = []
            for _cc, g in self.FEAT.groupby("concentration"):
                x = g[col].to_numpy(float); y = g[RESP].to_numpy(float); m = np.isfinite(x) & np.isfinite(y)
                if m.sum() < 5 or np.std(x[m]) == 0 or np.std(y[m]) == 0:
                    continue
                zs.append(np.arctanh(np.clip(np.corrcoef(x[m], y[m])[0, 1], -0.999, 0.999)))
            return float(np.tanh(np.mean(zs))) if zs else np.nan
        ABS = pd.Series({c: abs_align(c) for c in self.STATE}, name="abs_align").dropna()
        da = float(ALIGN["abs_alignment"].max()); aa = float(ABS.abs().max())
        print(f"max |DRIFT alignment|    (within-sensor change) = {da:.2f}")
        print(f"max |ABSOLUTE alignment| (cross-sensor level)   = {aa:.2f}   [same scale]")
        verdict = ("ABSOLUTE >> DRIFT: signal is in the LEVEL; test d0_normalize=False (Track 3)."
                   if aa > da + 0.1 else ("comparable: the drift (D0) framing is not discarding an obvious "
                   "absolute signal." if abs(aa - da) <= 0.1 else
                   "DRIFT > ABSOLUTE: temporal co-movement is the stronger signal; D0 (drift) framing appropriate."))
        print("  ->", verdict)
        top = ALIGN.reindex(ALIGN["abs_alignment"].sort_values(ascending=False).index).head(15)
        fig, ax = plt.subplots(figsize=(7.5, 4.6), dpi=140)
        yp = np.arange(len(top))[::-1]
        ax.barh(yp, top["abs_alignment"].to_numpy(), color="#E6A817", label="|drift alignment|")
        ax.scatter(top.index.map(lambda f: abs(ABS.get(f, np.nan))), yp, color="#3B6EA5", zorder=3, s=28,
                   label="|absolute alignment|")
        ax.set_yticks(yp); ax.set_yticklabels(top.index, fontsize=6); ax.set_xlim(0, 1)
        ax.set_xlabel("|Pearson r| (same scale)"); ax.set_title("top 15 features by |drift alignment|")
        ax.legend(fontsize=7); plt.tight_layout()
        if show:
            viz.emit("diag_alignment_sanity_checks")
        return {"max_drift_alignment": da, "max_absolute_alignment": aa}

    # ---- Section 4: verdict --------------------------------------------------------------------
    def verdict(self) -> None:
        drift = float(self.VAR["within_time"].median()); struct = float(self.VAR["between_sensor"].median())
        ntp = int(self.FEAT.timepoint.nunique())
        tps_per_sensor = self.FEAT.groupby("sensor")["timepoint"].nunique()
        print("Diagnosis (this selection):")
        print(f"  • Features {struct:.0%} structural / {drift:.0%} drift (median within-time). "
              "D0-normalization removes structure; the model learns from the drift slice.")
        if self.R is not None and np.isfinite(self.R):
            rd = (f"{(self._valid['R_drift']>0.5).mean():.0%}" if self._valid is not None else "n/a")
            print(f"  • Response reliability R = {self.R:.2f} "
                  f"({'clean' if self.R>=0.8 else 'noisy: RMSEP floor is real' if self.R<0.5 else 'moderate'}); "
                  f"drift resolvable for {rd} of features.")
        if self._maxa is not None:
            print(f"  • Best predictor-response drift alignment |r| = {self._maxa:.2f} "
                  f"({'a feature tracks the response drift' if self._maxa>0.5 else 'NOTHING tracks the response drift; recalibration unlikely'}).")
        print(f"  • Timepoints: {ntp} distinct. Forward-chained nested CV needs ≥2 -> "
              f"{'OK' if ntp>=2 else 'INSUFFICIENT'}; per-sensor coverage median {int(tps_per_sensor.median())}.")
        print("  • Well-posed IF: drift resolvable above noise (§2), ≥1 feature's drift tracks the response "
              "(§3), and enough timepoints for CV. Else expect a high RMSEP floor regardless of architecture.")

    # ---- Section 5: D0-normalization effect ----------------------------------------------------
    def plot_d0_effect(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
        from .features.normalize import d0_normalize_frame
        FEAT_D0 = d0_normalize_frame(self.FEAT.copy(), self.STATE)
        VAR_D0 = pd.DataFrame({c: self._vh(FEAT_D0, c) for c in self.STATE}).T[
            ["between_sensor", "within_time", "within_dose"]]
        self.VAR_D0, self.FEAT_D0 = VAR_D0, FEAT_D0
        keys = ["between_sensor", "within_time", "within_dose"]; labs = ["between-sensor", "within-time", "within-dose"]
        cols = ["#8C8C8C", "#4C78A8", "#7B3FA0"]; xr = np.arange(len(keys)); w = 0.38
        raw = [float(self.VAR[k].median()) for k in keys]; d0 = [float(VAR_D0[k].median()) for k in keys]
        fig, ax = plt.subplots(figsize=(6.4, 3.4), dpi=140)
        ax.bar(xr - w / 2, raw, w, color=cols, alpha=0.45, edgecolor="0.3")
        ax.bar(xr + w / 2, d0, w, color=cols, alpha=1.0, edgecolor="0.3")
        for i, (a, b) in enumerate(zip(raw, d0)):
            ax.text(i - w / 2, a + 0.01, f"{a:.2f}", ha="center", fontsize=6)
            ax.text(i + w / 2, b + 0.01, f"{b:.2f}", ha="center", fontsize=6)
        ax.set_xticks(xr); ax.set_xticklabels(labs); ax.set_ylabel("median variance share"); ax.set_ylim(0, 1)
        ax.set_title("variance decomposition: raw vs D0-normalized")
        ax.legend(handles=[Patch(facecolor="0.6", alpha=0.45, label="raw"),
                           Patch(facecolor="0.6", alpha=1.0, label="D0-normalized")], fontsize=7)
        plt.tight_layout()
        if show:
            viz.emit("diag_d0_effect_1")
        topfeat = self.VAR["within_time"].idxmax()
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), dpi=140, sharex=True)
        for si, s in enumerate(sorted(self.FEAT.sensor.unique())):
            col = plt.get_cmap("tab20")(si % 20)
            gr = self.FEAT[self.FEAT.sensor == s].groupby("timepoint")[topfeat].mean()
            gd = FEAT_D0[FEAT_D0.sensor == s].groupby("timepoint")[topfeat].mean()
            ax[0].plot(gr.index, gr.values, "-o", ms=4, lw=1.3, color=col)
            ax[1].plot(gd.index, gd.values, "-o", ms=4, lw=1.3, color=col)
        ax[0].set_title(f"'{topfeat}': raw (per sensor)"); ax[1].set_title(f"'{topfeat}': D0-normalized")
        for a in ax:
            a.set_xlabel("timepoint (days)")
        fig.suptitle("top-drift feature: fabrication offset (raw) collapses to a common D0 baseline (right)", y=1.02)
        plt.tight_layout()
        if show:
            viz.emit("diag_d0_effect_2")
        print(f"between-sensor median: raw {self.VAR['between_sensor'].median():.2f} -> D0 {VAR_D0['between_sensor'].median():.2f}"
              f"  |  within-time: raw {self.VAR['within_time'].median():.2f} -> D0 {VAR_D0['within_time'].median():.2f}")

    def plot_d0_invariance(self, show: bool = True):
        viz.ensure_style()
        import matplotlib.pyplot as plt
        if getattr(self, "FEAT_D0", None) is None:
            self.plot_d0_effect(show=False)
        if self.ALIGN is None:
            self.plot_drift_alignment(show=False)
        ad0 = {c: drift_alignment(self.FEAT_D0[c].to_numpy(float), self.FEAT_D0[RESP].to_numpy(float),
                                  self.FEAT_D0.sensor.to_numpy(), self.FEAT_D0.timepoint.to_numpy(),
                                  self.FEAT_D0.concentration.to_numpy(), method=self.cfg.mono_method)["alignment_r"]
               for c in self.STATE}
        ALIGN_D0 = pd.Series(ad0).dropna()
        common = [f for f in self.ALIGN.index if f in ALIGN_D0.index]
        fig, ax = plt.subplots(1, 2, figsize=(12, 3.8), dpi=140)
        ax[0].plot([-1, 1], [-1, 1], "k--", lw=1)
        ax[0].scatter(self.ALIGN.loc[common, "alignment_r"], ALIGN_D0.loc[common], s=14, alpha=0.6,
                      c=[self.type_color.get(feat_type(f), "#888888") for f in common])
        ax[0].set_xlim(-1, 1); ax[0].set_ylim(-1, 1)
        ax[0].set_xlabel("alignment r (raw)"); ax[0].set_ylabel("alignment r (D0-normalized)")
        ax[0].set_title("drift↔response alignment is invariant to D0-norm")
        cf = [f for f in self.STATE if f in self.VAR.index and f in self.VAR_D0.index]
        ax[1].plot([0, 1], [0, 1], "k--", lw=1)
        ax[1].scatter(self.VAR.loc[cf, "within_time"], self.VAR_D0.loc[cf, "within_time"], s=14, alpha=0.6,
                      c=[self.type_color.get(feat_type(f), "#888888") for f in cf])
        ax[1].set_xlim(0, 1); ax[1].set_ylim(0, 1)
        ax[1].set_xlabel("within-time share (raw)"); ax[1].set_ylabel("within-time share (D0-normalized)")
        ax[1].set_title("within-time (drift) share rises after D0-norm")
        ax[1].legend(handles=self._type_legend_handles(), fontsize=4.3, loc="center left", bbox_to_anchor=(1.02, 0.5))
        plt.tight_layout()
        if show:
            viz.emit("diag_d0_invariance")
        dmed = float((ALIGN_D0.loc[common] - self.ALIGN.loc[common, "alignment_r"]).abs().median())
        print(f"median |Δ alignment| raw→D0 = {dmed:.3f}  (≈0 ⇒ D0-norm does not distort the drift↔response signal).")

    # ---- Section 6: response / deformation modes ----------------------------------------------
    def plot_response_modes(self, show: bool = True) -> pd.DataFrame:
        viz.ensure_style()
        import matplotlib.pyplot as plt
        from .features.targets import sensitivity_featureset
        REL = self._ensure_reliability()

        def mode_row(col, level, note, df, conc, has_rel):
            v = df[col].to_numpy(float); m = np.isfinite(v)
            h = variance_hierarchy(v[m], df.sensor.to_numpy()[m], df.timepoint.to_numpy()[m], conc[m])
            dr = (drift_reliability(v[m], df.sensor.to_numpy()[m], df.timepoint.to_numpy()[m], conc[m],
                                    REL.loc[col, "sigma2_meas"], REL.loc[col, "n_rep"])
                  if has_rel and REL is not None and col in REL.index
                  else {"R_drift": float("nan"), "drift_snr": float("nan")})
            return {"response": col, "level": level, "between_sensor": h["between_channel"],
                    "within_time": h["within_temporal"], "within_dose": h["within_concentration"],
                    "R_drift": dr["R_drift"], "drift_snr": dr["drift_snr"], "note": note}
        rows = []; conc = self.FEAT.concentration.to_numpy(float)
        for col, note in [("NormIpeak", "ratio (default response)"), ("peak_height", "numerator: bg-subtracted peak"),
                          ("mean_Ibg", "denominator: background"), ("peak_area", "peak charge / area"),
                          ("peak_fwhm", "peak width (kinetics)")]:
            if col in self.FEAT.columns:
                rows.append(mode_row(col, "per-dose", note, self.FEAT, conc, has_rel=True))
        S = sensitivity_featureset(self.FEAT.drop(columns=[c for c in ("sensor", "devicetype") if c in self.FEAT.columns]))
        if len(S):
            S["sensor"] = S.device.astype(str) + ":" + S.channel.astype(str)
            zero = np.zeros(len(S))
            for col, note in [("sensitivity_intercept", "DC level (intercept)"), ("sensitivity", "sensitivity (slope)"),
                              ("sensitivity_curvature", "shape (curvature)")]:
                if col in S.columns and int(S[col].notna().sum()) >= 3:
                    rows.append(mode_row(col, "per-sensor-timepoint", note, S, zero, has_rel=False))
        MODES = pd.DataFrame(rows).set_index("response")
        fig, ax = plt.subplots(figsize=(max(8, 0.85 * len(MODES) + 2), 4.2), dpi=140)
        xp = np.arange(len(MODES)); lefts = np.zeros(len(MODES))
        for key, lab, col in [("within_time", "within-time (drift)", "#4C78A8"),
                              ("between_sensor", "between-sensor", "#8C8C8C"), ("within_dose", "within-dose", "#7B3FA0")]:
            vv = MODES[key].fillna(0.0).to_numpy(float); ax.bar(xp, vv, bottom=lefts, color=col, label=lab, width=0.8)
            lefts += vv
        for i, (_, r) in enumerate(MODES.iterrows()):
            if np.isfinite(r["R_drift"]):
                ax.text(i, 1.03, f"R_drift {r['R_drift']:.2f}", ha="center", va="bottom", fontsize=5.5, rotation=90)
        ax.set_xticks(xp); ax.set_xticklabels(MODES.index, rotation=45, ha="right", fontsize=7)
        ax.set_ylim(0, 1.25); ax.axhline(1.0, color="0.7", lw=0.6)
        ax.set_ylabel("variance share"); ax.set_title("Response / deformation modes: decomposition + R_drift")
        ax.legend(fontsize=7, ncol=3, loc="lower center", bbox_to_anchor=(0.5, 1.10)); plt.tight_layout()
        if show:
            viz.emit("diag_response_modes")
        return MODES[["level", "between_sensor", "within_time", "within_dose", "R_drift", "drift_snr", "note"]]
