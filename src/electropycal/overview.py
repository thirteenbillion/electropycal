"""Finalized-dataset overview plots (folds discovery_checkpointed §2.1).

Three views of the QC-valid featureset that discovery will train on: a per-sensor-timepoint retention
grid (how many concentrations survived), and NormIpeak-vs-time per sensor and per device. Each takes the
selected featureset frame; all are robust when a ``device`` column is absent (synthetic data → sensor is
the channel, and the per-device view is skipped).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _with_sensor(sel: pd.DataFrame) -> pd.DataFrame:
    ov = sel.copy()
    if "sensor" not in ov.columns:
        if "device" in ov.columns:
            ov["sensor"] = ov.device.astype(str) + ":" + ov.channel.astype(str)
        else:
            ov["sensor"] = ov.channel.astype(str)
    return ov


def plot_qc_grid(sel: pd.DataFrame, show: bool = True):
    """Sensor × timepoint grid, colored by #concentrations retained (green = all, red = 0/dropped),
    positioned at true day spacing."""
    import matplotlib.pyplot as plt
    ov = _with_sensor(sel)
    if not len(ov):
        print("no valid channel-timepoints to preview."); return
    nconc_max = int(sel.concentration.nunique())
    grid = ov.groupby(["sensor", "timepoint"])["concentration"].nunique().unstack("timepoint")
    grid = grid.reindex(sorted(grid.index), axis=0).reindex(sorted(grid.columns), axis=1)
    rows = list(grid.index); days = [float(t) for t in grid.columns]
    xs, ys, cs = [], [], []
    for ri, _ in enumerate(rows):
        for ci, d in enumerate(days):
            v = grid.iloc[ri, ci]
            if np.isfinite(v):
                xs.append(d); ys.append(ri); cs.append(v)
    span = (max(days) - min(days)) if len(days) > 1 else 1.0
    fig, ax = plt.subplots(figsize=(min(2 + 0.30 * span, 15), min(1.5 + 0.26 * len(rows), 12)), dpi=140)
    sc = ax.scatter(xs, ys, c=cs, cmap="RdYlGn", vmin=0, vmax=nconc_max, marker="s", s=64,
                    edgecolors="0.6", linewidths=0.3)
    ax.set_yticks(range(len(rows))); ax.set_yticklabels(rows, fontsize=5); ax.invert_yaxis()
    ax.set_xlabel("timepoint (days)"); ax.set_ylabel("sensor (device:channel)"); ax.margins(x=0.02, y=0.02)
    ax.set_title(f"QC-valid data: #concentrations retained per sensor-timepoint (green = all {nconc_max}, red = 0)")
    fig.colorbar(sc, label="# concentrations retained", fraction=0.03, pad=0.02)
    plt.tight_layout()
    if show:
        plt.show()


def plot_normipeak_per_sensor(sel: pd.DataFrame, yclip_pct: float | None = 99, ymax: float | None = None,
                              show: bool = True):
    """NormIpeak vs timepoint (true day spacing), one line per sensor, one panel per concentration.
    Per-panel y-top clipped to ``yclip_pct`` (or a hard ``ymax``) so outliers don't compress traces."""
    import matplotlib.pyplot as plt
    ov = _with_sensor(sel)
    if not len(ov):
        return
    concs = sorted(ov.concentration.unique()); sensors = sorted(ov.sensor.unique())
    cm = plt.get_cmap("turbo")
    fig, axes = plt.subplots(1, len(concs), figsize=(max(3.0 * len(concs), 4), 3.6), dpi=140, squeeze=False)
    for ax, cc in zip(axes[0], concs):
        sub = ov[ov.concentration == cc]; vals = []
        for si, sen in enumerate(sensors):
            g = sub[sub.sensor == sen].groupby("timepoint")["NormIpeak"].mean().sort_index()
            if len(g):
                ax.plot(g.index.astype(float), g.values, "-o", ms=3, lw=1.0,
                        color=cm(si / max(len(sensors) - 1, 1)))
                vals.extend(g.values)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        elif yclip_pct is not None and vals:
            top = float(np.nanpercentile(vals, yclip_pct))
            if np.isfinite(top) and top > ax.get_ylim()[0]:
                ax.set_ylim(top=top * 1.05)
        ax.set_title(f"{int(cc)} nM", fontsize=8); ax.set_xlabel("timepoint (days)")
    axes[0][0].set_ylabel("mean NormIpeak\n(per sensor-timepoint-dose)")
    h = [plt.Line2D([], [], color=cm(i / max(len(sensors) - 1, 1)), lw=2) for i in range(len(sensors))]
    fig.legend(h, sensors, fontsize=4.5, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(sensors), 8), title="sensor")
    fig.suptitle(f"NormIpeak vs timepoint (true day spacing; per-panel y clipped at p{yclip_pct}) "
                 f"- one line per sensor", y=1.04)
    plt.tight_layout()
    if show:
        plt.show()


def plot_normipeak_per_device(sel: pd.DataFrame, ymax: float | None = None, show: bool = True):
    """Per-device mean NormIpeak vs timepoint (band = ±1 s.d.), device outliers excluded before
    averaging (median ± 3·MAD within each device × concentration). Skipped if no ``device`` column."""
    import matplotlib.pyplot as plt
    if "device" not in sel.columns:
        print("per-device overview skipped (no 'device' column — synthetic data)."); return
    ov = _with_sensor(sel)
    if not len(ov):
        return
    concs = sorted(ov.concentration.unique()); devices = sorted(ov.device.unique())
    cmd = plt.get_cmap("tab10")

    def keep_inliers(v):
        v = np.asarray(v, float); m = np.nanmedian(v)
        mad = np.nanmedian(np.abs(v - m)) or np.nanstd(v) or 0.0
        return np.ones_like(v, bool) if mad == 0 else (np.abs(v - m) <= 3.0 * 1.4826 * mad)
    fig, axes = plt.subplots(1, len(concs), figsize=(max(3.0 * len(concs), 4), 3.6), dpi=140, squeeze=False)
    for ax, cc in zip(axes[0], concs):
        sub = ov[ov.concentration == cc]
        for di, dev in enumerate(devices):
            cell = sub[sub.device == dev].groupby(["sensor", "timepoint"])["NormIpeak"].mean().reset_index()
            if cell.empty:
                continue
            cell = cell[keep_inliers(cell["NormIpeak"].to_numpy())]
            st = cell.groupby("timepoint")["NormIpeak"].agg(["mean", "std"]).sort_index()
            if len(st):
                x = st.index.astype(float); m = st["mean"].to_numpy(); s = st["std"].fillna(0).to_numpy()
                ax.plot(x, m, "-o", ms=3, lw=1.6, color=cmd(di % 10), label=str(dev))
                ax.fill_between(x, m - s, m + s, color=cmd(di % 10), alpha=0.15)
        if ymax is not None:
            ax.set_ylim(top=ymax)
        ax.set_title(f"{int(cc)} nM", fontsize=8); ax.set_xlabel("timepoint (days)")
    axes[0][0].set_ylabel("mean NormIpeak\n(per device: mean +/- s.d. over sensors, outliers excluded)")
    h = [plt.Line2D([], [], color=cmd(i % 10), lw=2) for i in range(len(devices))]
    fig.legend(h, [str(d) for d in devices], fontsize=6, loc="upper center", bbox_to_anchor=(0.5, -0.02),
               ncol=min(len(devices), 8), title="device")
    fig.suptitle("NormIpeak vs timepoint - per-device average (outliers excluded; band = +/-1 s.d.)", y=1.04)
    plt.tight_layout()
    if show:
        plt.show()


def conditions_table(queue) -> pd.DataFrame:
    """The batch queue as a tidy table: batch, task, architecture, k_LVs, selector, evaluation track."""
    label = {1: "1. Baselines", 2: "2. Channel-specific", 3: "3. P>N feature selection",
             4: "4. New-channel generalization"}

    def task(name):
        parts = name.split("_"); return parts[1] if len(parts) > 1 else "?"
    return pd.DataFrame([
        {"batch": label.get(c.batch, str(c.batch)), "task": task(c.name), "architecture": c.architecture,
         "k_LVs": list(c.k_grid), "selector": c.selector or "—", "evaluation": c.track}
        for c in queue]).sort_values(["batch", "task"]).reset_index(drop=True)
