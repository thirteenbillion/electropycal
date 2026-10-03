"""Stabilization review: inventory + settling plots as a library API (folds stabilization_review).

Wraps the ``data.stabilization`` primitives (``stabilization_traces``, ``round_drift_series``,
``check_converged``, ``estimate_settle_tau``) with a filename inventory of the stabilization files under
a data ``ROOT`` and the review figures, so the notebook is thin calls. A :class:`StabilizationIndex`
holds the inventory + the settling parameters (``v_target``, ``cycles_per_round``, ``tol``, ``patience``,
``smooth_window``, ``region``, ``drop_coldstart``); ``plot_*`` / ``convergence_table`` take it plus a list
of ``(device, timepoint)`` pairs.
"""

from __future__ import annotations

from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd

from .data.pstrace import parse_filename, parse_folder, read_pstrace
from .data.stabilization import (check_converged, estimate_settle_tau, round_drift_series,
                                 stabilization_traces)
from .viz import channel_color, panel_grid, timepoint_label
from . import viz

_PHASE_LABEL = {"start": "first 3 (start)", "end": "last 3 (end)", "full": "full"}
_PHASE_ORDER = ["full", "end", "start"]        # richest first

#: The one place the stabilization filename pattern is written down. The error message
#: below quotes this constant rather than restating it, so the two cannot drift apart.
_STAB_GLOB = "*_fscv_stabilization*.csv"


class NoStabilizationFiles(FileNotFoundError):
    """Raised when a tree holds no stabilization exports for :class:`StabilizationIndex`.

    Stabilization runs are optional -- extraction never reads them and the modelling stack
    does not need them -- so an export legitimately may not have any. This is a clear stop
    rather than a downstream ``AttributeError`` on an empty, column-less inventory.
    """


class StabilizationIndex:
    """Inventory + settling-parameter config + per-phase loaders for stabilization files (see module)."""

    def __init__(self, root, *, v_target: float = 0.7, cycles_per_round: int = 20, tol: float = 0.02,
                 patience: int = 3, smooth_window: int = 3, region: str = "anodic",
                 drop_coldstart: bool = True):
        self.root = str(root)
        self.v_target = v_target
        self.cycles_per_round = cycles_per_round
        self.tol = tol
        self.patience = patience
        self.smooth_window = smooth_window
        self.region = region
        self.drop_coldstart = drop_coldstart

        from .data.paths import validate_path
        rp = validate_path(root, 'stabilization root')
        if not rp.is_dir():
            raise NoStabilizationFiles(
                f"stabilization root is not a directory: {self.root!r}\n"
                f"  resolved to : {rp.resolve()}\n"
                f"  cwd         : {Path.cwd()}\n"
                f"  Expected a PSTrace export root holding <date>_<devicetype>_<testtype>/ "
                f"session folders.")

        sig_dates = defaultdict(set)
        recs = []
        n_dirs = n_sessions = 0
        for folder in sorted(p for p in rp.iterdir() if p.is_dir()):
            n_dirs += 1
            fm = parse_folder(folder.name)
            if not fm:
                continue
            n_sessions += 1
            if fm.get("testtype") == "signal":
                for csv in folder.glob("*.csv"):
                    mm = parse_filename(csv.name)
                    if mm:
                        sig_dates[mm["deviceid"]].add(fm["date"])
            for csv in folder.glob(_STAB_GLOB):
                m = parse_filename(csv.name)
                if not m:
                    continue
                phase = m["dose"].split("-", 1)[1] if "-" in m["dose"] else "end"
                recs.append((m["deviceid"], fm.get("devicetype", "?"), fm["date"], phase, str(csv)))
        d0 = {d: min(s) for d, s in sig_dates.items()}
        rows = [dict(device=dev, devicetype=dt, date=date,
                     timepoint=(date - d0.get(dev, date)).days, phase=ph, path=pth)
                for dev, dt, date, ph, pth in recs]
        if not rows:
            # Without this, self.inv is an empty frame with NO COLUMNS, and every consumer
            # dies far from the cause -- load_phases raised
            # ``AttributeError: 'DataFrame' object has no attribute 'device'``, which names
            # neither the tree nor what was missing from it.
            # Only offer causes the counts are actually consistent with -- listing
            # "no folder parsed" next to "14 parsed" would send the reader the wrong way.
            causes = []
            if n_sessions == 0:
                causes.append(f"    - none of the {n_dirs} subdirectory(ies) parsed as a session, so "
                              f"nothing was\n      searched: folder names must look like "
                              f"<date>_<devicetype>_<testtype>")
            else:
                causes.append("    - this export genuinely has no stabilization runs -- they are "
                              "optional,\n      and neither extraction nor the modelling stack "
                              "needs them")
                causes.append("    - the files use a spelling this glob does not match; the "
                              "convention has\n      already moved once, from "
                              "'stabilization.csv' to 'stabilization-full.csv'")
            raise NoStabilizationFiles(
                f"no stabilization files found under {self.root!r}\n"
                f"  resolved to : {rp.resolve()}\n"
                f"  looked for  : {_STAB_GLOB}  (in every parsable session folder, not just "
                f"*_signal)\n"
                f"  scanned     : {n_dirs} subdirectory(ies), {n_sessions} of which parsed as "
                f"sessions\n"
                f"\n"
                f"  Likely causes:\n" + "\n".join(causes))

        self.inv = pd.DataFrame(rows)
        self.dtype_of = {r.device: r.devicetype for r in self.inv.itertuples()}
        self.date_of = {(r.device, r.timepoint): r.date for r in self.inv.itertuples()}

    # ---- identity / selection ------------------------------------------------------------------
    def title(self, dev) -> str:
        return f"{self.dtype_of.get(dev, '?')} {dev}"

    def label_devices(self, df: pd.DataFrame) -> pd.DataFrame:
        if "device" not in df.columns:
            return df
        out = df.copy(); out.insert(0, "devicetype", out["device"].map(self.dtype_of))
        return out.rename(columns={"device": "deviceid"})

    def available_table(self) -> pd.DataFrame:
        if not len(self.inv):
            return pd.DataFrame(columns=["device", "date", "timepoint", "phases"])
        av = (self.inv.groupby(["device", "date", "timepoint"])["phase"]
              .agg(lambda s: ", ".join(sorted(set(s)))).reset_index(name="phases"))
        av["timepoint"] = av["timepoint"].map(timepoint_label)
        return av[["device", "date", "timepoint", "phases"]]

    def pairs(self, select=None, device_types=None) -> list[tuple]:
        ps = sorted(set(zip(self.inv.device, self.inv.timepoint))) if select is None else list(select)
        if device_types is not None:
            ps = [(d, t) for (d, t) in ps if self.dtype_of.get(d) in device_types]
        return ps

    def describe(self) -> str:
        L = ["StabilizationIndex settings:",
             f"  ROOT             = {self.root}",
             f"  v_target         = {self.v_target} V   (oxidation potential tracked)",
             f"  cycles_per_round = {self.cycles_per_round}",
             f"  tol / patience   = {self.tol} / {self.patience}   (convergence: rounds below tol)",
             f"  smooth_window    = {self.smooth_window} rounds   region = {self.region!r}   "
             f"drop_coldstart = {self.drop_coldstart}",
             "customize by rebuilding, e.g.:",
             "  IDX = StabilizationIndex(ROOT, v_target=0.7, tol=0.02, patience=3, region='anodic')"]
        return "\n".join(L)

    def print_settings(self) -> None:
        print(self.describe(), flush=True)

    # ---- loaders -------------------------------------------------------------------------------
    def load_phases(self, dev, tp) -> dict:
        """``{phase: traces_df}`` for a device-timepoint, ordered full > end > start."""
        got = {r.phase: r.path for r in self.inv[(self.inv.device == dev) & (self.inv.timepoint == tp)].itertuples()}
        return {ph: stabilization_traces(read_pstrace(got[ph]), v_target=self.v_target,
                                         cycles_per_round=self.cycles_per_round)
                for ph in _PHASE_ORDER if ph in got}

    @staticmethod
    def chosen_phases(ph_tr) -> list[tuple]:
        """Traces to draw: full alone (solid) if present, else end (solid) + start (dashed)."""
        if "full" in ph_tr:
            return [("full", "-")]
        return [(ph, ls) for ph, ls in (("end", "-"), ("start", "--")) if ph in ph_tr]

    def drift_series(self, t):
        return round_drift_series(t, region=self.region, drop_coldstart=self.drop_coldstart)

    def round_edges(self, ph_tr, ch):
        """Last cycle of the first round and last cycle of the last round for channel ``ch``."""
        order = ["full"] if "full" in ph_tr else [p for p in ("start", "end") if p in ph_tr]

        def last_of_round(ph, pick):
            d = ph_tr[ph]; d = d[d.channel == ch]
            r = d[d["round"] == pick(d["round"])]
            return r.sort_values("cycle_in_round").iloc[-1]
        return last_of_round(order[0], lambda s: s.min()), last_of_round(order[-1], lambda s: s.max())

    def cycle_no(self, row) -> int:
        """1-based global cycle index for a traces row, taken from the data.

        This used to be ``round * self.cycles_per_round + cycle_in_round + 1``, which silently
        assumed the acquisition used exactly ``self.cycles_per_round`` (default 20) cycles per
        round. When it did not — a different protocol, or a demo tree generated with a smaller
        round — every printed cycle number was wrong, and nothing detected the disagreement.
        ``stabilization_traces`` already derives ``cycle`` by counting within each channel in
        temporal order, so read that instead and the two cannot disagree by construction.
        """
        return int(row["cycle"]) + 1


def _phase_legend(ax, phs):
    from matplotlib.lines import Line2D
    ax.legend(handles=[Line2D([0], [0], color="0.3", ls=ls, label=_PHASE_LABEL[ph])
                       for ph, ls in phs], fontsize=6, loc="best")


def plot_raw_cycles(idx: StabilizationIndex, pairs):
    """§1 — raw FSCV loops: first-round vs last-round last cycle per channel (pick V_target)."""
    viz.ensure_style()
    from matplotlib.lines import Line2D
    import matplotlib.pyplot as plt
    for dev, tp in pairs:
        ph_tr = idx.load_phases(dev, tp)
        if not ph_tr:
            continue
        chs = sorted(next(iter(ph_tr.values())).channel.unique())
        fig, axes = panel_grid(len(chs))
        leg_first = leg_last = None
        for a, ch in zip(axes, chs):
            col = channel_color(ch)
            first, last = idx.round_edges(ph_tr, ch)
            if leg_first is None:
                leg_first, leg_last = idx.cycle_no(first), idx.cycle_no(last)
            a.plot(first["_voltage"], first["_current"], "-", color=col, lw=0.9, alpha=0.35)
            a.plot(last["_voltage"], last["_current"], "-", color=col, lw=1.1, alpha=1.0)
            a.axvspan(0.6, 0.8, color="0.85", alpha=0.5)
            a.axvline(idx.v_target, color="crimson", lw=0.8, ls="--")
            a.set_title(f"ch {ch}", fontsize=7); a.set_xlabel("E (V)"); a.set_ylabel("i (µA)")
        axes[0].legend(handles=[
            Line2D([0], [0], color="0.3", alpha=0.35, label=f"first round, last cycle (cycle {leg_first})"),
            Line2D([0], [0], color="0.3", label=f"last round, last cycle (cycle {leg_last})")],
            fontsize=6, loc="best")
        fig.suptitle(f"{idx.title(dev)} - timepoint {timepoint_label(tp)} - raw FSCV loops", y=1.02)
        fig.tight_layout(); viz.emit("stab_raw_cycles")


def plot_i_vtarget(idx: StabilizationIndex, pairs):
    """§2 — I(V_target) vs cycle per channel."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    for dev, tp in pairs:
        ph_tr = idx.load_phases(dev, tp)
        if not ph_tr:
            continue
        phs = idx.chosen_phases(ph_tr)
        chs = sorted(next(iter(ph_tr.values())).channel.unique())
        fig, axes = panel_grid(len(chs))
        for a, ch in zip(axes, chs):
            col = channel_color(ch)
            for ph, ls in phs:
                t = ph_tr[ph][ph_tr[ph].channel == ch].sort_values("cycle")
                a.plot(t.cycle, t.i_target, ls, color=col, lw=1.0)
            a.margins(x=0.02); a.autoscale(enable=True, axis="x", tight=True)
            a.set_title(f"ch {ch}", fontsize=7); a.set_xlabel("cycle"); a.set_ylabel(f"I({idx.v_target} V) µA")
        _phase_legend(axes[0], phs)
        fig.suptitle(f"{idx.title(dev)} - timepoint {timepoint_label(tp)} - I(V_target) over cycles", y=1.02)
        fig.tight_layout(); viz.emit("stab_i_vtarget")


def plot_round_drift(idx: StabilizationIndex, pairs):
    """§3 — round-averaged drift (convergence metric) per channel, tol line drawn."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    for dev, tp in pairs:
        ph_tr = idx.load_phases(dev, tp)
        if not ph_tr:
            continue
        phs = idx.chosen_phases(ph_tr)
        chs = sorted(next(iter(ph_tr.values())).channel.unique())
        fig, axes = panel_grid(len(chs))
        for a, ch in zip(axes, chs):
            col = channel_color(ch)
            for ph, ls in phs:
                t = ph_tr[ph][ph_tr[ph].channel == ch]
                dr = idx.drift_series(t)
                sm = check_converged(dr, tol=idx.tol, patience=idx.patience,
                                     smooth_window=idx.smooth_window).smoothed
                a.semilogy(np.arange(1, len(sm) + 1), np.clip(sm, 1e-6, None), ls,
                           color=col, lw=1.0, marker="o", ms=2.5)
            a.axhline(idx.tol, color="crimson", lw=0.8, ls="--")
            a.set_title(f"ch {ch}", fontsize=7); a.set_xlabel("round"); a.set_ylabel("norm. drift")
        _phase_legend(axes[0], phs)
        fig.suptitle(f"{idx.title(dev)} - timepoint {timepoint_label(tp)} - round-averaged {idx.region} "
                     f"drift (tol={idx.tol}, patience={idx.patience})", y=1.02)
        fig.tight_layout(); viz.emit("stab_round_drift")


def convergence_table(idx: StabilizationIndex, pairs) -> pd.DataFrame:
    """§4 — per-channel convergence summary: settle_tau, final_drift, n_stable, converged."""
    rows = []
    for dev, tp in pairs:
        ph_tr = idx.load_phases(dev, tp)
        if not ph_tr:
            continue
        tr = next(ph_tr[p] for p in _PHASE_ORDER if p in ph_tr)
        for ch in sorted(tr.channel.unique()):
            dr = idx.drift_series(tr[tr.channel == ch])
            res = check_converged(dr, tol=idx.tol, patience=idx.patience, smooth_window=idx.smooth_window)
            rows.append({"device": dev, "date": idx.date_of.get((dev, tp)), "timepoint": timepoint_label(tp),
                         "channel": int(ch), "settle_tau": round(estimate_settle_tau(dr), 1),
                         "final_drift": round(res.final_drift, 5), "n_stable": res.n_stable,
                         "converged": res.converged})
    return pd.DataFrame(rows)
