"""Deployment plots (folded from the deployment_domain_shift notebook).

Two figures for the recalibration + domain-shift monitor, so the notebook is thin calls and the same
plots are reproducible from a script. Both take a summary frame (from ``recalibrate`` rows, or from
``recalibrate_invivo``) and draw the recalibrated response next to the CORAL domain-distance monitor,
the confidence signal that flags when the frozen model is extrapolating beyond its validated regime.
"""

from __future__ import annotations
from .. import viz


def plot_domain_monitor(df, flag: float = 2.0, day_col: str = "day",
                        distance_col: str = "domain_distance", value_col: str = "mean_recal",
                        confidence_col: str = "confidence"):
    """Two-panel monitor from a per-session summary frame: CORAL domain distance vs day (left) and
    the recalibrated output bars colored by confidence (right; red = extrapolating past ``flag``)."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6))
    if confidence_col in df:
        colors = ["#55A868" if c == "ok" else "#C44E52" for c in df[confidence_col]]
    else:
        colors = ["#55A868" if d < flag else "#C44E52" for d in df[distance_col]]
    ax1.plot(df[day_col], df[distance_col], "o-", color="#4C72B0")
    ax1.axhline(flag, ls="--", color="crimson", label=f"flag threshold {flag}")
    ax1.set_xlabel("in-vivo day"); ax1.set_ylabel("CORAL domain distance")
    ax1.set_title("Drift / confidence monitor"); ax1.legend()
    ax2.bar(df[day_col].astype(str), df[value_col], color=colors)
    ax2.set_xlabel("in-vivo day"); ax2.set_ylabel("mean recalibrated NormIpeak")
    ax2.set_title("Recalibrated output (red = extrapolating)")
    plt.tight_layout(); viz.emit("deploy_domain_monitor")


def plot_invivo_recalibration(res, flag: float = 2.0):
    """Two-panel figure from a ``recalibrate_invivo`` result frame (``timepoint``,
    ``mean_norm_ipeak``, ``domain_distance``): recalibrated response and domain shift vs in-vitro."""
    viz.ensure_style()
    import matplotlib.pyplot as plt
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 3.6))
    a1.plot(res.timepoint, res.mean_norm_ipeak, "o-")
    a1.set_xlabel("in-vivo day"); a1.set_ylabel("mean recal NormIpeak")
    a1.set_title("recalibrated response")
    a2.plot(res.timepoint, res.domain_distance, "s-", color="crimson")
    a2.axhline(flag, ls="--", color="0.5")
    a2.set_xlabel("in-vivo day"); a2.set_ylabel("CORAL domain distance")
    a2.set_title("domain shift vs in-vitro")
    plt.tight_layout(); viz.emit("deploy_invivo_recalibration")
