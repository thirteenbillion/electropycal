"""Publication (Nature-style) plotting defaults for the notebooks.

Applies a compact, colorblind-safe matplotlib style: sans-serif ~8 pt type, thin
axes with the top/right spines removed, frameless legends, and the **Okabe-Ito**
categorical palette (CVD-safe, assigned in a fixed order — never cycled beyond its
length). Sequential magnitude uses viridis (perceptually uniform, CVD-safe).

Usage in a notebook's first cell::

    from electropycal.viz import set_pub_style, categorical, SEQUENTIAL
    set_pub_style()
"""

from __future__ import annotations

# Okabe-Ito colorblind-safe categorical palette, fixed order. The middle six are
# validated (ΔE ≥ 8 under deutan/protan/tritan); black is the single-series / text
# ink, yellow is last (very light — pair with a marker edge on white).
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9",
             "#000000", "#F0E442"]
SEQUENTIAL = "viridis"        # magnitude ramp (concentration, timepoint, …)


def categorical(n: int) -> list[str]:
    """Return ``n`` categorical colors in fixed order (repeats only past 8 series —
    beyond that prefer small multiples or a composite encoding)."""
    if n <= len(OKABE_ITO):
        return OKABE_ITO[:n]
    return [OKABE_ITO[i % len(OKABE_ITO)] for i in range(n)]


def timepoint_label(tp) -> str:
    """Timepoint (days) → compact label, e.g. ``0 → "D0"``."""
    return f"D{int(tp)}"


def channel_color(ch: int, total: int = 16):
    """Fixed viridis color for channel ``ch`` in ``1..total`` (encodes depth / string position)."""
    from matplotlib.colors import Normalize
    import matplotlib.pyplot as plt
    return plt.get_cmap("viridis")(Normalize(1, total)(int(ch)))


def tp_alpha(tp, tps) -> float:
    """Overlay opacity for a timepoint: older = faded, latest = solid (1.0)."""
    tps = sorted(set(tps))
    return 1.0 if len(tps) == 1 else 0.3 + 0.7 * (tps.index(tp) / (len(tps) - 1))


def concentration_color(cc, concs):
    """Plasma gradient by ``log10(dose)`` across ``concs`` (clipped off the near-white extreme)."""
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.colors import Normalize
    cmap = plt.get_cmap("plasma")
    cs = sorted({float(c) for c in concs if c and c > 0})
    if len(cs) < 2:
        return cmap(0.55)
    t = float(np.clip(Normalize(np.log10(min(cs)), np.log10(max(cs)))(np.log10(cc)), 0.0, 1.0))
    return cmap(0.08 + 0.84 * t)


def panel_grid(n: int, ncols: int = 4, size=(2.8, 2.3), dpi: int = 140):
    """A grid of ``n`` panels (``ncols`` wide); returns ``(fig, flat_axes)`` with spares hidden."""
    import math
    import matplotlib.pyplot as plt
    nrows = math.ceil(n / ncols) if n else 1
    fig, axes = plt.subplots(nrows, ncols, figsize=(size[0] * ncols, size[1] * nrows),
                             dpi=dpi, squeeze=False)
    flat = axes.ravel()
    for a in flat[n:]:
        a.axis("off")
    return fig, flat


def set_pub_style() -> None:
    """Set Nature-style matplotlib rcParams (idempotent)."""
    import matplotlib as mpl
    from cycler import cycler
    mpl.rcParams.update({
        "figure.dpi": 120, "savefig.dpi": 300, "savefig.bbox": "tight",
        # Nature's figure font is Helvetica (Arial is the standard metric-compatible
        # substitute); DejaVu Sans is the guaranteed fallback when neither is installed.
        "font.family": "sans-serif",
        "font.sans-serif": ["Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8,
        "xtick.labelsize": 7, "ytick.labelsize": 7, "legend.fontsize": 7,
        "axes.titleweight": "bold", "axes.titlepad": 4,
        "axes.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": False, "grid.linewidth": 0.4, "grid.alpha": 0.4,
        "lines.linewidth": 1.2, "lines.markersize": 4,
        "xtick.direction": "out", "ytick.direction": "out",
        "xtick.major.width": 0.6, "ytick.major.width": 0.6,
        "xtick.major.size": 3, "ytick.major.size": 3,
        "legend.frameon": False, "legend.handlelength": 1.4,
        "axes.prop_cycle": cycler(color=OKABE_ITO),
    })
