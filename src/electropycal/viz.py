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


# ---------------------------------------------------------------------------------------
# Figure output
# ---------------------------------------------------------------------------------------
# Every plotting function in this package used to end in ``plt.show()`` and nothing else:
# 48 sites, zero ``savefig``, zero ``plt.close()``. Under a notebook kernel with
# ``%matplotlib inline`` that renders. Everywhere else, which means CI, a plain script, and
# a chat code-execution sandbox, ``show()`` on the Agg backend is a silent no-op: the code
# completes, reports success, and produces nothing.
#
# One mechanism rather than 48 edits' worth of decisions. Each plotting function ends in
# ``viz.emit("<stable name>")``, which saves, closes, and shows only when showing can work.
#
# Filenames are STABLE and overwrite on re-run. A timestamped name is how you end up with
# forty near-identical PNGs and no idea which one is in the draft; versioning belongs in git
# or in a run directory, not in a filename.

import os as _os
from pathlib import Path as _Path

#: Where :func:`emit` writes. ``None`` means "not configured", and on first use resolves to
#: ``<repo-or-cwd>/figures``, which is gitignored.
_OUT_DIR: _Path | None = None
#: Raster first, vector alongside it, because a figure headed for a paper needs the vector
#: and a figure headed for a chat needs the raster.
_FORMATS: tuple[str, ...] = ("png", "pdf")
#: Tri-state. None = decide from the backend, which is the right default in both a notebook
#: and a headless runner.
_SHOW: bool | None = None
#: Emitted paths, in order, for tests and for a caller that wants to list what it made.
_WRITTEN: list[_Path] = []


def _default_out_dir() -> _Path:
    """``figures/`` beside the nearest ``pyproject.toml``, else beside the cwd."""
    env = _os.environ.get("ELECTROPYCAL_FIGURES")
    if env:
        return _Path(env)
    here = _Path.cwd().resolve()
    for cand in (here, *here.parents):
        if (cand / "pyproject.toml").exists() or (cand / ".git").exists():
            return cand / "figures"
    return here / "figures"


#: "Argument not supplied", distinct from ``None``. Needed because ``show=None`` is itself
#: a meaningful value: it means "sniff the backend". Without this sentinel, ``emit`` calling
#: ``output_dir()`` -> ``configure_output()`` silently reset a caller's ``show=True`` back to
#: sniffing, which is a bug the headless tests caught.
_UNSET = object()


def configure_output(out_dir=_UNSET, formats=_UNSET, show=_UNSET) -> _Path:
    """Set where :func:`emit` writes, what formats, and whether to also display.

    Returns the resolved output directory. Safe to call repeatedly: an argument not supplied
    is left unchanged, and ``out_dir`` resolves to the default on first use. Pass
    ``show=None`` to restore backend-sniffing, ``show=True``/``False`` to force it.
    """
    global _OUT_DIR, _FORMATS, _SHOW
    if out_dir is not _UNSET and out_dir is not None:
        _OUT_DIR = _Path(out_dir)
    elif _OUT_DIR is None:
        _OUT_DIR = _default_out_dir()
    if formats is not _UNSET and formats is not None:
        _FORMATS = tuple(formats)
    if show is not _UNSET:
        _SHOW = show
    return _OUT_DIR


def output_dir() -> _Path:
    """The configured output directory, resolving the default on first use."""
    return configure_output()


def written() -> list[_Path]:
    """Paths written so far this session, in order."""
    return list(_WRITTEN)


def _can_show() -> bool:
    """True when ``plt.show()`` will actually display something.

    An inline notebook backend displays; Agg does not. Sniffed rather than assumed so the
    same call works in a notebook, a terminal and a runner without a flag.
    """
    if _SHOW is not None:
        return _SHOW
    import matplotlib
    backend = matplotlib.get_backend().lower()
    if backend in ("agg", "pdf", "ps", "svg", "template"):
        return False
    if backend.startswith("module://"):          # ipympl, inline, widget
        return True
    # An interactive GUI backend with no display is the remaining trap.
    return bool(_os.environ.get("DISPLAY")) or _os.name == "nt"


def _slug(name: str) -> str:
    keep = [c if (c.isalnum() or c in "-_.") else "_" for c in str(name)]
    out = "".join(keep).strip("_")
    while "__" in out:
        out = out.replace("__", "_")
    return out or "figure"


def emit(name: str, fig=None, *, formats=None, close: bool = True) -> list[_Path]:
    """Save the current figure under ``name``, then close it. Show too where that works.

    ``name`` must be stable for a given plot: re-running overwrites rather than
    accumulating. Returns the paths written.
    """
    import matplotlib.pyplot as plt

    if fig is None:
        fig = plt.gcf()
    out = output_dir()
    out.mkdir(parents=True, exist_ok=True)
    exts = tuple(formats) if formats is not None else _FORMATS

    paths = []
    stem = _slug(name)
    for ext in exts:
        p = out / f"{stem}.{ext}"
        fig.savefig(p)
        paths.append(p)
        _WRITTEN.append(p)

    if _can_show():
        plt.show()
    if close:
        # Without this a multi-figure pass accumulates every figure it ever made, and
        # matplotlib starts warning about it around 20.
        plt.close(fig)
    return paths


_STYLE_APPLIED = False


def ensure_style(force: bool = False) -> bool:
    """Apply :func:`set_pub_style` once per process. Returns True if it applied it.

    Called at the top of every plotting entry point, which is the only place it can work:
    rcParams are read when a figure and its artists are *created*, so applying the style
    inside :func:`emit` would be too late for the figure being saved and would silently
    style only the next one.

    Deliberately not applied at import. A library that rewrites global matplotlib state
    just because it was imported is a library that breaks somebody else's figure.
    """
    global _STYLE_APPLIED
    if _STYLE_APPLIED and not force:
        return False
    set_pub_style()
    _STYLE_APPLIED = True
    return True
