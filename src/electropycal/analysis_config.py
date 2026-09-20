"""Shared analysis configuration for the review/QC/discovery notebooks.

The three pre-made notebooks — ``raw_spectra_review``, ``quality_filtering_dashboard``,
and ``discovery_checkpointed`` — run as **separate** Colab kernels, so a config set in
one cannot be seen by the others. To keep the EIS ``band`` and the QC / extraction
parameters from drifting apart between them, this module persists a single
``electropycal_analysis_config.json`` **inside the data ``ROOT``**. Each notebook loads it
(falling back to defaults if absent) and may still override any field in its setup cell.

Workflow:
  1. In ``raw_spectra_review`` set the band + params and call ``save_analysis_config(ROOT, cfg)``.
  2. ``quality_filtering_dashboard`` and ``discovery_checkpointed`` call
     ``load_analysis_config(ROOT)`` and get exactly those values.

The library primitives (``extract_dataset``, ``channel_quality_report``, the CLI) still take
explicit parameters — this is a convenience layer for the notebooks, not a new gate.
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

CONFIG_FILENAME = "electropycal_analysis_config.json"
QC_STATS_FILENAME = "electropycal_qc_stats.json"


@dataclass
class AnalysisConfig:
    """Analysis **parameters** shared by the raw-spectra / QC-dashboard / discovery notebooks.

    This holds only processing/QC knobs that must agree across notebooks (band, peak method,
    acceptance, gates). It deliberately does **not** hold data *selection* (device types / ids /
    channels / timepoints): selection is per-notebook-run (raw_spectra_review is often run one
    device at a time, while discovery runs across many), so each notebook keeps its own selection
    and only the parameters are shared.
    """

    band: tuple[float, float] | str = (2.0, 2000.0)   # (lo, hi) Hz; "auto" for data-driven. Study
    #: default 2–2000 Hz: the f-dependent drift-alignment review (diagnostics §3.1) shows aligned
    #: drift concentrated ~2 Hz–2 kHz, while retention is flat below the ~10 kHz inductive-onset cliff,
    #: so 2 kHz captures the signal at full retention and 2 Hz reaches the low-f aligned band.
    peak_method: str = "direct"                   # "direct" | "chord"
    acceptance: str = "monotonic"                 # extract_dataset acceptance test
    mono_tol: float = 0.10                         # EIS.2 |Z|-monotonicity tolerance
    min_norm_snr: float = 3.0                      # per-dose reproducibility-SNR cutoff
    monotonic_r_min: float = 0.6                   # dose-monotonicity min log-conc correlation
    mono_method: str = "pearson"                   # dose-response corr: "pearson" | "spearman" (rank)
    max_reps: int | None = 3                       # first N FSCV replicate cycles
    gate_on: dict = field(default_factory=lambda: {
        "eis": True, "monotonic": True, "snr_all": False, "peak_in_window": False})


def config_path(root: str | Path) -> Path:
    """Where the shared config lives for a given data ``root``."""
    return Path(root) / CONFIG_FILENAME


def load_analysis_config(root: str | Path, **overrides) -> AnalysisConfig:
    """Load the shared config from ``<root>/electropycal_analysis_config.json``.

    Returns defaults when the file is absent. Any keyword in ``overrides`` that is not
    ``None`` replaces the loaded value (so a notebook cell can override a field inline).
    Unknown keys in the file are ignored, so old configs keep loading as fields are added.
    """
    data: dict = {}
    p = config_path(root)
    if p.exists():
        data = json.loads(p.read_text())
    known = {f for f in AnalysisConfig().__dataclass_fields__}
    cfg = AnalysisConfig(**{k: v for k, v in data.items() if k in known})
    if isinstance(cfg.band, list):                # JSON has no tuples
        cfg.band = tuple(cfg.band)
    for k, v in overrides.items():
        if v is not None and k in known:
            setattr(cfg, k, v)
    return cfg


def save_analysis_config(root: str | Path, cfg: AnalysisConfig) -> Path:
    """Write ``cfg`` to ``<root>/electropycal_analysis_config.json`` and return the path."""
    p = config_path(root)
    p.write_text(json.dumps(asdict(cfg), indent=2, default=list))
    return p


def qc_stats_path(root: str | Path) -> Path:
    """Where the QC-stats companion artifact lives for a given data ``root``."""
    return Path(root) / QC_STATS_FILENAME


def save_qc_stats(root: str | Path, stats: dict) -> Path:
    """Write the QC-statistics companion to ``<root>/electropycal_qc_stats.json``.

    This is the *evidence* alongside the config (``save_analysis_config``): the config records
    **what** QC parameters were used, this records **what they did** on the data (yields, per-gate
    dropout, the stringency-sweep numbers) so a finalized run documents both the knobs and their
    effect. ``stats`` should embed the config that produced it (e.g. under a ``"config"`` key).
    Non-JSON scalars (numpy types) are coerced via ``default=str``.
    """
    p = qc_stats_path(root)
    p.write_text(json.dumps(stats, indent=2, default=str))
    return p


def load_qc_stats(root: str | Path) -> dict | None:
    """Load the QC-stats companion, or ``None`` if it has not been written yet."""
    p = qc_stats_path(root)
    return json.loads(p.read_text()) if p.exists() else None


def format_provenance(root: str | Path, target: str | None = None,
                      cfg: AnalysisConfig | None = None) -> str:
    """One human-readable block: the pre-processing **parameters**, the **decisions**, and (if the
    QC-stats companion exists) their **impact** on the dataset. Printed at the start of every discovery
    run — whatever the entry point (notebook / script / CLI) — so a run always states what
    pre-processing produced the featureset it trained on, and why."""
    cfg = cfg or load_analysis_config(root)
    stats = load_qc_stats(root)
    b = cfg.band if isinstance(cfg.band, str) else f"({cfg.band[0]:g}, {cfg.band[1]:g}) Hz"
    L = ["=" * 74,
         "ElectroPyCal — run provenance  (pre-processing parameters · decisions · impact)",
         "=" * 74,
         "config (electropycal_analysis_config.json):",
         f"  band = {b}   peak_method = {cfg.peak_method}   acceptance = {cfg.acceptance}",
         f"  EIS.2 mono_tol = {cfg.mono_tol}   FSCV.1 monotonic_r_min = {cfg.monotonic_r_min} "
         f"({cfg.mono_method})   min_norm_snr = {cfg.min_norm_snr}   max_reps = {cfg.max_reps}",
         f"  gate_on = {cfg.gate_on}",
         "decisions / conventions:",
         f"  target = {target or 'NormIpeak'}",
         "  D0-normalization: ON (per-sensor drift-from-baseline; additive for phase/bounded types, "
         "multiplicative for magnitudes)",
         "  leakage-safe predictors: EIS + 0 nM-background FSCV only; faradaic peak features are "
         "RESERVED targets",
         "  NormIpeak < 0 dose rows dropped (drop_negative, non-physical)"]
    if stats:
        t = stats.get("totals", {})
        L.append("QC impact (electropycal_qc_stats.json):")
        L.append(f"  paired = {t.get('channel_timepoints_paired')}   retained = "
                 f"{t.get('overall_valid')} ({t.get('overall_valid_pct')}%)")
        for g in (stats.get("gate_impact") or []):
            L.append(f"    {g.get('gate')}: {g.get('n_channeltimepoints_failed')} failed "
                     f"({g.get('failed_%')}%)")
    else:
        L.append("QC impact: electropycal_qc_stats.json not found "
                 "(run quality_filtering_dashboard §7 to generate it)")
    L += ["see docs/DESIGN.md for the method rationale.", "=" * 74]
    return "\n".join(L)


def print_provenance(root: str | Path, target: str | None = None,
                     cfg: AnalysisConfig | None = None) -> None:
    """Print :func:`format_provenance` (safe: never raises — provenance is informational)."""
    try:
        print(format_provenance(root, target=target, cfg=cfg), flush=True)
    except Exception as e:                                     # never let provenance break a run
        print(f"(provenance unavailable: {e})", flush=True)


def resolve_band(root: str | Path, cfg: AnalysisConfig) -> tuple[float, float]:
    """Resolve ``cfg.band`` to a concrete ``(lo, hi)`` — ``"auto"`` → ``recommended_band(root)``."""
    if isinstance(cfg.band, str):
        if cfg.band != "auto":
            raise ValueError(f"band must be a (lo, hi) tuple or 'auto', got {cfg.band!r}")
        from .data.inventory import recommended_band
        return recommended_band(root)
    return (float(cfg.band[0]), float(cfg.band[1]))
