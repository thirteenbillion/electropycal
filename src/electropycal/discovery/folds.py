"""Per-fold model bundles of one condition: one archive and one index, not files per fold.

A condition writes its outer folds' fitted models into two files in its own directory:

    fold_models.npz   every fold's arrays, keyed ``<fold>__<array>``, e.g. ``ch3_t28__coef``
    folds.json        ``{"layout": 2, "folds": {<fold>: {"manifest", "hyperparams", "metrics"}}}``

A fold is named ``ch<channel>_t<t_test>``. ``np.load`` reads an archive member only when it
is asked for, so loading one fold from a condition of several hundred reads that fold alone.

Run directories written before 0.11.0 hold the same content as one directory per fold,
``folds/<fold>/{model_arrays.npz, manifest.json, hyperparams.json, metrics.json}``. Every
reader here accepts both layouts, so an older run directory needs no conversion.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from ..data.io import load_npz, read_json, write_json

FOLD_MODELS = "fold_models.npz"
FOLD_INDEX = "folds.json"
_SEP = "__"
_LAYOUT = 2


def write_fold_bundles(cond_dir: str | Path, payloads: list[dict]) -> None:
    """Write the fold bundles of one condition. ``payloads`` are the runner's fold payloads,
    each carrying ``fold_name``, ``arrays``, ``manifest``, ``hyperparams`` and ``metrics``;
    ``None`` entries (unusable folds) are skipped. Writes nothing when no fold is usable."""
    usable = [p for p in payloads if p is not None]
    if not usable:
        return
    cond_dir = Path(cond_dir)
    cond_dir.mkdir(parents=True, exist_ok=True)
    arrays: dict[str, np.ndarray] = {}
    index: dict[str, dict] = {}
    for p in usable:
        name = p["fold_name"]
        for key, value in p["arrays"].items():
            arrays[f"{name}{_SEP}{key}"] = value
        index[name] = {"manifest": p["manifest"], "hyperparams": p["hyperparams"],
                       "metrics": p["metrics"]}
    # np.savez, not savez_compressed: the arrays are float64 measurement data that compress
    # to ~87% at best, and an uncompressed member is read without inflating it.
    np.savez(cond_dir / FOLD_MODELS, **arrays)
    write_json(cond_dir / FOLD_INDEX, {"layout": _LAYOUT, "folds": index})


def _legacy_dirs(cond_dir: Path) -> list[Path]:
    root = cond_dir / "folds"
    return sorted(p for p in root.iterdir() if p.is_dir()) if root.is_dir() else []


def fold_names(cond_dir: str | Path) -> list[str]:
    """The condition's usable folds, sorted by name. Empty if it produced none."""
    cond_dir = Path(cond_dir)
    if (cond_dir / FOLD_INDEX).exists():
        return sorted(read_json(cond_dir / FOLD_INDEX)["folds"])
    return [p.name for p in _legacy_dirs(cond_dir)]


def fold_records(cond_dir: str | Path) -> dict[str, dict]:
    """``{fold: {"manifest", "hyperparams", "metrics"}}`` for every usable fold, sorted by
    fold name, without loading any arrays."""
    cond_dir = Path(cond_dir)
    if (cond_dir / FOLD_INDEX).exists():
        folds = read_json(cond_dir / FOLD_INDEX)["folds"]
        return {k: folds[k] for k in sorted(folds)}
    out = {}
    for d in _legacy_dirs(cond_dir):
        out[d.name] = {name: read_json(d / f"{name}.json")
                       for name in ("manifest", "hyperparams", "metrics")
                       if (d / f"{name}.json").exists()}
    return out


def load_fold_bundle(cond_dir: str | Path, fold: str) -> tuple[dict[str, np.ndarray], dict]:
    """``(arrays, manifest)`` for one fold, the same pair ``models.base.load_model_bundle``
    returns, so ``predict_from_bundle(arrays, manifest, X)`` applies it. Reads only that
    fold's members of the archive."""
    cond_dir = Path(cond_dir)
    if (cond_dir / FOLD_INDEX).exists():
        record = read_json(cond_dir / FOLD_INDEX)["folds"].get(fold)
        if record is None:
            raise KeyError(f"no fold {fold!r} in {cond_dir}; have {fold_names(cond_dir)}")
        prefix = f"{fold}{_SEP}"
        with np.load(cond_dir / FOLD_MODELS, allow_pickle=False) as z:
            arrays = {k[len(prefix):]: z[k] for k in z.files if k.startswith(prefix)}
        return arrays, record["manifest"]
    legacy = cond_dir / "folds" / fold
    if not legacy.is_dir():
        raise KeyError(f"no fold {fold!r} in {cond_dir}; have {fold_names(cond_dir)}")
    return load_npz(legacy / "model_arrays.npz"), read_json(legacy / "manifest.json")
