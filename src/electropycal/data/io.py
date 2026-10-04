"""Serialization helpers (DESIGN §8): npy/npz, Parquet, JSON (no pickle).

JSON writing casts NumPy scalars/arrays to native Python so metrics/config are
portable. Parquet is used for feature/metric tables (dtype- and precision-safe).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _jsonable(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, np.generic):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def write_json(path: str | Path, obj: dict) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_jsonable(obj), indent=2))
    return path


def read_json(path: str | Path) -> dict:
    return json.loads(Path(path).read_text())


def write_parquet(path: str | Path, df) -> Path:
    """Write a pandas DataFrame to Parquet (falls back to CSV if pyarrow absent)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        df.to_parquet(path, index=False)
    except Exception:  # pragma: no cover - pyarrow optional
        path = path.with_suffix(".csv")
        df.to_csv(path, index=False)
    return path


def read_parquet(path: str | Path):
    import pandas as pd
    path = Path(path)
    if path.suffix == ".csv" or not path.exists():
        return pd.read_csv(path.with_suffix(".csv"))
    return pd.read_parquet(path)


def save_npz(path: str | Path, **arrays: np.ndarray) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)
    return path


def load_npz(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(Path(path), allow_pickle=False) as z:
        return {k: z[k] for k in z.files}
