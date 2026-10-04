"""Shared joblib helper for the library's parallel work units.

Two callers: the session-parallel raw walks (extraction, QC, reliability), one unit per
device-timepoint session, and discovery's fold loop, one unit per CV fold. Both are
embarrassingly parallel, so they parallelize cleanly. The one hazard for **non-notebook / direct-API** callers is
BLAS/OpenMP oversubscription: without the thread pin the notebooks set, each worker process would
spin up a full multi-threaded BLAS pool, so ``n_jobs`` workers × ``cores`` threads fight over the
cores and the "speedup" can be a slowdown. :func:`map_sessions` closes that hole by capping every
worker to a single inner thread (``inner_max_num_threads=1``) itself, so a plain
``extract_dataset(n_jobs=8)`` from a script is safe with no environment setup required.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any


def should_parallelize(n_jobs: int, n_items: int) -> bool:
    """True when a joblib pool is worth spawning: more than one worker requested (``n_jobs`` not
    0/1; negatives like -1 mean "all cores") and more than one item to spread over it. Below that,
    the worker spawn + pickling overhead outweighs the gain, so the caller runs serially."""
    return n_jobs not in (0, 1) and n_items > 1


def map_sessions(n_jobs: int, tasks: Iterable[Any]) -> list:
    """Run ``tasks`` (an iterable of ``joblib.delayed(...)`` calls) across worker processes, with
    **each worker capped to one BLAS/OpenMP thread** so parallel sessions never oversubscribe the
    cores, even when the caller has not pinned BLAS in its environment. joblib preserves input
    order, so the returned list is in ``tasks`` order (the raw walks rely on this for deterministic,
    serial-identical output). The caller decides *whether* to parallelize (see
    :func:`should_parallelize`); this only runs the pool once that decision is made.

    Named for its first caller; ``tasks`` are any independent work units: raw-walk sessions or
    discovery CV folds."""
    from joblib import Parallel, parallel_config
    # inner_max_num_threads=1: the library defends itself against oversubscription regardless of the
    # caller's environment (notebook pin, script with nothing set, CI). loky = process backend.
    with parallel_config(backend="loky", inner_max_num_threads=1):
        return list(Parallel(n_jobs=n_jobs)(tasks))
