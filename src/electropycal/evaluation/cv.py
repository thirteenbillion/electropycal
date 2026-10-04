"""Forward-chained cross-validation splitters.

Temporal causality is enforced everywhere: a fold's training set only contains
timepoints strictly earlier than the held-out timepoint, so evaluation simulates
forward deployment and cannot leak future data. ``timepoint``
must be *numeric and comparable* (e.g. days: D0→0, D20→20).

- ``outer_folds(mode="loto_c_ac")`` (Track 1/2 outer split): test one channel at
  ``t_test``; train on all channels at timepoints ``< t_test`` (strict variant).
- ``outer_folds(mode="loco")`` (Track 3): the test channel is removed from
  training entirely; train on *other* channels at ``< t_test``.
- ``inner_split``: nested-CV inner split for hyperparameter/feature selection on
  the outer-training set only (``t_val`` = latest timepoint ``< t_test``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterator, Literal

import numpy as np


@dataclass(frozen=True)
class Fold:
    """One outer CV fold: index arrays into the observation table + fold identity."""

    channel: int
    t_test: float
    train_idx: np.ndarray
    test_idx: np.ndarray


def outer_folds(
    channel: np.ndarray,
    timepoint: np.ndarray,
    mode: Literal["loto_c_ac", "loco", "random"] = "loto_c_ac",
    min_train_times: int = 3,
    n_splits: int = 5,
    seed: int = 0,
) -> Iterator[Fold]:
    """Yield outer folds.

    ``loto_c_ac`` / ``loco`` are **forward-chained** (train on timepoints ``< t_test``): the
    deployment-faithful scheme (never trains on the future). ``min_train_times`` requires at least that
    many earlier timepoints before a ``t_test`` is evaluated, which also drops the
    underpowered earliest folds from the metric; raise it for a steady-state read.

    ``random`` is a **diagnostic only** (NOT deployment-valid): it holds out whole ``(channel,
    timepoint)`` units (the *same* test unit as forward-chaining) but assigns them to ``n_splits``
    folds **at random**, ignoring time order (so it may train on later timepoints to predict earlier
    ones). Comparing ``random`` Q² vs forward-chained Q² measures how much the temporal constraint (early
    underpowering + non-stationarity) costs: ``random ≫ forward`` ⇒ the map is learnable but the
    forward task is hard; ``both ≈ 0`` ⇒ the signal is genuinely absent.
    """
    channel = np.asarray(channel)
    timepoint = np.asarray(timepoint, dtype=float)
    if mode == "random":
        groups = np.char.add(np.char.add(channel.astype(str), "|"), timepoint.astype(str))
        uniq = np.unique(groups)
        k = int(min(n_splits, uniq.size))
        if k < 2:
            return
        assign = dict(zip(uniq, np.random.default_rng(seed).integers(0, k, size=uniq.size)))
        fold_of = np.array([assign[g] for g in groups])
        for kk in range(k):
            test_idx = np.where(fold_of == kk)[0]
            train_idx = np.where(fold_of != kk)[0]
            if test_idx.size == 0 or train_idx.size == 0:
                continue
            yield Fold(channel=-1, t_test=float(kk), train_idx=train_idx, test_idx=test_idx)
        return
    times = np.unique(timepoint)
    for c in np.unique(channel):
        for ti, t_test in enumerate(times):
            if ti < min_train_times:
                continue
            test_idx = np.where((channel == c) & (timepoint == t_test))[0]
            if test_idx.size == 0:
                continue
            if mode == "loto_c_ac":
                train_mask = timepoint < t_test
            elif mode == "loco":
                train_mask = (channel != c) & (timepoint < t_test)
            else:  # pragma: no cover
                raise ValueError(f"unknown mode {mode!r}")
            train_idx = np.where(train_mask)[0]
            if train_idx.size == 0:
                continue
            yield Fold(channel=int(c), t_test=float(t_test),
                       train_idx=train_idx, test_idx=test_idx)


def inner_split(
    channel: np.ndarray,
    timepoint: np.ndarray,
    fold: Fold,
    track: Literal["channel", "global", "random"] = "global",
) -> tuple[np.ndarray, np.ndarray] | None:
    """Nested-CV inner split on ``fold``'s training set only.

    Picks ``t_val`` = the latest timepoint ``< t_test`` present in the outer-train
    set. Inner train = outer-train rows at timepoints ``< t_val``. Inner
    validation = the held-out channel at ``t_val`` (``track="channel"``, Track 1)
    or all channels at ``t_val`` (``track="global"``, Track 2). Returns
    ``(inner_train_idx, inner_val_idx)`` as positions *within the full table*, or
    ``None`` if no valid ``t_val`` exists.

    ``track="random"`` (paired with the ``random`` outer mode) ignores time and splits the outer-train
    into a random 75/25 inner-train / inner-validation (seeded deterministically from the fold's test
    indices, so it is reproducible).
    """
    timepoint = np.asarray(timepoint, dtype=float)
    channel = np.asarray(channel)
    train = fold.train_idx
    if track == "random":
        tr = np.asarray(train)
        if tr.size < 4:
            return None
        rng = np.random.default_rng(int(np.asarray(fold.test_idx).sum()) & 0xFFFFFFFF)
        perm = tr[rng.permutation(tr.size)]
        cut = max(1, int(round(0.75 * tr.size)))
        inner_train, inner_val = perm[:cut], perm[cut:]
        if inner_train.size == 0 or inner_val.size == 0:
            return None
        return inner_train, inner_val
    train_times = np.unique(timepoint[train])
    val_times = train_times[train_times < fold.t_test]
    if val_times.size == 0:
        return None
    t_val = val_times.max()

    inner_train = train[timepoint[train] < t_val]
    if track == "channel":
        inner_val = train[(timepoint[train] == t_val) & (channel[train] == fold.channel)]
    else:
        inner_val = train[timepoint[train] == t_val]
    if inner_train.size == 0 or inner_val.size == 0:
        return None
    return inner_train, inner_val
