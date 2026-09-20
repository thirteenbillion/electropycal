"""Track-level aggregation of fold results.

Track 1 (channel-specific) reports per-channel pooled metrics; Tracks 2/3
(global / universal) pool across all held-out folds. All use the sample-weighted
pooled RMSEP/Q² from ``evaluation.metrics``.
"""

from __future__ import annotations

from collections import defaultdict

from .metrics import FoldResult, aggregate, pooled_rmsep, pooled_q2


def aggregate_by_track(folds: list[FoldResult], track: str, n_boot: int = 2000) -> dict:
    """Aggregate fold results per the CV track's objective."""
    if not folds:
        return {"pooled_rmsep": float("nan"), "n_folds": 0}
    out = aggregate(folds, with_ci=True)
    out["track"] = track
    if track == "channel":
        by_ch: dict[int, list[FoldResult]] = defaultdict(list)
        for f in folds:
            by_ch[f.channel].append(f)
        out["per_channel"] = {
            int(ch): {"pooled_rmsep": pooled_rmsep(fs), "pooled_q2": pooled_q2(fs),
                      "n_folds": len(fs)}
            for ch, fs in by_ch.items()
        }
    return out


def track_objective(folds: list[FoldResult]) -> float:
    """Scalar the inner loop minimizes: pooled RMSEP over the (track-scoped) folds."""
    return pooled_rmsep(folds)
