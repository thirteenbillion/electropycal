"""Hierarchical / partial-pooling recalibration model (study experiment E6)."""

import numpy as np
import pandas as pd

from electropycal.evaluation.hierarchical import hierarchical_cv


def _panel(channel_structure: bool, seed: int = 0) -> pd.DataFrame:
    """10 channels x 6 timepoints across 4 devices. When ``channel_structure`` each channel carries a
    persistent random intercept the population map can't see (partial pooling should recover it);
    otherwise the target is fully explained by the shared state map (pooling has nothing to add)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(10):
        off = rng.normal(0, 0.03) if channel_structure else 0.0
        for tp in range(6):
            state = rng.normal(0, 1.0)
            y = 0.05 + 0.01 * state + off + rng.normal(0, 0.004)
            rows.append(dict(device=f"D{i // 3}", channel=i, timepoint=float(tp), sensitivity=y,
                             R_s_f00=1000 + 50 * state, C_s_f00=1e-6, mean_Ibg=1e-7))
    return pd.DataFrame(rows)


def test_partial_pooling_recovers_channel_random_effects():
    r = hierarchical_cv(_panel(channel_structure=True), "sensitivity", group_level="channel")
    assert r["n_folds"] >= 2 and r["n_test"] > 0
    assert r["q2"] > r["q2_fixed"] + 0.3          # pooling beats the population-only map by a lot
    assert r["q2"] > 0.5


def test_nested_grouping_is_at_least_as_good_as_channel_here():
    r_ch = hierarchical_cv(_panel(channel_structure=True), "sensitivity", group_level="channel")
    r_nd = hierarchical_cv(_panel(channel_structure=True), "sensitivity", group_level="channel_in_device")
    assert r_nd["q2"] >= r_ch["q2"] - 0.1         # nested captures the same channel effect (+device)


def test_pooling_does_not_help_when_no_group_structure():
    r = hierarchical_cv(_panel(channel_structure=False), "sensitivity", group_level="channel")
    # with no persistent per-channel effect, shrinkage keeps pooling close to the fixed map (no harm)
    assert abs(r["q2"] - r["q2_fixed"]) < 0.25


def test_reports_naive_rmsep_and_group_level_and_is_graceful_on_empty():
    r = hierarchical_cv(_panel(channel_structure=True), "sensitivity", group_level="device")
    assert np.isfinite(r["naive_rmsep"]) and r["group_level"] == "device"
    empty = hierarchical_cv(pd.DataFrame({"timepoint": [], "sensitivity": []}), "sensitivity")
    assert empty["n_folds"] == 0 and np.isnan(empty["q2"])


def test_unknown_group_level_raises():
    import pytest
    with pytest.raises(ValueError):
        hierarchical_cv(_panel(channel_structure=True), "sensitivity", group_level="bogus")
