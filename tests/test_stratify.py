"""Stratify-by-drift diagnostic (study experiment E8)."""

import numpy as np
import pandas as pd

from electropycal.evaluation.stratify import drift_magnitude, clean_trend_channels, stratify_by_drift


def test_drift_magnitude_is_cumulative_fraction_from_baseline():
    df = pd.DataFrame([
        dict(device="d", channel=1, timepoint=0.0, sensitivity=1.0),
        dict(device="d", channel=1, timepoint=1.0, sensitivity=0.8),   # 20%
        dict(device="d", channel=1, timepoint=2.0, sensitivity=0.5),   # 50%
    ])
    m = drift_magnitude(df, "sensitivity")
    assert np.allclose(m, [0.0, 0.2, 0.5])


def test_clean_trend_channels_picks_monotone_only():
    rows = []
    for tp in range(6):                                   # monotone decay -> kept
        rows.append(dict(device="d", channel=1, timepoint=float(tp), sensitivity=1.0 - 0.1 * tp))
    rng = np.random.default_rng(0)
    for tp in range(6):                                   # pure noise -> dropped
        rows.append(dict(device="d", channel=2, timepoint=float(tp), sensitivity=rng.normal(0.5, 0.2)))
    keep = clean_trend_channels(pd.DataFrame(rows), "sensitivity", min_abs_spearman=0.8)
    assert ("d", 1) in keep and ("d", 2) not in keep


def _panel(seed: int = 0) -> pd.DataFrame:
    """Half the channels have evolved (state predicts a real drifted target); half are flat plateau
    noise. State→drift signal should therefore live in the has_drifted stratum, not the still_flat one."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(12):
        evolving = (i % 2 == 0)
        for tp in range(6):
            state = rng.normal(0, 1.0)
            if evolving:
                y = 0.05 - 0.02 * tp + 0.02 * state       # drifts with time, coupled to state
            else:
                y = 0.05 + rng.normal(0, 0.002)           # flat plateau, ~no variance
            rows.append(dict(device=f"d{i}", channel=i, timepoint=float(tp), sensitivity=y,
                             R_s_f00=1000 + 50 * state, C_s_f00=1e-6, mean_Ibg=1e-7))
    return pd.DataFrame(rows)


def test_stratify_separates_evolved_from_flat():
    out = stratify_by_drift(_panel(), "sensitivity", drift_threshold=0.2).set_index("stratum")
    assert {"all", "has_drifted", "still_flat"} <= set(out.index)
    assert out.loc["has_drifted", "n"] > 0
    # the flat stratum has almost no target variance (nothing to predict there)
    assert out.loc["still_flat", "target_std"] < out.loc["has_drifted", "target_std"]


def test_stratify_graceful_on_empty():
    out = stratify_by_drift(pd.DataFrame({"timepoint": [], "sensitivity": []}), "sensitivity")
    assert list(out.columns) == ["stratum", "n", "q2", "rmsep", "target_std"] and len(out) == 0
