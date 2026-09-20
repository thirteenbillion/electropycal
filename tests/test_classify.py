"""Drift-beyond-trust classifier (study experiment E5): binary usable/degraded from electrode state."""

import numpy as np
import pandas as pd

from electropycal.evaluation.classify import degraded_labels, drift_classifier_cv


def _panel(signal: bool, seed: int = 0) -> pd.DataFrame:
    """12 sensors x 6 timepoints. Half drift hard, half stay stable -> class balance at late
    timepoints. When ``signal`` the electrode-state predictors track the drift; otherwise they are
    pure noise (the null control)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(12):
        rate = 0.6 if (i % 2 == 0) else 0.03      # alternating degrade / stable
        base_sens = 0.05 + 0.005 * i
        d0 = rng.normal(0, 0.05)
        for tp in range(6):
            state = d0 + rate * tp + rng.normal(0, 0.05)
            sens = base_sens * np.exp(-0.9 * max(0.0, state))
            if signal:
                r_s, c_s, ibg = 1000 + 50 * state, 1e-6 * (1 + 0.1 * state), 1e-7 * (1 + 0.2 * state)
            else:                                  # predictors decoupled from degradation
                r_s, c_s, ibg = rng.normal(1000, 50), rng.normal(1e-6, 1e-7), rng.normal(1e-7, 2e-8)
            rows.append(dict(device=f"d{i}", channel=1, timepoint=float(tp), sensitivity=sens,
                             R_s_f00=r_s, C_s_f00=c_s, mean_Ibg=ibg))
    return pd.DataFrame(rows)


def test_degraded_labels_flag_drift_from_baseline():
    df = pd.DataFrame([
        dict(device="d", channel=1, timepoint=0.0, sensitivity=1.0),   # baseline
        dict(device="d", channel=1, timepoint=1.0, sensitivity=0.9),   # -10% -> not degraded
        dict(device="d", channel=1, timepoint=2.0, sensitivity=0.4),   # -60% -> degraded
        dict(device="d", channel=1, timepoint=3.0, sensitivity=1.8),   # +80% -> degraded (either way)
    ])
    lab = degraded_labels(df, "sensitivity", frac=0.5)
    assert lab.tolist() == [0.0, 0.0, 1.0, 1.0]


def test_degraded_labels_nan_when_baseline_nonpositive():
    df = pd.DataFrame([
        dict(device="d", channel=1, timepoint=0.0, sensitivity=0.0),   # zero baseline -> undefined
        dict(device="d", channel=1, timepoint=1.0, sensitivity=0.5),
    ])
    assert degraded_labels(df, "sensitivity", frac=0.5).isna().all()


def test_drift_classifier_recovers_signal_and_is_chance_on_null():
    sig = drift_classifier_cv(_panel(signal=True), "sensitivity", frac=0.3)
    assert sig["n_folds"] >= 2 and sig["n_test"] > 0
    assert 0.4 < sig["base_rate"] < 0.6                # balanced problem, honest AUC
    assert sig["roc_auc"] > 0.8                        # state predicts degradation

    null = drift_classifier_cv(_panel(signal=False), "sensitivity", frac=0.3)
    assert abs(null["roc_auc"] - 0.5) < 0.2            # decoupled predictors -> chance


def test_drift_classifier_handles_single_class_gracefully():
    # every sensor identical & stable -> no positives -> undefined AUC, not a crash
    rows = [dict(device=f"d{i}", channel=1, timepoint=float(tp), sensitivity=0.05,
                 R_s_f00=1000.0, C_s_f00=1e-6, mean_Ibg=1e-7)
            for i in range(6) for tp in range(6)]
    res = drift_classifier_cv(pd.DataFrame(rows), "sensitivity", frac=0.3)
    assert np.isnan(res["roc_auc"]) and res["n_test"] >= 0
