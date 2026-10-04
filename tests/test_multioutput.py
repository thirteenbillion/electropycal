"""PLS2 dose-curve-vector model."""

import numpy as np
import pandas as pd

from electropycal.data.synthetic import make_dataset
from electropycal.evaluation.multioutput import dose_curve_matrix, pls2_forward_chained


def _frame():
    ds = make_dataset(random_state=0)
    df = pd.DataFrame(ds.X, columns=ds.feature_names)
    df["device"] = "d"; df["channel"] = ds.channel; df["timepoint"] = ds.timepoint
    df["concentration"] = ds.concentration; df["NormIpeak"] = ds.y
    return df


def test_dose_curve_matrix_shapes():
    df = _frame()
    meta, X, Y, feats, concs = dose_curve_matrix(df)
    assert len(meta) == X.shape[0] == Y.shape[0]                 # one row per (device,channel,timepoint)
    assert Y.shape[1] == len(concs)                             # one target column per concentration
    assert X.shape[1] == len(feats)


def test_pls2_forward_chained_runs_and_scores():
    df = _frame()
    r = pls2_forward_chained(df, k_grid=(2, 3), min_train_times=2)
    assert r["n_folds"] > 0 and r["n_test_curves"] > 0
    assert np.isfinite(r["pooled_q2"])
    assert set(r["per_dose_q2"]) == set(r["concs"])             # a Q² per concentration
