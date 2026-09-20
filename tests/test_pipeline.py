"""End-to-end smoke tests on the synthetic dataset."""

import numpy as np

from electropycal.data.synthetic import make_dataset, make_invivo_drift
from electropycal.discovery.baseline import run_track2_linear
from electropycal.diagnostics import variance
from electropycal.deployment import domain


def test_synthetic_dataset_shape():
    ds = make_dataset(random_state=0)
    assert ds.X.shape[1] == 157
    assert ds.X.shape[0] == ds.y.shape[0] == ds.channel.shape[0]
    assert 60 < ds.X.shape[0] < 160  # data gaps applied


def test_track2_baseline_runs_and_learns():
    ds = make_dataset(random_state=0)
    folds, agg = run_track2_linear(ds.X, ds.y, ds.channel, ds.timepoint, k=3)
    assert folds, "expected at least one evaluable fold"
    assert np.isfinite(agg["pooled_rmsep"])
    # informative signal exists → Q2 should beat predicting the mean
    assert agg["pooled_q2"] > 0.0
    assert "rmsep_ci_lo" in agg and agg["rmsep_ci_lo"] <= agg["rmsep_ci_hi"]


def test_reliability_high_for_informative_features():
    ds = make_dataset(random_state=0, meas_noise=0.01)
    # reliability of one informative feature across replicate spread
    j = int(ds.informative[0])
    reps = ds.replicates[:, j, :]                       # (N, R)
    vals = reps.ravel()
    obs = np.repeat(np.arange(reps.shape[0]), reps.shape[1])
    r = variance.measurement_reliability(vals, obs)["reliability"]
    assert 0.0 <= r <= 1.0 and r > 0.8


def test_invivo_drift_increases_coral_distance():
    ds = make_dataset(random_state=0)
    drifted = make_invivo_drift(ds, timepoints=(0, 7, 21, 42))
    path = domain.drift_path(ds.X, drifted)
    dists = [p["distance"] for p in path]
    assert dists[0] < dists[-1]  # drift grows with time
