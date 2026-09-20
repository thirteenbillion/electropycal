"""Tests for the implemented foundation modules."""

import numpy as np
import pytest

from electropycal.evaluation import metrics
from electropycal.evaluation.cv import outer_folds, inner_split
from electropycal.deployment import domain
from electropycal.diagnostics import variance


def test_pooled_rmsep_matches_concatenated_rmse():
    # Two folds; pooled RMSEP must equal RMSE over concatenated residuals.
    f1 = metrics.FoldResult.from_predictions(0, 20, [1.0, 2.0], [1.5, 2.5], mean_train=0.0)
    f2 = metrics.FoldResult.from_predictions(1, 20, [3.0], [3.0], mean_train=0.0)
    resid = np.array([0.5, 0.5, 0.0])
    assert metrics.pooled_rmsep([f1, f2]) == pytest.approx(np.sqrt((resid**2).mean()))


def test_micro_vs_macro_differ_with_imbalance():
    big = metrics.FoldResult.from_predictions(0, 20, [0.0] * 10, [1.0] * 10, 0.0)   # rmsep 1
    small = metrics.FoldResult.from_predictions(1, 20, [0.0], [0.0], 0.0)           # rmsep 0
    micro = metrics.pooled_rmsep([big, small])   # sample-weighted → near 1
    macro = metrics.macro_rmsep([big, small])    # channel-weighted → 0.5
    assert micro > macro
    assert macro == pytest.approx(0.5)


def test_q2_zero_when_predicting_train_mean():
    f = metrics.FoldResult.from_predictions(0, 20, [1.0, 3.0], [2.0, 2.0], mean_train=2.0)
    assert metrics.pooled_q2([f]) == pytest.approx(0.0)


def test_outer_folds_are_temporally_causal():
    # 2 channels × 4 timepoints (days 0,1,7,20)
    ch = np.repeat([0, 1], 4)
    tp = np.tile([0, 1, 7, 20], 2)
    folds = list(outer_folds(ch, tp, mode="loto_c_ac", min_train_times=3))
    assert folds, "expected at least one fold"
    for f in folds:
        # no training row may be at or after the tested timepoint
        assert np.all(tp[f.train_idx] < f.t_test)
        assert np.all(tp[f.test_idx] == f.t_test)


def test_loco_excludes_test_channel_from_training():
    ch = np.repeat([0, 1, 2], 4)
    tp = np.tile([0, 1, 7, 20], 3)
    for f in outer_folds(ch, tp, mode="loco", min_train_times=3):
        assert f.channel not in set(ch[f.train_idx])


def test_inner_split_respects_forward_chaining():
    ch = np.repeat([0, 1], 4)
    tp = np.tile([0, 1, 7, 20], 2)
    fold = next(outer_folds(ch, tp, min_train_times=3))
    split = inner_split(ch, tp, fold, track="global")
    assert split is not None
    inner_train, inner_val = split
    t_val = tp[inner_val].max()
    assert np.all(tp[inner_train] < t_val)
    assert np.all(tp[inner_val] == t_val)


def test_coral_distance_zero_for_identical_distributions():
    rng = np.random.default_rng(0)
    x = rng.normal(size=(200, 5))
    d = domain.coral_distance(x, x.copy())
    assert d["distance"] == pytest.approx(0.0, abs=1e-6)


def test_coral_transform_reduces_distance():
    rng = np.random.default_rng(1)
    src = rng.normal(size=(300, 4))
    tgt = rng.normal(loc=2.0, scale=3.0, size=(300, 4))
    before = domain.coral_distance(src, tgt)["distance"]
    after = domain.coral_distance(domain.coral_transform(src, tgt), tgt)["distance"]
    assert after < before


def test_measurement_reliability_high_for_low_noise():
    # Strong between-observation signal, tiny replicate noise → reliability ~1.
    rng = np.random.default_rng(2)
    obs_means = np.repeat(np.arange(20) * 10.0, 3)      # 20 obs, big spread
    obs_ids = np.repeat(np.arange(20), 3)
    vals = obs_means + rng.normal(scale=0.01, size=obs_means.size)
    r = variance.measurement_reliability(vals, obs_ids)["reliability"]
    assert r > 0.99


def test_variance_hierarchy_sums_to_one():
    rng = np.random.default_rng(3)
    ch = np.repeat([0, 1, 2], 8)
    tp = np.tile(np.repeat([0, 7], 4), 3)
    conc = np.tile([100, 250, 500, 1000], 6)
    vals = ch * 5.0 + rng.normal(size=ch.size)
    parts = variance.variance_hierarchy(vals, ch, tp, conc)
    assert sum(parts.values()) == pytest.approx(1.0, abs=1e-9)


def test_drift_reliability_high_when_drift_exceeds_noise():
    # Two sensors, three timepoints, a per-sensor linear drift far above the replicate noise floor.
    sensor = np.repeat(["a", "b"], 9)
    timepoint = np.tile(np.repeat([0.0, 1.0, 2.0], 3), 2)
    conc = np.tile([100.0, 250.0, 500.0], 6)
    vals = np.tile(np.repeat([0.0, 10.0, 20.0], 3), 2)      # drift only (same in both sensors)
    out = variance.drift_reliability(vals, sensor, timepoint, conc, sigma2_meas=0.01, n_rep=3)
    assert out["drift_snr"] > 100 and out["R_drift"] > 0.9


def test_drift_reliability_low_when_noise_dominates():
    sensor = np.repeat(["a", "b"], 9)
    timepoint = np.tile(np.repeat([0.0, 1.0, 2.0], 3), 2)
    conc = np.tile([100.0, 250.0, 500.0], 6)
    tiny_drift = np.tile(np.repeat([0.0, 0.1, 0.2], 3), 2)
    out = variance.drift_reliability(tiny_drift, sensor, timepoint, conc, sigma2_meas=100.0, n_rep=3)
    assert out["drift_snr"] < 1 and out["R_drift"] == 0.0


def test_drift_reliability_constant_feature_is_nan():
    sensor = np.repeat(["a", "b"], 9)
    timepoint = np.tile(np.repeat([0.0, 1.0, 2.0], 3), 2)
    conc = np.tile([100.0, 250.0, 500.0], 6)
    out = variance.drift_reliability(np.full(18, 5.0), sensor, timepoint, conc, sigma2_meas=0.0, n_rep=3)
    assert np.isnan(out["drift_snr"]) and np.isnan(out["R_drift"])


def test_drift_alignment_detects_tracking_feature():
    # A feature whose per-sensor temporal trajectory matches the response's tracks perfectly (r≈1);
    # an unrelated feature does not.
    sensor = np.repeat(["a", "b", "c"], 6)
    timepoint = np.tile(np.repeat([0.0, 1.0, 2.0], 2), 3)
    conc = np.tile([100.0, 500.0], 9)
    resp = np.tile(np.repeat([1.0, 2.0, 3.0], 2), 3)        # rises with time in every sensor
    tracks = 2.0 * resp + 0.5                                # affine of the response ⇒ |r|=1
    rng = np.random.default_rng(0)
    noise_feat = rng.normal(size=resp.size)
    a = variance.drift_alignment(tracks, resp, sensor, timepoint, conc)
    b = variance.drift_alignment(noise_feat, resp, sensor, timepoint, conc)
    assert a["abs_alignment"] > 0.99 and a["n_cells"] > 0
    assert b["abs_alignment"] < a["abs_alignment"]


def test_drift_alignment_spearman_matches_monotonic_but_nonlinear():
    # feature = exp(response) per (sensor,dose) time series: perfectly MONOTONIC but nonlinear.
    # Spearman should read ~1 while Pearson is < 1.
    sensor = np.repeat(["a", "b", "c"], 6)
    timepoint = np.tile(np.repeat([0.0, 1.0, 2.0], 2), 3)
    conc = np.tile([100.0, 500.0], 9)
    resp = np.tile(np.repeat([1.0, 2.0, 3.0], 2), 3)
    feat = np.exp(resp)                                     # monotone increasing, convex
    ap = variance.drift_alignment(feat, resp, sensor, timepoint, conc, method="pearson")
    asp = variance.drift_alignment(feat, resp, sensor, timepoint, conc, method="spearman")
    assert asp["abs_alignment"] > 0.99                     # rank-perfect (clamped at 0.999)
    assert asp["abs_alignment"] >= ap["abs_alignment"]     # spearman >= pearson for a monotone map


def test_dose_response_corr_pearson_vs_spearman_on_saturating_curve():
    from electropycal.features.fscv import dose_response_corr
    conc = np.array([100.0, 300.0, 1000.0, 3000.0, 10000.0])
    lc = np.log10(conc)
    y = np.log1p(lc - lc.min())                            # monotone increasing, saturating (concave)
    rp = dose_response_corr(conc, y, method="pearson")
    rs = dose_response_corr(conc, y, method="spearman")
    assert rs > 0.999 and rs >= rp                         # rank-monotonic sees it perfectly
    assert np.isnan(dose_response_corr([100.0, 200.0], [1.0, 2.0]))   # <3 doses -> NaN
