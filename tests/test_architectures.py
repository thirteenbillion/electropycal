"""Orthogonal + nonlinear PLSR: serialization round-trip, curvature, pipeline."""

import numpy as np
import pandas as pd
import pytest

from electropycal.data.synthetic import make_dataset
from electropycal.deployment.deploy import freeze_model, load_frozen_model, recalibrate
from electropycal.discovery.config import Condition, FAST, RunData, baseline_queue
from electropycal.discovery.runner import run_condition
from electropycal.models import base, variants


@pytest.mark.parametrize("name,kw", [("orthogonal_plsr", {"n_orth": 1}),
                                     ("nonlinear_plsr", {"degree": 2})])
def test_bundle_round_trip(name, kw):
    rng = np.random.default_rng(0)
    X = rng.normal(size=(80, 12))
    y = np.abs(X[:, 0] - X[:, 1]) + 1.0
    m = variants.build(name, k=3, **kw).fit(X, y)
    arrays, manifest = m.to_arrays(), m.manifest()
    p_bundle = base.predict_from_bundle(arrays, manifest, X)
    assert np.allclose(m.predict(X), p_bundle, atol=1e-6)


def test_nonlinear_beats_linear_on_curved_target():
    rng = np.random.default_rng(1)
    X = rng.normal(size=(120, 10))
    lat = 1.5 * X[:, 0] - X[:, 1]
    y = 3.0 * lat + lat ** 2 + 0.05 * rng.normal(size=120)   # monotonic + curvature
    lin = variants.build("linear_plsr", k=3).fit(X, y)
    nl = variants.build("nonlinear_plsr", k=3, degree=2).fit(X, y)
    r_lin = np.sqrt(((lin.predict(X) - y) ** 2).mean())
    r_nl = np.sqrt(((nl.predict(X) - y) ** 2).mean())
    assert r_nl < 0.5 * r_lin


@pytest.mark.parametrize("arch", ["orthogonal_plsr", "nonlinear_plsr"])
def test_run_condition_for_new_architectures(arch, tmp_path):
    ds = make_dataset(random_state=0)
    data = RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)
    rows, agg = run_condition(Condition(arch, arch, "global", None, k_grid=(2, 3)),
                              data, tmp_path, FAST)
    assert rows and np.isfinite(agg["pooled_rmsep"])


def test_freeze_and_recalibrate_nonlinear(tmp_path):
    ds = make_dataset(random_state=0)
    bundle = freeze_model(ds.X, ds.y, ds.feature_names, "nonlinear_plsr", k=3,
                          out_dir=tmp_path / "frozen_model")
    fm = load_frozen_model(bundle)
    assert fm.manifest["architecture"] == "nonlinear_plsr"
    out = recalibrate(fm, ds.X, with_domain=False)
    assert out["norm_ipeak"].shape[0] == ds.X.shape[0] and np.all(np.isfinite(out["norm_ipeak"]))


def test_full_queue_covers_architectures_and_log_transform():
    q = baseline_queue()
    archs = {c.architecture for c in q}
    # implemented + extension-point architectures are present
    assert {"linear_plsr", "weighted_plsr", "orthogonal_plsr", "nonlinear_plsr"} <= archs
    # the log model is now a TRANSFORM (full log model: log(x/d0) features + log target),
    # applicable to any architecture, not a separate 'log_plsr' architecture
    assert {c.transform for c in q} >= {"linear", "log"}


def test_kernel_plsr_is_registered_extension_point():
    with pytest.raises(NotImplementedError):
        variants.build("kernel_plsr", k=2)


def test_plsr_robust_to_zero_variance_and_collinear_columns():
    """Zero-variance / rank-deficient feature blocks must not produce NaN (band-empty features and
    collinear adjacent-frequency EIS columns): drop constant columns, cap k at the numerical rank."""
    from electropycal.models.plsr import PLSRModel
    rng = np.random.default_rng(0)
    n = 40
    sig = rng.normal(size=(n, 2))
    X = np.column_stack([
        sig, sig @ rng.normal(size=(2, 3)),          # 3 collinear combinations (rank stays 2)
        np.zeros(n), np.full(n, 7.0),                # two zero-variance columns
    ])
    y = (sig[:, 0] * 1.5 - sig[:, 1]).reshape(-1, 1)
    m = PLSRModel(k=5).fit(X, y)                      # k=5 but rank is 2 -> must not raise/NaN
    pred = m.predict(X)
    assert np.isfinite(pred).all()
    arr = m.to_arrays()
    assert arr["coef"].shape[0] == X.shape[1]         # full-width coef (dropped columns padded with 0)
    assert np.isfinite(arr["coef"]).all()
    assert (arr["coef"][-2:] == 0).all()              # the two constant columns carry zero coefficient
    assert np.isfinite(m.vip()).all() and m.vip().shape[0] == X.shape[1]


def test_plsr_raises_on_fully_degenerate_fold():
    from electropycal.models.plsr import PLSRModel
    X = np.ones((10, 4))                              # no feature variance at all -> rank 0
    with pytest.raises(ValueError, match="degenerate"):
        PLSRModel(k=2).fit(X, np.arange(10.0).reshape(-1, 1))


def test_log_transform_rejects_signed_target():
    """A log condition on a signed target (e.g. a calibration-curve intercept/curvature) must raise,
    not silently clamp negatives to 1e-12 and corrupt the fit."""
    import tempfile
    from electropycal.discovery.config import Condition, FAST, RunData
    from electropycal.discovery.runner import run_condition
    ds = make_dataset(random_state=0)
    F = pd.DataFrame(ds.X, columns=ds.feature_names)
    F["channel"] = ds.channel; F["timepoint"] = ds.timepoint
    F["concentration"] = ds.concentration; F["device"] = "d0"
    F["signed"] = ds.y - ds.y.mean()                       # centered -> has negatives
    data = RunData.from_frame(F, target="signed")
    with pytest.raises(ValueError, match="positive target"):
        run_condition(Condition("t", "linear_plsr", "global", None, k_grid=(2,), transform="log"),
                      data, tempfile.mkdtemp(), FAST)
