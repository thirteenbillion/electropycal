"""Deployment pipeline tests (freeze → load → recalibrate + domain shift)."""

import numpy as np

from electropycal.data.synthetic import make_dataset, make_invivo_drift
from electropycal.deployment.deploy import freeze_model, load_frozen_model, recalibrate
from electropycal.selection.cars import cars_select


def _frozen(tmp_path, seed=0):
    ds = make_dataset(random_state=seed)
    subset = cars_select(ds.X, ds.y, k_max=3, n_generations=12, random_state=0)
    bundle = freeze_model(ds.X, ds.y, ds.feature_names, "linear_plsr", k=3,
                          out_dir=tmp_path / "frozen_model", feature_index=subset)
    return ds, bundle


def test_freeze_writes_portable_bundle(tmp_path):
    _, bundle = _frozen(tmp_path)
    for f in ("model_arrays.npz", "scaler.npz", "scaler_center.npy", "scaler_scale.npy",
              "reference_features.npy", "feature_names.json", "manifest.json"):
        assert (bundle / f).exists()


def test_recalibrate_recovers_in_distribution(tmp_path):
    ds, bundle = _frozen(tmp_path)
    fm = load_frozen_model(bundle)
    out = recalibrate(fm, ds.X)
    rmse = float(np.sqrt(((out["norm_ipeak"] - ds.y) ** 2).mean()))
    assert rmse < 0.5
    assert out["domain_distance"] < 1e-6  # in-vitro vs itself


def test_domain_distance_grows_with_drift(tmp_path):
    ds, bundle = _frozen(tmp_path)
    fm = load_frozen_model(bundle)
    drift = make_invivo_drift(ds, timepoints=(0, 7, 21, 42))
    dists = [recalibrate(fm, drift[t])["domain_distance"] for t in (0, 7, 21, 42)]
    assert dists == sorted(dists)      # monotonically increasing
    assert dists[-1] > dists[0]


def test_recalibrate_accepts_bundle_path(tmp_path):
    ds, bundle = _frozen(tmp_path)
    out = recalibrate(str(bundle), ds.X)   # load from path, not object
    assert out["norm_ipeak"].shape[0] == ds.X.shape[0]


def test_recalibrate_invivo_end_to_end(tmp_path):
    import numpy as np

    from electropycal.data.synthetic import write_synthetic_invivo_dir
    from electropycal.features.extract import extract_invivo
    from electropycal.features.normalize import d0_normalize_frame
    from electropycal.deployment.deploy import freeze_model, recalibrate_invivo

    # train a model on a drifting in-vivo-shaped featureset (matching schema)
    tr = extract_invivo(write_synthetic_invivo_dir(tmp_path / "train", devices=("3-2",),
                                                   channels=(3, 5, 6), sessions=(0, 7, 21, 42)))
    feats = [c for c in tr.columns if c not in
             ("device", "channel", "timepoint", "time_index", "NormIpeak")]
    trn = d0_normalize_frame(tr, feats)
    X = trn[feats].to_numpy(float)
    y = trn["NormIpeak"].to_numpy(float)
    keep = np.isfinite(X).all(0) & (np.nanstd(X, 0) > 1e-9)
    feats = [f for f, k in zip(feats, keep) if k]
    X = X[:, keep]
    rows = np.isfinite(X).all(1) & np.isfinite(y)
    bundle = freeze_model(X[rows], y[rows], feats, "linear_plsr", k=2, out_dir=tmp_path / "fm")

    res = recalibrate_invivo(bundle, write_synthetic_invivo_dir(
        tmp_path / "deploy", devices=("4-2",), channels=(3, 5, 6), sessions=(0, 7)),
        flag_distance=2.0)
    assert list(res["timepoint"]) == [0.0, 7.0]
    assert {"mean_norm_ipeak", "domain_distance", "confidence"} <= set(res.columns)
    assert np.isfinite(res["domain_distance"]).all()


def test_freeze_top_from_discovery_run(tmp_path):
    from electropycal.discovery.config import RunData, FAST, baseline_queue
    from electropycal.discovery.scheduler import run_discovery
    from electropycal.deployment.deploy import freeze_top, load_frozen_model, recalibrate

    ds = make_dataset(random_state=0)
    data = RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)
    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=tmp_path, profile=FAST)

    bundle = freeze_top(run_dir, data, out_dir=tmp_path / "frozen_model")
    assert (bundle / "full_feature_names.json").exists()
    # the bundle records which seed set selected it, so a model chosen under a multi-seed
    # sweep can be traced back to that run
    import json
    prov = json.loads((bundle / "manifest.json").read_text())["provenance"]
    assert prov["seeds"] == [0] and prov["fold_seed"] == 0
    assert prov["condition"] and prov["stochastic_selector"] in (True, False)
    fm = load_frozen_model(bundle)
    out = recalibrate(fm, ds.X)
    assert np.isfinite(out["norm_ipeak"]).all()
    # a named selector condition freezes a consensus subset (fewer than all features)
    b2 = freeze_top(run_dir, data, condition="pNproblem_3.2_CARS", out_dir=tmp_path / "fm_cars")
    assert load_frozen_model(b2).feature_index.size <= ds.X.shape[1]
