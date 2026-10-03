"""Per-fold bundles: one archive and one index per condition, and older run directories."""

import numpy as np
import pytest

from electropycal.data.io import load_npz, write_json
from electropycal.data.synthetic import make_dataset
from electropycal.discovery.config import FAST, Condition, RunData
from electropycal.discovery.folds import (
    fold_names,
    fold_records,
    load_fold_bundle,
    write_fold_bundles,
)
from electropycal.discovery.runner import run_condition
from electropycal.models.base import predict_from_bundle, save_model_bundle


@pytest.fixture(scope="module")
def run(tmp_path_factory):
    ds = make_dataset(random_state=0)
    data = RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)
    out = tmp_path_factory.mktemp("run")
    cond = Condition("lin", "linear_plsr", "global", None, k_grid=(2, 3))
    rows, _ = run_condition(cond, data, out, FAST, seed=0)
    return data, out / "conditions" / "lin", rows


def test_a_condition_writes_two_fold_files_whatever_its_fold_count(run):
    _, cond_dir, rows = run
    assert len(rows) > 3
    written = {p.name for p in cond_dir.iterdir()}
    assert {"fold_models.npz", "folds.json"} <= written
    assert "folds" not in written


def test_index_matches_the_summary_rows(run):
    _, cond_dir, rows = run
    want = sorted(f"ch{r['channel']}_t{int(r['t_test'])}" for r in rows)
    assert fold_names(cond_dir) == want
    for r in rows:
        rec = fold_records(cond_dir)[f"ch{r['channel']}_t{int(r['t_test'])}"]
        assert rec["hyperparams"]["k"] == r["k"]
        assert rec["metrics"]["rmsep"] == pytest.approx(r["rmsep"], rel=0, abs=0)


def test_a_reloaded_fold_reproduces_its_own_test_predictions(run):
    """The bundle is a deployable model, not a record: applying it to the fold's own test
    rows must give back the predictions the run scored, to floating point."""
    import pandas as pd
    data, cond_dir, rows = run
    preds = pd.read_parquet(cond_dir / "predictions.parquet")
    r = rows[0]
    arrays, manifest = load_fold_bundle(cond_dir, f"ch{r['channel']}_t{int(r['t_test'])}")
    te = (data.channel == r["channel"]) & (data.timepoint == r["t_test"])
    X = data.X[te]
    Xz = (X - arrays["zscore_mean"]) / arrays["zscore_std"]
    got = predict_from_bundle(arrays, manifest, Xz[:, arrays["feature_index"]])
    want = preds[(preds["sensor"] == r["channel"]) & (preds["t_test"] == r["t_test"])]["y_pred"]
    np.testing.assert_allclose(got, want.to_numpy(), rtol=1e-12, atol=1e-12)


def test_loading_one_fold_returns_only_that_folds_arrays(run):
    _, cond_dir, _ = run
    a, b = fold_names(cond_dir)[:2]
    arrays_a, _ = load_fold_bundle(cond_dir, a)
    arrays_b, _ = load_fold_bundle(cond_dir, b)
    assert set(arrays_a) == set(arrays_b)
    assert not any("__" in k for k in arrays_a)
    with pytest.raises(KeyError, match="no fold"):
        load_fold_bundle(cond_dir, "ch999_t0")


def test_a_pre_0_11_run_directory_reads_unchanged(tmp_path):
    """Run directories written before 0.11.0 hold one directory per fold. Every reader
    accepts them, so no conversion is needed."""
    cond_dir = tmp_path / "old"
    arrays = {"coef": np.arange(3.0), "x_mean": np.zeros(3), "y_mean": np.array(1.0)}
    for name, k in (("ch1_t3", 2), ("ch0_t3", 3)):
        d = cond_dir / "folds" / name
        save_model_bundle(d, arrays, {"architecture": "linear_plsr"})
        write_json(d / "hyperparams.json", {"k": k, "threshold": None, "feature_index": [0, 1, 2]})
        write_json(d / "metrics.json", {"rmsep": 0.1, "k": k})
    assert fold_names(cond_dir) == ["ch0_t3", "ch1_t3"]
    assert [r["hyperparams"]["k"] for r in fold_records(cond_dir).values()] == [3, 2]
    got, manifest = load_fold_bundle(cond_dir, "ch1_t3")
    assert manifest == {"architecture": "linear_plsr"}
    np.testing.assert_array_equal(got["coef"], load_npz(cond_dir / "folds/ch1_t3/model_arrays.npz")["coef"])


def test_no_usable_fold_writes_nothing(tmp_path):
    write_fold_bundles(tmp_path / "c", [None, None])
    assert not (tmp_path / "c").exists()
    assert fold_names(tmp_path / "c") == []
