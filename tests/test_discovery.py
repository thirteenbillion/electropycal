"""Discovery pipeline tests (fast profile; the full profile is marked slow)."""

import numpy as np
import pytest

from electropycal.data.synthetic import make_dataset
from electropycal.discovery.config import Condition, FAST, Profile, RunData, baseline_queue
from electropycal.discovery.folds import fold_records
from electropycal.discovery.runner import run_condition
from electropycal.discovery.scheduler import run_discovery


@pytest.fixture
def data():
    ds = make_dataset(random_state=0)
    return RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)


@pytest.mark.parametrize("cond", [
    Condition("lin", "linear_plsr", "global", None, k_grid=(2, 3)),
    Condition("log", "log_plsr", "global", None, k_grid=(2,)),
    Condition("wt", "weighted_plsr", "global", None, k_grid=(2,), weighted_by="concentration"),
    Condition("sr", "linear_plsr", "global", "sr", k_grid=(2,), threshold_grid=(0.5, 1.0)),
    Condition("cars", "linear_plsr", "global", "cars", k_grid=(2, 3)),
    Condition("uni", "linear_plsr", "universal", "cars", k_grid=(2,)),
])
def test_run_condition_produces_valid_outputs(cond, data, tmp_path):
    rows, agg = run_condition(cond, data, tmp_path, FAST, seed=0)
    assert rows and agg["n_folds"] > 0
    assert np.isfinite(agg["pooled_rmsep"])
    cond_dir = tmp_path / "conditions" / cond.name
    assert (cond_dir / "aggregated_metrics.json").exists()
    assert (cond_dir / "feature_stability.parquet").exists()
    # every usable fold serialized a portable bundle, into one archive + one index
    assert (cond_dir / "fold_models.npz").exists() and (cond_dir / "folds.json").exists()
    assert not (cond_dir / "folds").exists()
    records = fold_records(cond_dir)
    assert len(records) == len(rows)
    assert all({"manifest", "hyperparams", "metrics"} <= set(r) for r in records.values())


def test_random_track_diagnostic_runs(data, tmp_path):
    """The `random` track (diagnostic-only random-split CV) produces usable folds + finite metrics,
    so it can be compared against forward-chained `global` to measure the temporal-constraint cost."""
    rows, agg = run_condition(Condition("rnd", "linear_plsr", "random", None, k_grid=(2, 3)),
                              data, tmp_path, FAST, seed=0)
    assert rows and agg["n_folds"] > 0 and np.isfinite(agg["pooled_rmsep"])


def test_multiseed_averages_stochastic_but_collapses_deterministic(data, tmp_path):
    """profile.seeds with >1 seed repeats the nested-CV per seed for a STOCHASTIC selector
    (CARS), but a deterministic condition (no selector) collapses to a single seed; repeating
    it would be pure waste.

    Both record the seed set they *actually* ran over, so a collapsed condition says so
    explicitly rather than by omitting the field."""
    import dataclasses
    prof = dataclasses.replace(FAST, seeds=(0, 1, 2))
    cars = run_condition(Condition("cars_ms", "linear_plsr", "global", "cars", k_grid=(2, 3)),
                         data, tmp_path / "a", prof, seed=0)[1]
    assert cars["n_seeds"] == 3 and cars["seeds"] == [0, 1, 2]
    assert np.isfinite(cars["pooled_rmsep"])
    lin = run_condition(Condition("lin_ms", "linear_plsr", "global", None, k_grid=(2, 3)),
                        data, tmp_path / "b", prof, seed=0)[1]
    assert lin["n_seeds"] == 1 and lin["seeds"] == [0]        # deterministic -> single seed


def test_snr_weighted_condition_in_batch1_and_weights_applied(tmp_path):
    import numpy as np, pandas as pd
    from electropycal.discovery.config import baseline_queue
    from electropycal.discovery.runner import _weights
    # the condition ships in Batch 1
    snr_conds = [c for c in baseline_queue()
                 if c.batch == 1 and c.weighted_by == "repeatability_snr"]
    assert len(snr_conds) == 1
    c = snr_conds[0]
    # featureset with a varied repeatability_snr column (+ an inf = perfect reproducibility)
    ds = make_dataset(random_state=0)
    df = pd.DataFrame(ds.X, columns=ds.feature_names)
    df["channel"] = ds.channel; df["timepoint"] = ds.timepoint
    df["concentration"] = ds.concentration; df["NormIpeak"] = ds.y
    df["repeatability_snr"] = np.linspace(0.5, 8.0, len(df)); df.iloc[0, -1] = np.inf
    data = RunData.from_frame(df)
    assert data.repeatability_snr is not None
    w = _weights(c, data, np.arange(len(df)))
    assert w is not None and np.all(np.isfinite(w)) and np.all(w >= 0)   # inf capped, non-negative
    rows, agg = run_condition(c, data, tmp_path, FAST, seed=0)
    assert rows and np.isfinite(agg["pooled_rmsep"])
    # no SNR column -> gracefully unweighted (None), doesn't crash
    plain = RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)
    assert _weights(c, plain, np.arange(5)) is None


def test_run_condition_writes_predictions_with_device_channel(tmp_path):
    import pandas as pd
    ds = make_dataset(random_state=0)
    df = pd.DataFrame(ds.X, columns=ds.feature_names)
    df["channel"] = ds.channel; df["timepoint"] = ds.timepoint
    df["concentration"] = ds.concentration; df["NormIpeak"] = ds.y
    df["device"] = ["2-2" if c < 3 else "2-3" for c in ds.channel]     # two synthetic devices
    data = RunData.from_frame(df)
    assert data.sensor_labels is not None and ":" in str(data.sensor_labels[0])
    run_condition(Condition("lin", "linear_plsr", "global", None, k_grid=(2,)),
                  data, tmp_path, FAST, seed=0)
    p = pd.read_parquet(tmp_path / "conditions" / "lin" / "predictions.parquet")
    assert {"device", "channel", "concentration", "y_true", "y_pred", "residual"} <= set(p.columns)
    assert set(p.device.unique()) <= {"2-2", "2-3"}                    # split back from device:channel
    assert (p.residual == p.y_pred - p.y_true).all()


def test_effective_min_train_times_caps():
    from electropycal.discovery.config import effective_min_train_times
    assert effective_min_train_times(3, 3) == 2      # 3 timepoints can only train on 2
    assert effective_min_train_times(3, 5) == 3      # enough history -> unchanged
    assert effective_min_train_times(3, 1) == 3      # <2 timepoints: caller surfaces separately
    assert effective_min_train_times(1, 3) == 1      # already small -> unchanged


def test_run_condition_caps_min_train_times_and_warns(tmp_path):
    # 3 timepoints but FAST.min_train_times=3 (needs >=4): the shared run_condition path caps
    # to 2 with a warning so folds are produced, and can be opted out of.
    ds = make_dataset(random_state=0)
    keep = np.isin(ds.timepoint, [0.0, 1.0, 7.0])
    data = RunData(ds.X[keep], ds.y[keep], ds.channel[keep], ds.timepoint[keep],
                   ds.concentration[keep], ds.feature_names)
    cond = Condition("lin", "linear_plsr", "global", None, k_grid=(2,))
    with pytest.warns(UserWarning, match="capping to 2"):
        _, agg = run_condition(cond, data, tmp_path / "capped", FAST, seed=0)
    assert agg["n_folds"] > 0 and np.isfinite(agg["pooled_rmsep"])       # folds, not nan
    _, agg0 = run_condition(cond, data, tmp_path / "raw", FAST, seed=0, cap_min_train_times=False)
    assert agg0["n_folds"] == 0                                          # verbatim -> 0 folds


def test_run_condition_progress_logs_without_changing_results(data, tmp_path, capsys):
    c = Condition("lin", "linear_plsr", "global", None, k_grid=(2,))
    _, agg_q = run_condition(c, data, tmp_path / "quiet", FAST, seed=0)
    assert capsys.readouterr().out == ""                       # silent by default
    _, agg_v = run_condition(c, data, tmp_path / "verbose", FAST, seed=0, progress=True)
    out = capsys.readouterr().out
    assert "START" in out and "DONE" in out and "ETA" in out   # timestamped progress emitted
    assert agg_v["pooled_rmsep"] == agg_q["pooled_rmsep"]       # logging doesn't change results


def test_to_log_representation_logs_multiplicative_only():
    from electropycal.features.normalize import to_log_representation
    names = ["mean_Ibg", "mean_Vpeak", "R_s_f00"]              # multiplicative, additive, multiplicative
    X = np.array([[2.0, 0.5, 4.0], [1.0, 0.3, 1.0]])
    Y = to_log_representation(X, names)
    assert np.allclose(Y[:, 0], np.log(X[:, 0]))              # mean_Ibg logged
    assert np.allclose(Y[:, 1], X[:, 1])                      # mean_Vpeak (additive) unchanged
    assert np.allclose(Y[:, 2], np.log(X[:, 2]))              # R_s logged


def test_log_transform_runs_on_any_architecture_and_guards(data, tmp_path):
    # transform="log" (full log model) runs on a linear-output architecture and gives finite RMSEP
    _, agg = run_condition(Condition("lg", "linear_plsr", "global", None, k_grid=(2,), transform="log"),
                           data, tmp_path / "log", FAST, seed=0)
    assert agg["n_folds"] > 0 and np.isfinite(agg["pooled_rmsep"])
    # log twins ship in the queue (each implemented architecture can run its log version)
    assert any(c.transform == "log" for c in baseline_queue())
    # guard: transform="log" + log_plsr would double-transform -> rejected
    with pytest.raises(ValueError, match="double-transform|full log model"):
        run_condition(Condition("bad", "log_plsr", "global", None, k_grid=(2,), transform="log"),
                      data, tmp_path / "bad", FAST, seed=0)


def test_track1_reports_per_channel(data, tmp_path):
    _, agg = run_condition(Condition("c1", "linear_plsr", "channel", None, k_grid=(2,)),
                           data, tmp_path, FAST)
    assert "per_channel" in agg and len(agg["per_channel"]) >= 1


def test_full_queue_runs_and_ranks(data, tmp_path):
    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=tmp_path,
                            profile=FAST, seed=0)
    assert (run_dir / "summary.parquet").exists()
    assert (run_dir / "report" / "condition_ranking.parquet").exists()
    assert (run_dir / "report" / "discovery_summary.md").exists()
    import pandas as pd
    ranking = pd.read_parquet(run_dir / "report" / "condition_ranking.parquet")
    assert len(ranking) == len(baseline_queue())
    assert ranking["pooled_rmsep"].is_monotonic_increasing  # sorted best-first


def test_run_discovery_progress_logs_per_condition(data, tmp_path, capsys):
    conds = baseline_queue()[:2]
    run_discovery(data, conditions=conds, out_root=tmp_path / "quiet", profile=FAST, seed=0)
    assert capsys.readouterr().out == ""                       # silent by default
    run_discovery(data, conditions=conds, out_root=tmp_path / "verbose", profile=FAST,
                  seed=0, progress=True)
    out = capsys.readouterr().out
    assert "CONDITION 1/2" in out and "START" in out and "DONE" in out   # per-condition + per-fold


def test_decision_gate_can_stop_queue(data, tmp_path):
    calls = []

    def gate(cond, agg, history):
        calls.append(cond.name)
        return len(calls) < 2  # stop after the 2nd condition

    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=tmp_path,
                            profile=FAST, gate=gate)
    assert len(calls) == 2
    assert (run_dir / "logs" / "run.log").exists()


def test_feature_ranking_written(data, tmp_path):
    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=tmp_path, profile=FAST)
    import pandas as pd
    fr = pd.read_parquet(run_dir / "report" / "feature_ranking.parquet")
    assert {"feature", "mean_selection_frequency", "n_conditions_selected"} <= set(fr.columns)
    assert fr["mean_selection_frequency"].is_monotonic_decreasing


def test_batch_gate_stops_after_batch(data, tmp_path):
    seen = []

    def gate(batch_num, ranking):
        seen.append(batch_num)
        return False                      # stop after the first completed batch

    run_discovery(data, conditions=baseline_queue(), out_root=tmp_path,
                  profile=FAST, batch_gate=gate)
    assert seen == [1]                    # only batch 1 ran before stopping


def test_batch_gate_can_edit_remaining_queue(data, tmp_path):
    # a 3-arg batch_gate that drops every remaining condition after batch 1
    ran = []

    def gate(batch_num, ranking, remaining):
        ran.append(batch_num)
        return [] if batch_num == 1 else remaining      # edit: empty the rest after batch 1

    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=tmp_path,
                            profile=FAST, batch_gate=gate)
    import pandas as pd
    ranking = pd.read_parquet(run_dir / "report" / "condition_ranking.parquet")
    # only batch-1 conditions ran (the rest were edited out)
    assert set(ranking["batch"]) == {1}
    assert ran == [1]


def test_auto_flag_conditions_flags_dominated_architecture():
    from electropycal.discovery.config import Condition
    from electropycal.discovery.scheduler import auto_flag_conditions
    ranking = [
        {"condition": "a", "architecture": "linear_plsr", "pooled_rmsep": 0.10,
         "rmsep_ci_lo": 0.08, "rmsep_ci_hi": 0.12},
        {"condition": "b", "architecture": "nonlinear_plsr", "pooled_rmsep": 0.50,
         "rmsep_ci_lo": 0.45, "rmsep_ci_hi": 0.55},          # clearly dominated
    ]
    remaining = [Condition("c", "nonlinear_plsr", "global"),
                 Condition("d", "linear_plsr", "channel")]
    flags = auto_flag_conditions(ranking, remaining)
    assert "c" in flags and "d" not in flags               # only the dominated arch is flagged


@pytest.mark.slow
def test_full_profile_path(data, tmp_path):
    # exercises the default (non-fast) profile end-to-end on a single condition
    run_condition(Condition("lin", "linear_plsr", "global", None, k_grid=(2,)),
                  data, tmp_path, Profile(n_boot=200))


def test_from_frame_leaves_nan_for_per_fold_imputation():
    """``from_frame`` must NOT impute: doing so there computes the median over train and test
    together, so held-out rows set the values the model is fitted on. It leaves missing cells
    in place so the statistic can be fitted per fold on training rows only."""
    import numpy as np
    import pandas as pd

    from electropycal.discovery.config import RunData
    df = pd.DataFrame({
        "device": ["a", "a", "b"], "channel": [1, 1, 2],
        "timepoint": [0.0, 1.0, 0.0], "concentration": [100.0, 100.0, 100.0],
        "NormIpeak": [0.1, 0.2, 0.3],
        "feat_ok": [1.0, 2.0, 3.0], "feat_nan": [np.nan, 4.0, 6.0],
        "feat_empty": [np.nan, np.nan, np.nan]})
    data = RunData.from_frame(df)
    j = data.feature_names.index("feat_nan")
    assert np.isnan(data.X[0, j])                        # preserved, not filled with 5.0
    assert "feat_empty" not in data.feature_names        # all-NaN column still dropped
    assert np.isfinite(data.X[:, data.feature_names.index("feat_ok")]).all()


def test_median_impute_is_fitted_on_train_rows_only():
    """The imputation median comes from the training rows and is applied to held-out rows.

    Built so the train-only answer and the old corpus-wide answer differ: training rows are
    [10, 20] (median 15) while the full column is [10, 20, 1000] (median 20).
    """
    import numpy as np

    from electropycal.features.normalize import apply_median_impute, fit_median_impute
    X = np.array([[10.0], [20.0], [1000.0], [np.nan]])
    tr, te = np.array([0, 1]), np.array([2, 3])
    med = fit_median_impute(X[tr])
    assert med[0] == 15.0                                 # train-only; corpus-wide would be 20.0
    filled = apply_median_impute(X[te], med)
    assert filled[1, 0] == 15.0                           # held-out NaN gets the TRAIN median
    assert filled[0, 0] == 1000.0                         # finite held-out values untouched
    # a column empty within the training rows has no median -> 0.0 rather than NaN
    assert fit_median_impute(np.array([[np.nan], [np.nan]]))[0] == 0.0
    # +/-inf is treated as missing too, so it cannot reach the fit
    assert apply_median_impute(np.array([[np.inf]]), med)[0, 0] == 15.0


def test_run_discovery_skips_log_conditions_for_signed_target(tmp_path, capsys):
    """A full queue on a signed target must drop the log conditions and still complete, not crash."""
    import numpy as np, pandas as pd
    from electropycal.data.synthetic import make_dataset
    from electropycal.discovery.config import RunData, FAST, Condition
    from electropycal.discovery.scheduler import run_discovery
    ds = make_dataset(random_state=0)
    F = pd.DataFrame(ds.X, columns=ds.feature_names)
    F["channel"] = ds.channel; F["timepoint"] = ds.timepoint
    F["concentration"] = ds.concentration; F["device"] = "d0"
    F["signed"] = ds.y - ds.y.mean()                                  # has negatives
    data = RunData.from_frame(F, target="signed")
    queue = [Condition("lin", "linear_plsr", "global", None, k_grid=(2,), transform="linear"),
             Condition("log", "linear_plsr", "global", None, k_grid=(2,), transform="log")]
    rd = run_discovery(data, conditions=queue, out_root=str(tmp_path), profile=FAST, seed=0)
    assert rd.exists()
    assert "skipping 1 log condition" in capsys.readouterr().out


def test_parallel_folds_match_serial_and_cap_inner_threads(data, tmp_path):
    """The fold loop runs through the shared BLAS-capped helper.

    Without the cap each of n_jobs workers opens its own multi-threaded BLAS pool and they
    contend for the same cores, which is why the benchmark had to pin OMP_NUM_THREADS in the
    workflow to get interpretable timings. Results must also be unchanged by parallelism.
    """
    import dataclasses

    import electropycal._parallel as par
    cond = Condition("par", "linear_plsr", "global", None, k_grid=(2, 3))
    serial = run_condition(cond, data, tmp_path / "s", dataclasses.replace(FAST, n_jobs=1),
                           seed=0)[1]

    calls = []
    real = par.map_sessions
    par.map_sessions = lambda n, tasks: calls.append(n) or real(n, tasks)
    try:
        par_agg = run_condition(cond, data, tmp_path / "p",
                                dataclasses.replace(FAST, n_jobs=2), seed=0)[1]
    finally:
        par.map_sessions = real
    assert calls == [2]                                   # the fold loop went through the helper
    assert par_agg["pooled_rmsep"] == pytest.approx(serial["pooled_rmsep"])
