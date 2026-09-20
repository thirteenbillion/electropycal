"""Sensitivity target framing: dose-response slope per sensor-timepoint."""

import numpy as np
import pandas as pd

from electropycal.discovery.config import RunData, Condition, FAST
from electropycal.discovery.runner import run_condition
from electropycal.features.targets import sensitivity_featureset


def _doses(state, slope, intercept, concs=(100.0, 500.0, 1000.0, 5000.0)):
    """Rows for one sensor-timepoint: NormIpeak = intercept + slope*log10(conc), fixed state."""
    return [{"device": state["device"], "channel": state["channel"], "timepoint": state["timepoint"],
             "concentration": c, "NormIpeak": intercept + slope * np.log10(c),
             "feat_a": state["a"], "feat_b": state["b"]} for c in concs]


def _frame():
    rows = []
    # two sensors, two timepoints each; both share slopes 0.02 (t0) and 0.05 (t1)
    for dev, ch in (("2-2", 1), ("2-3", 1)):
        for tp, sl in ((0.0, 0.02), (1.0, 0.05)):
            rows += _doses({"device": dev, "channel": ch, "timepoint": tp,
                            "a": sl * 10, "b": 1.0}, slope=sl, intercept=0.1)
    return pd.DataFrame(rows)


def _frame_many(n=6):
    rows = []
    # n sensors x 3 timepoints: enough samples per fold and >=2 training timepoints for the
    # nested (inner) CV; slope tracks the state features so state predicts sensitivity
    for i in range(n):
        for tp in (0.0, 1.0, 2.0):
            sl = 0.02 + 0.004 * i + 0.01 * tp
            rows += _doses({"device": f"d{i}", "channel": 1, "timepoint": tp,
                            "a": sl * 10, "b": 1.0 + 0.1 * i}, slope=sl, intercept=0.1)
    return pd.DataFrame(rows)


def test_sensitivity_featureset_fits_slope_per_sensor_timepoint():
    S = sensitivity_featureset(_frame())
    assert len(S) == 4                                        # 2 sensors x 2 timepoints
    assert {"sensitivity", "sensitivity_intercept", "dose_response_r", "n_conc"} <= set(S.columns)
    # recovered slopes match the ones we injected (0.02 and 0.05)
    assert set(np.round(S.sensitivity, 3)) == {0.02, 0.05}
    assert np.allclose(S.sensitivity_intercept, 0.1, atol=1e-6)
    assert (S.n_conc == 4).all() and np.allclose(S.dose_response_r, 1.0)
    # state features carried through; identity kept
    assert {"feat_a", "feat_b", "device", "channel", "timepoint"} <= set(S.columns)


def test_sensitivity_featureset_drops_underdetermined_groups():
    df = _frame()
    # a sensor-timepoint with only 2 concentrations can't fit a line -> dropped
    df = pd.concat([df, pd.DataFrame([
        {"device": "9-9", "channel": 2, "timepoint": 0.0, "concentration": 100.0,
         "NormIpeak": 0.1, "feat_a": 1.0, "feat_b": 1.0},
        {"device": "9-9", "channel": 2, "timepoint": 0.0, "concentration": 500.0,
         "NormIpeak": 0.2, "feat_a": 1.0, "feat_b": 1.0}])], ignore_index=True)
    S = sensitivity_featureset(df)
    assert not ((S.device == "9-9")).any()                   # the 2-dose group is gone


def test_compare_target_framings_metrics():
    from electropycal.evaluation.framing import compare_target_framings
    cmp = compare_target_framings(_frame_many(), min_train_times=3, include_interaction=True)
    by = cmp.set_index("framing")
    assert set(by.index) == {"current", "sensitivity", "sensitivity_quadratic", "interaction"}
    assert (cmp.n_folds > 0).all()
    # the quadratic arm reconstructs NormIpeak and recovers concentration (4 doses -> identifiable)
    assert np.isfinite(by.loc["sensitivity_quadratic", "normipeak_rmsep"])
    # 'current' is flat -> no calibration slope -> cannot recover concentration
    assert by.loc["current", "conc_recovery_frac"] == 0.0
    assert np.isnan(by.loc["current", "conc_recovery_rmse_log10"])
    # 'sensitivity' gives a finite NormIpeak RMSEP AND recovers concentration
    assert np.isfinite(by.loc["sensitivity", "normipeak_rmsep"])
    assert by.loc["sensitivity", "conc_recovery_frac"] > 0


def test_sensitivity_featureset_ignores_stray_string_columns():
    """A non-feature string column (e.g. a downstream 'devicetype'/'sensor' label) must not break the
    per-group mean — only numeric predictors are averaged."""
    import numpy as np, pandas as pd
    from electropycal.data.synthetic import make_dataset
    from electropycal.features.targets import sensitivity_featureset
    ds = make_dataset(random_state=0)
    df = pd.DataFrame(ds.X, columns=ds.feature_names)
    df["device"] = "2-2"; df["channel"] = ds.channel; df["timepoint"] = ds.timepoint
    df["concentration"] = ds.concentration; df["NormIpeak"] = ds.y
    df["devicetype"] = "neurostring"; df["sensor"] = "2-2:1"      # stray strings
    S = sensitivity_featureset(df)
    assert "sensitivity" in S and np.isfinite(S["sensitivity"]).any()
    assert "devicetype" not in S.columns and "sensor" not in S.columns


def test_from_frame_target_and_discovery_run(tmp_path):
    S = sensitivity_featureset(_frame_many())
    data = RunData.from_frame(S, target="sensitivity")
    assert "sensitivity" not in data.feature_names          # target excluded from X
    assert np.array_equal(data.y, S["sensitivity"].to_numpy(float))
    assert set(data.feature_names) == {"feat_a", "feat_b"}   # only real features remain
    rows, agg = run_condition(Condition("s", "linear_plsr", "global", None, k_grid=(2,)),
                              data, tmp_path, FAST, seed=0)
    assert rows and np.isfinite(agg["pooled_rmsep"])         # runs on the slope target


def test_sensitivity_featureset_emits_curvature_when_enough_doses():
    # 4 doses -> quadratic identifiable -> sensitivity_curvature finite; linear slope unchanged
    S = sensitivity_featureset(_frame())            # concs (100,500,1000,5000) = 4 distinct
    assert "sensitivity_curvature" in S.columns
    assert S["sensitivity_curvature"].notna().all()
    # a purely linear NormIpeak(log conc) -> curvature ~ 0, slope still the injected value
    assert np.allclose(S["sensitivity_curvature"].to_numpy(float), 0.0, atol=1e-6)
    assert set(np.round(S.sensitivity, 3)) == {0.02, 0.05}


def test_sensitivity_curvature_nan_with_three_doses():
    df = _frame().copy()
    df = df[df.concentration != 5000.0]             # drop to 3 distinct doses
    S = sensitivity_featureset(df)
    assert S["sensitivity_curvature"].isna().all()  # under-determined -> NaN


def test_sensitivity_curvature_recovers_injected_quadratic():
    # NormIpeak = 0.1 + 0.03*lc + 0.02*lc^2 over 4 doses -> curvature ~ 0.02
    concs = (100.0, 500.0, 1000.0, 5000.0)
    rows = []
    for lc_c in concs:
        lc = np.log10(lc_c)
        rows.append({"device": "d", "channel": 1, "timepoint": 0.0, "concentration": lc_c,
                     "NormIpeak": 0.1 + 0.03 * lc + 0.02 * lc ** 2, "feat_a": 1.0, "feat_b": 2.0})
    S = sensitivity_featureset(pd.DataFrame(rows))
    assert np.isclose(float(S["sensitivity_curvature"].iloc[0]), 0.02, atol=1e-6)


def test_reserved_columns_include_sensitivity_curvature():
    from electropycal.data.schema import RESERVED_COLUMNS
    assert "sensitivity_curvature" in RESERVED_COLUMNS


def test_langmuir_saturation_target():
    import numpy as np, pandas as pd
    from electropycal.features.targets import sensitivity_featureset, _fit_langmuir
    from electropycal.data.schema import RESERVED_COLUMNS
    conc = np.array([100, 250, 500, 1000, 5000.0])
    imax, kd = _fit_langmuir(conc, 2.0 * conc / (400.0 + conc))
    assert abs(imax - 2.0) < 0.05 and abs(kd - 400.0) < 20            # recovers known params
    # monotone (invertible): the fitted curve never decreases with dose
    fit = imax * conc / (kd + conc)
    assert np.all(np.diff(fit) > 0)
    rows = [dict(device="d0", channel=ch, timepoint=float(tp), concentration=float(c),
                 NormIpeak=1.5 * c / (300 + c), R_s_f00=1000.0 + ch, C_s_f00=1e-6)
            for ch in (1, 2) for tp in (0, 20) for c in conc]
    S = sensitivity_featureset(pd.DataFrame(rows))
    assert {"sat_imax", "sat_kd", "sat_logkd"} <= set(S.columns)
    assert (S["sat_imax"] > 0).all() and (S["sat_kd"] > 0).all()
    for c in ("sat_imax", "sat_kd", "sat_logkd"):
        assert c in RESERVED_COLUMNS                                  # targets, never predictors
    # opt-out
    assert "sat_imax" not in sensitivity_featureset(pd.DataFrame(rows), saturation=False).columns


def test_hill_fit_and_snr_weighting():
    import numpy as np, pandas as pd
    from electropycal.features.targets import _fit_hill, sensitivity_featureset
    from electropycal.data.schema import RESERVED_COLUMNS
    conc = np.array([100, 250, 500, 1000, 5000.0])
    hi, hk, hn = _fit_hill(conc, 2.0 * conc ** 1.5 / (400.0 ** 1.5 + conc ** 1.5))
    assert abs(hi - 2.0) < 0.1 and abs(hk - 400) < 40 and abs(hn - 1.5) < 0.2   # recovers Hill params
    for c in ("hill_imax", "hill_kd", "hill_n"):
        assert c in RESERVED_COLUMNS
    # SNR weighting shifts the fit away from noisy low-dose points
    rng = np.random.default_rng(0)
    base = []
    for c in conc:
        snr = 0.5 if c < 300 else 8.0
        noise = (0.3 if c < 300 else 0.0) * rng.standard_normal()
        base.append(dict(concentration=float(c), NormIpeak=1.5 * c / (300 + c) + noise,
                         repeatability_snr=snr, R_s_f00=1000.0))
    df = pd.concat([pd.DataFrame(base).assign(device="d0", channel=1, timepoint=t) for t in (0.0, 20.0)],
                   ignore_index=True)
    S_w = sensitivity_featureset(df)                              # weighted (default weight_col present)
    S_u = sensitivity_featureset(df, weight_col="__absent__")     # unweighted
    assert {"hill_imax", "hill_kd", "hill_n"} <= set(S_w.columns)
    assert not np.allclose(S_w["sensitivity"].to_numpy(), S_u["sensitivity"].to_numpy())
    assert "hill_imax" not in sensitivity_featureset(df, hill=False).columns


def test_compare_target_framings_tolerates_all_nan_state_columns():
    """A feature undefined for the chosen band is all-NaN. ``RunData.from_frame`` drops such
    columns; this path builds its own matrix, so without the same drop PLS raised
    ``Input X contains NaN`` on the project's own featureset while passing on synthetic
    frames that happen to have none."""
    import numpy as np

    from electropycal.evaluation.framing import compare_target_framings
    df = _frame_many()
    df["ideality_C_band_HF"] = np.nan        # undefined for this band, exactly like the real one
    df["partly_missing"] = np.where(np.arange(len(df)) % 7 == 0, np.nan, 1.23)
    cmp = compare_target_framings(df, min_train_times=3, include_interaction=True)
    assert (cmp.n_folds > 0).all()
    assert np.isfinite(cmp.set_index("framing").loc["sensitivity", "normipeak_rmsep"])


def test_quadratic_root_selection_does_not_use_the_held_out_truth():
    """The quadratic inversion has two roots. Disambiguating with the held-out group's own
    true log-concentrations uses the quantity being recovered; the reference must be the
    training dose grid, which a real deployment has."""
    import inspect

    import numpy as np

    from electropycal.evaluation import framing
    src = inspect.getsource(framing.compare_target_framings)
    assert "_invert_quad(aa_, bb_, cc_, y, x_ref_tr)" in src      # not `..., y, x)`
    assert 'x_ref_tr = tr["_lc"].to_numpy(float)' in src

    # and the helper genuinely keys off the reference it is handed
    a, b, c = 0.0, 0.0, 1.0                  # y = x^2 -> roots +/-sqrt(y), fully ambiguous
    y = np.array([4.0])
    assert framing._invert_quad(a, b, c, y, np.array([-10.0]))[0] == -2.0
    assert framing._invert_quad(a, b, c, y, np.array([+10.0]))[0] == +2.0
