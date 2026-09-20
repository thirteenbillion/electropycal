"""Trivial baselines (naïve-mean + time-only) that bound the electrode-state model."""

import numpy as np
import pandas as pd

from electropycal.evaluation.baselines import time_only_baseline_cv


def _panel(time_driven: bool, seed: int = 0) -> pd.DataFrame:
    """8 sensors x 6 timepoints. When ``time_driven`` the target is a clean function of age (a
    population time-trend exists); otherwise it is age-independent noise (time carries nothing)."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(8):
        base = 0.05 + 0.005 * i
        for tp in range(6):
            age = float(tp)
            y = base - 0.006 * age if time_driven else base + rng.normal(0, 0.02)
            rows.append(dict(device=f"d{i}", channel=1, timepoint=age, sensitivity=y))
    return pd.DataFrame(rows)


def test_time_only_recovers_population_age_trend():
    res = time_only_baseline_cv(_panel(time_driven=True), "sensitivity")
    assert res["n_folds"] >= 2 and res["n_test"] > 0
    assert res["time_q2"] > 0.5                       # a real age trend is learnable
    assert res["time_rmsep"] < res["naive_rmsep"]     # time beats the mean here


def test_time_only_is_chance_when_target_is_age_independent():
    res = time_only_baseline_cv(_panel(time_driven=False), "sensitivity")
    assert res["time_q2"] < 0.3                       # no age trend -> no better than the mean


def test_time_only_reports_naive_rmsep_and_target_std():
    res = time_only_baseline_cv(_panel(time_driven=True), "sensitivity")
    # naïve RMSEP is the do-nothing reference; target_std contextualizes whether a small RMSEP is skill
    assert np.isfinite(res["naive_rmsep"]) and res["naive_rmsep"] > 0
    assert np.isfinite(res["target_std"])


def test_time_only_empty_or_missing_target_is_graceful():
    res = time_only_baseline_cv(pd.DataFrame({"timepoint": [], "sensitivity": []}), "sensitivity")
    assert res["n_folds"] == 0 and np.isnan(res["time_q2"])


def test_fit_ridge_auto_alpha_shrinks_to_the_mean_on_a_signal_free_target():
    """The E6/E8 regime: p >> n and the predictors carry nothing about the target.

    A fixed small penalty fits the noise and scores clearly negative out of sample; the inner-CV
    penalty shrinks toward the training mean and lands at Q2 ~ 0. This is the toy version of the real
    failure the alpha=1.0 default produced (q2 ~ -45 on 145 predictors).
    """
    import numpy as np
    from electropycal.evaluation.metrics import ALPHA_GRID, fit_ridge

    rng = np.random.default_rng(0)
    n, p = 60, 145
    X = rng.standard_normal((n, p))
    y = rng.normal(0, 0.018, n)                  # target scale of `sensitivity`; no signal in X
    Xte = rng.standard_normal((300, p))
    yte = rng.normal(0, 0.018, 300)

    def q2(model):
        pred = model.predict(Xte)
        return 1.0 - np.sum((yte - pred) ** 2) / np.sum((yte - y.mean()) ** 2)

    auto = fit_ridge(X, y)
    fixed = fit_ridge(X, y, alpha=1.0)
    assert q2(auto) > -0.05                      # ~0: correctly declines to predict
    assert q2(fixed) < -0.15                     # fixed small penalty fits noise
    assert q2(auto) > q2(fixed)
    assert auto.alpha_ in set(ALPHA_GRID)


def test_fit_ridge_explicit_alpha_is_honoured():
    import numpy as np
    from electropycal.evaluation.metrics import fit_ridge
    rng = np.random.default_rng(1)
    X = rng.standard_normal((20, 5)); y = rng.standard_normal(20)
    assert fit_ridge(X, y, alpha=7.5).alpha == 7.5


def _drifting_frame(n_ch=8, n_t=8, seed=0):
    """Per-channel offsets + a shared downward drift, with pure-noise predictors."""
    import numpy as np, pandas as pd
    rng = np.random.default_rng(seed)
    rows = []
    off = rng.normal(0, 0.3, n_ch)
    for c in range(n_ch):
        for t in range(n_t):
            r = {"device": "d1", "channel": c, "timepoint": float(t),
                 "sensitivity": 1.0 + off[c] - 0.05 * t + rng.normal(0, 0.02)}
            for k in range(12):
                r[f"f{k}"] = rng.standard_normal()   # carry no signal
            rows.append(r)
    return pd.DataFrame(rows)


def test_channel_persistence_cv_recovers_channel_offsets():
    """chan_mean must find the per-channel offset with no features at all."""
    from electropycal.evaluation.baselines import channel_persistence_cv
    out = channel_persistence_cv(_drifting_frame(), target="sensitivity")
    by = {r.control: r for r in out.itertuples()}
    assert by["chan_mean"].value > 0.3            # strong per-channel autocorrelation
    assert set(by) >= {"chan_mean", "chan_trend", "persistence_frac0.5"}
    assert (out["n_test"] > 0).all()


def test_channel_persistence_cv_null_when_no_channel_structure():
    """With no per-channel offsets there is nothing for the control to exploit."""
    import numpy as np, pandas as pd
    from electropycal.evaluation.baselines import channel_persistence_cv
    rng = np.random.default_rng(3)
    rows = []
    for c in range(8):
        for t in range(8):
            rows.append({"device": "d1", "channel": c, "timepoint": float(t),
                         "sensitivity": rng.normal(1.0, 0.05), "f0": rng.standard_normal()})
    out = channel_persistence_cv(pd.DataFrame(rows), target="sensitivity")
    q2 = {r.control: r.value for r in out.itertuples()}["chan_mean"]
    assert q2 < 0.15                              # no exploitable channel structure


def test_hierarchical_cv_auto_alpha_is_sane():
    """The default (inner-CV) penalty must not produce the alpha=1.0 blow-up."""
    from electropycal.evaluation.hierarchical import hierarchical_cv
    r = hierarchical_cv(_drifting_frame(), target="sensitivity", group_level="channel")
    assert r["n_folds"] > 0
    assert r["q2"] > -2.0 and r["q2_fixed"] > -2.0          # not the -45 regime
    assert r["rmsep"] < 5 * r["naive_rmsep"]


def test_stratify_by_drift_auto_alpha_is_sane():
    from electropycal.evaluation.stratify import stratify_by_drift
    out = stratify_by_drift(_drifting_frame(), target="sensitivity")
    assert len(out) and set(out["stratum"]) >= {"all"}
    allrow = out[out.stratum == "all"].iloc[0]
    assert allrow["q2"] > -2.0
