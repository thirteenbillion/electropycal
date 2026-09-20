"""Target-admissibility screen (study experiment A2)."""

import numpy as np
import pandas as pd

from electropycal.evaluation.admissibility import target_admissibility, INVERTIBLE_TARGETS


def _doses(dev, ch, tp, slope, intercept, concs=(100.0, 500.0, 1000.0, 5000.0)):
    return [{"device": dev, "channel": ch, "timepoint": tp, "concentration": c,
             "NormIpeak": intercept + slope * np.log10(c), "R_s_f00": 1000.0, "C_s_f00": 1e-6}
            for c in concs]


def _frame():
    rows = []
    rng = np.random.default_rng(0)
    for i in range(8):
        for tp in (0.0, 1.0, 2.0):
            slope = 0.02 + 0.01 * i + 0.01 * tp        # varies across sensor-timepoints -> real range
            rows += _doses(f"d{i}", 1, tp, slope=slope, intercept=0.1)
    return pd.DataFrame(rows)


def test_admissibility_ranks_sensitivity_admissible():
    out = target_admissibility(_frame()).set_index("target")
    assert bool(out.loc["sensitivity", "admissible"])          # identifiable, invertible, moves
    assert out.loc["sensitivity", "identifiable"] == 1.0
    assert out.loc["sensitivity", "target_cv"] > 0


def test_curvature_and_intercept_not_invertible_so_not_admissible():
    out = target_admissibility(_frame()).set_index("target")
    # signed shape/level params are estimable but not standalone-invertible -> not admissible
    assert not bool(out.loc["sensitivity_curvature", "admissible"])
    assert not bool(out.loc["sensitivity_intercept", "admissible"])
    assert INVERTIBLE_TARGETS["sensitivity"] and not INVERTIBLE_TARGETS["sensitivity_curvature"]


def test_low_dynamic_range_target_fails_cv_gate():
    # a target that barely moves (all slopes nearly equal) -> low CV -> not admissible even if invertible
    rows = []
    for i in range(8):
        for tp in (0.0, 1.0, 2.0):
            rows += _doses(f"d{i}", 1, tp, slope=0.05, intercept=0.1)   # identical slope everywhere
    out = target_admissibility(pd.DataFrame(rows), min_cv=0.15).set_index("target")
    assert out.loc["sensitivity", "target_cv"] < 0.15
    assert not bool(out.loc["sensitivity", "admissible"])


def test_admissible_sorted_first_and_columns_present():
    out = target_admissibility(_frame())
    assert list(out.columns) == ["target", "identifiable", "n_estimable", "invertible",
                                 "target_std", "target_cv", "reliability", "admissible"]
    assert out["admissible"].iloc[0] in (True, np.True_)       # admissible rows sort to the top
