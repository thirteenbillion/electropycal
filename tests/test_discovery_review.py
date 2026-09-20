"""Discovery results-review plots (folded from discovery_results_review)."""

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import pytest

from electropycal.data.synthetic import make_dataset
from electropycal.discovery.config import RunData, FAST
from electropycal.discovery.scheduler import run_discovery
from electropycal.discovery import review


@pytest.fixture(scope="module")
def run_dir():
    ds = make_dataset(random_state=0)
    data = RunData(ds.X, ds.y, ds.channel, ds.timepoint, ds.concentration, ds.feature_names)
    return run_discovery(data, out_root=tempfile.mkdtemp(), profile=FAST)


def test_ranking_and_feature_plots(run_dir):
    cr = review.plot_condition_ranking(run_dir, show=False); plt.close("all")
    assert {"condition", "pooled_rmsep"} <= set(cr.columns) and len(cr) > 0
    review.plot_fold_spread(run_dir, show=False); plt.close("all")
    fr = review.plot_feature_ranking(run_dir, top=10, show=False); plt.close("all")
    assert "feature" in fr.columns and len(fr) <= 10


def test_calibration_review_returns_predictions(run_dir):
    P = review.plot_calibration_review(run_dir, show=False); plt.close("all")
    assert {"y_true", "y_pred", "residual", "concentration", "t_test"} <= set(P.columns)


def test_target_framing_comparison(run_dir):
    ds = make_dataset(random_state=1)
    F = pd.DataFrame(ds.X, columns=ds.feature_names)
    F["channel"] = ds.channel; F["timepoint"] = ds.timepoint
    F["concentration"] = ds.concentration; F["NormIpeak"] = ds.y
    cmp = review.plot_target_framing_comparison(F, show=False); plt.close("all")
    assert "framing" in cmp.columns and "normipeak_rmsep" in cmp.columns
