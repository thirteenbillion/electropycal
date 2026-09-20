"""Discovery overview plots + batch runner (folded from discovery_checkpointed)."""

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from electropycal.data.synthetic import make_dataset
from electropycal.discovery.config import RunData, FAST, baseline_queue
from electropycal import overview
from electropycal.discovery.batch import BatchRunner


@pytest.fixture(scope="module")
def sel():
    ds = make_dataset(random_state=0)
    F = pd.DataFrame(ds.X, columns=ds.feature_names)
    F["device"] = "2-2"; F["channel"] = ds.channel; F["timepoint"] = ds.timepoint
    F["concentration"] = ds.concentration; F["NormIpeak"] = ds.y
    return F


def test_overview_plots_and_table(sel):
    overview.plot_qc_grid(sel, show=False); plt.close("all")
    overview.plot_normipeak_per_sensor(sel, show=False); plt.close("all")
    overview.plot_normipeak_per_device(sel, show=False); plt.close("all")
    ct = overview.conditions_table(baseline_queue())
    assert {"batch", "architecture", "evaluation"} <= set(ct.columns) and len(ct) > 0


def test_overview_per_device_skips_without_device_column(sel, capsys):
    overview.plot_normipeak_per_device(sel.drop(columns="device"), show=False); plt.close("all")
    assert "skipped" in capsys.readouterr().out


def test_batch_runner_runs_and_ranks(sel):
    data = RunData.from_frame(sel)
    run = BatchRunner(baseline_queue(), data, tempfile.mkdtemp(), FAST)
    df = run.run(1)
    assert len(df) > 0 and {"condition", "rmsep", "q2", "folds"} <= set(df.columns)
    assert len(run.ranking) == len(df)                 # ranking accumulated
    run.plot(df, "Batch 1"); plt.close("all")
    flags = run.advise(2)
    assert isinstance(flags, dict)


def test_batch_runner_exclude(sel):
    data = RunData.from_frame(sel)
    run = BatchRunner(baseline_queue(), data, tempfile.mkdtemp(), FAST)
    n0 = len(run.queue)
    run.exclude("baselines_1.1_linearPLSR")
    assert len(run.queue) == n0 - 1
