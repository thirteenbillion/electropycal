"""Deployment monitor plots (folded from deployment_domain_shift)."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

from electropycal.deployment.plots import plot_domain_monitor, plot_invivo_recalibration


def test_plot_domain_monitor_runs():
    df = pd.DataFrame({"day": [0, 7, 21, 42], "domain_distance": [0.5, 1.2, 2.4, 3.1],
                       "mean_recal": [0.4, 0.42, 0.5, 0.6],
                       "confidence": ["ok", "ok", "EXTRAPOLATING", "EXTRAPOLATING"]})
    plot_domain_monitor(df, flag=2.0); plt.close("all")
    # falls back to threshold coloring when no confidence column
    plot_domain_monitor(df.drop(columns="confidence"), flag=2.0); plt.close("all")


def test_plot_invivo_recalibration_runs():
    res = pd.DataFrame({"timepoint": [0, 7, 21], "mean_norm_ipeak": [0.4, 0.45, 0.5],
                        "domain_distance": [0.6, 1.5, 2.5]})
    plot_invivo_recalibration(res, flag=2.0); plt.close("all")
