"""Diagnostics review (folded from diagnostics_review)."""

import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from electropycal.data.synthetic import write_synthetic_pstrace_dir
from electropycal.analysis_config import load_analysis_config, resolve_band
from electropycal.features.extract import extract_dataset
from electropycal.diagreview import DiagnosticsReview, feat_type, feat_fidx


@pytest.fixture(scope="module")
def dr():
    root = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    cfg = load_analysis_config(root); band = resolve_band(root, cfg)
    feat = extract_dataset(root, band=band, peak_method=cfg.peak_method, acceptance=cfg.acceptance,
                           mono_tol=cfg.mono_tol, min_norm_snr=cfg.min_norm_snr,
                           monotonic_r_min=cfg.monotonic_r_min, max_reps=cfg.max_reps,
                           d0_normalize=False, n_jobs=1, progress=False)
    return DiagnosticsReview(feat, cfg=cfg, root=root, band=band, n_jobs=1)


def test_taxonomy_and_variance_built(dr):
    assert dr.STATE and dr.predictor_types
    assert set(dr.VAR.columns) == {"between_sensor", "within_time", "within_dose"}
    assert feat_type("C_s_f03") == "C_s" and feat_fidx("C_s_f03") == 3
    assert feat_type("mean_Ibg") == "mean_Ibg" and feat_fidx("mean_Ibg") is None
    fd = dr.feature_dictionary()
    assert "in_this_featureset" in fd.columns


def test_section1_variance(dr):
    dr.plot_featuretype_distributions(show=False); plt.close("all")
    top = dr.plot_variance_breakdown(show=False); plt.close("all")
    assert "within_time_var" in top.columns and len(top) <= 10
    vs = dr.plot_variance_decomposition(show=False); plt.close("all")
    assert {"between-sensor", "within-time", "within-dose"} <= set(vs.index)
    dr.plot_feature_3d(show=False); plt.close("all")


def test_section2_reliability(dr):
    dr.plot_response_reliability(show=False); plt.close("all")
    assert dr.R is not None
    dr.plot_response_drift_reliability(show=False); plt.close("all")
    dr.plot_feature_measurement_reliability(show=False); plt.close("all")
    typ = dr.plot_feature_drift_reliability(show=False); plt.close("all")
    assert "median_R_drift" in typ.columns and dr.DRIFT is not None


def test_section3_alignment(dr):
    top = dr.plot_drift_alignment(show=False); plt.close("all")
    assert "alignment_r" in top.columns and dr._maxa is not None
    dr.plot_frequency_bandwidth_tradeoff(show=False); plt.close("all")
    sc = dr.plot_alignment_sanity_checks(show=False); plt.close("all")
    assert {"max_drift_alignment", "max_absolute_alignment"} <= set(sc)


def test_section4_5_6(dr, capsys):
    dr.verdict()
    assert "Diagnosis" in capsys.readouterr().out
    dr.plot_d0_effect(show=False); plt.close("all")
    assert dr.VAR_D0 is not None
    dr.plot_d0_invariance(show=False); plt.close("all")
    modes = dr.plot_response_modes(show=False); plt.close("all")
    assert {"level", "within_time", "R_drift"} <= set(modes.columns) and len(modes) > 0
