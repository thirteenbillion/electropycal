"""Shared notebook analysis-config: round-trip, defaults, overrides, band resolution."""

import pytest

from electropycal.analysis_config import (AnalysisConfig, config_path, load_analysis_config,
                                        load_qc_stats, qc_stats_path, resolve_band,
                                        save_analysis_config, save_qc_stats)
from electropycal.data.synthetic import write_synthetic_pstrace_dir


def test_defaults_when_absent(tmp_path):
    cfg = load_analysis_config(tmp_path)                       # no file yet
    assert cfg.band == (2.0, 2000.0) and cfg.peak_method == "direct"
    assert cfg.mono_method == "pearson"
    assert cfg.gate_on["eis"] and not cfg.gate_on["snr_all"]


def test_save_load_roundtrip_preserves_values(tmp_path):
    cfg = AnalysisConfig(band=(10.0, 10_000.0), peak_method="chord",
                         mono_tol=0.07, gate_on={"eis": True, "monotonic": True,
                                                 "snr_all": True, "peak_in_window": False})
    p = save_analysis_config(tmp_path, cfg)
    assert p == config_path(tmp_path) and p.exists()
    back = load_analysis_config(tmp_path)
    assert back.band == (10.0, 10_000.0)                       # tuple restored from JSON list
    assert back.peak_method == "chord" and back.mono_tol == 0.07
    assert back.gate_on["snr_all"] is True


def test_inline_overrides_take_precedence(tmp_path):
    save_analysis_config(tmp_path, AnalysisConfig(band=(10.0, 5000.0)))
    cfg = load_analysis_config(tmp_path, band=(10.0, 20_000.0), peak_method="chord",
                               mono_tol=None)                  # None override is ignored
    assert cfg.band == (10.0, 20_000.0) and cfg.peak_method == "chord"
    assert cfg.mono_tol == 0.10                                # unchanged by the None


def test_unknown_keys_in_file_are_ignored(tmp_path):
    config_path(tmp_path).write_text('{"band": [10, 9000], "legacy_removed_field": 1}')
    cfg = load_analysis_config(tmp_path)
    assert cfg.band == (10.0, 9000.0)                          # loads despite the stale key


def test_qc_stats_roundtrip_and_absent(tmp_path):
    assert load_qc_stats(tmp_path) is None                     # not written yet
    stats = {"config": {"mono_method": "pearson"}, "totals": {"overall_valid": 42}}
    p = save_qc_stats(tmp_path, stats)
    assert p == qc_stats_path(tmp_path) and p.exists()
    got = load_qc_stats(tmp_path)
    assert got["totals"]["overall_valid"] == 42 and got["config"]["mono_method"] == "pearson"


def test_resolve_band_auto_and_explicit(tmp_path):
    root = write_synthetic_pstrace_dir(tmp_path)               # capacitive synthetic -> default hi
    assert resolve_band(root, AnalysisConfig(band="auto")) == (10.0, 100_000.0)
    assert resolve_band(root, AnalysisConfig(band=(10.0, 8000.0))) == (10.0, 8000.0)
    with pytest.raises(ValueError, match="band must be"):
        resolve_band(root, AnalysisConfig(band="wide"))
