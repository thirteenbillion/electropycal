"""Quality-filtering dashboard (folded from quality_filtering_dashboard)."""

import json
import tempfile

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from electropycal.data.synthetic import write_synthetic_pstrace_dir
from electropycal.qcdash import QCDashboard


@pytest.fixture(scope="module")
def qc():
    root = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    return QCDashboard(root, n_jobs=1, progress=False)


def test_qc_tables_and_registry(qc):
    assert {"overall_valid", "fail_reasons", "has_eis", "has_fscv"} <= set(qc.CH_TBL.columns)
    assert len(qc.available_table()) > 0
    defs = qc.qc_definitions()
    assert {"code", "check", "gating"} <= set(defs.columns) and len(defs) == len(qc.dchecks)
    assert qc.quality_table().shape[0] == len(qc.CH_TBL)
    assert (qc.valid_sensor_summary()["valid_%"] <= 100).all()


def test_all_plots_run(qc):
    qc.plot_schedule(show=False); plt.close("all")
    qc.plot_dropout_by_check(show=False); plt.close("all")
    imp = qc.gate_impact(show=False); plt.close("all")
    assert "n_channeltimepoints_failed" in imp.columns
    qc.plot_dose_monotonicity_sweep(show=False); plt.close("all")
    qc.plot_z_monotonicity_sweep(show=False); plt.close("all")
    qc.plot_snr_sweep(show=False); plt.close("all")
    qc.plot_finalized_overview(show=False); plt.close("all")


def test_finalized_featureset_exposes_normipeak(qc):
    ov = qc.finalized_featureset()
    assert "NormIpeak" in ov.columns and "concentration" in ov.columns


def test_save_stats_writes_valid_json(qc):
    qc.gate_impact(show=False); plt.close("all")
    qc.plot_dose_monotonicity_sweep(show=False); plt.close("all")
    qc.valid_sensor_summary()
    p = qc.save_stats()
    stats = json.loads(p.read_text())
    assert stats["totals"]["channel_timepoints_paired"] == len(qc.CH_TBL[qc.CH_TBL.has_eis & qc.CH_TBL.has_fscv])
    assert stats["band_hz"] == [float(qc.band[0]), float(qc.band[1])]
    assert stats["gate_impact"] is not None            # cached from gate_impact()
