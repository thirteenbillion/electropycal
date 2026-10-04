"""RawSpectraIndex + plot functions (folded from the raw_spectra_review notebook)."""

import shutil
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pytest

from electropycal.data.synthetic import write_synthetic_pstrace_dir
from electropycal import rawspectra as rs


@pytest.fixture(scope="module")
def idx():
    root = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    return rs.RawSpectraIndex(root)


def test_index_discovers_devices_and_channels(idx):
    devs = idx.devices()
    assert devs == ["2-2", "2-3"]
    assert idx.title("2-2").endswith("2-2")
    dev = devs[0]
    assert idx.eis_channels(dev) and idx.fscv_channels(dev)
    assert 0.0 in idx.concs(dev) and 0.0 not in idx.concs(dev, include_zero=False)
    assert len(idx.available_table()) > 0


def test_device_type_filter(idx):
    # every synthetic device has the same devicetype; filtering to a bogus type yields none
    assert idx.devices(device_types=["__nope__"]) == []
    assert set(idx.devices(devices=["2-2"])) == {"2-2"}


def test_loaders_return_arrays(idx):
    dev = idx.devices()[0]; ch = idx.fscv_channels(dev)[0]; tp = idx.timepoints_with(dev, ch, "fscv")[0]
    v, y = idx.fscv_bgsub(dev, tp, ch, idx.concs(dev, include_zero=False)[0], smooth_window=idx.smooth_window)
    assert v.shape == y.shape and v.size > 0
    e = idx.eis_full(dev, tp, ch)
    assert e is not None and len(e) == 3


def test_peak_of_reports_clipped_flag(idx):
    dev = idx.devices()[0]; ch = idx.fscv_channels(dev)[0]; tp = idx.timepoints_with(dev, ch, "fscv")[0]
    pk = idx.peak_of(dev, tp, ch, idx.concs(dev, include_zero=False)[-1], idx.smooth_window)
    assert pk is None or {"v", "ipeak", "norm", "clipped"} <= set(pk)


def test_inductive_onset_monotone_semantics():
    f = np.array([1.0, 10.0, 100.0, 1000.0])
    zi = np.array([-5.0, -2.0, 1.0, 3.0])          # crosses >= 0 at 100 Hz
    assert rs.RawSpectraIndex.inductive_onset(f, zi) == 100.0
    assert np.isnan(rs.RawSpectraIndex.inductive_onset(f, np.array([-5, -4, -3, -2.0])))


def test_all_plots_and_tables_run(idx):
    dev = idx.devices()[0]
    for kind in ("phase", "mag", "nyquist"):
        rs.plot_eis(idx, dev, kind, band=idx.band); plt.close("all")
    rs.plot_eis_replicate_spread(idx, dev, "zr"); plt.close("all")
    rs.plot_inductive_onset_vs_time(idx, dev, band=idx.band); plt.close("all")
    for tr in ("raw", "bgsub", "norm"):
        rs.plot_fscv_loops(idx, dev, tr); plt.close("all")
    rs.plot_dose_response(idx, dev); plt.close("all")
    for k in ("norm", "vpeak", "snr"):
        rs.plot_vs_time(idx, dev, k); plt.close("all")
    assert len(rs.dose_stats_table(idx, idx.devices())) > 0
    assert len(rs.edge_pinning_table(idx, idx.devices())) > 0


def test_available_table_without_a_0nM_background_raises_naming_the_tree(tmp_path):
    """A tree can hold signal files and still build no session rows.

    ``available_table`` is keyed off 0 nM FSCV backgrounds alone, so EIS plus dosed FSCV
    clears the "any signal file" guard and then produces nothing. Left unguarded that
    surfaced as ``KeyError: 'device'`` from inside ``sort_values`` on a column-less frame,
    naming neither the tree nor the missing file.
    """
    sess = tmp_path / "20260915_neurostring_signal"
    sess.mkdir()
    full = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    src = next(iter(sorted(Path(full).rglob("*_eis_0n[mM].csv"))))
    dosed = sorted(Path(full).rglob("*_fscv_100n[mM].csv"))
    dev = rs.parse_filename(src.name)["deviceid"]
    shutil.copy2(src, sess / f"{dev}_eis_0nM.csv")
    if dosed:
        shutil.copy2(dosed[0], sess / f"{dev}_fscv_100nM.csv")

    idx = rs.RawSpectraIndex(str(tmp_path))
    assert len(idx.signal) > 0, "fixture must clear the any-signal-file guard"
    with pytest.raises(rs.NoSignalSessions) as ei:
        idx.available_table()
    msg = str(ei.value)
    assert str(tmp_path) in msg
    assert "0 nM FSCV" in msg
    assert "fscv_0nM.csv" in msg            # names what to add
    assert "found by type" in msg           # and what it did find


def test_available_table_still_returns_empty_frame_for_an_empty_tree(tmp_path):
    """An empty tree is a different situation and must not raise: columns, no rows."""
    (tmp_path / "20260915_neurostring_signal").mkdir()
    tbl = rs.RawSpectraIndex(str(tmp_path)).available_table()
    assert len(tbl) == 0
    assert list(tbl.columns) == ["device", "date", "timepoint", "n_channels", "channels"]
