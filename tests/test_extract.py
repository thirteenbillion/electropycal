"""End-to-end ingestion: raw PSTrace directory -> featureset -> RunData."""

import numpy as np
import pandas as pd

from electropycal.discovery.config import RunData
from electropycal.features.extract import extract_dataset, replicate_feature_reliability


def _write_eis(path, channels, Rs=5000.0, C=1e-8, freqs=None):
    freqs = np.logspace(1, 5, 15) if freqs is None else np.asarray(freqs, float)
    w = 2 * np.pi * freqs
    lines = ["Date and time:,2025-01-01 00:00:00", "Notes:",
             "Measurement:,Impedance Spectroscopy", "Notes:,",
             "Date and time:,2025-01-01 00:00:00"]
    for ch in channels:
        Cc = C * (1 + 0.1 * ch)
        im = -1.0 / (w * Cc)                            # Im(Z) < 0 (capacitive)
        zr = np.full_like(freqs, Rs)
        zmag = np.hypot(zr, im)
        stored = -im                                    # PSTrace stores -Im(Z) (positive)
        neg_phase = -np.degrees(np.arctan2(im, zr))
        lines.append(f"CH {ch}: Fixed at {len(freqs)} freqs")
        lines.append("freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F")
        for k in range(len(freqs) - 1, -1, -1):         # high -> low frequency
            lines.append(f"{freqs[k]},{neg_phase[k]},1.0,{zmag[k]},{zr[k]},{stored[k]},1e-9")
    path.write_text("\n".join(lines), encoding="utf-16")


def _write_fscv(path, channels, peak_scale, reps=2):
    v = np.linspace(-0.4, 1.1, 80)
    bg = 100.0 + 50.0 * v
    peak = peak_scale * np.exp(-((v - 0.7) / 0.05) ** 2)
    hdr, units = [], []
    for ch in channels:
        for r in range(reps):
            lab = f"Fast Cyclic Voltammetry: FCV i vs E Channel {ch}" + ("" if r == 0 else f" [{r}]")
            hdr += [lab, ""]
            units += ["V", "µA"]
    lines = ["Date and time:,2025-01-01 00:00:00", "Notes:", ",,,,",
             ",".join(hdr), ",".join(["Date and time measurement:"] * len(hdr)), ",".join(units)]
    for i in range(len(v)):
        cells = []
        for _ch in channels:
            for r in range(reps):
                cells += [f"{v[i]}", f"{bg[i] + peak[i] * (1 + 0.02 * r)}"]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines), encoding="utf-16")


def _session(root, date, device, channels, peak_at, devicetype="neurostring", freqs=None):
    d = root / f"{date}_{devicetype}_signal"
    d.mkdir(exist_ok=True)
    _write_eis(d / f"{device}_eis_0nm.csv", channels, freqs=freqs)
    _write_fscv(d / f"{device}_fscv_0nm.csv", channels, peak_scale=0.0)   # background
    for dose in (100, 500, 1000):
        _write_fscv(d / f"{device}_fscv_{dose}nm.csv", channels, peak_scale=peak_at * dose / 1000)


def _build_raw(root):
    # device 2-2: sessions 7 days apart -> timepoints 0 and 7
    _session(root, "20260715", "2-2", [3, 5], peak_at=20)
    _session(root, "20260722", "2-2", [3, 5], peak_at=18)
    # device 2-3: only the later session -> its own timepoint 0 (staggered)
    _session(root, "20260722", "2-3", [3, 5], peak_at=22)


def test_extract_builds_featureset(tmp_path):
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto")
    assert not df.empty
    for col in ("device", "channel", "timepoint", "concentration", "NormIpeak",
                "mean_Vpeak", "mean_Ibg", "R_s_f00", "R_s_integral", "tau_ratio"):
        assert col in df.columns
    assert set(df["concentration"]) == {100.0, 500.0, 1000.0}     # 0 nM excluded (background)


def test_extract_d0_normalizes_features_by_default(tmp_path):
    import inspect

    import numpy as np
    _build_raw(tmp_path)
    assert inspect.signature(extract_dataset).parameters["d0_normalize"].default is True
    d0 = extract_dataset(tmp_path, band="auto")                            # D0-normalized (default)
    raw = extract_dataset(tmp_path, band="auto", d0_normalize=False)       # raw features
    # target + QC columns are untouched by D0-normalization; features change
    assert np.allclose(d0["NormIpeak"], raw["NormIpeak"])
    assert not np.allclose(d0["R_s_f00"], raw["R_s_f00"])
    # at each sensor's earliest timepoint a magnitude feature normalizes to ~1 (division)
    for (dev, ch), g in d0.groupby(["device", "channel"]):
        d0rows = g[g.timepoint == g.timepoint.min()]
        assert np.allclose(d0rows["R_s_f00"].to_numpy(float), 1.0, atol=1e-6)


def test_extract_band_auto_is_opt_in_and_matches_recommended(tmp_path):
    """``band`` has no default: "auto" is a percentile over whatever corpus is present, so
    it cannot be reproduced over a staged subset and must be asked for explicitly."""
    import inspect

    import pytest

    from electropycal.data.inventory import recommended_band
    _build_raw(tmp_path)
    assert inspect.signature(extract_dataset).parameters["band"].default is None
    with pytest.raises(ValueError, match="band is required"):
        extract_dataset(tmp_path)                                      # no silent corpus pre-pass
    auto = extract_dataset(tmp_path, band="auto")                      # opt-in
    explicit = extract_dataset(tmp_path, band=recommended_band(tmp_path))
    assert not auto.empty and auto.equals(explicit)                   # auto == recommended_band
    with pytest.raises(ValueError, match="band must be"):
        extract_dataset(tmp_path, band="wide")


def test_extract_progress_logs_without_changing_results(tmp_path, capsys):
    _build_raw(tmp_path)
    df_q = extract_dataset(tmp_path, band="auto")
    assert capsys.readouterr().out == ""                       # silent by default
    df_v = extract_dataset(tmp_path, band="auto", progress=True)
    out = capsys.readouterr().out
    assert "extract_dataset" in out and "ETA" in out           # timestamped progress emitted
    assert df_v.equals(df_q)                                   # logging doesn't change the result


def test_extract_parallel_matches_serial(tmp_path):
    """n_jobs>1 feature-extracts sessions in parallel but must be byte-identical to serial:
    the shared EIS ref_grid is fixed by a deterministic pre-pass and rows are concatenated in
    sorted-session order."""
    import pandas as pd
    _build_raw(tmp_path)
    serial = extract_dataset(tmp_path, band="auto", n_jobs=1)
    parallel = extract_dataset(tmp_path, band="auto", n_jobs=2)
    assert list(serial.columns) == list(parallel.columns)
    pd.testing.assert_frame_equal(serial.reset_index(drop=True),
                                  parallel.reset_index(drop=True))


def test_min_dose_response_range_gate(tmp_path):
    """The amplitude floor is off by default (featureset unchanged) and, when set high, drops
    monotone-but-flat channel-timepoints that the scale-invariant r would otherwise keep."""
    _build_raw(tmp_path)
    base = extract_dataset(tmp_path, band="auto")
    assert extract_dataset(tmp_path, band="auto", min_dose_response_range=None).equals(base)   # default off = unchanged
    assert len(extract_dataset(tmp_path, band="auto", min_dose_response_range=1e9)) == 0       # nothing clears an absurd floor


def test_should_parallelize_policy():
    """Spawn a pool only when >1 worker is requested AND >1 item to spread over it; negatives
    (-1 = all cores) count as parallel."""
    from electropycal._parallel import should_parallelize
    assert not should_parallelize(1, 100)     # single worker -> serial
    assert not should_parallelize(0, 100)     # 0 -> serial
    assert not should_parallelize(8, 1)       # nothing to spread -> serial
    assert should_parallelize(8, 50)
    assert should_parallelize(-1, 50)         # all cores


def test_replicate_reliability_parallel_matches_serial(tmp_path):
    """replicate_feature_reliability(n_jobs>1) walks sessions in parallel but must be identical to
    serial: the EIS ref_grid is fixed by a pre-pass and observation ids are assigned in
    sorted-session order afterwards, so the per-feature variance grouping is unchanged."""
    import pandas as pd
    _build_raw(tmp_path)
    serial = replicate_feature_reliability(tmp_path, n_jobs=1)
    parallel = replicate_feature_reliability(tmp_path, n_jobs=2)
    pd.testing.assert_frame_equal(serial, parallel)


def test_timepoints_auto_derived_and_staggered(tmp_path):
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto")
    tp22 = set(df[df.device == "2-2"]["timepoint"])
    tp23 = set(df[df.device == "2-3"]["timepoint"])
    assert tp22 == {0.0, 7.0}          # first session = day 0, second = +7 days
    assert tp23 == {0.0}               # staggered: 2-3's first session is its own day 0


def test_normipeak_increases_with_concentration(tmp_path):
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto")
    one = df[(df.device == "2-2") & (df.channel == 3) & (df.timepoint == 0.0)]
    by_conc = one.sort_values("concentration")["NormIpeak"].to_numpy()
    assert np.all(np.diff(by_conc) > 0)


def test_extract_dataset_smooths_raw_fscv_by_default(tmp_path):
    # raw-signal Savitzky-Golay smoothing is ON by default; disabling it with
    # smooth_window=0 still runs, and both keep the dose-response monotone.
    import inspect

    from electropycal.features.fscv import FSCV_SMOOTH_WINDOW
    assert inspect.signature(extract_dataset).parameters["smooth_window"].default == FSCV_SMOOTH_WINDOW
    assert FSCV_SMOOTH_WINDOW > 0
    _build_raw(tmp_path)
    default = extract_dataset(tmp_path, band="auto")
    raw = extract_dataset(tmp_path, band="auto", smooth_window=0)
    assert not default.empty and not raw.empty
    for d in (default, raw):
        one = d[(d.device == "2-2") & (d.channel == 3) & (d.timepoint == 0.0)]
        assert np.all(np.diff(one.sort_values("concentration")["NormIpeak"].to_numpy()) > 0)


def test_acceptance_monotonic_keeps_dose_response_channel(tmp_path):
    _build_raw(tmp_path)
    # the synthetic channels have a clean, monotone dose response
    df = extract_dataset(tmp_path, band="auto", peak_method="direct", acceptance="monotonic")
    assert not df.empty
    one = df[(df.device == "2-2") & (df.channel == 3) & (df.timepoint == 0.0)]
    assert set(one["concentration"]) == {100.0, 500.0, 1000.0}   # all doses kept together


def test_qc_columns_present_and_not_features(tmp_path):
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto", acceptance="none")
    qc = {"noise_floor", "snr", "rms_snr", "repeatability_snr", "rep_std",
          "peak_at_edge", "dose_response_r"}
    for col in qc:
        assert col in df.columns
    assert df["snr"].equals(df["repeatability_snr"])          # snr defaults to reproducibility SNR
    # QC metadata (incl. the bool peak_at_edge) must not leak into the model feature matrix
    data = RunData.from_frame(df)
    assert not (qc & set(data.feature_names))


def test_snr_acceptance_uses_reproducibility_not_rms_floor(tmp_path):
    # replicates are near-identical here (tiny 2%/rep scaling) -> high reproducibility SNR ->
    # 'snr' acceptance keeps rows that the residual-dominated RMS floor would have cut.
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto", acceptance="snr", min_norm_snr=3.0)
    assert not df.empty
    assert (df["repeatability_snr"] >= 3.0).all()


def test_acceptance_none_keeps_all_finite_rows(tmp_path):
    _build_raw(tmp_path)
    df_none = extract_dataset(tmp_path, band="auto", acceptance="none")
    df_snr = extract_dataset(tmp_path, band="auto", acceptance="snr")
    assert len(df_none) >= len(df_snr)                            # 'none' never drops more


def _write_fscv_below_bg(path, ch, offset, reps=2):
    """An FSCV file whose current sits a constant ``offset`` BELOW the (100+50v) background —
    so (signal - background) is negative across the whole window and NormIpeak(direct) < 0."""
    v = np.linspace(-0.4, 1.1, 80)
    cur = (100.0 + 50.0 * v) - offset
    hdr, units = [], []
    for r in range(reps):
        lab = f"Fast Cyclic Voltammetry: FCV i vs E Channel {ch}" + ("" if r == 0 else f" [{r}]")
        hdr += [lab, ""]
        units += ["V", "µA"]
    lines = ["Date and time:,2025-01-01 00:00:00", "Notes:", ",,,,",
             ",".join(hdr), ",".join(["Date and time measurement:"] * len(hdr)), ",".join(units)]
    for i in range(len(v)):
        cells = []
        for _r in range(reps):
            cells += [f"{v[i]}", f"{cur[i]}"]
        lines.append(",".join(cells))
    path.write_text("\n".join(lines), encoding="utf-16")


def test_drop_negative_removes_nonphysical_rows(tmp_path):
    # a dose whose signal sits BELOW background at V_ox yields NormIpeak < 0 (non-physical)
    d = tmp_path / "20260715_neurostring_signal"
    d.mkdir()
    _write_eis(d / "9-9_eis_0nm.csv", [3])
    _write_fscv(d / "9-9_fscv_0nm.csv", [3], peak_scale=0.0)      # background
    _write_fscv(d / "9-9_fscv_100nm.csv", [3], peak_scale=10.0)   # real positive peak
    _write_fscv_below_bg(d / "9-9_fscv_500nm.csv", 3, offset=8.0)  # entirely below bg -> negative
    _write_fscv(d / "9-9_fscv_1000nm.csv", [3], peak_scale=20.0)
    keep = extract_dataset(tmp_path, band="auto", acceptance="none", drop_negative=True)
    allrows = extract_dataset(tmp_path, band="auto", acceptance="none", drop_negative=False)
    assert (allrows["NormIpeak"] < 0).any()                      # the negative dose exists
    assert not (keep["NormIpeak"] < 0).any()                     # default drops it
    assert len(keep) < len(allrows)


def test_peak_at_edge_column_present_and_flags_edge(tmp_path):
    # peak far above the window sits near the upper bound -> flagged; a mid-window peak -> not.
    d = tmp_path / "20260715_neurostring_signal"
    d.mkdir()
    _write_eis(d / "8-8_eis_0nm.csv", [3])
    _write_fscv(d / "8-8_fscv_0nm.csv", [3], peak_scale=0.0)
    for dose in (100, 500, 1000):
        _write_fscv(d / f"8-8_fscv_{dose}nm.csv", [3], peak_scale=15.0 * dose / 1000)
    df = extract_dataset(tmp_path, band="auto", acceptance="none")
    assert "peak_at_edge" in df.columns
    assert df["peak_at_edge"].dtype == bool
    assert not df["peak_at_edge"].all()                          # a 0.7 V peak is mid-window


def test_direct_and_chord_both_run(tmp_path):
    _build_raw(tmp_path)
    for pm in ("chord", "direct"):
        df = extract_dataset(tmp_path, band="auto", peak_method=pm, acceptance="none")
        assert not df.empty and np.all(np.isfinite(df["NormIpeak"]))


def test_extract_rejects_bad_params(tmp_path):
    import pytest
    _build_raw(tmp_path)
    with pytest.raises(ValueError):
        extract_dataset(tmp_path, band="auto", peak_method="bogus")
    with pytest.raises(ValueError):
        extract_dataset(tmp_path, band="auto", acceptance="bogus")


def test_from_frame_builds_composite_sensor_identity(tmp_path):
    _build_raw(tmp_path)
    df = extract_dataset(tmp_path, band="auto")
    data = RunData.from_frame(df)
    # 2 devices × 2 channels = 4 distinct sensors
    assert len(np.unique(data.channel)) == 4
    assert data.X.shape[0] == len(df)


def _write_paired_invivo(tmp_path):
    from electropycal.data.synthetic import write_synthetic_invivo_dir
    return write_synthetic_invivo_dir(tmp_path, devices=("3-2",), channels=(3, 5), sessions=(0, 7))


def test_extract_invivo_builds_time_indexed_featureset(tmp_path):
    from electropycal.features.extract import extract_invivo
    df = extract_invivo(_write_paired_invivo(tmp_path))
    assert not df.empty
    for col in ("device", "channel", "timepoint", "time_index", "NormIpeak", "mean_Vpeak", "R_s_f00"):
        assert col in df.columns
    assert "concentration" not in df.columns          # in vivo: unlabeled
    assert set(df["timepoint"]) == {0.0, 7.0}         # two sessions, auto-derived


def test_d0_normalize_frame_zeroes_baseline(tmp_path):
    import numpy as np

    from electropycal.features.extract import extract_invivo
    from electropycal.features.normalize import d0_normalize_frame
    df = extract_invivo(_write_paired_invivo(tmp_path))
    feats = [c for c in df.columns if c not in
             ("device", "channel", "timepoint", "time_index", "NormIpeak")]
    n = d0_normalize_frame(df, feats)
    # a magnitude feature (multiplicative D0-normalization) is ~1.0 at the baseline
    base = n[n.timepoint == n.timepoint.min()]
    assert np.abs(base["R_s_f00"].mean() - 1.0) < 1e-6


def test_avg_fscv_handles_uneven_replicate_lengths():
    import numpy as np

    from electropycal.data.pstrace import PSTraceExport
    from electropycal.features.extract import _avg_fscv
    exp = PSTraceExport(
        fscv={(3, 0): {"voltage": np.linspace(-0.4, 1.1, 80), "current": np.ones(80)},
              (3, 1): {"voltage": np.linspace(-0.4, 1.1, 79), "current": 3 * np.ones(79)}},
        eis={})
    avg = _avg_fscv(exp, 3)
    assert avg is not None and len(avg["current"]) == 79      # truncated to the common length
    assert np.allclose(avg["current"], 2.0)                   # mean of 1 and 3


def test_avg_fscv_max_reps_keeps_first_n():
    import numpy as np

    from electropycal.data.pstrace import PSTraceExport
    from electropycal.features.extract import _avg_fscv
    v = np.linspace(-0.4, 1.1, 80)
    exp = PSTraceExport(  # 3 intended cycles (~1) + 2 erroneous appended (~5)
        fscv={(3, 0): {"voltage": v, "current": np.ones(80)},
              (3, 1): {"voltage": v, "current": np.ones(80)},
              (3, 2): {"voltage": v, "current": np.ones(80)},
              (3, 3): {"voltage": v, "current": 5 * np.ones(80)},
              (3, 4): {"voltage": v, "current": 5 * np.ones(80)}}, eis={})
    assert np.isclose(_avg_fscv(exp, 3, max_reps=None)["current"].mean(), 2.6)  # all 5 → biased
    assert np.isclose(_avg_fscv(exp, 3)["current"].mean(), 1.0)             # default 3 → intended only


def test_replicate_feature_reliability_covers_eis_and_fscv():
    import tempfile

    from electropycal.data.synthetic import write_synthetic_pstrace_dir
    root = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    rel = replicate_feature_reliability(root, band="auto")
    assert not rel.empty
    assert {"eis", "fscv"} <= set(rel["kind"].unique())            # both feature families measured
    assert {"mean_Vpeak", "mean_Ibg", "NormIpeak"} <= set(rel.index)   # FSCV predictors + response
    assert (rel["sigma2_meas"] >= 0).all() and rel["sigma2_meas"].notna().all()
    assert (rel["n_rep"] >= 1).all()


def test_replicate_feature_reliability_keep_restricts_observations():
    import tempfile

    from electropycal.data.synthetic import write_synthetic_pstrace_dir
    root = write_synthetic_pstrace_dir(tempfile.mkdtemp())
    full = replicate_feature_reliability(root, band="auto")
    ds = extract_dataset(root, band="auto", d0_normalize=False)
    one = next(iter({(r.device, int(r.channel), float(r.timepoint)) for r in ds.itertuples()}))
    restricted = replicate_feature_reliability(root, band="auto", keep={one})
    # restricting to a single (device, channel, timepoint) cannot increase observation counts
    common = full.index.intersection(restricted.index)
    assert len(common) > 0
    assert (restricted.loc[common, "n_obs"] <= full.loc[common, "n_obs"]).all()


def test_ref_grid_is_anchored_per_device_type(tmp_path):
    """Each device type anchors its own EIS reference grid.

    The anchor used to be the globally first-sorted quality-passing channel, so one device
    type's measurement grid became the grid every other type had to match to ``rtol=1e-3`` or
    be skipped entirely. Device ``1-6`` (cfme) sorts before ``2-2``, so here it would win and
    take every neurostring row down with it.
    """
    grid_cfme = np.logspace(1, 5, 15)
    grid_neuro = np.logspace(1.05, 5, 15)        # same length, off-grid beyond rtol=1e-3
    assert not np.allclose(grid_cfme, grid_neuro, rtol=1e-3)
    _session(tmp_path, "20260715", "1-6", [3], peak_at=20, devicetype="cfme", freqs=grid_cfme)
    _session(tmp_path, "20260715", "2-2", [3, 5], peak_at=20, freqs=grid_neuro)

    df = extract_dataset(tmp_path, band=(10.0, 100_000.0), d0_normalize=False)
    assert set(df["device"]) == {"1-6", "2-2"}           # neither type starves the other
    grids = df.attrs["extraction_pin"]["extraction"]["ref_grid_hz"]
    assert set(grids) == {"cfme", "neurostring"}
    assert np.allclose(grids["cfme"], grid_cfme, rtol=1e-3)
    assert np.allclose(grids["neurostring"], grid_neuro, rtol=1e-3)


def test_empty_session_is_reported_not_silent(tmp_path):
    """A session that yields no rows must say so: absent from the featureset otherwise looks
    identical to never measured, which is how three zero-row sessions went unnoticed."""
    import warnings

    import pytest

    from electropycal.features.extract import EmptySessionError, EmptySessionWarning
    _build_raw(tmp_path)
    # strip 2-3's EIS -> that session can produce nothing (an incomplete fetch looks like this)
    (tmp_path / "20260722_neurostring_signal" / "2-3_eis_0nm.csv").unlink()

    with pytest.warns(EmptySessionWarning, match="2-3@2026-07-22"):
        df = extract_dataset(tmp_path, band="auto", d0_normalize=False)
    assert "2-3" not in set(df["device"])                  # still absent, but now announced

    with pytest.raises(EmptySessionError, match="produced 0 rows"):
        extract_dataset(tmp_path, band="auto", d0_normalize=False, on_empty_session="raise")

    with warnings.catch_warnings():
        warnings.simplefilter("error")                     # 'ignore' restores the old silence
        extract_dataset(tmp_path, band="auto", d0_normalize=False, on_empty_session="ignore")

    with pytest.raises(ValueError, match="on_empty_session must be"):
        extract_dataset(tmp_path, band="auto", on_empty_session="bogus")


def test_empty_session_distinguishes_missing_files_from_failed_channels(tmp_path):
    """The two causes need different responses, so they are reported differently."""
    import pytest

    from electropycal.features.extract import EmptySessionWarning
    _build_raw(tmp_path)
    (tmp_path / "20260722_neurostring_signal" / "2-3_eis_0nm.csv").unlink()
    with pytest.warns(EmptySessionWarning, match="no EIS file"):
        extract_dataset(tmp_path, band="auto", d0_normalize=False)

    # channels present but all off-grid for their device type -> the other message
    other = tmp_path / "other"
    other.mkdir()
    _session(other, "20260715", "2-2", [3], peak_at=20, freqs=np.logspace(1, 5, 15))
    _session(other, "20260716", "2-2", [3], peak_at=20, freqs=np.logspace(1.05, 5, 15))
    with pytest.warns(EmptySessionWarning, match="dropped by a gate"):
        extract_dataset(other, band=(10.0, 100_000.0), d0_normalize=False)


def _two_type_tree(root):
    """cfme device 1-6 (sorts first) + neurostring 2-2, on deliberately different grids."""
    _session(root, "20260715", "1-6", [3], peak_at=20, devicetype="cfme",
             freqs=np.logspace(1, 5, 15))
    _session(root, "20260715", "2-2", [3, 5], peak_at=20, freqs=np.logspace(1.05, 5, 15))


def test_device_types_filters_and_is_recorded(tmp_path):
    _two_type_tree(tmp_path)
    both = extract_dataset(tmp_path, band=(10.0, 100_000.0), d0_normalize=False)
    assert set(both["device"]) == {"1-6", "2-2"}
    assert both.attrs["extraction_pin"]["extraction"]["device_types"] is None    # no filter

    only = extract_dataset(tmp_path, band=(10.0, 100_000.0), d0_normalize=False,
                           device_types=["neurostring"])
    assert set(only["device"]) == {"2-2"}
    e = only.attrs["extraction_pin"]["extraction"]
    assert e["device_types"] == ["neurostring"]
    assert set(e["ref_grid_hz"]) == {"neurostring"}      # cfme never gets an anchor


def test_device_types_reaches_the_band_pre_pass(tmp_path):
    """Filtering is not a late row filter: it has to reach the ``band="auto"`` percentile, or
    excluded device types would still help set the band the kept ones are measured in.

    Asserts the argument is forwarded rather than that the band changes numerically: this
    fixture's spectra never go inductive, so ``recommended_band`` falls back to the same value
    for every subset of it. ``test_recommended_band_device_type_filter`` covers the filtering
    itself.
    """
    import electropycal.data.inventory as inv
    _two_type_tree(tmp_path)
    seen = []
    real = inv.recommended_band
    inv.recommended_band = lambda root, **kw: (seen.append(kw.get("devicetype")),
                                               real(root, **kw))[1]
    try:
        extract_dataset(tmp_path, band="auto", d0_normalize=False)
        extract_dataset(tmp_path, band="auto", d0_normalize=False,
                        device_types=["neurostring"])
    finally:
        inv.recommended_band = real
    assert seen == [None, ["neurostring"]]


def test_recommended_band_device_type_filter():
    """The band percentile can be restricted to a set of device types, not just one name."""
    from pathlib import Path

    import pytest

    from electropycal.data.inventory import recommended_band
    # This one needs a real multi-session PSTrace corpus, which is far too large to bundle.
    # Resolved absolutely rather than from the cwd, and skipped rather than failed when it is
    # not there, so the suite passes from a plain install.
    root = Path(__file__).resolve().parents[1] / "data" / "invitro"
    if not root.exists():
        pytest.skip("needs a local raw PSTrace corpus; none bundled")
    both = recommended_band(root)
    assert recommended_band(root, devicetype="neurostring") == both
    assert recommended_band(root, devicetype=["neurostring"]) == both
    # no cfme sessions there, so restricting to it leaves no onsets -> fallback
    assert recommended_band(root, devicetype=["cfme"]) == (10.0, 100_000.0) != both


def test_pinned_device_types_are_reproduced_and_conflicts_raise(tmp_path):
    import pytest

    from electropycal.features.pin import PinMismatch
    _two_type_tree(tmp_path)
    pin_path = tmp_path / "pin.json"
    only = extract_dataset(tmp_path, band=(10.0, 100_000.0), d0_normalize=False,
                           device_types=["neurostring"], pin_out=pin_path)
    # the pinned selection is reapplied without being restated
    again = extract_dataset(tmp_path, d0_normalize=False, pin=pin_path)
    pd.testing.assert_frame_equal(only.reset_index(drop=True), again.reset_index(drop=True))
    with pytest.raises(PinMismatch, match="conflicts with the pinned selection"):
        extract_dataset(tmp_path, d0_normalize=False, pin=pin_path, device_types=["cfme"])
