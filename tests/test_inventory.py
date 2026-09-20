"""Raw-dataset inventory + full-cycle FSCV handling."""

import numpy as np

from electropycal.data.inventory import (band_retention_curve, channel_survival, index_raw,
                                        quality_filtering)
from electropycal.data.synthetic import write_synthetic_pstrace_dir
from electropycal.features.fscv import anodic_sweep, norm_ipeak


def test_index_raw_covers_channeltest_and_signal(tmp_path):
    root = write_synthetic_pstrace_dir(tmp_path)
    idx = index_raw(root)
    assert set(idx.testtype) == {"channeltest", "signal"}
    assert {"device", "timepoint", "channel", "signaltype", "dose", "eis_valid", "path"} <= set(idx.columns)
    # EIS rows carry a quality verdict; FSCV rows don't
    assert idx[idx.signaltype == "eis"]["eis_valid"].notna().all()


def test_index_raw_path_points_at_the_right_file(tmp_path):
    # regression: every row's `path` must be the file it describes (not the last-globbed
    # file), and reading it back must yield the row's signaltype/channel.
    from electropycal.data.pstrace import parse_filename, read_pstrace
    idx = index_raw(write_synthetic_pstrace_dir(tmp_path))
    assert idx["path"].nunique() > 1
    for r in idx.sample(min(8, len(idx)), random_state=0).itertuples():
        nm = parse_filename(r.path.rsplit("/", 1)[-1])
        assert nm["signaltype"] == r.signaltype
        exp = read_pstrace(r.path)
        blocks = exp.eis if r.signaltype == "eis" else exp.fscv
        assert any(ch == r.channel for (ch, _) in blocks)


def test_band_retention_curve_monotone_nonincreasing(tmp_path):
    root = write_synthetic_pstrace_dir(tmp_path)
    df = band_retention_curve(root)
    assert {"upper_hz", "kept", "dropped_vs_lowest", "kept_frac"} <= set(df.columns)
    assert len(df) > 0
    # raising the ceiling can only drop channel-timepoints, never add them
    assert df.sort_values("upper_hz")["kept"].is_monotonic_decreasing
    assert (df["kept"] >= 0).all() and (df["kept_frac"] <= 1.0).all()


def test_channel_survival_shows_dropout(tmp_path):
    idx = index_raw(write_synthetic_pstrace_dir(tmp_path))
    surv = channel_survival(idx)
    s = surv[surv.device == "2-2"].sort_values("timepoint")["n_survived"].to_numpy()
    assert s[0] >= s[-1] and s[0] > s[-1]        # channels drop out over time


def test_quality_filtering_flags_rejections(tmp_path):
    idx = index_raw(write_synthetic_pstrace_dir(tmp_path))
    qf = quality_filtering(idx)
    assert (qf["n_rejected"] > 0).any()           # the injected late failure is caught
    assert (qf["n_valid"] + qf["n_rejected"] == qf["n_channels"]).all()


def test_inductive_onset_finds_first_nonnegative_imag():
    from electropycal.data.quality import inductive_onset
    f = np.array([10.0, 100.0, 1000.0, 10000.0])
    assert inductive_onset(f, np.array([-5.0, -3.0, -1.0, 2.0])) == 10000.0   # first Im>=0
    assert inductive_onset(f, np.array([2.0, -3.0, -1.0, -0.5])) == 10.0       # inductive at low f
    assert np.isnan(inductive_onset(f, np.array([-5.0, -3.0, -1.0, -0.5])))    # stays capacitive
    # order-independent: unsorted input gives the same onset frequency
    assert inductive_onset(f[::-1], np.array([2.0, -1.0, -3.0, -5.0])) == 10000.0


def test_recommended_band_defaults_when_capacitive(tmp_path):
    # the synthetic Randles cell never goes inductive -> fall back to the default upper bound
    from electropycal.data.inventory import recommended_band
    root = write_synthetic_pstrace_dir(tmp_path)
    lo, hi = recommended_band(root)
    assert lo == 10.0 and hi == 100_000.0
    assert recommended_band(root, devicetype="neurostring")[1] == 100_000.0     # devicetype filter
    assert recommended_band(root, devicetype="cfme") == (10.0, 100_000.0)       # no such folders


def _write_onset_eis(path, onset_by_ch, grid=(10.0, 100.0, 1000.0, 10_000.0, 100_000.0)):
    # minimal EIS export: each channel is capacitive (Im<0) below its onset and inductive
    # (Im>=0, i.e. stored Z''<=0) at/above it — so inductive_onset(ch) == onset_by_ch[ch].
    L = ["Date and time:,2025-01-01 00:00:00", "Notes:", "Measurement:,Impedance Spectroscopy",
         "Notes:,", "Date and time:,2025-01-01"]
    for ch, onset in onset_by_ch.items():
        L.append(f"CH {ch}: Fixed at {len(grid)} freqs")
        L.append("freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F")
        for f in sorted(grid, reverse=True):
            im = 50.0 if f >= onset else -1000.0            # Im>=0 == inductive
            L.append(f"{f},0,1,{(1e6 + im**2) ** 0.5},1000,{-im},1e-9")   # stored Z'' = -Im
    path.write_text("\n".join(L), encoding="utf-16")


def test_recommended_band_percentile_lets_poor_channels_fail(tmp_path):
    # 18 good channels go inductive only at 100 kHz; 2 poor ones at 1 kHz. The strict-min band
    # (onset_percentile=0) drops to protect the 2 poor channels; the default (10th pct) keeps a
    # wider band and lets those 2 fall in-band -> fail EIS.1 instead of shrinking it for all.
    from electropycal.data.inventory import recommended_band
    from electropycal.data.quality import eis_quality
    from electropycal.data.pstrace import read_pstrace
    sig = tmp_path / "20260101_neurostring_signal"
    sig.mkdir(parents=True)
    onsets = {ch: (1000.0 if ch in (1, 2) else 100_000.0) for ch in range(1, 21)}
    _write_onset_eis(sig / "5-5_eis_0nM.csv", onsets)

    strict = recommended_band(tmp_path, onset_percentile=0)[1]
    default = recommended_band(tmp_path)[1]                     # onset_percentile=10
    assert strict < default                                     # tail no longer drags the band down
    assert strict < 1000.0 <= default                           # poor onset in-band only for default
    # under the default band the 2 poor channels fail EIS.1; a good one passes
    exp = read_pstrace(sig / "5-5_eis_0nM.csv")
    band = (10.0, default)
    q = lambda ch: eis_quality(*(lambda s: (s["freq"], s["z_real"], s["z_imag"]))(exp.eis[(ch, 0)]), band=band)
    assert not q(1)["A_capacitive"] and not q(2)["A_capacitive"]
    assert q(10)["A_capacitive"]


def test_normipeak_invariant_to_full_triangle_sweep():
    up = np.linspace(-0.4, 1.1, 80)
    v_full = np.concatenate([up, up[::-1]])       # anodic + cathodic
    bg_full = 100.0 + 50.0 * v_full
    peak = 20.0 * np.exp(-((v_full - 0.7) / 0.05) ** 2)
    sig_full = bg_full + peak
    ni_full = norm_ipeak(sig_full, bg_full, v_full)
    # feeding the already-anodic sweep gives the same answer (extraction is idempotent)
    va, sa, ba = anodic_sweep(v_full, sig_full, bg_full)
    ni_anodic = norm_ipeak(sa, ba, va)
    assert np.isfinite(ni_full) and ni_full > 0
    assert np.isclose(ni_full, ni_anodic)


def test_channel_quality_report_verdicts_and_dose_table(tmp_path):
    from electropycal.data.inventory import channel_quality_report
    ct, dt = channel_quality_report(write_synthetic_pstrace_dir(tmp_path))
    # one row per measured channel-timepoint, with per-check + overall verdicts
    assert {"device", "timepoint", "channel", "eis_A", "eis_B", "eis_C",
            "fscv_monotonic", "fscv_noise", "overall_valid", "fail_reasons"} <= set(ct.columns)
    # F.2 (noise floor) is reported but does NOT gate overall_valid (acceptance="monotonic")
    from electropycal.data.inventory import QUALITY_CHECKS
    assert "fscv_noise" not in QUALITY_CHECKS
    # the injected late EIS failure (check C) is present and makes the channel invalid
    bad = ct[(ct.eis_C == False)]
    assert len(bad) and (~bad.overall_valid).all()
    assert bad.fail_reasons.str.contains("EIS.3").all()
    # a clean early channel passes everything
    good = ct[(ct.timepoint == 0)]
    assert good.overall_valid.all()
    # overall_valid iff all four checks pass (paired channels)
    paired = ct[ct.has_eis & ct.has_fscv]
    expect = paired.eis_A & paired.eis_B & paired.eis_C & paired.fscv_monotonic
    assert (paired.overall_valid == expect).all()
    # dose table: per concentration, averaging happened (std defined), noise floor present,
    # plus the peak-window diagnostics
    assert {"concentration", "NormIpeak_mean", "NormIpeak_std", "noise_floor",
            "snr", "rms_snr", "repeatability_snr", "conc_pass", "v_peak",
            "peak_at_edge", "negative"} <= set(dt.columns)
    assert (dt.NormIpeak_std >= 0).all()
    # peak-in-window diagnostic present and non-gating (overall_valid ignores it)
    assert "fscv_peak_inwindow" in ct.columns
    assert (paired.overall_valid == expect).all()   # still == the four gating checks only


def test_channel_quality_report_devices_filter(tmp_path):
    from electropycal.data.inventory import channel_quality_report
    root = write_synthetic_pstrace_dir(tmp_path)
    ct_all, _ = channel_quality_report(root)
    picks = sorted(ct_all.device.unique())[:1]                 # restrict to one device
    ct_sel, dt_sel = channel_quality_report(root, devices=picks)
    assert set(ct_sel.device.unique()) == set(picks)           # only the selected device processed
    assert set(dt_sel.device.unique()) <= set(picks)
    assert len(ct_sel) < len(ct_all)


def test_channel_quality_report_progress_logs_without_changing_results(tmp_path, capsys):
    from electropycal.data.inventory import channel_quality_report
    root = write_synthetic_pstrace_dir(tmp_path)
    ct_q, _ = channel_quality_report(root)
    assert capsys.readouterr().out == ""                       # silent by default
    ct_v, _ = channel_quality_report(root, progress=True)
    out = capsys.readouterr().out
    assert "channel_quality_report" in out and "ETA" in out    # timestamped progress emitted
    assert ct_v.equals(ct_q)                                   # logging doesn't change the result


def test_channel_quality_report_parallel_matches_serial(tmp_path):
    """n_jobs>1 QCs sessions in parallel worker processes but must be byte-identical to serial —
    the sessions are independent (every channel kept, no shared ref_grid) and rows are
    concatenated in sorted-session order."""
    import pandas as pd
    from electropycal.data.inventory import channel_quality_report
    root = write_synthetic_pstrace_dir(tmp_path)
    ct_s, dt_s = channel_quality_report(root, n_jobs=1)
    ct_p, dt_p = channel_quality_report(root, n_jobs=2)
    pd.testing.assert_frame_equal(ct_s.reset_index(drop=True), ct_p.reset_index(drop=True))
    pd.testing.assert_frame_equal(dt_s.reset_index(drop=True), dt_p.reset_index(drop=True))


def test_channel_quality_report_nonpaired_fails_without_crashing(tmp_path):
    # EIS present for a channel with NO matching FSCV background, and vice-versa:
    # the quality table must mark them incomplete/invalid, not raise.
    import datetime
    from electropycal.data.synthetic import _write_eis_file, _write_fscv_file
    from electropycal.data.inventory import channel_quality_report
    sig = tmp_path / "20260101_neurostring_signal"
    sig.mkdir(parents=True)
    _write_eis_file(sig / "9-9_eis_0nm.csv", channels=[3, 8])          # EIS ch3, ch8
    _write_fscv_file(sig / "9-9_fscv_0nm.csv", channels=[3, 5], conc_nM=0.0)  # FSCV ch3, ch5
    _write_fscv_file(sig / "9-9_fscv_100nm.csv", channels=[3, 5], conc_nM=100.0)
    ct, _ = channel_quality_report(sig.parent)
    row = lambda ch: ct[(ct.channel == ch)].iloc[0]
    assert row(8).has_eis and not row(8).has_fscv and not row(8).overall_valid  # EIS-only
    assert row(5).has_fscv and not row(5).has_eis and not row(5).overall_valid  # FSCV-only
    assert "incomplete" in row(8).fail_reasons and "incomplete" in row(5).fail_reasons


def test_index_raw_records_per_check_eis_flags(tmp_path):
    from electropycal.data.inventory import index_raw
    from electropycal.data.synthetic import write_synthetic_pstrace_dir
    idx = index_raw(write_synthetic_pstrace_dir(tmp_path, channels=(3, 5, 6)))
    for col in ("eis_A", "eis_B", "eis_C"):
        assert col in idx.columns
    eis = idx[idx.signaltype == "eis"]
    # valid == (A and B and C) on rows where checks ran
    ran = eis.dropna(subset=["eis_A"])
    assert (ran["eis_valid"] == (ran.eis_A & ran.eis_B & ran.eis_C)).all()
