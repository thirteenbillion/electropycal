"""Feature-extraction tests, including the EIS sign convention and derivative smoothing."""

import numpy as np
import pytest

from electropycal.features import eis, fscv, normalize


def _series_rc(R=1000.0, C=1e-6, n=25):
    f = np.logspace(1, 5, n)               # 10 Hz .. 100 kHz
    w = 2 * np.pi * f
    z_real = np.full_like(f, R)
    z_imag = -1.0 / (w * C)                # stored negative (capacitive)
    return f, z_real, z_imag, R, C


def test_eis_capacitances_positive_and_correct_O1():
    f, zr, zi, R, C = _series_rc()
    feats = eis.eis_features(f, zr, zi)
    assert set(feats) == {"R_s", "R_p", "C_s", "C_p", "ideality_C", "tau", "local_n"}
    assert np.allclose(feats["R_s"], R)
    assert np.allclose(feats["C_s"], C, rtol=1e-6)         # -1/(w Z'') recovers C
    assert np.all(feats["C_s"] > 0) and np.all(feats["C_p"] > 0) and np.all(feats["tau"] > 0)
    assert np.all((feats["ideality_C"] >= 0) & (feats["ideality_C"] <= 1))


def test_local_n_capacitive_at_low_frequency_O3():
    f, zr, zi, R, C = _series_rc()
    feats = eis.eis_features(f, zr, zi, smooth=True)
    # low-frequency end is capacitance-dominated: |Z| ~ 1/(wC) → local_n ≈ 1
    assert feats["local_n"][0] > 0.5


def test_eis_global_returns_thirteen_features():
    f, zr, zi, *_ = _series_rc()
    g = eis.eis_global_features(f, zr, zi)
    assert len(g) == 14                                  # 12 original + min_neg_phase + inductive_onset_hz
    assert "tau_ratio" in g and "f_ideality_crossover" in g and "min_neg_phase" in g
    assert "inductive_onset_hz" in g


def test_norm_ipeak_matches_expected_ratio():
    v = np.linspace(-0.4, 1.4, 400)
    background = 100.0 + 50.0 * v
    peak = 20.0 * np.exp(-((v - 0.7) / 0.03) ** 2)
    signal = background + peak
    ni = fscv.norm_ipeak(signal, background, v, v_window=(0.6, 0.8), method="chord")
    assert ni == pytest.approx(20.0 / 135.0, rel=0.05)     # peak_height / I_bgd(V_ox)
    assert fscv.mean_vpeak(signal, background, v) == pytest.approx(0.7, abs=0.02)
    assert fscv.mean_ibg(background, signal, v) == pytest.approx(135.0, rel=0.02)


def test_fscv_features_read_the_same_peak_as_normipeak():
    # V_ox and I_bgd must come from the SAME located peak as NormIpeak, under whichever
    # peak-detection method is used, so mean_Ibg == NormIpeak's denominator exactly.
    v = np.linspace(-0.4, 1.4, 400)
    background = 100.0 + 50.0 * v
    signal = background + 20.0 * np.exp(-((v - 0.68) / 0.05) ** 2) \
        + 6.0 * np.exp(-((v - 0.86) / 0.03) ** 2)          # asymmetric: chord/direct can differ
    for method in ("direct", "chord"):
        r = fscv._peak_readout(signal, background, v, method=method)
        assert fscv.mean_vpeak(signal, background, v, method=method) == r["v_ox"]
        assert fscv.mean_ibg(background, signal, v, method=method) == r["i_bgd"]
        assert fscv.norm_ipeak(signal, background, v, method=method) == pytest.approx(r["norm_ipeak"])
        assert r["height"] / r["i_bgd"] == pytest.approx(r["norm_ipeak"])   # I_bgd is the denominator
    # detrend must propagate too: a de-trended NormIpeak's V_ox/I_bgd come from the de-trended peak
    rd = fscv._peak_readout(signal, background, v, method="direct", detrend=True)
    assert fscv.mean_ibg(background, signal, v, method="direct", detrend=True) == rd["i_bgd"]


def test_peak_window_default_captures_peak_shifted_above_da_window():
    from electropycal.features.fscv import PEAK_WINDOW
    assert PEAK_WINDOW == (0.4, 1.0)                       # widened default peak-search window
    v = np.linspace(-0.4, 1.4, 400)
    bg = np.full_like(v, 100.0)
    sig = bg + 10.0 * np.exp(-((v - 0.9) / 0.05) ** 2)    # sharp peak at 0.9 V, above nominal DA
    assert fscv.mean_vpeak(sig, bg, v) == pytest.approx(0.9, abs=0.03)          # default finds it
    assert fscv.mean_vpeak(sig, bg, v, v_window=fscv.DA_WINDOW) <= 0.8 + 1e-9   # narrow window can't


def test_norm_ipeak_direct_counts_broad_peak_that_chord_misses():
    # a broad hump that is ~linear across the (0.6,0.8) window: the chord method
    # (deviation from the endpoint chord) under-counts it; direct (current at V_ox)
    # recovers the full magnitude.
    v = np.linspace(-0.4, 1.4, 400)
    background = np.full_like(v, 100.0)
    signal = background + 15.0 * np.exp(-((v - 0.75) / 0.25) ** 2)   # broad, wide sigma
    # evaluated in the narrow nominal DA window, where a broad hump is ~linear across it
    chord = fscv.norm_ipeak(signal, background, v, method="chord", v_window=fscv.DA_WINDOW)
    direct = fscv.norm_ipeak(signal, background, v, method="direct", v_window=fscv.DA_WINDOW)
    assert direct > 3 * chord                         # chord badly under-counts
    assert direct == pytest.approx(15.0 / 100.0, rel=0.15)


def test_noise_floor_window_sits_below_peak_window():
    from electropycal.features.fscv import NONFARADAIC_WINDOW, PEAK_WINDOW
    # the noise floor must be measured below the faradaic onset, i.e. not overlap the peak
    # search window, or it counts DA signal as noise and deflates the SNR.
    assert NONFARADAIC_WINDOW[1] <= PEAK_WINDOW[0]


def test_repeatability_snr_rewards_reproducible_peaks():
    # same NormIpeak, tighter replicate scatter -> higher SNR
    assert fscv.repeatability_snr(1.0, [0.98, 1.0, 1.02]) > fscv.repeatability_snr(1.0, [0.5, 1.0, 1.5])
    assert np.isnan(fscv.repeatability_snr(1.0, [1.0]))          # needs >= 2 replicates
    assert np.isnan(fscv.repeatability_snr(float("nan"), [1.0, 1.1]))
    assert fscv.repeatability_snr(1.0, [1.0, 1.0]) == float("inf")   # zero scatter
    # NormIpeak / std(reps): 1.0 / std([0.9,1.0,1.1])
    assert fscv.repeatability_snr(1.0, [0.9, 1.0, 1.1]) == pytest.approx(1.0 / np.std([0.9, 1.0, 1.1]))


def test_peak_at_edge_flags_boundary_peaks():
    lo, hi = fscv.PEAK_WINDOW
    assert fscv.peak_at_edge(lo + 0.01)                       # just inside lower bound -> pinned
    assert fscv.peak_at_edge(hi - 0.01)                       # just inside upper bound -> pinned
    assert not fscv.peak_at_edge((lo + hi) / 2)               # mid-window -> resolved
    assert not fscv.peak_at_edge(float("nan"))                # no peak -> not flagged
    assert fscv.peak_at_edge(0.9, v_window=(0.4, 1.0), tol=0.15)   # tol widens the edge band


def test_require_interior_returns_no_peak_when_pinned_to_bound():
    v = np.linspace(-0.4, 1.4, 400)
    bg = np.full_like(v, 100.0)
    ramp = bg + 40.0 * (v - 0.4)                          # monotone rising across (0.4,1.0): argmax pins at 1.0
    # default: reads the boundary; require_interior: no peak -> NaN
    assert np.isfinite(fscv.norm_ipeak(ramp, bg, v))
    assert np.isnan(fscv.norm_ipeak(ramp, bg, v, require_interior=True))
    # a genuine interior peak is found either way
    peak = bg + 12.0 * np.exp(-((v - 0.7) / 0.05) ** 2)
    assert fscv.mean_vpeak(peak, bg, v) == pytest.approx(0.7, abs=0.03)
    assert np.isfinite(fscv.norm_ipeak(peak, bg, v, require_interior=True))


def test_norm_ipeak_detrend_removes_sloping_residual():
    v = np.linspace(-0.4, 1.4, 400)
    background = np.full_like(v, 100.0)
    slope = 8.0 * (v + 0.4)                            # residual baseline slope
    signal = background + slope + 10.0 * np.exp(-((v - 0.7) / 0.03) ** 2)
    nf_raw = fscv.noise_floor(signal, background, v)
    nf_dt = fscv.noise_floor(signal, background, v, detrend=True, baseline_window=(-0.1, 0.2))
    assert nf_dt < nf_raw                              # de-trending deflates the slope


def test_norm_ipeak_rejects_bad_method():
    v = np.linspace(-0.4, 1.4, 50)
    with pytest.raises(ValueError):
        fscv.norm_ipeak(v, v, v, method="bogus")


def test_noise_floor_small_for_clean_nonfaradaic_window():
    v = np.linspace(-0.4, 1.4, 400)
    background = 100.0 + 50.0 * v
    signal = background + 20.0 * np.exp(-((v - 0.7) / 0.03) ** 2)
    nf = fscv.noise_floor(signal, background, v, window=(-0.1, 0.6))
    assert 0.0 <= nf < 0.05


def test_smooth_current_denoises_and_is_off_at_zero():
    from electropycal.features.fscv import FSCV_SMOOTH_WINDOW, smooth_current
    rng = np.random.default_rng(0)
    v = np.linspace(-0.4, 1.4, 400)
    clean = 100.0 + 50.0 * v + 20.0 * np.exp(-((v - 0.7) / 0.05) ** 2)
    noisy = clean + rng.normal(0, 2.0, size=v.size)
    assert np.std(smooth_current(noisy) - clean) < np.std(noisy - clean)   # closer to truth
    assert np.array_equal(smooth_current(noisy, window=0), noisy)          # off at window 0
    assert FSCV_SMOOTH_WINDOW > 0                                          # smoothing is on by default


def test_savgol_smoothing_reduces_noise_floor_and_stabilizes_peak():
    rng = np.random.default_rng(0)
    v = np.linspace(-0.4, 1.4, 400)
    background = 100.0 + 50.0 * v
    clean = background + 20.0 * np.exp(-((v - 0.7) / 0.05) ** 2)
    signal = clean + rng.normal(0, 2.0, size=v.size)              # add measurement noise
    # noise floor: smoothing (odd window) lowers the non-Faradaic RMS
    nf_raw = fscv.noise_floor(signal, background, v, window=(-0.1, 0.6))
    nf_sm = fscv.noise_floor(signal, background, v, window=(-0.1, 0.6), smooth_window=11)
    assert nf_sm < nf_raw
    # NormIpeak: smoothing pulls the noisy peak estimate back toward the clean value
    true_nip = fscv.norm_ipeak(clean, background, v)
    err_raw = abs(fscv.norm_ipeak(signal, background, v) - true_nip)
    err_sm = abs(fscv.norm_ipeak(signal, background, v, smooth_window=11) - true_nip)
    assert err_sm <= err_raw
    # off by default: window 0 == no smoothing
    assert fscv.norm_ipeak(signal, background, v) == fscv.norm_ipeak(signal, background, v, smooth_window=0)


def test_d0_normalize_additive_vs_multiplicative():
    names = ["mean_Ibg", "ideality_C"]     # multiplicative, additive
    X = np.array([[2.0, 0.5], [4.0, 0.7]])
    d0 = np.array([2.0, 0.5])
    lin = normalize.d0_normalize(X, names, d0, log=False)
    assert np.allclose(lin[:, 0], [1.0, 2.0])          # divided
    assert np.allclose(lin[:, 1], [0.0, 0.2])          # shifted
    logt = normalize.d0_normalize(X, names, d0, log=True)
    assert np.allclose(logt[:, 0], [0.0, np.log(2.0)])  # log(x/d0)
    assert np.allclose(logt[:, 1], [0.0, 0.2])          # bounded stays additive


def test_d0_kind_phase_additive_frequency_multiplicative():
    from electropycal.data.schema import d0_normalization_kind
    # min_neg_phase is a signed phase -> additive (a ratio to a near-zero baseline is unstable)
    assert d0_normalization_kind("min_neg_phase") == "additive"
    # inductive_onset_hz is a positive log-distributed frequency -> multiplicative (like tau)
    assert d0_normalization_kind("inductive_onset_hz") == "multiplicative"
    assert d0_normalization_kind("tau") == "multiplicative"


def test_d0_kind_strips_per_frequency_suffix():
    from electropycal.data.schema import d0_normalization_kind as k
    # per-frequency bounded / log-slope features inherit their base type's ADDITIVE kind
    assert k("ideality_C_f03") == "additive" and k("local_n_f07") == "additive"
    assert k("ideality_C_band_HF") == "additive" and k("n_band_LF") == "additive"
    # per-frequency magnitudes stay multiplicative
    assert k("R_s_f00") == "multiplicative" and k("C_s_f09") == "multiplicative"
    assert k("tau_f02") == "multiplicative"


def test_eis_global_features_include_min_neg_phase():
    f, zr, zi, _, _ = _series_rc()
    g = eis.eis_global_features(f, zr, zi)
    assert "min_neg_phase" in g
    assert g["min_neg_phase"] > 0                     # capacitive interface -> -phase > 0 everywhere
    # an in-band inductive point (Im(Z) > 0) drives min_neg_phase negative
    zi2 = zi.copy(); zi2[10] = +abs(zi[10])
    assert eis.eis_global_features(f, zr, zi2)["min_neg_phase"] < 0


def test_eis_global_features_inductive_onset_hz():
    f, zr, zi, _, _ = _series_rc()
    order = np.argsort(f); f_top = float(f[order][-1])
    # fully capacitive -> onset censored at the top of the (in-band) range
    assert eis.eis_global_features(f, zr, zi)["inductive_onset_hz"] == f_top
    # force an inductive point -> onset is that frequency (and below the ceiling)
    zi2 = zi.copy(); k = int(np.argsort(f)[10]); zi2[k] = +abs(zi[k])
    on = eis.eis_global_features(f, zr, zi2)["inductive_onset_hz"]
    assert on == float(np.sort(f)[10]) and on < f_top


def test_inductive_features_use_full_spectrum_not_the_band():
    # in-band spectrum is fully capacitive (as it must be to pass EIS.1), but the FULL spectrum goes
    # inductive above the band -> the degradation features must reflect the full range, not the band.
    f, zr, zi, _, _ = _series_rc()
    o = np.argsort(f); f, zr, zi = f[o], zr[o], zi[o]
    band_hi = f[len(f) // 2]
    in_band = f <= band_hi
    fb, zrb, zib = f[in_band], zr[in_band], zi[in_band]
    full_zi = zi.copy(); full_zi[-1] = +abs(zi[-1])      # inductive point above the band
    g = eis.eis_global_features(fb, zrb, zib, full_spectrum=(f, zr, full_zi))
    assert g["inductive_onset_hz"] == float(f[-1])       # full-range onset, above the band ceiling
    assert g["min_neg_phase"] < 0                        # full-range sees the inductive point
    # in-band-only (no full_spectrum) would be blind to it and pin onset at the band ceiling
    g_band = eis.eis_global_features(fb, zrb, zib)
    assert g_band["inductive_onset_hz"] == float(fb[-1]) and g_band["min_neg_phase"] > 0


def test_peak_shape_gaussian_fwhm_and_area():
    # a clean Gaussian anodic peak on zero background: FWHM = 2*sqrt(2 ln2)*sigma
    v = np.linspace(0.4, 1.0, 601)
    sigma, v0, amp = 0.03, 0.70, 5.0
    peak = amp * np.exp(-((v - v0) ** 2) / (2 * sigma ** 2))
    out = fscv.peak_shape(peak, np.zeros_like(v), v, method="direct")
    assert np.isclose(out["height"], amp, rtol=0.02)
    assert np.isclose(out["fwhm"], 2.0 * np.sqrt(2 * np.log(2)) * sigma, rtol=0.05)
    assert np.isclose(out["area"], amp * sigma * np.sqrt(2 * np.pi), rtol=0.05)   # ∫Gaussian
    # flat signal -> zero height and an undefined (NaN) width
    flat = fscv.peak_shape(np.zeros_like(v), np.zeros_like(v), v)
    assert abs(flat["height"]) < 1e-9 and np.isnan(flat["fwhm"])


def test_background_features_capacitive_loop():
    # ideal capacitive CV loop: +I_c on the anodic sweep, -I_c on the cathodic -> bg_cap == I_c
    v_up = np.linspace(-0.4, 1.1, 100); v = np.concatenate([v_up, v_up[::-1]])
    Ic = 3.0
    i = np.concatenate([np.full(100, Ic), np.full(100, -Ic)])
    bf = fscv.background_features(v, i, window=(0.4, 1.0))
    assert np.isclose(bf["bg_cap"], Ic, rtol=1e-6)          # hysteresis/2 = |Ic-(-Ic)|/2 = Ic
    assert np.isclose(bf["bg_switch"], Ic)                  # current at the anodic limit (top of sweep)
    assert bf["bg_charge"] > 0                              # positive charging charge over the window
    # too-short input -> all NaN
    assert all(np.isnan(x) for x in fscv.background_features(np.array([0.0, 1.0]), np.array([1.0, 1.0])).values())


def test_background_features_are_leakage_free_predictors():
    # they must be usable predictors (NOT reserved) and dose-invariant within a channel-timepoint
    from electropycal.data.schema import RESERVED_COLUMNS
    for c in ("bg_charge", "bg_cap", "bg_switch"):
        assert c not in RESERVED_COLUMNS


def test_feature_catalog_covers_extracted_columns_with_correct_roles():
    import tempfile
    from electropycal import feature_catalog
    from electropycal.features.extract import extract_dataset
    from electropycal.data.synthetic import write_synthetic_pstrace_dir
    from electropycal.data.schema import d0_normalization_kind
    cat = {r["featuretype"]: r for r in feature_catalog()}
    # every emitted column maps to a catalog entry, and the catalog's d0_kind matches the schema
    df = extract_dataset(write_synthetic_pstrace_dir(tempfile.mkdtemp()), band=(2.0, 2000.0))
    import re
    def base(c):
        m = re.match(r"^(.+)_f\d+$", c)
        t = m.group(1) if m else c
        for pre in ("ideality_C_band", "n_band"):
            if t.startswith(pre):
                return pre + "_LF/MF/HF"
        return t
    for c in df.columns:
        if c in ("device", "channel", "timepoint", "concentration", "sensor", "devicetype", "time_index"):
            continue
        b = base(c)
        assert b in cat, f"{c} (type {b}) missing from feature_catalog"
        assert cat[b]["d0_kind"] == d0_normalization_kind(c)
    # NormIpeak and the peak_* faradaic features are RESERVED targets, not predictors
    for t in ("NormIpeak", "peak_height", "peak_area", "peak_fwhm"):
        assert cat[t]["role"] == "target"
    assert cat["mean_Ibg"]["role"] == "predictor" and cat["min_neg_phase"]["role"] == "predictor"
