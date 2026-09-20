"""Peak edge-clip detector: flag a faradaic lobe truncated by the sweep/window edge."""

import numpy as np

from electropycal.features.fscv import peak_edge_clipped, peak_shape, PEAK_WINDOW


def _gauss(v, center, width, amp=1.0):
    return amp * np.exp(-((v - center) ** 2) / (2 * width ** 2))


def test_interior_peak_not_clipped():
    v = np.linspace(0.0, 1.2, 241)
    y = _gauss(v, center=0.7, width=0.05)          # fully inside (0.4, 1.0), returns to ~0 at bounds
    clip = peak_edge_clipped(y, v, height=1.0, v_window=PEAK_WINDOW)
    assert not clip["clipped"]


def test_peak_still_rising_at_upper_edge_is_clipped_high():
    v = np.linspace(0.0, 1.2, 241)
    y = _gauss(v, center=0.95, width=0.1)          # crest near 1.0 V; lobe extends past the upper bound
    clip = peak_edge_clipped(y, v, height=1.0, v_window=PEAK_WINDOW)
    assert clip["clipped"] and clip["clipped_high"] and not clip["clipped_low"]


def test_low_onset_peak_is_clipped_low():
    v = np.linspace(0.0, 1.2, 241)
    y = _gauss(v, center=0.45, width=0.1)          # broad low-dose peak rising from below 0.4 V
    clip = peak_edge_clipped(y, v, height=1.0, v_window=PEAK_WINDOW)
    assert clip["clipped"] and clip["clipped_low"]


def test_no_peak_returns_all_false():
    v = np.linspace(0.0, 1.2, 241)
    assert peak_edge_clipped(np.zeros_like(v), v, height=float("nan")) == {
        "clipped_high": False, "clipped_low": False, "clipped": False}


def test_peak_shape_reports_area_clipped():
    v = np.linspace(0.0, 1.2, 241)
    bg = np.zeros_like(v)
    clean = peak_shape(_gauss(v, 0.7, 0.05), bg, v, method="direct")
    clipped = peak_shape(_gauss(v, 0.98, 0.12), bg, v, method="direct")
    assert clean["area_clipped"] is False and np.isfinite(clean["area"])
    assert clipped["area_clipped"] is True        # truncated lobe -> area is an underestimate
