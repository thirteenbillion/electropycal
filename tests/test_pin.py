"""Pinned extraction parameters: reproduce over a subset, or fail loudly.

The property under test is the one that makes analysis over selectively-staged data valid:
re-extracting a subset with the pin from a full run yields *exactly* those sessions' rows,
unchanged — and every way that could silently stop being true raises instead.
"""

import pandas as pd
import pytest
from test_extract import _build_raw, _session

from electropycal.features.extract import extract_dataset
from electropycal.features.pin import PIN_SCHEMA_VERSION, PinMismatch


def _full(tmp_path):
    """The reference tree + its pin: devices 2-2 (t0, t7) and 2-3 (t0)."""
    root = tmp_path / "full"
    root.mkdir()
    _build_raw(root)
    pin_path = tmp_path / "pin.json"
    df = extract_dataset(root, band=(10.0, 100_000.0), d0_normalize=False, pin_out=pin_path)
    assert not df.empty
    return root, df, pin_path


def test_pin_records_the_five_derived_globals(tmp_path):
    _root, df, pin_path = _full(tmp_path)
    import json
    rec = json.loads(pin_path.read_text())
    assert rec["schema_version"] == PIN_SCHEMA_VERSION
    e = rec["extraction"]
    assert e["band_hz"] == [10.0, 100_000.0] and e["band_source"] == "explicit"
    assert set(e["device_d0"]) == {"2-2", "2-3"}
    assert e["device_d0"]["2-2"] == "2026-07-15"      # its own earliest session
    assert e["device_d0"]["2-3"] == "2026-07-22"      # staggered: a later, device-local D0
    assert e["ref_grid_hz"] and all(len(g) for g in e["ref_grid_hz"].values())
    assert e["feature_columns"] == list(df.columns)
    assert e["sessions"]                              # per-session row counts, for the fetch check
    # the same record is always on the frame, so a caller need not write a file
    assert df.attrs["extraction_pin"] == rec


def test_pin_round_trips_the_same_tree_exactly(tmp_path):
    root, df, pin_path = _full(tmp_path)
    again = extract_dataset(root, d0_normalize=False, pin=pin_path)
    pd.testing.assert_frame_equal(df.reset_index(drop=True), again.reset_index(drop=True))


def test_pinned_subset_reproduces_those_sessions_unchanged(tmp_path):
    """The whole point: a two-session subset must yield exactly its rows from the full run.

    Without the pin the same subset silently re-anchors — dropping 2-2's earliest session
    moves its ``device_d0``, so what was ``timepoint=7`` becomes ``timepoint=0``.
    """
    _root, full, pin_path = _full(tmp_path)
    subset = tmp_path / "subset"
    subset.mkdir()
    _session(subset, "20260722", "2-2", [3, 5], peak_at=18)      # 2-2's LATER session only
    _session(subset, "20260722", "2-3", [3, 5], peak_at=22)

    pinned = extract_dataset(subset, d0_normalize=False, pin=pin_path)
    expected = full[(full.device == "2-3") | (full.timepoint == 7.0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, pinned.reset_index(drop=True))
    assert set(pinned.loc[pinned.device == "2-2", "timepoint"]) == {7.0}

    unpinned = extract_dataset(subset, band=(10.0, 100_000.0), d0_normalize=False)
    assert set(unpinned.loc[unpinned.device == "2-2", "timepoint"]) == {0.0}   # re-anchored


def test_pin_raises_on_device_absent_from_the_pin(tmp_path):
    _root, _df, pin_path = _full(tmp_path)
    other = tmp_path / "other"
    other.mkdir()
    _session(other, "20260722", "9-9", [3, 5], peak_at=20)       # never pinned
    with pytest.raises(PinMismatch, match="absent from the pin"):
        extract_dataset(other, d0_normalize=False, pin=pin_path)


def test_pin_raises_on_session_predating_its_pinned_d0(tmp_path):
    """An earlier-dated session is the latent hazard: it would shift every timepoint."""
    _root, _df, pin_path = _full(tmp_path)
    earlier = tmp_path / "earlier"
    earlier.mkdir()
    _session(earlier, "20260701", "2-2", [3, 5], peak_at=20)     # predates d0 2026-07-15
    with pytest.raises(PinMismatch, match="predates its pinned device_d0"):
        extract_dataset(earlier, d0_normalize=False, pin=pin_path)


def test_pin_raises_on_a_partial_fetch(tmp_path):
    """A session short its EIS returns zero rows without raising anywhere else."""
    _root, _df, pin_path = _full(tmp_path)
    staged = tmp_path / "staged"
    staged.mkdir()
    _session(staged, "20260722", "2-3", [3, 5], peak_at=22)
    (staged / "20260722_neurostring_signal" / "2-3_eis_0nm.csv").unlink()
    with pytest.raises(PinMismatch, match="different row count"):
        extract_dataset(staged, d0_normalize=False, pin=pin_path)


def test_pin_raises_on_feature_column_mismatch(tmp_path):
    root, _df, pin_path = _full(tmp_path)
    import json
    rec = json.loads(pin_path.read_text())
    rec["extraction"]["feature_columns"] = rec["extraction"]["feature_columns"][:-1] + ["nope"]
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(rec))
    with pytest.raises(PinMismatch, match="feature columns differ"):
        extract_dataset(root, d0_normalize=False, pin=bad)


def test_pin_raises_on_explicit_band_conflicting_with_the_pin(tmp_path):
    root, _df, pin_path = _full(tmp_path)
    with pytest.raises(PinMismatch, match="conflicts with the pinned band"):
        extract_dataset(root, band=(1.0, 5000.0), d0_normalize=False, pin=pin_path)


def test_pinned_band_is_used_verbatim_without_a_corpus_pre_pass(tmp_path):
    """``band="auto"`` must not re-run its percentile pre-pass when a pin is present —
    that percentile would be taken over the subset instead of the pinned corpus."""
    root, _df, pin_path = _full(tmp_path)
    called = []
    import electropycal.data.inventory as inv
    real = inv.recommended_band
    inv.recommended_band = lambda *a, **k: called.append(1) or real(*a, **k)
    try:
        out = extract_dataset(root, band="auto", d0_normalize=False, pin=pin_path)
    finally:
        inv.recommended_band = real
    assert not called                                            # never consulted
    assert out.attrs["extraction_pin"]["extraction"]["band_hz"] == [10.0, 100_000.0]


def test_pinned_d0_rows_hold_the_baseline_across_a_subset(tmp_path):
    """A sensor's d0_row is its earliest timepoint *in the input*, so a subset that drops
    its true D0 re-baselines it. Pinned rows keep the original baseline."""
    root = tmp_path / "full"
    root.mkdir()
    _build_raw(root)
    pin_path = tmp_path / "pin.json"
    full = extract_dataset(root, band=(10.0, 100_000.0), d0_normalize=True, pin_out=pin_path)
    import json
    assert json.loads(pin_path.read_text())["extraction"]["d0_rows"]     # recorded when on

    subset = tmp_path / "subset"
    subset.mkdir()
    _session(subset, "20260722", "2-2", [3, 5], peak_at=18)              # 2-2's LATER session
    pinned = extract_dataset(subset, d0_normalize=True, pin=pin_path)
    expected = full[(full.device == "2-2") & (full.timepoint == 7.0)].reset_index(drop=True)
    pd.testing.assert_frame_equal(expected, pinned.reset_index(drop=True))

    # Compare the baselines actually used. Without the pin the surviving timepoint becomes its
    # own baseline; with it, the original D0 is kept. (Asserting on the normalized values
    # instead would only test the fixture, whose EIS is identical at both timepoints.)
    unpinned = extract_dataset(subset, band=(10.0, 100_000.0), d0_normalize=True)
    full_rows = full.attrs["extraction_pin"]["extraction"]["d0_rows"]
    pinned_rows = pinned.attrs["extraction_pin"]["extraction"]["d0_rows"]
    unpinned_rows = unpinned.attrs["extraction_pin"]["extraction"]["d0_rows"]
    import numpy as np
    assert set(pinned_rows) == set(unpinned_rows) == {"2-2|3", "2-2|5"}
    # equal_nan: a baseline legitimately carries NaN for a feature with no finite D0 value
    assert all(np.allclose(pinned_rows[k], full_rows[k], equal_nan=True)     # baseline preserved
               for k in pinned_rows)


def test_pinned_d0_rows_are_used_rather_than_recomputed(tmp_path):
    """Feed back a pin whose baselines have been doubled: a magnitude feature must halve.

    Asserting this directly, rather than comparing a subset against a full run, because the
    predictor features in this fixture are identical at every timepoint — so a re-baselined
    run is numerically indistinguishable from a correctly-baselined one there, and only a
    changed baseline can show whether the pinned values are consumed at all.
    """
    import json
    root = tmp_path / "full"
    root.mkdir()
    _build_raw(root)
    pin_path = tmp_path / "pin.json"
    base = extract_dataset(root, band=(10.0, 100_000.0), d0_normalize=True, pin_out=pin_path)

    rec = json.loads(pin_path.read_text())
    rec["extraction"]["d0_rows"] = {k: [2.0 * x for x in v]
                                    for k, v in rec["extraction"]["d0_rows"].items()}
    doubled_pin = tmp_path / "doubled.json"
    doubled_pin.write_text(json.dumps(rec))

    doubled = extract_dataset(root, d0_normalize=True, pin=doubled_pin)
    # R_s_f00 is a magnitude feature -> D0-normalized by division, so 2x baseline halves it
    import numpy as np
    assert np.allclose(doubled["R_s_f00"].to_numpy(float),
                       base["R_s_f00"].to_numpy(float) / 2.0, equal_nan=True)
