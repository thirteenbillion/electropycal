"""Session-folder file discovery: ``_redo`` supersession, and nothing dropped silently.

Every reader walks a session folder through ``session_files``, so these rules are tested once
here, through extraction end to end, and through one review reader to show the wiring.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from test_extract import _session, _write_fscv

from electropycal.data.pstrace import (
    PartialRedoWarning,
    UnknownFileSuffix,
    UnparsedFileWarning,
    resolve_session_names,
    session_files,
)
from electropycal.features.extract import extract_dataset
from electropycal.features.fscv import align_background

BAND = (10.0, 100_000.0)
KEY = ["device", "channel", "timepoint", "concentration"]


def _extract(root):
    df = extract_dataset(root, band=BAND, d0_normalize=False)
    return df.sort_values(KEY).reset_index(drop=True)


def _two_sessions(root, later_peak):
    _session(root, "20260715", "2-2", [3, 5], peak_at=20)
    _session(root, "20260722", "2-2", [3, 5], peak_at=later_peak)
    return root / "20260722_neurostring_signal"


def _redo_everything(folder, peak_at):
    """Write a ``_redo`` twin of every 2-2 file in ``folder``, measured with ``peak_at``."""
    tmp = folder.parent / "_redo_tmp"
    tmp.mkdir()
    _session(tmp, "20260722", "2-2", [3, 5], peak_at=peak_at)
    for p in (tmp / folder.name).glob("*.csv"):
        p.replace(folder / f"{p.stem}_redo.csv")


# ---- the rule, on names alone ----------------------------------------------------------

def test_redo_replaces_its_original_file_by_file():
    res = resolve_session_names(["2-2_eis_0nM.csv", "2-2_eis_0nM_redo.csv",
                                 "2-2_fscv_100nM.csv", "2-3_fscv_100nM.csv",
                                 "2-2_fscv_500nM_REDO.csv"])
    used = [n for n, _ in res.files]
    assert used == ["2-2_eis_0nM_redo.csv", "2-2_fscv_100nM.csv", "2-2_fscv_500nM_REDO.csv",
                    "2-3_fscv_100nM.csv"]
    assert res.superseded == {"2-2_eis_0nM.csv": "2-2_eis_0nM_redo.csv"}
    meta = dict(res.files)
    assert meta["2-2_eis_0nM_redo.csv"] == {"deviceid": "2-2", "signaltype": "eis",
                                            "dose": 0.0, "redo": True}
    assert meta["2-2_fscv_500nM_REDO.csv"]["dose"] == 500.0      # a redo with no original
    assert meta["2-2_fscv_100nM.csv"]["redo"] is False


@pytest.mark.parametrize("name", ["2-2_fscv_100nM_redo2.csv", "2-2_fscv_100nM_redo_redo.csv",
                                  "2-2_eis_0nM_old.csv", "2-2_fscv_100nM-redo.csv"])
def test_an_unknown_suffix_raises_naming_the_file(name):
    with pytest.raises(UnknownFileSuffix, match=name):
        resolve_session_names(["2-2_fscv_100nM.csv", name])


def test_protocol_tokens_are_not_reported_but_unknown_ones_are():
    res = resolve_session_names(["2-2_fscv_stabilization-full.csv", "2-2_fscv_bgd.csv",
                                 "2-2_fscv_typo.csv", "readme.csv"])
    assert res.unknown_tokens == ["2-2_fscv_typo.csv"]
    assert res.unparsed == ["readme.csv"]


# ---- through extraction ----------------------------------------------------------------

def test_redo_files_supersede_their_originals_in_extraction(tmp_path):
    a = tmp_path / "a"
    b = tmp_path / "b"
    a.mkdir()
    b.mkdir()
    folder = _two_sessions(a, later_peak=18)
    _redo_everything(folder, peak_at=30)
    _two_sessions(b, later_peak=30)                    # what the redo measured, as originals
    with warnings.catch_warnings():
        warnings.simplefilter("error")                 # a complete redo set warns about nothing
        got = _extract(a)
    pd.testing.assert_frame_equal(got, _extract(b))

    c = tmp_path / "c"
    c.mkdir()
    _two_sessions(c, later_peak=18)
    assert not got.equals(_extract(c))                 # the originals were not what was read


def test_partial_redo_warns_and_still_reads(tmp_path):
    folder = _two_sessions(tmp_path, later_peak=18)
    _write_fscv(folder / "2-2_fscv_500nm_redo.csv", [3, 5], peak_scale=15.0)
    with pytest.warns(PartialRedoWarning, match="original 0 nM background"):
        df = _extract(tmp_path)
    assert set(df[df.timepoint == 7].concentration) == {100.0, 500.0, 1000.0}


def test_an_unparsed_file_in_a_signal_folder_warns_naming_it(tmp_path):
    folder = _two_sessions(tmp_path, later_peak=18)
    (folder / "notes.csv").write_text("not an export", encoding="utf-8")
    with pytest.warns(UnparsedFileWarning, match="20260722_neurostring_signal/notes.csv"):
        df = _extract(tmp_path)
    assert not df.empty


def test_an_unknown_suffix_stops_extraction(tmp_path):
    folder = _two_sessions(tmp_path, later_peak=18)
    _write_fscv(folder / "2-2_fscv_500nm_redo2.csv", [3, 5], peak_scale=15.0)
    with pytest.raises(UnknownFileSuffix, match="_redo2"):
        extract_dataset(tmp_path, band=BAND, d0_normalize=False)


def test_review_readers_see_the_redo_file(tmp_path):
    from electropycal.rawspectra import RawSpectraIndex

    folder = _two_sessions(tmp_path, later_peak=18)
    _redo_everything(folder, peak_at=30)
    paths = {Path(p).resolve() for p in RawSpectraIndex(tmp_path, band=BAND).signal.path}
    redo = {p.resolve() for p in folder.glob("*_redo.csv")}
    originals = {p.resolve() for p in folder.glob("*.csv")} - redo
    assert redo and redo <= paths
    assert not originals & paths
    assert {p.resolve() for p, _ in session_files(folder)} == redo


# ---- background alignment ------------------------------------------------------------

def test_background_is_read_from_the_same_sweep_at_tied_voltages():
    up = np.linspace(-0.4, 1.3, 18)
    v = np.concatenate([up, up[-2::-1]])               # every voltage but the apex visited twice
    bg = np.concatenate([np.full(up.size, 10.0), np.full(up.size - 1, -10.0)])
    got = align_background(v, v, bg)
    np.testing.assert_array_equal(got, bg)
    # the whole-cycle interpolation this replaces returns the falling-sweep value at a tie
    o = np.argsort(v, kind="stable")
    assert (np.interp(v, v[o], bg[o])[: up.size - 1] == -10.0).all()


def test_tied_branches_no_longer_change_extraction(tmp_path, monkeypatch):
    import electropycal.data.synthetic as syn

    def tree(name):
        root = tmp_path / name
        syn.write_synthetic_pstrace_dir(root, devices=("2-2",), timepoints=(0, 7),
                                        channels=(3, 5), concentrations=(0, 100, 500, 1000))
        return extract_dataset(root, band=BAND, d0_normalize=False).sort_values(KEY)

    offset = tree("offset")
    monkeypatch.setattr(syn, "BRANCH_EPS", 0.0)        # rising and falling samples tie exactly
    tied = tree("tied")
    assert len(offset) and len(tied) == len(offset)
    for col in ("NormIpeak", "peak_height", "peak_fwhm", "mean_Ibg"):
        np.testing.assert_allclose(tied[col].to_numpy(), offset[col].to_numpy(), rtol=1e-9)
