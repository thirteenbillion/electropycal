"""Stabilization review index + plots (folded from stabilization_review)."""

import os
import tempfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest

from electropycal.data.synthetic import write_synthetic_pstrace_dir
from electropycal import stabreview as sr


@pytest.fixture(scope="module")
def idx():
    return sr.StabilizationIndex(write_synthetic_pstrace_dir(tempfile.mkdtemp()))


def test_inventory_and_pairs(idx):
    assert len(idx.inv) > 0 and len(idx.available_table()) > 0
    pairs = idx.pairs()
    assert pairs and all(len(p) == 2 for p in pairs)
    assert idx.pairs(select=[pairs[0]]) == [pairs[0]]


def test_settings_describe(idx):
    d = idx.describe()
    assert "v_target" in d and "tol / patience" in d


def test_no_stabilization_files_raises_a_named_error():
    """A tree with sessions but no stabilization exports must stop with a clear message.

    Previously the empty inventory was a column-less DataFrame and the failure surfaced
    far downstream as ``AttributeError: 'DataFrame' object has no attribute 'device'``,
    which named neither the tree nor what was missing.
    """
    root = Path(write_synthetic_pstrace_dir(tempfile.mkdtemp()))
    removed = 0
    for p in root.rglob("*_fscv_stabilization*.csv"):
        os.remove(p); removed += 1
    assert removed > 0, "fixture should have had stabilization files to remove"

    with pytest.raises(sr.NoStabilizationFiles) as ei:
        sr.StabilizationIndex(root)

    msg = str(ei.value)
    assert "no stabilization files found" in msg
    assert str(root) in msg or str(root.resolve()) in msg   # names the tree
    assert sr._STAB_GLOB in msg                             # names what it looked for
    assert "parsed as sessions" in msg                      # names how far it got
    # The sessions themselves are still there, so this is not "empty directory".
    assert "0 subdirectory" not in msg


def test_unparsable_folders_get_the_cause_that_fits():
    """Causes are conditional: don't tell the reader nothing parsed when things did."""
    d = Path(tempfile.mkdtemp())
    (d / "not-a-session").mkdir(); (d / "junk").mkdir()
    with pytest.raises(sr.NoStabilizationFiles) as ei:
        sr.StabilizationIndex(d)
    msg = str(ei.value)
    assert "0 of which parsed as sessions" in msg
    assert "none of the 2 subdirectory(ies) parsed" in msg
    # the "wrong spelling" cause is meaningless when nothing was searched
    assert "stabilization-full.csv" not in msg


def test_missing_root_raises_a_path_error_not_a_no_files_error():
    """Behaviour change in 0.10.0, and a deliberate sharpening.

    0.9.0 answered a missing root with ``NoStabilizationFiles``, which conflated "this tree
    has no stabilization sweeps in it" with "this tree does not exist". They want different
    fixes, so they are now different errors. Both still subclass ``FileNotFoundError``, so
    anything catching that is unaffected.
    """
    from electropycal.data.paths import PathNotFound
    missing = Path(tempfile.mkdtemp()) / "does-not-exist"
    with pytest.raises(PathNotFound) as ei:
        sr.StabilizationIndex(missing)
    msg = str(ei.value)
    assert str(missing) in msg
    assert "does not exist" in msg
    assert issubclass(PathNotFound, FileNotFoundError)
    # It must not claim there are NO stabilization files, which was the old conflation.
    # ("stabilization root" in the message is the argument's name, which is fine.)
    assert not isinstance(ei.value, sr.NoStabilizationFiles)
    assert "no stabilization" not in msg.lower()
    assert "_STAB_GLOB" not in msg and "stabilization*" not in msg


def test_plots_and_convergence_table_run(idx):
    pairs = idx.pairs()[:1]
    sr.plot_raw_cycles(idx, pairs); plt.close("all")
    sr.plot_i_vtarget(idx, pairs); plt.close("all")
    sr.plot_round_drift(idx, pairs); plt.close("all")
    tbl = sr.convergence_table(idx, idx.pairs())
    assert {"device", "channel", "settle_tau", "final_drift", "converged"} <= set(tbl.columns)
    assert len(tbl) > 0
