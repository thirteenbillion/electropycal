"""Figures must reach disk, headless.

The bug this covers is silence, not a crash. Every plotting function in the package ended
in ``plt.show()`` and nothing else: 48 sites, zero ``savefig``, zero ``plt.close()``. Under
an inline notebook kernel that renders, so it looked fine. On the Agg backend, which is what
CI and a chat code-execution sandbox get, ``show()`` is a no-op: the function returns, the
caller reports success, and no file exists.

So these tests assert on the **filesystem**, never on a return value.
"""

import matplotlib
import pytest

matplotlib.use("Agg", force=True)
import matplotlib.pyplot as plt

from electropycal import viz


@pytest.fixture(autouse=True)
def _headless(tmp_path, monkeypatch):
    """No DISPLAY, Agg backend, output into a temp dir. Reset viz state each test."""
    monkeypatch.delenv("DISPLAY", raising=False)
    monkeypatch.delenv("ELECTROPYCAL_FIGURES", raising=False)
    viz._WRITTEN.clear()
    viz.configure_output(tmp_path, formats=("png", "pdf"), show=None, footer=False)
    yield tmp_path
    plt.close("all")


def _a_figure():
    fig, ax = plt.subplots()
    ax.plot([1, 2, 3], [1, 4, 9])
    ax.set_xlabel("timepoint (days)")
    ax.set_ylabel("NormIpeak")
    return fig


def test_emit_actually_writes_both_formats(tmp_path):
    _a_figure()
    paths = viz.emit("normipeak_vs_time")
    assert [p.name for p in paths] == ["normipeak_vs_time.png", "normipeak_vs_time.pdf"]
    for p in paths:
        assert p.exists(), f"{p} was not written"
        assert p.stat().st_size > 1000, f"{p} is suspiciously small"


def test_a_vector_format_is_offered_alongside_the_raster(tmp_path):
    _a_figure()
    viz.emit("fig")
    assert (tmp_path / "fig.png").exists()
    assert (tmp_path / "fig.pdf").exists()          # the one a paper needs
    head = (tmp_path / "fig.pdf").read_bytes()[:5]
    assert head == b"%PDF-", "the .pdf is not a PDF"


def test_emit_closes_the_figure_so_a_multi_figure_pass_does_not_accumulate():
    assert plt.get_fignums() == []
    for i in range(25):                             # matplotlib warns around 20
        _a_figure()
        viz.emit(f"many_{i}")
    assert plt.get_fignums() == [], "figures accumulated; plt.close is not being called"


def test_show_is_not_attempted_on_a_headless_agg_backend(monkeypatch):
    """The point of sniffing: no flag needed, and no spurious warning in CI."""
    assert viz._can_show() is False
    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))
    _a_figure()
    viz.emit("no_show")
    assert called == [], "plt.show() was called on a headless Agg backend"


def test_show_is_attempted_when_the_caller_asks_for_it(monkeypatch, tmp_path):
    viz.configure_output(tmp_path, show=True)
    called = []
    monkeypatch.setattr(plt, "show", lambda *a, **k: called.append(1))
    _a_figure()
    viz.emit("shown")
    assert called == [1]
    assert (tmp_path / "shown.png").exists(), "showing must not replace saving"


def test_filenames_are_stable_and_overwrite_rather_than_accumulate(tmp_path):
    """A timestamped name is how you get 40 near-identical PNGs and no idea which is which."""
    for _ in range(3):
        _a_figure()
        viz.emit("normipeak_vs_time")
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == ["normipeak_vs_time.png"], f"expected one stable name, got {pngs}"


def test_a_name_with_awkward_characters_is_slugged_not_rejected(tmp_path):
    _a_figure()
    paths = viz.emit("EIS |Z| vs f  (device 2-2/ch 3)")
    assert paths[0].exists()
    assert "/" not in paths[0].name and "|" not in paths[0].name


def test_default_output_dir_is_figures_at_the_repo_root(monkeypatch, tmp_path):
    repo = tmp_path / "repo"
    (repo / "sub" / "deeper").mkdir(parents=True)
    (repo / "pyproject.toml").write_text("[project]\nname='x'\n", encoding="utf-8")
    monkeypatch.chdir(repo / "sub" / "deeper")
    viz._OUT_DIR = None
    assert viz._default_out_dir() == repo / "figures"


def test_the_figures_env_var_overrides_the_default(monkeypatch, tmp_path):
    monkeypatch.setenv("ELECTROPYCAL_FIGURES", str(tmp_path / "elsewhere"))
    viz._OUT_DIR = None
    assert viz._default_out_dir() == tmp_path / "elsewhere"


def test_ensure_style_applies_once_and_is_idempotent():
    viz._STYLE_APPLIED = False
    assert viz.ensure_style() is True
    assert viz.ensure_style() is False          # already applied
    assert viz.ensure_style(force=True) is True
    assert matplotlib.rcParams["savefig.dpi"] == 300
    # Okabe-Ito is actually in the cycle, not just defined in the module
    cycle = matplotlib.rcParams["axes.prop_cycle"].by_key()["color"]
    assert cycle[:3] == viz.OKABE_ITO[:3]


def test_written_records_what_was_emitted(tmp_path):
    _a_figure(); viz.emit("one")
    _a_figure(); viz.emit("two")
    names = [p.name for p in viz.written()]
    assert names == ["one.png", "one.pdf", "two.png", "two.pdf"]


# --------------------------------------------------------------- the real plotting modules

def test_a_real_plotting_function_writes_a_file(tmp_path):
    """End to end through an actual module, not just the helper."""
    import pandas as pd

    from electropycal.discovery import review

    ranking = pd.DataFrame({
        "condition": ["baselines_1.1_linearPLSR", "baselines_1.2_linearPLSR"],
        "pooled_rmsep": [0.041, 0.054],
        "rmsep_ci_lo": [0.020, 0.030],
        "rmsep_ci_hi": [0.058, 0.070],
        "pooled_q2": [-0.20, -1.01],
        "n_folds": [341, 341],
    })
    rd = tmp_path / "run"
    (rd / "report").mkdir(parents=True)
    ranking.to_parquet(rd / "report" / "condition_ranking.parquet")
    review.plot_condition_ranking(rd)
    written = sorted(p.name for p in tmp_path.glob("*"))
    # The run directory here has no sweep_manifest.json, which is the self-provisioned
    # example case, so the filename must carry the unverified marker rather than the bare
    # name: an unlabelled figure is indistinguishable from one drawn off real measurements.
    assert "review_condition_ranking__unverified-run.png" in written, written
    assert "review_condition_ranking__unverified-run.pdf" in written, written
    assert "review_condition_ranking.png" not in written, "unstamped copy would be misread"
    assert plt.get_fignums() == [], "the module left a figure open"


def test_no_plotting_module_still_calls_plt_show_directly():
    """A regression guard: 48 bare show() calls is how this started."""
    import pathlib

    import electropycal
    root = pathlib.Path(electropycal.__file__).parent
    offenders = []
    for f in sorted(root.rglob("*.py")):
        if f.name == "viz.py":
            continue                    # viz.emit is the one legitimate caller
        for i, line in enumerate(f.read_text(encoding="utf-8").splitlines(), 1):
            if "plt.show()" in line and not line.lstrip().startswith("#"):
                offenders.append(f"{f.relative_to(root)}:{i}")
    assert offenders == [], f"bare plt.show() found at {offenders}"


def test_emit_without_provenance_keeps_the_bare_filename(tmp_path, monkeypatch):
    """The default must not move: 0.10.0 shipped these exact names."""
    monkeypatch.setenv("ELECTROPYCAL_FIGURES", str(tmp_path))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from electropycal import viz
    plt.figure(); plt.plot([0, 1], [0, 1])
    names = {p.name for p in viz.emit("some_plot")}
    assert names == {"some_plot.png", "some_plot.pdf"}


def test_emit_provenance_stops_two_sources_overwriting_each_other(tmp_path, monkeypatch):
    """The actual bug: same plot, two data sources, one filename, second silently wins."""
    monkeypatch.setenv("ELECTROPYCAL_FIGURES", str(tmp_path))
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from electropycal import viz
    for prov in ("demo-synthetic", "run-20261001_004258"):
        plt.figure(); plt.plot([0, 1], [0, 1])
        viz.emit("review_calibration", provenance=prov)
    pngs = sorted(p.name for p in tmp_path.glob("*.png"))
    assert pngs == ["review_calibration__demo-synthetic.png",
                    "review_calibration__run-20261001_004258.png"], pngs
    # and the bare name is NOT also written, so there is no unlabelled copy to misread
    assert not (tmp_path / "review_calibration.png").exists()


def test_review_provenance_flags_a_run_with_no_manifest(tmp_path):
    """A self-provisioned example run has no manifest. That is the case that must shout."""
    from electropycal.discovery.review import _provenance
    prov, stage = _provenance(tmp_path / "model_discovery_x")
    assert "unverified" in prov
    assert "UNVERIFIED" in stage and "synthetic" in stage


def test_review_provenance_distinguishes_released_from_checkout(tmp_path):
    import json
    rel, dev = tmp_path / "rel", tmp_path / "dev"
    for d, env in ((rel, {"electropycal": "0.10.0", "electropycal_released": True}),
                   (dev, {"electropycal": "0.10.0", "electropycal_released": False,
                          "electropycal_commit": "abcdef1234"})):
        d.mkdir()
        (d / "sweep_manifest.json").write_text(json.dumps({"environment": env}), encoding="utf-8")
    from electropycal.discovery.review import _provenance
    p_rel, s_rel = _provenance(rel)
    p_dev, s_dev = _provenance(dev)
    assert "rel0.10.0" in p_rel and "released" in s_rel
    assert "devabcdef1" in p_dev and "NOT attributable" in s_dev
    assert p_rel != p_dev


# ------------------------------------------------------------------ the figure log and index

def test_a_saved_figure_carries_no_bookkeeping_text_by_default(tmp_path):
    """Provenance goes to the filename and the log, not into the image."""
    fig = _a_figure()
    viz.emit("clean", fig, provenance="run-a", stage="extracted features", close=False)
    assert fig.texts == [], [t.get_text() for t in fig.texts]
    assert (tmp_path / "clean__run-a.png").exists()


def test_the_footer_is_drawn_only_when_asked_for(tmp_path):
    fig = _a_figure()
    viz.emit("per_call", fig, provenance="run-a", stage="raw export", footer=True, close=False)
    assert any("raw export" in t.get_text() and "run-a" in t.get_text() for t in fig.texts)
    viz.configure_output(footer=True)
    fig2 = _a_figure()
    viz.emit("global", fig2, provenance="run-b", close=False)
    assert any("run-b" in t.get_text() for t in fig2.texts)


def test_every_save_is_logged_with_its_source_stage_and_params(tmp_path):
    _a_figure()
    viz.emit("heatmap", provenance="featureset abc1234", stage="QC-gated features",
             params={"devices": "neurostring", "doses": "separate"})
    log = (tmp_path / viz.LOG_NAME).read_text(encoding="utf-8")
    assert "heatmap__featureset_abc1234  [png, pdf]" in log
    assert "data      featureset abc1234" in log
    assert "stage     QC-gated features" in log
    assert "  param     devices=neurostring\n  param     doses=separate\n" in log
    assert "library   electropycal " in log
    assert "replaces" not in log


def test_an_overwrite_keeps_the_old_entry_and_says_what_it_replaced(tmp_path):
    for _ in range(2):
        _a_figure()
        viz.emit("same_name", provenance="src")
    entries = [e for e in viz._read_log(tmp_path / viz.LOG_NAME) if e["stem"] == "same_name__src"]
    assert len(entries) == 2
    assert entries[1]["replaces"] == f"version saved {entries[0]['saved']}"


def test_a_save_without_provenance_says_so_rather_than_leaving_a_blank(tmp_path):
    _a_figure()
    viz.emit("bare")
    log = (tmp_path / viz.LOG_NAME).read_text(encoding="utf-8")
    assert "data      (not recorded: pass provenance=)" in log


def test_the_index_lists_what_is_on_disk_and_flags_what_the_log_cannot_vouch_for(tmp_path):
    _a_figure(); viz.emit("kept", provenance="src", params={"k": 2})
    _a_figure(); viz.emit("deleted", provenance="src")
    (tmp_path / "deleted__src.png").unlink()
    (tmp_path / "deleted__src.pdf").unlink()
    _a_figure(); plt.savefig(tmp_path / "made_by_hand.png"); plt.close("all")

    text = viz.figure_index(tmp_path)
    assert text == (tmp_path / viz.INDEX_NAME).read_text(encoding="utf-8")
    assert "kept__src  [pdf, png]" in text and "param     k=2" in text
    assert "deleted__src  [" not in text
    assert "1 logged figure(s) no longer on disk" in text
    unlogged = text.split("NOT IN THE LOG")[1]
    assert "made_by_hand  [png]" in unlogged
    # The history still has the deleted figure.
    assert "deleted__src" in (tmp_path / viz.LOG_NAME).read_text(encoding="utf-8")


def test_the_index_is_newest_first(tmp_path):
    import time
    _a_figure(); viz.emit("older")
    time.sleep(1.1)                                  # the log has one-second resolution
    _a_figure(); viz.emit("newer")
    text = (tmp_path / viz.INDEX_NAME).read_text(encoding="utf-8")
    assert text.index("newer  [") < text.index("older  [")
