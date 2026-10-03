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
    viz.configure_output(tmp_path, formats=("png", "pdf"), show=None)
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
    assert "review_condition_ranking.png" in written
    assert "review_condition_ranking.pdf" in written
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
