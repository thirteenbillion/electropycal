"""CLI wiring: argument parsing and end-to-end extract, kept consistent with the
library defaults (data-driven band, progress logging)."""

import argparse

import pandas as pd
import pytest

from electropycal.cli import _parse_band, main
from electropycal.data.synthetic import write_synthetic_pstrace_dir


def test_parse_band_auto_and_tuple():
    assert _parse_band("auto") == "auto"
    assert _parse_band("10,21544") == (10.0, 21544.0)
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_band("10")                        # needs exactly lo,hi
    with pytest.raises(argparse.ArgumentTypeError):
        _parse_band("a,b,c")


def test_cli_help_renders_for_every_subcommand(capsys):
    # argparse formats help via `help_string % params`; a literal '%' in a help=
    # string (e.g. "100% of") is read as a format spec and blows up only on --help.
    for argv in ([], ["extract"], ["discover"], ["freeze"], ["deploy"]):
        with pytest.raises(SystemExit) as e:
            main([*argv, "--help"])
        assert e.value.code == 0
        assert "usage:" in capsys.readouterr().out


def test_cli_extract_defaults_to_auto_band(tmp_path, capsys):
    raw = write_synthetic_pstrace_dir(tmp_path)
    out = tmp_path / "fs.parquet"
    main(["extract", "--raw", str(raw), "--out", str(out)])       # --band defaults to 'auto'
    assert out.exists()
    df = pd.read_parquet(out)
    assert not df.empty and {"device", "channel", "concentration", "NormIpeak"} <= set(df.columns)


def test_cli_load_data_sensitivity_target():
    from electropycal.cli import _load_data
    normi = _load_data("synthetic")                          # per-dose NormIpeak (default)
    sens = _load_data("synthetic", target="sensitivity")     # per sensor-timepoint slope
    assert sens.X.shape[0] < normi.X.shape[0]                # collapsed to sensor-timepoints
    assert "sensitivity" not in sens.feature_names and "concentration" not in sens.feature_names


def test_cli_extract_progress_and_explicit_band(tmp_path, capsys):
    raw = write_synthetic_pstrace_dir(tmp_path)
    out = tmp_path / "fs.parquet"
    main(["extract", "--raw", str(raw), "--out", str(out), "--band", "10,50000", "--progress"])
    out_text = capsys.readouterr().out
    assert "extract_dataset" in out_text and "featureset written" in out_text
    assert out.exists()


def test_discover_seeds_flag_reaches_the_stochastic_selector(tmp_path, capsys):
    """Before --seeds existed, no CLI path could produce a multi-seed run: runner honours
    profile.seeds only when the selector is stochastic AND more than one distinct seed is set,
    and both profiles ship seeds=(0,) -- so the three-seed sweep was notebook-only."""
    from electropycal.cli import _parse_seeds, main
    assert _parse_seeds("0,1,2") == (0, 1, 2)

    import pytest
    with pytest.raises(SystemExit):
        main(["discover", "--data", "synthetic", "--profile", "fast", "--seeds", "nope",
              "--out", str(tmp_path / "bad")])

    main(["discover", "--data", "synthetic", "--profile", "fast", "--seeds", "0,1,2",
          "--min-train-times", "1", "--progress", "--out", str(tmp_path / "out")])
    out = capsys.readouterr().out
    assert "seeds=(0, 1, 2)" in out                      # a stochastic condition ran all three
    assert "(selector deterministic -> 1 seed)" in out   # a deterministic one collapsed

    import json
    run_dir = next((tmp_path / "out").glob("model_discovery_*"))
    assert json.loads((run_dir / "run_config.json").read_text())["profile"]["seeds"] == [0, 1, 2]
