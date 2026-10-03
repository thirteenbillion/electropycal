"""Path validation: three failure modes, three mutually exclusive messages.

The property under test is not just that each case raises. It is that each case raises the
*right* message and **not** either of the other two, because the original bug was a path
error surfacing as a ``FileNotFoundError`` from inside a band percentile, and the fix is
worthless if it replaces that with a message about the working directory when the user
actually pasted a URL.
"""

import pytest

from electropycal.data.paths import (
    NoSessionFolders,
    NotAFilesystemPath,
    PathNotFound,
    looks_like_url,
    validate_path,
    validate_raw_root,
)

#: Fragments that identify each message. Each test asserts its own is present and the
#: other two are absent, so the three can never quietly merge.
MISSING = "does not exist"
URL = "is a URL, not a filesystem path"
NO_SESSIONS = "contains no session folders"


def _assert_only(msg: str, expected: str) -> None:
    assert expected in msg, f"expected {expected!r} in {msg!r}"
    for other in (MISSING, URL, NO_SESSIONS):
        if other != expected:
            assert other not in msg, f"{other!r} leaked into the {expected!r} message"


def _session(root, name="20260716_neurostring_signal"):
    d = root / name
    d.mkdir(parents=True)
    (d / "3-2_eis_0nM.csv").write_text("x", encoding="utf-8")
    return d


# ---------------------------------------------------------------------------- case 1

def test_absent_path_says_so_and_nothing_else(tmp_path):
    with pytest.raises(PathNotFound) as ei:
        validate_raw_root(tmp_path / "nope")
    _assert_only(str(ei.value), MISSING)


def test_a_relative_absent_path_shows_what_it_resolved_to(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    with pytest.raises(PathNotFound) as ei:
        validate_raw_root("some/relative/dir")
    msg = str(ei.value)
    _assert_only(msg, MISSING)
    assert "relative to the current directory" in msg
    assert str(tmp_path) in msg          # names what it actually looked at


# ---------------------------------------------------------------------------- case 2

@pytest.mark.parametrize("url", [
    "https://drive.google.com/drive/folders/1AbC",
    "http://example.org/data",
    "gs://bucket/path",
])
def test_a_url_is_named_as_a_url(url):
    with pytest.raises(NotAFilesystemPath) as ei:
        validate_raw_root(url)
    _assert_only(str(ei.value), URL)


def test_a_drive_link_gets_the_colab_recipe():
    with pytest.raises(NotAFilesystemPath) as ei:
        validate_raw_root("https://drive.google.com/drive/folders/1AbC")
    msg = str(ei.value)
    assert "drive.google.com" in msg
    assert "from google.colab import drive" in msg
    assert "drive.mount('/content/drive')" in msg
    assert "/content/drive/MyDrive/" in msg


def test_a_non_cloud_url_does_not_get_the_colab_recipe():
    with pytest.raises(NotAFilesystemPath) as ei:
        validate_raw_root("https://example.org/data")
    assert "google.colab" not in str(ei.value)


def test_a_windows_drive_letter_is_not_mistaken_for_a_url():
    """``C:\\data`` parses as scheme ``c``. Calling that a URL would be a worse error."""
    assert not looks_like_url(r"C:\data\invitro")
    assert not looks_like_url("/home/me/data")
    with pytest.raises(PathNotFound) as ei:
        validate_raw_root(r"C:\definitely\not\here\at\all")
    _assert_only(str(ei.value), MISSING)


# ---------------------------------------------------------------------------- case 3

def test_a_directory_with_no_sessions_says_so_and_nothing_else(tmp_path):
    (tmp_path / "notes").mkdir()
    with pytest.raises(NoSessionFolders) as ei:
        validate_raw_root(tmp_path)
    _assert_only(str(ei.value), NO_SESSIONS)


def test_one_level_too_high_points_down(tmp_path):
    """The commonest real mistake: a Drive mount point above the session folders."""
    inner = tmp_path / "neurostring_model" / "data"
    _session(inner)
    with pytest.raises(NoSessionFolders) as ei:
        validate_raw_root(tmp_path / "neurostring_model")
    msg = str(ei.value)
    _assert_only(msg, NO_SESSIONS)
    assert "one level DOWN" in msg
    assert str(inner) in msg


def test_one_level_too_low_points_up(tmp_path):
    corpus = tmp_path / "corpus"
    sess = _session(corpus)
    with pytest.raises(NoSessionFolders) as ei:
        validate_raw_root(sess)          # pointed at the session itself
    msg = str(ei.value)
    _assert_only(msg, NO_SESSIONS)
    assert "PARENT directory" in msg
    assert str(corpus) in msg


def test_a_file_is_not_a_directory(tmp_path):
    f = tmp_path / "featureset.parquet"
    f.write_text("x", encoding="utf-8")
    with pytest.raises(NoSessionFolders) as ei:
        validate_raw_root(f)
    assert "is a file, not a directory" in str(ei.value)


def test_a_real_corpus_passes_and_returns_a_path(tmp_path):
    _session(tmp_path)
    out = validate_raw_root(tmp_path)
    assert out == tmp_path
    assert validate_path(tmp_path) == tmp_path


# ---------------------------------------------------------- the entry points use it

def test_extract_dataset_rejects_a_url_before_inferring_anything():
    """The original bug: --band auto ran a corpus pre-pass before any validation."""
    from electropycal.features.extract import extract_dataset
    with pytest.raises(NotAFilesystemPath) as ei:
        extract_dataset("https://drive.google.com/drive/folders/1AbC", band="auto")
    _assert_only(str(ei.value), URL)


def test_extract_dataset_rejects_an_absent_path_before_inferring_anything(tmp_path):
    from electropycal.features.extract import extract_dataset
    with pytest.raises(PathNotFound) as ei:
        extract_dataset(tmp_path / "nope", band="auto")
    _assert_only(str(ei.value), MISSING)


def test_cli_load_frame_rejects_a_url():
    from electropycal.cli import _load_frame
    with pytest.raises(NotAFilesystemPath) as ei:
        _load_frame("https://drive.google.com/drive/folders/1AbC")
    _assert_only(str(ei.value), URL)


@pytest.mark.parametrize("factory", ["qcdash", "rawspectra"])
def test_review_module_roots_validate(factory):
    import electropycal.qcdash as qcdash
    import electropycal.rawspectra as rawspectra
    cls = {"qcdash": qcdash.QCDashboard, "rawspectra": rawspectra.RawSpectraIndex}[factory]
    with pytest.raises(NotAFilesystemPath) as ei:
        cls("https://drive.google.com/drive/folders/1AbC")
    _assert_only(str(ei.value), URL)


def test_stabreview_root_validates():
    from electropycal.stabreview import StabilizationIndex
    with pytest.raises(NotAFilesystemPath) as ei:
        StabilizationIndex("https://drive.google.com/drive/folders/1AbC")
    _assert_only(str(ei.value), URL)


def test_run_data_from_frame_rejects_a_path_with_a_useful_message():
    from electropycal.discovery.config import RunData
    with pytest.raises(TypeError) as ei:
        RunData.from_frame("data/featureset_raw.parquet")
    msg = str(ei.value)
    assert "takes a DataFrame, not a path" in msg
    assert "pd.read_parquet" in msg
