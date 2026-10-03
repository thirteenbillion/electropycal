"""Validate a user-supplied path before anything infers from it.

Every entry point that takes a path used to infer first and validate never. The most
expensive instance: ``electropycal extract --raw <bad path>`` reached
``_resolve_cli_band`` -> ``recommended_band`` -> ``root.iterdir()`` because ``--band auto``
runs a full-corpus pre-pass, so the first thing a new user saw was a twenty-line traceback
ending in a bare ``FileNotFoundError`` from inside the band percentile.

Three failure modes need three different answers, and conflating them is what made the
errors useless:

======================  ==================================================================
does not exist          a typo, or a relative path from the wrong directory
is a URL                a cloud share link pasted where a filesystem path goes. The fix is
                        to mount or download, and a message about the working directory
                        actively misleads here
exists, wrong contents  a real directory that is not a PSTrace corpus, usually one level
                        above or below the session folders
======================  ==================================================================

The messages are mutually exclusive by construction, and the tests assert that each one's
text is *absent* from the other two cases.
"""

from __future__ import annotations

import re
from pathlib import Path

#: A session folder: ``<YYYYMMDD>_<devicetype>_<testtype>``. Only the date prefix and the
#: trailing ``_signal`` are load-bearing for "is this a corpus root", so the middle is left
#: open rather than enumerating device types that will change.
_SESSION_RE = re.compile(r"^\d{8}_.+_signal$")

#: URL schemes worth naming. A single letter is excluded on purpose: ``C:\data`` parses as
#: scheme ``c``, and calling a Windows path a URL would be a worse error than the original.
_URL_RE = re.compile(r"^(?P<scheme>[a-zA-Z][a-zA-Z0-9+.-]+)://")

_CLOUD_HOSTS = ("drive.google.com", "docs.google.com", "dropbox.com",
                "onedrive.live.com", "sharepoint.com", "box.com")

_COLAB_RECIPE = (
    "        from google.colab import drive\n"
    "        drive.mount('/content/drive')\n"
    "        ROOT = '/content/drive/MyDrive/<your-folder>'"
)


class PathNotFound(FileNotFoundError):
    """A supplied path does not exist on this filesystem."""


class NotAFilesystemPath(ValueError):
    """A URL or cloud share link was supplied where a filesystem path is required."""


class NoSessionFolders(FileNotFoundError):
    """A directory exists but holds no ``<YYYYMMDD>_<devicetype>_signal`` folders."""


def looks_like_url(value) -> bool:
    """True for ``http://...``, ``https://...``, ``gs://...`` and friends.

    Deliberately not a general URL test: a bare hostname or a UNC path is not caught,
    because guessing wrong turns a clear "does not exist" into a confusing lecture.
    """
    return bool(_URL_RE.match(str(value)))


def _url_message(value: str, argname: str) -> str:
    host = str(value).split("://", 1)[1].split("/", 1)[0].lower()
    cloud = next((h for h in _CLOUD_HOSTS if h in host), None)
    lines = [
        f"{argname} is a URL, not a filesystem path: {value!r}.",
        "",
        "Nothing here downloads for you. The data has to be on the local filesystem",
        "before it can be read.",
    ]
    if cloud:
        lines += [
            "",
            f"That link points at {cloud}. On Colab, mount the drive and use the mounted",
            "path instead:",
            "",
            _COLAB_RECIPE,
            "",
            "Note the mount point is a directory *above* your session folders, so point",
            "at the folder that directly contains the <YYYYMMDD>_..._signal directories.",
            "",
            "Off Colab, download or sync the folder first and pass the local path.",
        ]
    else:
        lines += [
            "",
            "Download or mount it first, then pass the local path.",
        ]
    return "\n".join(lines)


def validate_path(value, argname: str = "path", *, must_exist: bool = True) -> Path:
    """Check ``value`` is an existing filesystem path. Returns it as a :class:`Path`.

    Raises :class:`NotAFilesystemPath` for a URL and :class:`PathNotFound` for a path that
    is simply absent. Says nothing about contents; see :func:`validate_raw_root`.
    """
    if looks_like_url(value):
        raise NotAFilesystemPath(_url_message(str(value), argname))
    p = Path(value)
    if must_exist and not p.exists():
        cwd = Path.cwd()
        hint = ""
        if not p.is_absolute():
            hint = (f"\n\nIt was read as relative to the current directory, {cwd}, giving\n"
                    f"{(cwd / p)}.\nPass an absolute path if that is not what you meant.")
        # Native form, not repr: repr of a WindowsPath prints forward slashes inside a class
        # name, which does not look like what the user typed. The raw input is echoed only
        # when normalisation changed it, which is how a trailing space stays visible.
        shown = str(p)
        if str(value).strip() != shown:
            shown += f"   (from {str(value)!r})"
        raise PathNotFound(f"{argname} does not exist: {shown}{hint}")
    return p


def session_folders(root: Path) -> list[Path]:
    """Immediate subdirectories of ``root`` that parse as signal session folders."""
    try:
        return sorted(d for d in root.iterdir()
                      if d.is_dir() and _SESSION_RE.match(d.name))
    except (OSError, PermissionError):
        return []


def validate_raw_root(value, argname: str = "raw root") -> Path:
    """Check ``value`` is a directory holding PSTrace session folders.

    Call this at every entry point taking a raw corpus path, **before** any inference over
    the corpus: a band percentile, a reference grid, a device origin. All three iterate the
    directory, so without this the first error a user sees comes from inside the inference
    and names neither their argument nor their mistake.
    """
    p = validate_path(value, argname)
    if not p.is_dir():
        raise NoSessionFolders(
            f"{argname} is a file, not a directory: {p}.\n"
            f"Pass the directory that contains the <YYYYMMDD>_<devicetype>_signal folders.")
    found = session_folders(p)
    if found:
        return p

    # Exists, is a directory, holds no sessions. The two useful diagnoses are "you are one
    # level too high" and "you are one level too low", so look for both rather than just
    # reporting the miss.
    subdirs = [d for d in p.iterdir() if d.is_dir()]
    nested = []
    for d in subdirs[:50]:
        if session_folders(d):
            nested.append(d)
    parent_has = session_folders(p.parent) if p.parent != p else []

    lines = [f"{argname} contains no session folders: {p}.",
             "",
             "Expected immediate subdirectories named <YYYYMMDD>_<devicetype>_signal,",
             "for example 20260716_neurostring_signal."]
    if nested:
        lines += ["",
                  f"Sessions were found one level DOWN, in {len(nested)} subdirector"
                  f"{'y' if len(nested) == 1 else 'ies'}. You probably meant:",
                  ""]
        lines += [f"    {d}" for d in nested[:5]]
    elif parent_has:
        lines += ["",
                  f"The PARENT directory holds {len(parent_has)} session folder(s). You",
                  "probably meant:", "", f"    {p.parent}"]
    else:
        listing = sorted(d.name for d in subdirs)[:8]
        lines += ["",
                  f"It holds {len(subdirs)} subdirector"
                  f"{'y' if len(subdirs) == 1 else 'ies'}"
                  + (f": {listing}" if listing else " and no subdirectories")
                  + ".",
                  "",
                  "If this really is the corpus, the session folders are misnamed: the date",
                  "prefix must be 8 digits and the name must end in _signal."]
    raise NoSessionFolders("\n".join(lines))


def validate_data_path(value, argname: str = "--data") -> Path:
    """Check ``value`` is an existing featureset file or a raw corpus directory.

    The ``--data`` flag accepts either, so this only rules out the two cases that are
    unambiguously wrong: a URL, and a path that is not there.
    """
    return validate_path(value, argname)
