"""PSTrace CSV ingestion: potentiostat export to structured FSCV / EIS arrays.

Parses a PSTrace potentiostat export (UTF-16, CRLF) into structured raw FSCV and
EIS data. Two block layouts observed in real exports:

- **FSCV**: horizontal blocks on one header row, each labelled
  ``Fast Cyclic Voltammetry: FCV i vs E Channel N [rep]`` and occupying two columns
  ``(V, µA)`` = (voltage, current). ``[rep]`` marks replicate cycles (absent = 0).
- **EIS**: vertical blocks, each headed ``CH N: Fixed at K freqs`` followed by a
  column-name row (``freq / Hz``, ``neg. Phase / °``, ``Idc / uA``, ``Z / Ohm``,
  ``Z' / Ohm``, ``Z'' / Ohm``, ``Cs / F``) and ``K`` data rows. The ``Z''`` column holds
  ``-Im(Z)`` and is **negated on ingest** to match ``features.eis``'s ``Im(Z) < 0`` convention
  for a capacitive interface.

A single export corresponds to one device. Measurement protocol / timepoint /
concentration are read from the file and folder names, which follow a fixed convention:

- **file**: ``<deviceid>_<signaltype>_<dose>.csv`` (:func:`parse_filename`), e.g.
  ``2-2_fscv_100nM.csv``, ``2-2_eis_0nM.csv``, ``2-2_fscv_stabilization-end.csv``,
  ``2-2_paired_live.csv``. ``signaltype`` is ``eis``/``fscv`` (in vitro) or ``paired``
  (in vivo); the final field is a concentration (``0nM``, ``100nm``, case-insensitive)
  or a protocol token (``stabilization``/``baseline``/``live``).
- **folder**: ``<YYYYMMDD>_<devicetype>_<testtype>`` (:func:`parse_folder`), e.g.
  ``20260720_neurostring_signal``. The date supplies the timepoint; ``devicetype``
  separates ``neurostring`` from ``cfme``; ``testtype`` is ``signal`` or ``channeltest``.

The **content** parser is convention-agnostic: only the name parsers above depend on it, so a
different naming scheme can be supported by replacing them alone.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_FSCV_RE = re.compile(r"Channel\s+(\d+)")
_SCAN_RE = re.compile(r"Scan\s+(\d+)")            # stabilization cycles: "... Scan k"
_BRACKET_RE = re.compile(r"\[(\d+)\]\s*$")         # trailing "[n]": replicate (dose) or round (stabilization)
_EIS_RE = re.compile(r"CH\s+(\d+):\s*Fixed at\s+(\d+)\s*freqs")


@dataclass
class PSTraceExport:
    """Structured raw export for one device/file."""

    fscv: dict[tuple[int, int], dict]  # (channel, rep) -> {"voltage", "current"}
    eis: dict[tuple[int, int], dict]   # (channel, rep) -> {"freq", "z_real", "z_imag"}
    meta: dict = field(default_factory=dict)

    @property
    def fscv_channels(self) -> list[int]:
        return sorted({ch for ch, _ in self.fscv})

    @property
    def eis_channels(self) -> list[int]:
        return sorted({ch for ch, _ in self.eis})


def _f(x: str) -> float:
    try:
        return float(x)
    except (ValueError, TypeError):
        return np.nan


class PSTraceFormatError(ValueError):
    """A PSTrace export could not be located in, or read from, a file.

    Raised rather than returning an empty export, because both failure modes this
    covers are *silent* otherwise: a header the column finder cannot match yields zero
    blocks, and a data row it cannot parse yields blocks of zero samples. Either way the
    session simply disappears from the featureset, indistinguishable from one that was
    never measured.
    """


def _split_row(line: str) -> list[str]:
    """Split one export line into fields, tolerating whole-line double quoting.

    Exports are sometimes encountered with their lines wrapped in double quotes, and must
    read identically to the unwrapped ones. It is worth assuming this is a per-file
    accident of handling rather than a second exporter: it has been seen on a single EIS
    export while the FSCV export recorded beside it was unwrapped, so the style cannot be
    inferred from a sibling file, or from anything but the line in hand.

    **Wrapping is selective WITHIN a file, not whole-file.** Measured on such a file,
    per line class:

        quoted    "File date:,2026-09-15 13:11:36"
        quoted    "Measurement:,ImpedanceSpectroscopy"
        bare      CH 10: Fixed at 37 freqs
        quoted    "freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F,"
        quoted    "1000000,-163.0100583515,2.62260437011719,8765.02,...,"

    So a reader must tolerate both styles *interleaved*, which is why the wrapper is
    stripped per line rather than decided once per file.

    The wrapper is around the whole line, not each field, so a plain ``split(",")``
    returns the right number of fields but leaves a stray quote on the first and last of
    them. On the real mixed-quoting file that is enough to break the EIS column finder:
    the bare ``CH N:`` line means ``_EIS_RE`` *does* match and the block *is* found, then
    the quoted header splits to ``['"freq / Hz', ..., '"']`` and ``'"freq / Hz'`` does not
    start with ``freq``. Before the guards below existed that surfaced as a bare
    ``StopIteration`` naming nothing.

    Stripping the wrapper before splitting keeps the unquoted path byte-identical, which
    is why it is preferred here over ``csv.reader``: that would collapse a quoted line to
    a single field needing re-splitting, and would change field-level behaviour for the
    files that already parse.
    """
    s = line.strip()
    if len(s) >= 2 and s[0] == '"' and s[-1] == '"':
        # Whole-line wrapper. Undouble any escaped quotes inside it, per RFC 4180, so a
        # future export that does escape them is read correctly rather than silently.
        s = s[1:-1].replace('""', '"')
        # A stray BOM can sit INSIDE the wrapper, because whatever applied the wrapper
        # wrapped a line that already began with one: the real file's first line is
        # `"<BOM>File date:,...`. Harmless there only by luck of which line came first:
        # on `"<BOM>Date and time:,...` it would defeat the exact-string date lookup in
        # read_pstrace and lose export_date with no error. A BOM is never data.
        return s.lstrip("﻿").split(",")
    return line.split(",")


def read_pstrace(path: str | Path, encoding: str = "utf-16") -> PSTraceExport:
    """Parse a PSTrace CSV export into :class:`PSTraceExport`."""
    path = Path(path)
    text = path.read_text(encoding=encoding)
    rows = [_split_row(ln) for ln in text.splitlines()]

    fscv = _parse_fscv(rows, path)
    eis = _parse_eis(rows, path)
    if not fscv and not eis:
        raise PSTraceFormatError(
            f"{path}: no FSCV or EIS blocks found. An export should contain at least one "
            f"'Fast Cyclic Voltammetry: ... Channel N' header row or one "
            f"'CH N: Fixed at K freqs' row. First 3 lines were: "
            f"{[ln[:70] for ln in text.splitlines()[:3]]}")
    date = next((r[1] for r in rows if len(r) > 1 and r[0].strip() == "Date and time:"), None)
    return PSTraceExport(fscv=fscv, eis=eis, meta={"source": path.name, "export_date": date})


def _parse_fscv(rows: list[list[str]], path: Path | None = None) -> dict[tuple[int, int], dict]:
    hdr_idx = next((i for i, r in enumerate(rows)
                    if any("Fast Cyclic" in c for c in r)), None)
    if hdr_idx is None:
        return {}
    # Each "Fast Cyclic ... Channel N ..." header cell is one cycle block. ``rep``
    # is the per-channel column-appearance index (0,1,2,…), robust to the two
    # observed label styles: dose replicates ``Channel N [k]`` and stabilization
    # cycles ``Channel N Scan k [round]`` (the trailing bracket is a replicate in a
    # dose file, a stabilization round in a stabilization file). We keep ``scan`` and
    # ``bracket`` as metadata (used by the stabilization review) but never key on
    # them, so 60 same-channel stabilization cycles don't collapse.
    blocks = []  # (voltage_col, channel, rep, scan|None, bracket)
    counter: dict[int, int] = {}
    for col, cell in enumerate(rows[hdr_idx]):
        m = _FSCV_RE.search(cell)
        if not (m and "Fast Cyclic" in cell):
            continue
        ch = int(m.group(1))
        rep = counter.get(ch, 0)
        counter[ch] = rep + 1
        scan_m = _SCAN_RE.search(cell)
        br_m = _BRACKET_RE.search(cell)
        scan = int(scan_m.group(1)) if scan_m else None
        bracket = int(br_m.group(1)) if br_m else 0
        blocks.append((col, ch, rep, scan, bracket))

    data = []
    rejected = []                                      # kept only for the error message
    for r in rows[hdr_idx + 3:]:                       # skip header, date, units rows
        if r and (_EIS_RE.match(r[0]) or r[0].strip() == "Measurement:"):
            break
        if r and not np.isnan(_f(r[0])):
            data.append(r)
        elif r and any(c.strip() for c in r):
            rejected.append(r)

    out: dict[tuple[int, int], dict] = {}
    for col, ch, rep, scan, bracket in blocks:
        v = np.array([_f(r[col]) if col < len(r) else np.nan for r in data])
        i = np.array([_f(r[col + 1]) if col + 1 < len(r) else np.nan for r in data])
        keep = ~np.isnan(v)
        out[(ch, rep)] = {"voltage": v[keep], "current": i[keep],
                          "scan": scan, "bracket": bracket}
    # Headers located but not one sample recovered is the signature of an unrecognised row
    # format, not of an empty measurement. Unlike the EIS guards this one is precautionary:
    # line wrapping has been observed on an EIS export but not on an FSCV one. It is kept
    # because it is exactly what a wrapped FSCV export would do -- every row fails the
    # numeric test, so none reach ``data``, giving blocks of 0 samples and a session that
    # vanishes from the featureset indistinguishably from one never measured. The
    # rejected-row count is what makes that diagnosable rather than merely detected.
    if blocks and not any(len(b["voltage"]) for b in out.values()):
        where = f"{path}" if path is not None else "<in-memory rows>"
        sample = (rejected or data or [[]])[0][:6]
        raise PSTraceFormatError(
            f"{where}: found {len(blocks)} FSCV block(s) but recovered 0 samples "
            f"({len(data)} row(s) parsed as numeric, {len(rejected)} rejected). "
            f"First unusable row read as {sample!r}. The usual cause is an export row "
            f"format the splitter did not recognise.")
    return out


def _parse_eis(rows: list[list[str]], path: Path | None = None) -> dict[tuple[int, int], dict]:
    out: dict[tuple[int, int], dict] = {}
    rep_counter: dict[int, int] = {}
    i = 0
    while i < len(rows):
        cell = rows[i][0] if rows[i] else ""
        m = _EIS_RE.match(cell)
        if not m:
            i += 1
            continue
        ch, k = int(m.group(1)), int(m.group(2))
        colhdr = [c.strip() for c in rows[i + 1]]
        ci = _eis_columns(colhdr, path, ch)
        block = rows[i + 2:i + 2 + k]
        freq = np.array([_f(r[ci["freq"]]) for r in block])
        zr = np.array([_f(r[ci["z_real"]]) for r in block])
        # PSTrace's "Z'' / Ohm" column is -Im(Z) (positive for a capacitive
        # interface at low frequency, as seen in real exports). We negate to the
        # math convention Z = Z' + j·Im(Z) that features.eis expects (capacitive
        # Im < 0). NOTE: the column's NAME suggests it already holds Im(Z); real exports show
        # otherwise, so verify this sign on a known-capacitive channel before trusting any
        # derived capacitance; it is a property of the instrument software, not the physics.
        zi = -np.array([_f(r[ci["z_imag"]]) for r in block])
        rep = rep_counter.get(ch, 0)
        rep_counter[ch] = rep + 1
        out[(ch, rep)] = {"freq": freq, "z_real": zr, "z_imag": zi}
        i += 2 + k
    return out


def _eis_columns(header: list[str], path: Path | None = None,
                 channel: int | None = None) -> dict[str, int]:
    """Locate freq / Z' / Z'' columns by name (robust to column order).

    A bare ``next()`` here used to raise ``StopIteration`` naming nothing, which cost a
    day of debugging when a new export style arrived. Every failure now names the file,
    the column sought, and the header actually found.
    """
    def find(pred, wanted):
        j = next((j for j, h in enumerate(header) if pred(h)), None)
        if j is None:
            where = f"{path}" if path is not None else "<in-memory rows>"
            ch = f" (channel {channel})" if channel is not None else ""
            raise PSTraceFormatError(
                f"{where}{ch}: EIS column {wanted!r} not found. "
                f"Header actually read as {header!r}. "
                f"If the first field looks like '\"freq / Hz' the line is double-quoted "
                f"and _split_row should have stripped it; if the whole header arrived as "
                f"one field, the export is field-quoted and needs a different reader.")
        return j
    return {
        "freq": find(lambda h: h.startswith("freq"), "freq / Hz"),
        "z_real": find(lambda h: h.startswith("Z' "), "Z' / Ohm"),
        "z_imag": find(lambda h: h.startswith("Z'' "), "Z'' / Ohm"),
    }


import datetime as _dt  # noqa: E402

_DOSE_RE = re.compile(r"([\d.]+)\s*n[mM]")


def parse_filename(name: str) -> dict | None:
    """Parse ``<deviceid>_<signaltype>_<dose>.csv`` (deviceid may contain ``-``).

    Splits on ``_`` from the right so ``deviceid`` may itself contain underscores.
    ``signaltype`` is ``eis``/``fscv`` (in-vitro) or ``paired`` (in-vivo:
    time-sequential EIS+FSCV per channel in one file). The last field is either a
    concentration like ``0nm``/``100nM`` → ``float`` nM, or a non-numeric token (the
    lowercased string), covering the in-vitro protocol token ``stabilization`` and
    the in-vivo periods ``baseline``/``live``. Returns ``None`` if it doesn't match.
    e.g. ``2-2_fscv_100nm.csv`` → ``{deviceid: '2-2', signaltype: 'fscv', dose: 100.0}``;
    ``2-2_fscv_stabilization.csv`` → ``{..., dose: 'stabilization'}``;
    ``2-2_paired_live.csv`` → ``{deviceid: '2-2', signaltype: 'paired', dose: 'live'}``.
    """
    stem = Path(name).stem
    parts = stem.split("_")
    if len(parts) < 3:
        return None
    dose_token = parts[-1].strip()
    signaltype = parts[-2].lower()
    if signaltype not in ("eis", "fscv", "paired"):
        return None
    dose_m = _DOSE_RE.fullmatch(dose_token)
    dose = float(dose_m.group(1)) if dose_m else dose_token.lower()
    return {"deviceid": "_".join(parts[:-2]), "signaltype": signaltype, "dose": dose}


def parse_folder(name: str) -> dict | None:
    """Parse ``<YYYYMMDD>_<devicetype>_<testtype>`` → date, devicetype, testtype."""
    parts = name.split("_")
    if len(parts) < 3 or not re.fullmatch(r"\d{8}", parts[0]):
        return None
    try:
        date = _dt.datetime.strptime(parts[0], "%Y%m%d").date()
    except ValueError:
        return None
    return {"date": date, "devicetype": "_".join(parts[1:-1]), "testtype": parts[-1].lower()}


# ---- session file discovery ------------------------------------------------------------
#
# Every reader of a corpus (extraction, the QC report, the review modules, the band and
# reliability pre-passes) walks a session folder through session_files(), so the rules
# below hold everywhere at once rather than per caller.

#: Suffix marking a re-measurement: ``<stem>_redo.csv`` replaces ``<stem>.csv``.
REDO_TOKEN = "redo"

#: Non-numeric final fields a recognised export may carry. Anything else in that position
#: is reported rather than silently treated as a protocol token.
PROTOCOL_TOKENS = ("bgd", "baseline", "live")
_PROTOCOL_PREFIXES = ("stabilization",)

_DOSE_PREFIX_RE = re.compile(r"[\d.]+\s*n[mM]")


class UnknownFileSuffix(ValueError):
    """A recognised export name carries a trailing token other than ``_redo``.

    ``2-2_fscv_100nM_redo2.csv`` or ``2-2_eis_0nM_old.csv``: the file is plainly a
    measurement, but which measurement it should stand for is not knowable from its name.
    Guessing either way changes the data, so it raises.
    """


class UnparsedFileWarning(UserWarning):
    """A CSV in a signal session folder does not follow the naming convention and is skipped."""


class PartialRedoWarning(UserWarning):
    """A device's files in one session are a mix of redone and original measurements."""


@dataclass
class SessionNames:
    """What :func:`resolve_session_names` decided about one session folder's file names."""

    files: list = field(default_factory=list)           # [(name, meta)], the files to read
    superseded: dict = field(default_factory=dict)      # original name -> redo name
    unparsed: list = field(default_factory=list)        # names matching no convention
    unknown_tokens: list = field(default_factory=list)  # parsed, but an odd final field
    partial: dict = field(default_factory=dict)         # device -> {"redone", "original"}


def _is_known_token(token: str) -> bool:
    return token in PROTOCOL_TOKENS or token.startswith(_PROTOCOL_PREFIXES)


def resolve_session_names(names) -> SessionNames:
    """Apply the session-folder naming rules to the CSV file names of one folder.

    Pure: it reads names only, so it applies equally to a directory and to a remote listing
    of files that are not on disk. The rules, in order:

    1. A name :func:`parse_filename` accepts is a measurement.
    2. ``<stem>_redo.csv``, where ``<stem>.csv`` would parse, is a re-measurement of the
       same thing and **replaces** ``<stem>.csv`` in that folder, file by file. A redo with
       no original is simply that measurement.
    3. Any other trailing token on a parseable stem (``_redo2``, ``_redo_redo``, ``_old``),
       or a dose field with trailing text (``100nM-redo``), raises
       :class:`UnknownFileSuffix`.
    4. Anything else is recorded in ``unparsed``.

    ``partial`` lists each device whose numeric-dose files mix redone and original ones,
    the case where, for example, a redone dose would be background-subtracted with an
    original 0 nM background from a different run.
    """
    names = sorted(names)
    out = SessionNames()
    chosen: dict[str, tuple[str, dict]] = {}            # stem, lowercased -> (name, meta)
    redone: dict[str, str] = {}                         # original stem -> redo name
    for name in names:
        stem = Path(name).stem
        meta = parse_filename(name)
        if meta is not None:
            if isinstance(meta["dose"], str):
                token = stem.split("_")[-1].strip()
                if _DOSE_PREFIX_RE.match(token):
                    raise UnknownFileSuffix(
                        f"{name}: the dose field {token!r} has trailing text. Only a whole "
                        f"_{REDO_TOKEN} token marks a re-measurement; rename the file to say "
                        f"which measurement it is.")
                if not _is_known_token(meta["dose"]):
                    out.unknown_tokens.append(name)
            chosen.setdefault(stem.lower(), (name, {**meta, "redo": False}))
            continue
        parts = stem.split("_")
        base, extra = None, []
        for k in range(len(parts) - 1, 2, -1):          # longest parseable prefix first
            if parse_filename("_".join(parts[:k]) + ".csv") is not None:
                base, extra = parts[:k], parts[k:]
                break
        if base is None:
            out.unparsed.append(name)
            continue
        if [t.lower() for t in extra] != [REDO_TOKEN]:
            raise UnknownFileSuffix(
                f"{name}: unrecognised suffix {'_' + '_'.join(extra)!r} after "
                f"{'_'.join(base)!r}. Only _{REDO_TOKEN} is understood, and it replaces the "
                f"original file; rename the file to say which measurement it is.")
        key = "_".join(base).lower()
        redone[key] = name
        chosen[key] = (name, {**parse_filename("_".join(base) + ".csv"), "redo": True})
    for name in names:
        key = Path(name).stem.lower()
        if key in redone and name != redone[key]:
            out.superseded[name] = redone[key]
    out.files = sorted(chosen.values())

    by_device: dict[str, dict] = {}
    for name, meta in out.files:
        if not isinstance(meta["dose"], float):
            continue                                    # stabilization etc. pair with nothing
        slot = by_device.setdefault(meta["deviceid"], {"redone": [], "original": []})
        slot["redone" if meta["redo"] else "original"].append(name)
    out.partial = {d: s for d, s in sorted(by_device.items()) if s["redone"] and s["original"]}
    return out


def session_files(folder) -> list[tuple[Path, dict]]:
    """The CSV files to read in one session folder, each with its parsed name.

    Returns ``[(path, meta)]`` sorted by file name, where ``meta`` is
    :func:`parse_filename`'s dict plus ``redo`` (``True`` for a ``_redo`` replacement). The
    rules are :func:`resolve_session_names`'. In a ``signal`` folder, a CSV matching no
    convention, or carrying an unrecognised protocol token, warns
    (:class:`UnparsedFileWarning`) naming the file rather than vanishing. A device mixing
    redone and original files warns (:class:`PartialRedoWarning`) and is still read.
    """
    import warnings

    folder = Path(folder)
    found = {p.name: p for p in folder.glob("*.csv")}
    res = resolve_session_names(found)
    fmeta = parse_folder(folder.name)
    if fmeta is not None and fmeta["testtype"] == "signal":
        for name in res.unparsed:
            warnings.warn(f"{folder.name}/{name}: does not match "
                          f"<device>_<signaltype>_<dose>.csv, so it is skipped.",
                          UnparsedFileWarning, stacklevel=2)
        for name in res.unknown_tokens:
            warnings.warn(f"{folder.name}/{name}: the final field is neither a concentration "
                          f"nor a known protocol token, so no reader treats it as a dose.",
                          UnparsedFileWarning, stacklevel=2)
    meta_of = dict(res.files)
    for dev, s in res.partial.items():
        bg_original = any(meta_of[n]["signaltype"] == "fscv" and meta_of[n]["dose"] == 0.0
                          for n in s["original"])
        fscv_redone = any(meta_of[n]["signaltype"] == "fscv" for n in s["redone"])
        why = (" A redone dose is background-subtracted with the original 0 nM background, "
               "from a different run." if bg_original and fscv_redone else "")
        warnings.warn(f"{folder.name}: device {dev} has redone files {s['redone']} alongside "
                      f"original files {s['original']}, so its rows mix two runs.{why}",
                      PartialRedoWarning, stacklevel=2)
    return [(found[name], meta) for name, meta in res.files]
