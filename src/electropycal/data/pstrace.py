"""PSTrace CSV ingestion: potentiostat export to structured FSCV / EIS arrays.

Parses a PSTrace potentiostat export (UTF-16, CRLF) into structured raw FSCV and
EIS data. Two block layouts observed in real exports:

- **FSCV** — horizontal blocks on one header row, each labelled
  ``Fast Cyclic Voltammetry: FCV i vs E Channel N [rep]`` and occupying two columns
  ``(V, µA)`` = (voltage, current). ``[rep]`` marks replicate cycles (absent = 0).
- **EIS** — vertical blocks, each headed ``CH N: Fixed at K freqs`` followed by a
  column-name row (``freq / Hz``, ``neg. Phase / °``, ``Idc / uA``, ``Z / Ohm``,
  ``Z' / Ohm``, ``Z'' / Ohm``, ``Cs / F``) and ``K`` data rows. The ``Z''`` column holds
  ``-Im(Z)`` and is **negated on ingest** to match ``features.eis``'s ``Im(Z) < 0`` convention
  for a capacitive interface.

A single export corresponds to one device. Measurement protocol / timepoint /
concentration are read from the file and folder names, which follow a fixed convention:

- **file** — ``<deviceid>_<signaltype>_<dose>.csv`` (:func:`parse_filename`), e.g.
  ``2-2_fscv_100nM.csv``, ``2-2_eis_0nM.csv``, ``2-2_fscv_stabilization-end.csv``,
  ``2-2_paired_live.csv``. ``signaltype`` is ``eis``/``fscv`` (in vitro) or ``paired``
  (in vivo); the final field is a concentration (``0nM``, ``100nm`` — case-insensitive)
  or a protocol token (``stabilization``/``baseline``/``live``).
- **folder** — ``<YYYYMMDD>_<devicetype>_<testtype>`` (:func:`parse_folder`), e.g.
  ``20260720_neurostring_signal``. The date supplies the timepoint; ``devicetype``
  separates ``neurostring`` from ``cfme``; ``testtype`` is ``signal`` or ``channeltest``.

The **content** parser is convention-agnostic — only the name parsers above depend on it, so a
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


def read_pstrace(path: str | Path, encoding: str = "utf-16") -> PSTraceExport:
    """Parse a PSTrace CSV export into :class:`PSTraceExport`."""
    path = Path(path)
    rows = [ln.split(",") for ln in path.read_text(encoding=encoding).splitlines()]

    fscv = _parse_fscv(rows)
    eis = _parse_eis(rows)
    date = next((r[1] for r in rows if r and r[0].strip() == "Date and time:"), None)
    return PSTraceExport(fscv=fscv, eis=eis, meta={"source": path.name, "export_date": date})


def _parse_fscv(rows: list[list[str]]) -> dict[tuple[int, int], dict]:
    hdr_idx = next((i for i, r in enumerate(rows)
                    if any("Fast Cyclic" in c for c in r)), None)
    if hdr_idx is None:
        return {}
    # Each "Fast Cyclic ... Channel N ..." header cell is one cycle block. ``rep``
    # is the per-channel column-appearance index (0,1,2,…) — robust to the two
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
    for r in rows[hdr_idx + 3:]:                       # skip header, date, units rows
        if r and (_EIS_RE.match(r[0]) or r[0].strip() == "Measurement:"):
            break
        if r and not np.isnan(_f(r[0])):
            data.append(r)

    out: dict[tuple[int, int], dict] = {}
    for col, ch, rep, scan, bracket in blocks:
        v = np.array([_f(r[col]) if col < len(r) else np.nan for r in data])
        i = np.array([_f(r[col + 1]) if col + 1 < len(r) else np.nan for r in data])
        keep = ~np.isnan(v)
        out[(ch, rep)] = {"voltage": v[keep], "current": i[keep],
                          "scan": scan, "bracket": bracket}
    return out


def _parse_eis(rows: list[list[str]]) -> dict[tuple[int, int], dict]:
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
        ci = _eis_columns(colhdr)
        block = rows[i + 2:i + 2 + k]
        freq = np.array([_f(r[ci["freq"]]) for r in block])
        zr = np.array([_f(r[ci["z_real"]]) for r in block])
        # PSTrace's "Z'' / Ohm" column is -Im(Z) (positive for a capacitive
        # interface at low frequency, as seen in real exports). We negate to the
        # math convention Z = Z' + j·Im(Z) that features.eis expects (capacitive
        # Im < 0). NOTE: the column's NAME suggests it already holds Im(Z); real exports show
        # otherwise, so verify this sign on a known-capacitive channel before trusting any
        # derived capacitance — it is a property of the instrument software, not the physics.
        zi = -np.array([_f(r[ci["z_imag"]]) for r in block])
        rep = rep_counter.get(ch, 0)
        rep_counter[ch] = rep + 1
        out[(ch, rep)] = {"freq": freq, "z_real": zr, "z_imag": zi}
        i += 2 + k
    return out


def _eis_columns(header: list[str]) -> dict[str, int]:
    """Locate freq / Z' / Z'' columns by name (robust to column order)."""
    def find(pred):
        return next(j for j, h in enumerate(header) if pred(h))
    return {
        "freq": find(lambda h: h.startswith("freq")),
        "z_real": find(lambda h: h.startswith("Z' ")),
        "z_imag": find(lambda h: h.startswith("Z'' ")),
    }


import datetime as _dt  # noqa: E402

_DOSE_RE = re.compile(r"([\d.]+)\s*n[mM]")


def parse_filename(name: str) -> dict | None:
    """Parse ``<deviceid>_<signaltype>_<dose>.csv`` (deviceid may contain ``-``).

    Splits on ``_`` from the right so ``deviceid`` may itself contain underscores.
    ``signaltype`` is ``eis``/``fscv`` (in-vitro) or ``paired`` (in-vivo:
    time-sequential EIS+FSCV per channel in one file). The last field is either a
    concentration like ``0nm``/``100nM`` → ``float`` nM, or a non-numeric token — the
    lowercased string — covering the in-vitro protocol token ``stabilization`` and
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
