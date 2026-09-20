"""Run-wide extraction parameters, pinned so a later extraction cannot drift silently.

Feature extraction derives five parameters from *incidental properties of whatever input
set is present*, applies them run-wide, and records none of them:

=========================  =======================================================
``band`` (when ``"auto"``) a percentile over every EIS spectrum in the corpus
``device_d0``              ``min(dates)`` per device — the origin of every timepoint
``ref_grid``               the first quality-passing channel's EIS frequency grid
``d0_row``                 per-sensor baseline = its earliest timepoint in the input
feature column set         whatever columns the walk happened to emit
=========================  =======================================================

Every one is a function of the *input set*, so extracting over a subset silently
produces a differently-anchored featureset rather than an error: drop a device's
earliest session and every timepoint for that device shifts; stage two sessions and
each sensor's ``d0_row`` re-baselines onto whichever timepoint survived.

A **pin** is that set of derived parameters, captured from one extraction and fed back
into the next. With a pin, extraction uses the recorded values verbatim and never
recomputes them, so a later run over a different subset either reproduces the original
rows exactly or raises :class:`PinMismatch`. That is the property that makes analysis
over selectively-staged data valid.

The record nests under an ``"extraction"`` key so it can live inside a run record
(``run_config.json``) beside ``seed`` and ``profile``; :func:`load_pin` accepts either
the whole run record or the extraction block alone.
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path

import numpy as np

#: The supported public surface. Everything else in this module is an implementation detail of
#: ``extract_dataset`` and may change without notice -- pinning down the names now, because at a
#: public release they would otherwise become a compatibility promise by accident. Re-exported
#: here (rather than only from ``features.extract``) so the reproducibility API reads as one unit.
__all__ = ["PIN_SCHEMA_VERSION", "PinMismatch", "load_pin", "session_key",
           "EmptySessionWarning", "EmptySessionError"]

#: Bumped when the pinned-parameter *shape* changes. Schema 1 recorded only ``band_hz``
#: (the old ``data/manifest.json`` build record); schema 2 adds the four remaining
#: derived globals plus the per-session row counts that catch a partial fetch.
PIN_SCHEMA_VERSION = 2

#: Frequency-grid comparison tolerance — the same ``rtol`` the extraction loop uses when
#: it decides whether a channel sits on the reference grid.
GRID_RTOL = 1e-3


def __getattr__(name):
    """Lazily re-export the empty-session types from ``features.extract``.

    They belong to the same reproducibility contract as the pin, but live in ``extract`` where
    they are raised. Done lazily to avoid a circular import at module load.
    """
    if name in ("EmptySessionWarning", "EmptySessionError"):
        from .extract import EmptySessionError, EmptySessionWarning
        return {"EmptySessionWarning": EmptySessionWarning,
                "EmptySessionError": EmptySessionError}[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


class PinMismatch(ValueError):
    """A pinned extraction parameter disagrees with the data being extracted.

    Raised rather than warned: every condition it covers means the featureset under
    construction is *not* the one the pin describes, so continuing would emit rows that
    look comparable to the pinned featureset but are anchored differently.
    """


def _date(value) -> _dt.date:
    """Coerce a pinned date (ISO string, or a ``date``) to ``datetime.date``."""
    if isinstance(value, _dt.datetime):
        return value.date()
    if isinstance(value, _dt.date):
        return value
    return _dt.date.fromisoformat(str(value))


def session_key(device: str, date) -> str:
    """Identity of one extraction unit: ``"<device>|<YYYY-MM-DD>"``.

    Keyed on the *input* identity (device + folder date) rather than on ``timepoint``,
    because ``timepoint`` is itself derived from ``device_d0`` — a key that moves with
    the thing being pinned cannot verify it.
    """
    return f"{device}|{_date(date).isoformat()}"


def build_pin(*, band, band_source: str, device_d0: dict, ref_grid_hz: dict,
              feature_columns, sessions: dict, params: dict,
              device_types=None, d0_rows: dict | None = None) -> dict:
    """Assemble a schema-2 pin record from the values an extraction just used.

    ``ref_grid_hz`` maps *device type* → frequency grid (see
    :func:`~electropycal.features.extract._first_ref_grid`); ``sessions`` maps
    :func:`session_key` → ``{"timepoint", "n_rows"}``; ``d0_rows`` (only when
    ``d0_normalize`` was on) maps ``"<device>|<channel>"`` → the per-feature baseline
    vector, in ``feature_columns`` order.
    """
    return {
        "schema_version": PIN_SCHEMA_VERSION,
        "extraction": {
            "band_hz": [float(band[0]), float(band[1])],
            # "auto" means the band was a percentile over the corpus present at the time,
            # so it is meaningless to recompute over a subset -- hence worth recording.
            "band_source": band_source,
            "device_d0": {str(d): _date(v).isoformat() for d, v in sorted(device_d0.items())},
            "ref_grid_hz": {str(t): [float(f) for f in np.asarray(g).ravel()]
                            for t, g in sorted(ref_grid_hz.items()) if g is not None},
            "feature_columns": [str(c) for c in feature_columns],
            "device_types": (None if device_types is None
                             else sorted(str(t) for t in device_types)),
            "sessions": {k: {"timepoint": float(v["timepoint"]), "n_rows": int(v["n_rows"])}
                         for k, v in sorted(sessions.items())},
            "d0_rows": (None if d0_rows is None else
                        {str(k): [float(x) for x in np.asarray(v).ravel()]
                         for k, v in sorted(d0_rows.items())}),
            "params": dict(params),
        },
    }


def load_pin(pin) -> dict:
    """Return the ``extraction`` block from ``pin``.

    Accepts a path to a JSON run record, an already-parsed run record (with or without
    the ``"extraction"`` wrapper), or an extraction block directly — so callers can pass
    ``run_config.json`` straight through without knowing how it is nested.
    """
    if isinstance(pin, (str, Path)):
        pin = json.loads(Path(pin).read_text())
    if not isinstance(pin, dict):
        raise TypeError(f"pin must be a path or dict, got {type(pin).__name__}")
    block = pin.get("extraction", pin)
    missing = {"band_hz", "device_d0", "feature_columns"} - set(block)
    if missing:
        raise PinMismatch(
            f"pin is missing required key(s) {sorted(missing)} — it does not look like a "
            f"schema-{PIN_SCHEMA_VERSION} extraction pin. Regenerate it with "
            f"extract_dataset(..., pin_out=...).")
    ver = pin.get("schema_version")
    if ver is not None and int(ver) > PIN_SCHEMA_VERSION:
        raise PinMismatch(f"pin schema_version={ver} is newer than this library supports "
                          f"({PIN_SCHEMA_VERSION}); upgrade electropycal.")
    return block


def pinned_band(block: dict) -> tuple[float, float]:
    lo, hi = block["band_hz"]
    return (float(lo), float(hi))


def pinned_device_d0(block: dict) -> dict:
    """``device_d0`` as ``{device: date}``, ready for the timepoint subtraction."""
    return {str(d): _date(v) for d, v in block["device_d0"].items()}


def pinned_ref_grids(block: dict) -> dict:
    """``{device_type: np.ndarray}`` reference EIS frequency grids."""
    return {str(t): np.asarray(g, dtype=float)
            for t, g in (block.get("ref_grid_hz") or {}).items()}


#: ``ref_grid_hz`` key meaning "one grid shared by every device type" — the pre-per-type
#: behaviour, kept so pins written before the grid was split per device type still load.
ANY_DEVICE_TYPE = "*"


def ref_grid_for(block: dict, devicetype: str):
    """The pinned reference grid for ``devicetype``, or ``None`` if the pin has none.

    Falls back to the :data:`ANY_DEVICE_TYPE` entry, which is how a pin written before
    the grid was anchored per device type records its single shared grid.
    """
    grids = pinned_ref_grids(block)
    got = grids.get(str(devicetype))
    return got if got is not None else grids.get(ANY_DEVICE_TYPE)


def pinned_d0_rows(block: dict) -> dict | None:
    rows = block.get("d0_rows")
    if not rows:
        return None
    return {str(k): np.asarray(v, dtype=float) for k, v in rows.items()}


def check_sessions_against_pin(block: dict, items) -> None:
    """Validate the *input* set against the pin, before any extraction work.

    ``items`` is an iterable of ``(device, date)``. Raises on a device the pin has never
    seen (its ``device_d0`` is unknown, so its timepoints would be invented) or on a
    session dated earlier than its pinned ``d0`` (which is what would have moved the
    baseline had the pin not fixed it).
    """
    d0 = pinned_device_d0(block)
    for dev, date in items:
        dev = str(dev)
        if dev not in d0:
            raise PinMismatch(
                f"device {dev!r} is absent from the pin (pinned devices: "
                f"{sorted(d0)}). Its device_d0 is unknown, so every timepoint it "
                f"produced would be anchored to this input set instead of the pinned "
                f"one. Re-pin over a corpus that includes {dev!r}.")
        if _date(date) < d0[dev]:
            raise PinMismatch(
                f"session {dev}@{_date(date).isoformat()} predates its pinned "
                f"device_d0 ({d0[dev].isoformat()}), so it would shift every timepoint "
                f"for {dev!r} and make these rows incomparable to the pinned featureset. "
                f"Re-pin over the corpus that includes this session.")


def check_feature_columns(block: dict, columns) -> list[str]:
    """Validate emitted columns against the pin; return the pinned order.

    Compares as sets and reports the asymmetric difference, because the useful signal is
    *which* columns appeared or vanished (a schema change, a feature type that produced
    no values on this subset) rather than that the lists differ.
    """
    pinned = [str(c) for c in block["feature_columns"]]
    got = [str(c) for c in columns]
    if set(pinned) != set(got):
        extra, absent = sorted(set(got) - set(pinned)), sorted(set(pinned) - set(got))
        raise PinMismatch(
            "feature columns differ from the pin"
            + (f"; {len(absent)} pinned column(s) absent: {absent[:8]}" if absent else "")
            + (f"; {len(extra)} unpinned column(s) present: {extra[:8]}" if extra else "")
            + ". A featureset with a different column set cannot be scored by a model "
              "frozen against the pinned one.")
    return pinned


def check_row_counts(block: dict, counts: dict) -> None:
    """Validate per-session row counts against the pin.

    ``counts`` maps :func:`session_key` → rows produced. This is the check that catches a
    **partial data fetch**: a session missing its EIS file or its 0 nM background yields
    zero rows without raising anywhere else, so a staged extraction would otherwise
    produce a quietly smaller featureset that still looks well-formed.
    """
    pinned = block.get("sessions") or {}
    bad = []
    for key, value in sorted(counts.items()):
        got = value["n_rows"] if isinstance(value, dict) else value
        if key not in pinned:
            raise PinMismatch(
                f"session {key!r} is absent from the pin (it records "
                f"{len(pinned)} session(s)). Extracting a session the pin has never "
                f"seen means these rows were never part of the pinned featureset.")
        want = int(pinned[key]["n_rows"])
        if int(got) != want:
            bad.append((key, want, int(got)))
    if bad:
        detail = "; ".join(f"{k}: pinned {w} rows, got {g}" for k, w, g in bad[:8])
        raise PinMismatch(
            f"{len(bad)} session(s) produced a different row count than the pin — "
            f"{detail}"
            + ("; ..." if len(bad) > 8 else "")
            + ". The usual cause is an incomplete fetch: a session short its EIS or its "
              "0 nM background returns zero rows without raising, so the featureset "
              "silently shrinks instead of failing.")
