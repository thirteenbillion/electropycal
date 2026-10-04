"""PSTrace ingestion tests against a synthetic fixture in the real export format."""

import numpy as np

from electropycal.data.pstrace import parse_filename, read_pstrace
from electropycal.data.quality import eis_quality

# Minimal PSTrace-format export: FSCV channel 3 with 2 replicate cycles (horizontal
# blocks, V/µA columns), then an EIS block for CH 3 (vertical, 3 freqs). EIS "Z''"
# is stored as -Im(Z): positive at low freq (capacitive), negative at high freq.
FIXTURE = "\n".join([
    "Date and time:,2025-01-01 12:00:00",
    "Notes:",
    ",,,,",
    "Fast Cyclic Voltammetry: FCV i vs E Channel 3,,Fast Cyclic Voltammetry: FCV i vs E Channel 3 [1],",
    "Date and time measurement:,2025-01-01,Date and time measurement:,2025-01-01",
    "V,µA,V,µA",
    "-0.4,0.10,-0.4,0.11",
    "0.7,0.50,0.7,0.52",
    "1.1,0.20,1.1,0.19",
    "Measurement:,Impedance Spectroscopy",
    "Notes:,",
    "Date and time:,2025-01-01 12:01:00",
    "CH 3: Fixed at 3 freqs",
    "freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z' / Ohm,Z'' / Ohm,Cs / F",
    "100000,-5,1,4000,3990,-60,1e-11",
    "1000,-40,1,50000,40000,30000,1e-9",
    "10,-70,1,600000,200000,560000,1e-8",
])


def _write(tmp_path, encoding="utf-16"):
    p = tmp_path / "M11_dev_D0.csv"
    p.write_text(FIXTURE, encoding=encoding)
    return p


def test_parses_fscv_channels_and_replicates(tmp_path):
    exp = read_pstrace(_write(tmp_path))
    assert exp.fscv_channels == [3]
    assert (3, 0) in exp.fscv and (3, 1) in exp.fscv          # base + replicate cycle
    cyc = exp.fscv[(3, 0)]
    assert len(cyc["voltage"]) == 3
    assert cyc["voltage"][1] == 0.7 and cyc["current"][1] == 0.50


def test_parses_eis_and_applies_sign_convention(tmp_path):
    exp = read_pstrace(_write(tmp_path))
    assert exp.eis_channels == [3]
    s = exp.eis[(3, 0)]
    assert len(s["freq"]) == 3
    # stored Z'' = -Im(Z); parser returns Im(Z): capacitive (low freq) must be < 0
    low = np.argmin(s["freq"])
    assert s["z_imag"][low] < 0            # 10 Hz stored +560000 -> Im = -560000
    high = np.argmax(s["freq"])
    assert s["z_imag"][high] > 0           # 100 kHz stored -60 -> Im = +60 (inductive)


def test_eis_quality_passes_for_capacitive_spectrum(tmp_path):
    # the fixture is capacitive at 10 Hz & 1 kHz but inductive at 100 kHz (Im=+60);
    # with a band that excludes that high-f inductive tail the spectrum is valid.
    s = read_pstrace(_write(tmp_path)).eis[(3, 0)]
    q = eis_quality(s["freq"], s["z_real"], s["z_imag"], band=(10, 1000))
    assert q["valid"] and q["A_capacitive"] and q["B_monotonic"] and q["C_environment"]


def test_eis_quality_A_flags_inband_inductive_excursion(tmp_path):
    # check A is now all-in-band: a band that INCLUDES the 100 kHz inductive point
    # (Im > 0) fails A even though the lowest frequency is capacitive; a single
    # low-f anchor would have wrongly passed it.
    s = read_pstrace(_write(tmp_path)).eis[(3, 0)]
    q = eis_quality(s["freq"], s["z_real"], s["z_imag"], band=(10, 100_000))
    assert not q["A_capacitive"] and not q["valid"]


def test_utf8_export_also_parses(tmp_path):
    # be tolerant of either encoding
    exp = read_pstrace(_write(tmp_path, encoding="utf-8"), encoding="utf-8")
    assert exp.fscv_channels == [3] and exp.eis_channels == [3]


def test_parse_filename_numeric_dose():
    assert parse_filename("2-2_fscv_100nm.csv") == {
        "deviceid": "2-2", "signaltype": "fscv", "dose": 100.0}
    assert parse_filename("2-2_eis_0nM.csv")["dose"] == 0.0


def test_parse_filename_stabilization_token():
    # non-numeric protocol tokens (e.g. stabilization) parse to the lowercased string
    m = parse_filename("2-2_fscv_stabilization.csv")
    assert m == {"deviceid": "2-2", "signaltype": "fscv", "dose": "stabilization"}
    assert isinstance(m["dose"], str)


def test_parse_filename_invivo_paired_and_periods():
    assert parse_filename("2-2_paired_baseline.csv") == {
        "deviceid": "2-2", "signaltype": "paired", "dose": "baseline"}
    assert parse_filename("2-2_paired_live.csv")["dose"] == "live"
    assert parse_filename("2-2_eis_baseline.csv")["signaltype"] == "eis"


def test_parse_filename_rejects_bad_signaltype():
    assert parse_filename("2-2_impedance_100nm.csv") is None
    assert parse_filename("only_two.csv") is None


def test_eis_quality_exposes_z_rise_statistic():
    import numpy as np
    freq = np.logspace(1, 4, 12)
    zr = 1000.0 / np.sqrt(freq)                 # |Z| decreasing with frequency
    zi = -1e-3 * np.ones_like(freq)             # tiny capacitive term -> |Z| ~ zr, A passes
    q = eis_quality(freq, zr, zi, band=(10, 10000), mono_tol=0.05)
    assert q["z_rise_max"] <= 0.05 and q["B_monotonic"]
    # force a ~30% |Z| rise at one step -> B fails, z_rise_max reflects it
    zr2 = zr.copy(); zr2[6] = zr[5] * 1.3
    q2 = eis_quality(freq, zr2, zi, band=(10, 10000), mono_tol=0.05)
    assert q2["z_rise_max"] > 0.2 and not q2["B_monotonic"]
    # B passes iff z_rise_max <= mono_tol (the sweep uses exactly this equivalence)
    assert eis_quality(freq, zr2, zi, band=(10, 10000), mono_tol=q2["z_rise_max"] + 1e-9)["B_monotonic"]


# ---------------------------------------------------------------------------------------
# Wrapped exports. An export is sometimes encountered with its lines wrapped in double
# quotes, and has to read identically to an unwrapped one.
#
# The wrapping is SELECTIVE WITHIN A FILE. Measured per line class on a real 0 nM EIS
# export of 969 lines and 24 blocks: the metadata lines, the EIS column header and the
# data rows are wrapped, while the `CH N:` block header is BARE, as are blank lines.
#
# That distinction is the whole bug, so the fixture has to carry it. With the `CH` line
# wrapped, `_EIS_RE` (anchored, `match`) never matches, no block is found, and the file
# parses to nothing quietly. With the `CH` line BARE, which is the real layout, the block
# IS found, the wrapped header then splits to `['"freq / Hz', ..., '"']`, and the column
# finder fails on it. A fixture that wrapped every line tested the wrong failure.
# ---------------------------------------------------------------------------------------

import pytest

from electropycal.data.pstrace import PSTraceFormatError, _eis_columns, _split_row

#: The real variant: the same measurements as ``FIXTURE``, wrapped the way a real export
#: is wrapped. Faithful in four details taken off the bytes of one: the bare ``CH`` line,
#: the trailing comma on the EIS header and data rows (so each carries a trailing empty
#: field), a ``File date:`` line, and a stray BOM *inside* the wrapper.
#:
#: The FSCV half is left unwrapped on purpose. That is the observed pairing, an EIS export
#: wrapped while the FSCV export recorded beside it is not, and it makes this one file
#: exercise both styles interleaved, which is the property ``_split_row`` has to hold.
FIXTURE_MIXED = "\n".join([
    '"﻿File date:,2025-01-01 12:00:00"',
    "Date and time:,2025-01-01 12:00:00",
    "Notes:",
    ",,,,",
    "Fast Cyclic Voltammetry: FCV i vs E Channel 3,,Fast Cyclic Voltammetry: FCV i vs E Channel 3 [1],",
    "Date and time measurement:,2025-01-01,Date and time measurement:,2025-01-01",
    "V,µA,V,µA",
    "-0.4,0.10,-0.4,0.11",
    "0.7,0.50,0.7,0.52",
    "1.1,0.20,1.1,0.19",
    '"Measurement:,Impedance Spectroscopy"',
    '"Notes:,"',
    '"Date and time:,2025-01-01 12:01:00"',
    "CH 3: Fixed at 3 freqs",
    '"freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z\' / Ohm,Z\'\' / Ohm,Cs / F,"',
    '"100000,-5,1,4000,3990,-60,1e-11,"',
    '"1000,-40,1,50000,40000,30000,1e-9,"',
    '"10,-70,1,600000,200000,560000,1e-8,"',
])

#: Every non-blank line wrapped, the ``CH`` line included. This layout has not been seen;
#: it is kept because it is the other thing a wrapping tool could plausibly do, and
#: because it is the case that fails *silently* rather than loudly, so the "no blocks"
#: guard needs a fixture. Do not read it as evidence about any real export.
FIXTURE_ALL_WRAPPED = "\n".join(
    ('"' + ln + '"') if ln else ln for ln in FIXTURE.splitlines())


def _write_mixed(tmp_path, encoding="utf-16"):
    p = tmp_path / "M11_dev_D0_mixed.csv"
    p.write_text(FIXTURE_MIXED, encoding=encoding)
    return p


def _write_all_wrapped(tmp_path, encoding="utf-16"):
    p = tmp_path / "M11_dev_D0_allwrapped.csv"
    p.write_text(FIXTURE_ALL_WRAPPED, encoding=encoding)
    return p


def test_the_mixed_fixture_really_is_selectively_wrapped():
    """Guard the fixture itself, so the tests below cannot pass for the wrong reason.

    Specifically: the ``CH`` line must stay BARE. If a future edit wraps it, every test
    here would still pass while exercising the silent failure instead of the real one.
    """
    lines = FIXTURE_MIXED.splitlines()

    def wrapped(ln):
        return len(ln) >= 2 and ln.startswith('"') and ln.endswith('"')

    ch = next(ln for ln in lines if ln.startswith("CH "))
    assert not wrapped(ch), "the CH block header is bare in a real export; keep it bare"

    hdr = next(ln for ln in lines if "freq / Hz" in ln)
    assert wrapped(hdr)
    # the exact breakage: a naive split leaves a stray quote on the first field
    assert hdr.split(",")[0] == '"freq / Hz'
    assert not hdr.split(",")[0].startswith("freq")

    # data rows wrapped, metadata wrapped, FSCV half bare: all three in one file
    assert wrapped('"10,-70,1,600000,200000,560000,1e-8,"')
    assert wrapped(next(ln for ln in lines if ln.startswith('"Measurement:')))
    assert not wrapped(next(ln for ln in lines if "Fast Cyclic" in ln))
    assert not wrapped(next(ln for ln in lines if ln.startswith("V,")))

    # and the trailing comma the real rows carry, giving a trailing empty field
    assert _split_row(hdr)[-1] == ""
    assert len(_split_row(hdr)) == 8


@pytest.mark.parametrize("encoding", ["utf-16", "utf-8"])
def test_mixed_and_unwrapped_exports_parse_identically(tmp_path, encoding):
    """The property that matters: same measurements, two styles, bit-identical arrays.

    Both encodings are exercised because the reader takes an ``encoding`` argument and
    real exports are UTF-16; a fix that only worked for one would be a trap.
    """
    plain = read_pstrace(_write(tmp_path, encoding), encoding=encoding)
    mixed = read_pstrace(_write_mixed(tmp_path, encoding), encoding=encoding)

    assert sorted(mixed.fscv) == sorted(plain.fscv)
    assert sorted(mixed.eis) == sorted(plain.eis)
    for key in plain.fscv:
        for f in ("voltage", "current"):
            # tobytes(), not allclose: the claim is bit-identity, not agreement
            assert mixed.fscv[key][f].tobytes() == plain.fscv[key][f].tobytes()
        assert mixed.fscv[key]["scan"] == plain.fscv[key]["scan"]
        assert mixed.fscv[key]["bracket"] == plain.fscv[key]["bracket"]
    for key in plain.eis:
        for f in ("freq", "z_real", "z_imag"):
            assert mixed.eis[key][f].tobytes() == plain.eis[key][f].tobytes()
    assert mixed.meta["export_date"] == plain.meta["export_date"]


def test_all_wrapped_export_also_parses_identically(tmp_path):
    """The unobserved layout still has to work; ``_split_row`` is per line, not per file."""
    plain = read_pstrace(_write(tmp_path))
    allw = read_pstrace(_write_all_wrapped(tmp_path))
    assert sorted(allw.eis) == sorted(plain.eis)
    assert sorted(allw.fscv) == sorted(plain.fscv)
    for key in plain.eis:
        for f in ("freq", "z_real", "z_imag"):
            assert allw.eis[key][f].tobytes() == plain.eis[key][f].tobytes()


def test_mixed_export_recovers_real_values_not_just_matching_emptiness(tmp_path):
    """Equality above would also hold if both parsed to nothing. Pin actual numbers."""
    exp = read_pstrace(_write_mixed(tmp_path))
    assert exp.eis_channels == [3]
    assert exp.fscv_channels == [3]
    eis = exp.eis[(3, 0)]
    assert len(eis["freq"]) == 3
    assert eis["freq"][0] == 100000.0
    assert eis["z_real"][1] == 40000.0
    assert eis["z_imag"][1] == -30000.0          # negated on ingest
    cyc = exp.fscv[(3, 0)]
    assert cyc["voltage"][1] == 0.7 and cyc["current"][1] == 0.50


def test_the_trailing_empty_field_does_not_shift_eis_columns(tmp_path):
    """The real rows end in a comma. Columns are found by name, so it must not matter."""
    mixed = read_pstrace(_write_mixed(tmp_path))
    plain = read_pstrace(_write(tmp_path))
    assert mixed.eis[(3, 0)]["z_imag"].tobytes() == plain.eis[(3, 0)]["z_imag"].tobytes()


def test_a_bom_inside_the_wrapper_does_not_lose_the_export_date(tmp_path):
    """A real wrapped first line reads ``"<BOM>File date:,...``: the BOM is INSIDE.

    It is harmless on that line only by luck of which line came first. On the
    ``Date and time:`` line the same quirk would defeat an exact-string lookup and drop
    ``export_date`` with no error, so the BOM is stripped rather than left to chance.
    """
    assert _split_row('"﻿File date:,2026-09-15 13:11:36"')[0] == "File date:"
    p = tmp_path / "bom_on_the_date_line.csv"
    p.write_text("\n".join([
        '"﻿Date and time:,2026-09-15 10:00:00"',
        "CH 7: Fixed at 2 freqs",
        '"freq / Hz,neg. Phase / °,Idc / uA,Z / Ohm,Z\' / Ohm,Z\'\' / Ohm,Cs / F,"',
        '"1000,-40,1,50000,40000,30000,1e-9,"',
        '"10,-70,1,600000,200000,560000,1e-8,"',
    ]), encoding="utf-16")
    assert read_pstrace(p).meta["export_date"] == "2026-09-15 10:00:00"


def test_a_wrapped_header_reaching_the_column_finder_raises_not_stopiteration(tmp_path):
    """The exact site that used to raise a bare ``StopIteration`` naming nothing.

    This is the real failure path: the bare ``CH`` line means the block IS found, so a
    wrapped header reaches the column finder. If ``_split_row`` ever stops stripping the
    wrapper, this is the error that must come out instead.
    """
    with pytest.raises(PSTraceFormatError) as ei:
        _eis_columns(['"freq / Hz', "neg. Phase / °", "Z' / Ohm", "Z'' / Ohm", '"'],
                     path=tmp_path / "wrapped_header.csv", channel=10)
    msg = str(ei.value)
    assert "wrapped_header.csv" in msg and "channel 10" in msg
    assert "freq / Hz" in msg
    assert "double-quoted" in msg


def test_an_export_with_no_blocks_raises_naming_the_file(tmp_path):
    p = tmp_path / "empty_export.csv"
    p.write_text("File date:,2026-09-15 10:00:00\nNotes:,\n", encoding="utf-16")
    with pytest.raises(PSTraceFormatError) as ei:
        read_pstrace(p)
    msg = str(ei.value)
    assert "empty_export.csv" in msg
    assert "no FSCV or EIS blocks found" in msg
    # must not be confused with the other two failures
    assert "not found" not in msg and "recovered 0 samples" not in msg


def test_an_unlocatable_eis_column_raises_naming_column_and_header(tmp_path):
    """The bare ``next()`` that used to raise StopIteration naming nothing."""
    p = tmp_path / "odd_columns.csv"
    p.write_text("\n".join([
        "Date and time:,2026-09-15 10:00:00",
        "CH 7: Fixed at 2 freqs",
        "frequency,phase,Idc,Zmag,ZReal,ZImag,Cs",
        "1000,-40,1,50000,40000,30000,1e-9",
        "10,-70,1,600000,200000,560000,1e-8",
    ]), encoding="utf-16")
    with pytest.raises(PSTraceFormatError) as ei:
        read_pstrace(p)
    msg = str(ei.value)
    assert "odd_columns.csv" in msg
    assert "channel 7" in msg
    assert "freq / Hz" in msg                    # the column it wanted
    assert "ZReal" in msg                        # the header it actually found
    assert "no FSCV or EIS blocks found" not in msg
    assert "recovered 0 samples" not in msg


def test_fscv_blocks_with_no_parseable_samples_raise(tmp_path):
    """Precautionary rather than observed: wrapping has been seen on an EIS export only.

    Kept because it is exactly what a wrapped FSCV export would do, every row failing the
    numeric test so the blocks are found and hold nothing.
    """
    p = tmp_path / "unparseable_rows.csv"
    p.write_text("\n".join([
        "Date and time:,2026-09-15 10:00:00",
        "Fast Cyclic Voltammetry: FCV i vs E Channel 1,",
        "Date and time measurement:,2026-09-15",
        "V,µA",
        "not-a-number,also-not",
        "still-not,nope",
    ]), encoding="utf-16")
    with pytest.raises(PSTraceFormatError) as ei:
        read_pstrace(p)
    msg = str(ei.value)
    assert "unparseable_rows.csv" in msg
    assert "recovered 0 samples" in msg
    assert "2 rejected" in msg
    assert "no FSCV or EIS blocks found" not in msg
    assert "not found" not in msg


def test_escaped_inner_quotes_are_undoubled(tmp_path):
    """RFC 4180 doubling inside a whole-line wrapper, so a future export is read right."""
    assert _split_row('"a,b""c,d"') == ["a", 'b"c', "d"]
    assert _split_row("a,b,c") == ["a", "b", "c"]
    assert _split_row('"a,b,c"') == ["a", "b", "c"]
