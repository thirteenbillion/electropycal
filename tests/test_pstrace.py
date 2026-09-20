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
    # (Im > 0) fails A even though the lowest frequency is capacitive — a single
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
