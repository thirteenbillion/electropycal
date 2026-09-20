"""FSCV stabilization convergence check."""

import numpy as np

from electropycal.data import stabilization as stab


def test_settling_series_converges():
    t = np.arange(60)
    drift = 0.5 * np.exp(-t / 8.0) + 0.001                # settles well below tol
    res = stab.check_converged(drift, tol=0.01, patience=10)
    assert res.converged and res.converged_at is not None
    assert res.final_drift < 0.01


def test_nonsettling_series_does_not_converge():
    rng = np.random.default_rng(0)
    drift = 0.1 + 0.02 * rng.normal(size=60)              # stays high / noisy
    assert not stab.check_converged(drift, tol=0.01, patience=10).converged


def test_cycle_drift_anodic_region_uses_rising_half():
    v = np.concatenate([np.linspace(-0.4, 1.1, 40), np.linspace(1.1, -0.4, 40)])  # triangle
    # cathodic half is noisy across cycles, anodic half is stable
    cycles = np.array([np.concatenate([np.ones(40), np.ones(40) + 5 * (k % 2)]) for k in range(6)])
    full = stab.cycle_drift(cycles, voltage=v, region="full")
    anodic = stab.cycle_drift(cycles, voltage=v, region="anodic")
    assert np.nanmean(anodic) < np.nanmean(full)     # anodic ignores the noisy cathodic half


def test_cycle_drift_decreases_for_settling_cycles():
    n_samp = 50
    steady = np.linspace(0, 1, n_samp)
    transient = np.ones(n_samp)
    cycles = np.array([steady + transient * np.exp(-k / 6.0) for k in range(40)])
    drift = stab.cycle_drift(cycles)
    assert drift[0] > drift[-1]
    assert np.isfinite(stab.estimate_settle_tau(drift))


def test_drift_series_drops_round_start_coldstart_spikes():
    # Build one channel's traces: 3 rounds x 6 cycles. Every round's FIRST cycle is a
    # cold-start (waveform offset by a large step); the rest are near-identical. Real
    # PSTrace exports behave this way (scan 1 of each round sits off the settled value).
    import pandas as pd
    n_samp = 40
    v = np.concatenate([np.linspace(-0.4, 1.1, n_samp // 2),
                        np.linspace(1.1, -0.4, n_samp // 2)])
    rows, cyc = [], 0
    for rnd in range(3):
        for k in range(6):
            base = np.ones(n_samp) + 0.001 * k
            wave = base + (5.0 if k == 0 else 0.0)          # cold-start on scan 1
            rows.append({"channel": 1, "cycle": cyc, "round": rnd, "cycle_in_round": k,
                         "scan": k + 1, "_voltage": v, "_current": wave})
            cyc += 1
    t = pd.DataFrame(rows)

    raw = stab.drift_series(t, region="anodic", drop_round_boundaries=False)
    filt = stab.drift_series(t, region="anodic", drop_round_boundaries=True)
    # cold-starts create large spikes in the raw series...
    assert np.nanmax(raw) > 1.0
    # ...and the boundary filter removes every one of them (incl. the first round's).
    assert np.nanmax(filt) < 0.1
    assert filt.size < raw.size


def test_round_drift_smoother_than_per_cycle_and_converges():
    # 12 rounds x 20 cycles for one channel: a round-level settling envelope, plus a
    # per-round cold-start (scan 1), occasional glitch cycles, and per-cycle noise.
    import pandas as pd
    rng = np.random.default_rng(1)
    n_samp = 60
    v = np.concatenate([np.linspace(-0.4, 1.1, 30), np.linspace(1.1, -0.4, 30)])
    rows, cyc = [], 0
    for rnd in range(12):
        settle = 0.6 * np.exp(-rnd / 3.0)
        for k in range(20):
            base = (1 + settle) * np.sin(np.pi * (v + 0.4) / 1.5) + 1.0
            wave = base + (0.5 if k == 0 else 0.0)                 # cold-start
            if rnd < 6 and rng.random() < 0.08:                    # glitches only while settling
                wave = wave + 0.4
            wave = wave + 0.01 * rng.normal(size=n_samp)           # per-cycle noise
            rows.append({"channel": 1, "cycle": cyc, "round": rnd, "cycle_in_round": k,
                         "scan": k + 1, "_voltage": v, "_current": wave})
            cyc += 1
    t = pd.DataFrame(rows)

    rounds, avg, volt = stab.round_average_cycles(t)
    assert rounds.tolist() == list(range(12)) and avg.shape[0] == 12

    per_cycle = stab.drift_series(t, region="anodic")
    per_round = stab.round_drift_series(t, region="anodic")
    assert per_round.size == 11                                    # n_rounds - 1
    # round-averaging is far smoother: much smaller peak drift than per-cycle...
    assert np.nanmax(per_round) < 0.5 * np.nanmax(per_cycle)
    # ...it decreases monotonically-ish (envelope settles, not jitter)...
    assert np.nanmean(per_round[-3:]) < np.nanmean(per_round[:3])
    # ...and it settles to convergence (in round units)
    assert stab.check_converged(per_round, tol=0.02, patience=3, smooth_window=3).converged


def test_round_drift_empty_for_single_round():
    import pandas as pd
    v = np.linspace(-0.4, 1.1, 20)
    rows = [{"channel": 1, "cycle": k, "round": 0, "cycle_in_round": k, "scan": k + 1,
             "_voltage": v, "_current": np.ones(20) + 0.01 * k} for k in range(20)]
    assert stab.round_drift_series(pd.DataFrame(rows)).size == 0


def test_stabilization_table_shape():
    drift = 0.3 * np.exp(-np.arange(30) / 5.0)
    df = stab.stabilization_table(drift)
    assert list(df.columns) == ["cycle", "drift", "smoothed", "below_tol"]
    assert len(df) == 30


def test_i_at_v_target_interpolates_on_anodic_sweep():
    up = np.linspace(-0.4, 1.1, 40)
    v = np.concatenate([up, up[::-1]])          # full triangle
    cur = np.concatenate([up, up[::-1]]) * 10   # current tracks voltage
    val = stab.i_at_v_target(v, cur, v_target=0.7)
    assert np.isclose(val, 7.0, atol=0.3)


def test_synthetic_writes_start_and_end_files(tmp_path):
    import pathlib

    from electropycal.data.pstrace import parse_filename, read_pstrace
    from electropycal.data.synthetic import write_synthetic_pstrace_dir

    root = write_synthetic_pstrace_dir(tmp_path)
    starts = sorted(pathlib.Path(root).rglob("*_fscv_stabilization-start.csv"))
    ends = sorted(pathlib.Path(root).rglob("*_fscv_stabilization-end.csv"))
    assert starts and ends
    assert parse_filename(starts[0].name)["dose"] == "stabilization-start"
    assert parse_filename(ends[0].name)["dose"] == "stabilization-end"

    # the 'end' file should be better-converged than the 'start' file
    def n_converged(path):
        tr = stab.stabilization_traces(read_pstrace(path), cycles_per_round=20)
        n = 0
        for ch, t in tr.groupby("channel"):
            dr = stab.cycle_drift(np.vstack(t.sort_values("cycle")["_current"].to_numpy()))
            n += stab.check_converged(dr, tol=0.02, patience=10).converged
        return n
    assert n_converged(ends[0]) >= n_converged(starts[0])


def test_stabilization_traces_from_synthetic_export(tmp_path):
    import pathlib

    from electropycal.data.pstrace import read_pstrace
    from electropycal.data.synthetic import write_synthetic_pstrace_dir

    root = write_synthetic_pstrace_dir(tmp_path)
    stab_file = sorted(pathlib.Path(root).rglob("*_fscv_stabilization-end.csv"))[0]
    exp = read_pstrace(stab_file)
    df = stab.stabilization_traces(exp, cycles_per_round=20)
    # 3 rounds x 20 cycles = 60 cycles per channel; round derived from cycle index
    per_ch = df.groupby("channel").size()
    assert (per_ch == 60).all()
    assert set(df["round"].unique()) == {0, 1, 2}
    # I(V_target) settles: last cycle below the first for each channel
    for ch, t in df.groupby("channel"):
        t = t.sort_values("cycle")
        assert t.i_target.iloc[-1] < t.i_target.iloc[0]
        # full-cycle drift converges
        cyc = np.vstack(t["_current"].to_numpy())
        assert stab.check_converged(stab.cycle_drift(cyc), tol=0.01, patience=10).converged
