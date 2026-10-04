"""Discovery orchestration: run the task queue, checkpoint, and rank the results.

Outer layer: conditions run **sequentially** as a decision-gated task queue.
Intermediate layer: joblib parallelizes outer CV folds inside each condition
(``Profile.n_jobs``), one single-threaded worker each; call
:func:`pin_blas_single_threaded` *before* importing numpy in the entry process to
avoid oversubscription. Writes the run directory (``run_config.json``, per-
condition outputs, ``summary.parquet``, ``report/``).
"""

from __future__ import annotations

import datetime as _dt
import os
from pathlib import Path

import numpy as np
import pandas as pd

from ..data.io import read_json, write_json, write_parquet
from .config import Condition, Profile, RunData, baseline_queue
from .runner import run_condition


def pin_blas_single_threaded() -> None:
    """Set BLAS/OpenMP thread env vars to 1. Call before importing numpy.

    The outer CV folds are parallelized with joblib, so letting each worker also spin up a full
    BLAS thread pool oversubscribes the machine and runs slower than single-threaded workers.
    The env vars only take effect if set before numpy is imported.
    """
    for var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS", "NUMEXPR_NUM_THREADS"):
        os.environ.setdefault(var, "1")


def _call_batch_gate(batch_gate, batch_num: int, ranking: list[dict],
                     remaining: list[Condition]):
    """Call ``batch_gate`` with 2 or 3 args depending on its signature."""
    import inspect
    try:
        nparams = len(inspect.signature(batch_gate).parameters)
    except (TypeError, ValueError):
        nparams = 2
    return (batch_gate(batch_num, ranking, remaining) if nparams >= 3
            else batch_gate(batch_num, ranking))


def auto_flag_conditions(ranking: list[dict], remaining: list[Condition]) -> dict[str, str]:
    """Advisory downstream exclusions from completed results.

    Flags a remaining condition when **every** completed condition sharing its
    architecture is dominated by the current leader under interval separation: its
    best-case CI (``rmsep_ci_lo``) is still worse than the leader's worst-case CI
    (``rmsep_ci_hi``). Returns ``{condition_name: reason}``. Purely advisory: the
    caller decides whether to exclude (accept the flag) or include anyway.
    """
    import math
    from collections import defaultdict
    done = [r for r in ranking if r.get("pooled_rmsep") is not None
            and math.isfinite(r["pooled_rmsep"])]
    if not done:
        return {}
    leader = min(done, key=lambda r: r["pooled_rmsep"])
    lead_hi = leader.get("rmsep_ci_hi") or leader["pooled_rmsep"]
    arch_best_lo: dict[str, float] = defaultdict(lambda: math.inf)
    for r in done:                                        # best (lowest) CI-lo per architecture
        lo = r.get("rmsep_ci_lo") or r["pooled_rmsep"]
        arch_best_lo[r["architecture"]] = min(arch_best_lo[r["architecture"]], lo)
    dominated = {a for a, lo in arch_best_lo.items()
                 if lo > lead_hi and a != leader["architecture"]}
    return {c.name: f"{c.architecture} dominated by '{leader['condition']}' (CIs separate)"
            for c in remaining if c.architecture in dominated}


def run_discovery(data: RunData, conditions: list[Condition] | None = None,
                  out_root: str | Path = "outputs", profile: Profile | None = None,
                  seed: int = 0, gate=None, batch_gate=None, progress: bool = False,
                  cap_min_train_times: bool = True, provenance_root: str | Path | None = None,
                  target: str | None = None) -> Path:
    """Run the discovery queue; return the timestamped run directory.

    Decision gating is **opt-in**: by default every condition runs.
    - ``gate(condition, agg, history) -> bool``: per-condition; return False to stop.
    - ``batch_gate(...) -> bool | list[Condition]``: called after each task-queue
      batch completes. Accepts either ``(batch_num, ranking)`` or
      ``(batch_num, ranking, remaining)`` (arity is auto-detected). Return value:
      ``True`` proceed unchanged; ``False``/``None`` stop the queue; a
      ``list[Condition]`` **replaces the remaining queue** (drop conditions to
      exclude, or re-insert flagged ones to include); this is how batches are
      edited between runs. Use :func:`auto_flag_conditions` to get advisory
      exclusions from the results so far. The CLI's ``--batched`` mode passes an
      interactive ``batch_gate`` that shows the ranking + flags and applies edits.

    ``progress=True`` emits timestamped ``[HH:MM:SS]`` lines (a per-condition
    banner plus each condition's own START/fold/DONE lines) for CLI/script runs
    (default silent, so notebooks and tests stay quiet).
    """
    conditions = conditions or baseline_queue()
    profile = profile or Profile()
    # A full queue can mix linear and log conditions; the log model needs a strictly positive target.
    # For a signed target (e.g. a calibration-curve intercept/curvature, or a slope that can go
    # negative), drop the log conditions here so the run proceeds on the compatible ones instead of
    # aborting; a single explicit log-on-signed-target call still raises in run_condition.
    if np.any(np.asarray(data.y, float) <= 0):
        _log_conds = [c for c in conditions if getattr(c, "transform", "linear") == "log"]
        if _log_conds:
            conditions = [c for c in conditions if getattr(c, "transform", "linear") != "log"]
            print(f"note: target has non-positive values; skipping {len(_log_conds)} log condition(s) "
                  f"(log needs a strictly positive target); running {len(conditions)} linear conditions.",
                  flush=True)
    if provenance_root is not None:                           # provenance of the pre-processing (informational)
        from ..analysis_config import print_provenance
        print_provenance(provenance_root, target=target)
    if cap_min_train_times:                                # cap ONCE for the whole run (not per condition)
        import dataclasses
        import warnings

        from .config import effective_min_train_times
        _n_tp = int(np.unique(np.asarray(data.timepoint)).size)
        _eff = effective_min_train_times(profile.min_train_times, _n_tp)
        if _eff != profile.min_train_times:
            warnings.warn(
                f"min_train_times={profile.min_train_times} needs >= {profile.min_train_times + 1} "
                f"distinct timepoints but the data has {_n_tp}; capping to {_eff} so forward-chained "
                f"CV yields folds (otherwise 0 folds -> nan). Pass cap_min_train_times=False to opt out.",
                stacklevel=2)
            profile = dataclasses.replace(profile, min_train_times=_eff)
    ts = _dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = Path(out_root) / f"model_discovery_{ts}"
    (run_dir / "logs").mkdir(parents=True, exist_ok=True)
    log = (run_dir / "logs" / "run.log").open("w", encoding="utf-8")

    # Carry the featureset's extraction pin into this run's record, when the data root has one.
    # The run record then states BOTH halves of what produced these numbers: the modelling knobs
    # here, and the run-wide parameters extraction derived (band, device_d0, ref_grid, d0_rows,
    # feature columns). Without it a run is reproducible only as far back as the parquet.
    _extraction = None
    if provenance_root is not None:
        _src = Path(provenance_root) / "run_config.json"
        if _src.exists():
            try:
                from ..features.pin import load_pin
                _extraction = load_pin(read_json(_src))
            except Exception as _e:                    # provenance must never break a run
                print(f"(extraction pin not carried forward: {_e})", flush=True)
    write_json(run_dir / "run_config.json", {
        "timestamp": ts, "seed": seed, "target": target,
        "profile": {"seeds": list(profile.seeds), "n_jobs": profile.n_jobs,
                    "n_boot": profile.n_boot, "min_train_times": profile.min_train_times,
                    "cap_min_train_times": cap_min_train_times, "cars": profile.cars},
        "conditions": [c.name for c in conditions],
        "dataset": {"n_samples": int(data.X.shape[0]), "n_features": int(data.X.shape[1])},
        "extraction": _extraction,
    })

    all_rows: list[dict] = []
    ranking: list[dict] = []
    i = 0
    while i < len(conditions):                            # while-loop: the queue is editable
        cond = conditions[i]
        if progress:
            from .runner import _log
            _log(f"CONDITION {i + 1}/{len(conditions)}: {cond.name} (batch {cond.batch})")
        rows, agg = run_condition(cond, data, run_dir, profile, seed=seed, progress=progress,
                                  cap_min_train_times=False)   # already capped once above
        all_rows.extend(rows)
        ranking.append({"condition": cond.name, "architecture": cond.architecture,
                        "track": cond.track, "selector": cond.selector,
                        "batch": cond.batch,
                        "pooled_rmsep": agg.get("pooled_rmsep"),
                        "rmsep_ci_lo": agg.get("rmsep_ci_lo"),
                        "rmsep_ci_hi": agg.get("rmsep_ci_hi"),
                        "macro_rmsep": agg.get("macro_rmsep"),
                        "pooled_q2": agg.get("pooled_q2"), "n_folds": agg.get("n_folds"),
                        "n_seeds": agg.get("n_seeds")})
        _rmsep, _q2 = agg.get("pooled_rmsep"), agg.get("pooled_q2")
        log.write(f"{cond.name}: RMSEP={_rmsep:.4f} " if _rmsep is not None
                  else f"{cond.name}: RMSEP=n/a ")
        log.write(f"Q2={_q2:.4f} " if _q2 is not None else "Q2=n/a ")
        log.write(f"folds={agg.get('n_folds')}\n")
        log.flush()
        if gate is not None and not gate(cond, agg, ranking):
            log.write(f"gate stopped queue after {cond.name}\n")
            break
        is_batch_end = (i == len(conditions) - 1) or (conditions[i + 1].batch != cond.batch)
        if batch_gate is not None and is_batch_end:
            remaining = conditions[i + 1:]
            decision = _call_batch_gate(batch_gate, cond.batch, ranking, remaining)
            if decision is False or decision is None:
                log.write(f"batch_gate stopped queue after batch {cond.batch}\n")
                break
            if isinstance(decision, list):               # edited remaining queue
                conditions = conditions[:i + 1] + list(decision)
                log.write(f"batch_gate edited queue after batch {cond.batch} -> "
                          f"{[c.name for c in decision]}\n")
        i += 1

    if all_rows:
        write_parquet(run_dir / "summary.parquet", pd.DataFrame(all_rows))
    _write_report(run_dir, ranking)
    log.close()
    return run_dir


def _write_feature_ranking(run_dir: Path) -> None:
    """Aggregate per-condition feature_stability into a cross-condition ranking."""
    frames = []
    for p in sorted((run_dir / "conditions").glob("*/feature_stability.parquet")):
        d = pd.read_parquet(p)
        d["condition"] = p.parent.name
        frames.append(d)
    if not frames:
        return
    allf = pd.concat(frames, ignore_index=True)
    ranked = (allf.groupby("feature")["selection_frequency"]
              .agg(mean_selection_frequency="mean", n_conditions_selected=lambda s: (s > 0).sum())
              .reset_index()
              .sort_values("mean_selection_frequency", ascending=False)
              .reset_index(drop=True))
    write_parquet(run_dir / "report" / "feature_ranking.parquet", ranked)


def _write_report(run_dir: Path, ranking: list[dict]) -> None:
    """Write the ``report/`` rankings + a hypothesis-generating summary."""
    rep = run_dir / "report"
    df = pd.DataFrame(ranking).sort_values("pooled_rmsep", na_position="last").reset_index(drop=True)
    write_parquet(rep / "condition_ranking.parquet", df)
    _write_feature_ranking(run_dir)

    lines = ["# Discovery summary (hypothesis-generating)", "",
             "Candidate models ranked by pooled RMSEP (with bootstrap CI). Treat as",
             "candidates to validate, not a finalized model. Compare CIs, not point",
             "estimates, at small N.", "",
             "| rank | condition | track | RMSEP | 95% CI | Q² | folds |",
             "|---|---|---|---|---|---|---|"]
    for i, r in df.iterrows():
        ci = f"[{r['rmsep_ci_lo']:.3f}, {r['rmsep_ci_hi']:.3f}]" if pd.notna(r["rmsep_ci_lo"]) else "n/a"
        rmsep = f"{r['pooled_rmsep']:.3f}" if pd.notna(r["pooled_rmsep"]) else "n/a"
        q2 = f"{r['pooled_q2']:.3f}" if pd.notna(r["pooled_q2"]) else "n/a"
        lines.append(f"| {i + 1} | {r['condition']} | {r['track']} | {rmsep} | {ci} | {q2} | {r['n_folds']} |")
    (rep / "discovery_summary.md").parent.mkdir(parents=True, exist_ok=True)
    (rep / "discovery_summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
