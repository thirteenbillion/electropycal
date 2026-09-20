"""Command-line interface: ``electropycal discover`` and ``electropycal deploy``.

Designed for a checkpointed, batch-by-batch workflow: run a queue (or a single batch),
inspect the ``report/`` outputs, then decide whether to proceed. BLAS threads are pinned
before numpy-heavy work to keep joblib fold-parallelism from oversubscribing.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from .discovery.scheduler import pin_blas_single_threaded

pin_blas_single_threaded()


def _parse_band(s: str):
    """CLI ``--band``: ``'auto'`` (data-driven) or ``'lo,hi'`` in Hz."""
    if s == "auto":
        return "auto"
    parts = s.split(",")
    if len(parts) != 2:
        raise argparse.ArgumentTypeError("band must be 'auto' or 'lo,hi' (e.g. 10,21544)")
    return (float(parts[0]), float(parts[1]))


def _parse_seeds(s: str):
    """CLI ``--seeds``: a comma-separated seed set, e.g. ``'0,1,2'`` -> ``(0, 1, 2)``.

    Distinct from ``--seed``, which is the single seed used to build the CV folds. This is the
    set the *stochastic* selectors (CARS / MI) are repeated over and averaged across;
    deterministic conditions collapse to one seed internally regardless.
    """
    try:
        seeds = tuple(int(x) for x in s.split(",") if x.strip() != "")
    except ValueError:
        raise argparse.ArgumentTypeError(
            f"seeds must be comma-separated integers (e.g. 0,1,2), got {s!r}") from None
    if not seeds:
        raise argparse.ArgumentTypeError("seeds must list at least one integer (e.g. 0,1,2)")
    return seeds


def _resolve_cli_band(root, band):
    """Resolve the CLI's ``--band`` to a concrete ``(lo, hi)``, announcing it when derived.

    The library primitive deliberately has no ``band`` default: ``"auto"`` is a percentile over
    whatever corpus is present, so it cannot be reproduced over a staged subset. The CLI keeps
    ``--band auto`` as its default because there it is a *stated* choice — it appears in
    ``--help`` and in the docs — but it must not be a silent one, so the resolved band is
    always printed and a concrete tuple is passed down. The value also lands in the run's
    extraction pin, so a later run can reuse it rather than re-deriving it.
    """
    if not isinstance(band, str):
        return (float(band[0]), float(band[1]))
    from .data.inventory import recommended_band
    lo, hi = recommended_band(root)
    print(f"--band auto -> ({lo:.0f}, {hi:.0f}) Hz  (data-driven, from {root}); "
          f"pass --band lo,hi to pin it", flush=True)
    return (float(lo), float(hi))


def _load_frame(source: str, progress: bool = False, band="auto", n_jobs: int = 1):
    """Resolve a data source to a per-dose featureset DataFrame."""
    if source == "synthetic":
        import pandas as pd
        from .data.synthetic import make_dataset
        ds = make_dataset(random_state=0)
        df = pd.DataFrame(ds.X, columns=ds.feature_names)
        df["channel"] = ds.channel; df["timepoint"] = ds.timepoint
        df["concentration"] = ds.concentration; df["NormIpeak"] = ds.y
        return df
    if Path(source).is_dir():
        from .features.extract import extract_dataset
        return extract_dataset(source, band=_resolve_cli_band(source, band),
                               n_jobs=n_jobs, progress=progress)
    import pandas as pd
    return pd.read_parquet(source) if source.endswith(".parquet") else pd.read_csv(source)


def _load_data(source: str, progress: bool = False, band="auto", target: str = "normipeak",
               n_jobs: int = 1):
    """Resolve a data source to RunData: 'synthetic', a featureset parquet/csv, or a raw
    PSTrace directory (auto-extracted, ``n_jobs`` worker processes). ``target="sensitivity"``
    reframes to the per-sensor-timepoint dose-response slope
    (:func:`~electropycal.features.targets.sensitivity_featureset`)."""
    from .discovery.config import RunData
    # per-sensor-timepoint calibration-curve targets all reframe via sensitivity_featureset
    CURVE_TARGETS = {"sensitivity", "sensitivity_intercept", "sensitivity_curvature",
                     "sat_imax", "sat_kd", "sat_logkd", "hill_imax", "hill_kd", "hill_n",
                     "power_a", "power_beta"}
    if target in CURVE_TARGETS:
        from .features.targets import sensitivity_featureset
        df = _load_frame(source, progress=progress, band=band, n_jobs=n_jobs)
        return RunData.from_frame(sensitivity_featureset(df), target=target)
    return RunData.from_frame(_load_frame(source, progress=progress, band=band, n_jobs=n_jobs))


def _extract(args) -> None:
    from .features.extract import extract_dataset
    df = extract_dataset(args.raw, band=_resolve_cli_band(args.raw, args.band),
                         peak_method=args.peak_method, detrend=args.detrend,
                         acceptance=args.acceptance, min_norm_snr=args.min_norm_snr,
                         mono_tol=args.mono_tol, max_reps=args.max_reps,
                         min_dose_response_range=args.min_dose_response_range,
                         d0_normalize=not args.no_d0_normalize, n_jobs=args.n_jobs,
                         progress=args.progress)
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(dest, index=False)
    n_sensors = df.groupby(["device", "channel"]).ngroups if not df.empty else 0
    print(f"featureset written to: {dest}  ({len(df)} rows, {n_sensors} sensors, "
          f"{df.shape[1]} columns)")


def _interactive_batch_gate(batch_num, ranking, remaining):
    """Show the completed batch's ranking + auto-flags, then apply the user's
    exclude/include edits to the remaining queue (terminal ``--batched`` mode).

    Returns ``False`` to stop, or the (possibly edited) remaining ``list[Condition]``.
    """
    import pandas as pd

    from .discovery.scheduler import auto_flag_conditions
    df = pd.DataFrame([r for r in ranking if r.get("batch") == batch_num])
    cols = [c for c in ("condition", "track", "pooled_rmsep", "rmsep_ci_lo",
                        "rmsep_ci_hi", "pooled_q2", "n_folds") if c in df.columns]
    print(f"\n=== Batch {batch_num} complete ===")
    print(df[cols].sort_values("pooled_rmsep", na_position="last").to_string(index=False))
    if not remaining:
        return True

    flags = auto_flag_conditions(ranking, remaining)
    print(f"\nUpcoming conditions ({len(remaining)}):")
    for c in remaining:
        print(f"  batch {c.batch}  {c.name}" + (f"   [FLAGGED: {flags[c.name]}]"
                                                if c.name in flags else ""))
    print("\nActions (space-separated): <Enter>=proceed (exclude flagged) | "
          "keep:<name>=include a flagged one | drop:<name>=exclude another | "
          "all=keep everything | stop=end run")
    resp = input("> ").strip()
    if resp.lower() == "stop":
        return False
    exclude = set() if resp.lower() == "all" else set(flags)
    if resp.lower() not in ("", "all"):
        for tok in resp.replace(",", " ").split():
            if tok.startswith("keep:"):
                exclude.discard(tok[5:])
            elif tok.startswith("drop:"):
                exclude.add(tok[5:])
    if exclude:
        print("excluding:", ", ".join(sorted(exclude)))
    return [c for c in remaining if c.name not in exclude]


def _features(args) -> None:
    from .features.catalog import print_feature_catalog
    print_feature_catalog()


def _discover(args) -> None:
    from .discovery.config import FAST, Profile, baseline_queue
    from .discovery.scheduler import run_discovery
    data = _load_data(args.data, progress=args.progress, band=args.band, target=args.target,
                      n_jobs=args.n_jobs)
    import dataclasses
    # FAST hardcodes n_jobs=1, so selecting it used to silently discard --n-jobs. Apply the
    # flag to whichever profile was chosen instead of only to the full one.
    profile = dataclasses.replace(FAST, n_jobs=args.n_jobs) if args.profile == "fast" \
        else Profile(n_jobs=args.n_jobs)
    if args.seeds is not None:
        # Without this the multi-seed path was unreachable from the CLI: runner honours
        # profile.seeds only when the selector is stochastic AND more than one distinct seed is
        # set, and both profiles ship seeds=(0,) -- so the three-seed sweep was notebook-only.
        profile = dataclasses.replace(profile, seeds=args.seeds)
    if args.min_train_times is not None:
        profile = dataclasses.replace(profile, min_train_times=args.min_train_times)
    batch_gate = _interactive_batch_gate if args.batched else None
    from pathlib import Path as _Path
    _proot = _Path(args.data)
    _proot = _proot if _proot.is_dir() else _proot.parent      # config/stats live at the data root
    run_dir = run_discovery(data, conditions=baseline_queue(), out_root=args.out,
                            profile=profile, seed=args.seed, batch_gate=batch_gate,
                            progress=args.progress, provenance_root=_proot, target=args.target)
    print(f"\ndiscovery run written to: {run_dir}")
    print((run_dir / "report" / "discovery_summary.md").read_text())


def _freeze(args) -> None:
    from .deployment.deploy import freeze_top
    data = _load_data(args.data)
    out = freeze_top(args.run, data, condition=args.condition,
                     stability_min=args.stability_min, out_dir=args.out)
    print(f"frozen model written to: {out}")


def _deploy(args) -> None:
    import pandas as pd
    from .deployment.deploy import recalibrate, recalibrate_invivo
    dest = Path(args.out)
    dest.parent.mkdir(parents=True, exist_ok=True)
    if args.raw:                                          # raw in-vivo directory → per-session
        res = recalibrate_invivo(args.model, args.raw, flag_distance=args.flag_distance, out=dest)
        print(f"in-vivo recalibration (per session) written to: {dest}")
        print(res.to_string(index=False))
        return
    df = pd.read_parquet(args.data) if args.data.endswith(".parquet") else pd.read_csv(args.data)
    from .data.schema import RESERVED_COLUMNS
    feats = [c for c in df.columns if c not in RESERVED_COLUMNS and c != "time_index"]
    out = recalibrate(args.model, df[feats].to_numpy(float))
    df_out = df[[c for c in ("channel", "timepoint") if c in df.columns]].copy()
    df_out["NormIpeak_recal"] = out["norm_ipeak"]
    if "domain_distance" in out:
        df_out["domain_distance"] = out["domain_distance"]
    df_out.to_parquet(dest, index=False)
    print(f"recalibrated predictions written to: {dest}")
    if "domain_distance" in out:
        print(f"CORAL domain distance vs in-vitro reference: {out['domain_distance']:.3f}")


def main(argv=None) -> None:
    parser = argparse.ArgumentParser(prog="electropycal")
    sub = parser.add_subparsers(dest="command", required=True)

    e = sub.add_parser("extract", help="build a featureset from a raw PSTrace export directory")
    e.add_argument("--raw", required=True, help="root directory of <date>_<devicetype>_signal folders")
    e.add_argument("--out", default="featureset_extracted.parquet")
    e.add_argument("--band", type=_parse_band, default="auto",
                   help="EIS analysis band: 'auto' (default, data-driven upper bound below the "
                        "tightest inductive onset) or 'lo,hi' in Hz (e.g. 10,21544)")
    e.add_argument("--peak-method", choices=["chord", "direct"], default="direct", dest="peak_method",
                   help="FSCV peak height: 'direct' (current at V_ox; default, robust to broad "
                        "DA peaks) or 'chord' (chord-baseline height)")
    e.add_argument("--detrend", action="store_true",
                   help="subtract a non-Faradaic baseline slope before peak/noise (opt-in)")
    e.add_argument("--acceptance", choices=["monotonic", "monotonic+snr", "snr", "none"],
                   default="monotonic",
                   help="FSCV channel acceptance: dose-response monotonicity (default); "
                        "'monotonic+snr' adds a lenient per-dose SNR cut; 'snr'; or 'none'")
    e.add_argument("--min-norm-snr", type=float, default=3.0, dest="min_norm_snr",
                   help="SNR gate multiple (acceptance=snr)")
    e.add_argument("--mono-tol", type=float, default=0.10, dest="mono_tol",
                   help="EIS |Z|-monotonicity tolerance (check B)")
    e.add_argument("--max-reps", type=int, default=3, dest="max_reps",
                   help="keep only the first N FSCV replicate cycles per file "
                        "(the intended [0..N-1] rounds; drops erroneous extras; default 3)")
    e.add_argument("--n-jobs", type=int, default=1, dest="n_jobs",
                   help="worker processes to feature-extract sessions in parallel "
                        "(-1 = all cores; output is identical to serial, just faster)")
    e.add_argument("--min-dose-response-range", type=float, default=None, dest="min_dose_response_range",
                   help="amplitude floor: drop channel-timepoints whose NormIpeak dynamic range "
                        "(max-min over doses) is below this (excludes monotone-but-dead electrodes; "
                        "off by default)")
    e.add_argument("--no-d0-normalize", action="store_true", dest="no_d0_normalize",
                   help="skip per-sensor D0-normalization; default is to D0-normalize")
    e.add_argument("--progress", action="store_true",
                   help="print timestamped per-session progress with ETA during extraction")
    e.set_defaults(func=_extract)

    d = sub.add_parser("discover", help="run the model-discovery queue")
    d.add_argument("--data", default="synthetic",
                   help="'synthetic', a featureset parquet/csv, or a raw PSTrace directory")
    d.add_argument("--band", type=_parse_band, default="auto",
                   help="EIS analysis band when --data is a raw directory: 'auto' (default, "
                        "data-driven) or 'lo,hi' in Hz. Ignored for a featureset/synthetic --data")
    d.add_argument("--target", choices=["normipeak", "sensitivity"], default="normipeak",
                   help="prediction target: 'normipeak' (per-dose, default) or 'sensitivity' "
                        "(per sensor-timepoint dose-response slope — the recalibration signal)")
    d.add_argument("--out", default="outputs")
    d.add_argument("--profile", choices=["fast", "full"], default="full")
    d.add_argument("--n-jobs", type=int, default=1, dest="n_jobs",
                   help="worker processes over CV folds (default 1 = serial). -1 uses every "
                        "visible core, which is more robust than $(nproc) -- that returns 1 in "
                        "some containers and silently runs serially")
    d.add_argument("--seed", type=int, default=0,
                   help="seed for CV fold construction (a single value)")
    d.add_argument("--seeds", type=_parse_seeds, default=None,
                   help="seed SET for the stochastic selectors (CARS/MI), comma-separated, "
                        "e.g. '0,1,2'. Each such condition is repeated per seed and its "
                        "aggregates averaged; deterministic conditions collapse to one seed "
                        "internally, so this multiplies only the stochastic ones. Default: the "
                        "profile's own seeds=(0,)")
    d.add_argument("--min-train-times", type=int, default=None, dest="min_train_times",
                   help="min prior timepoints before a t_test is used (nested CV needs ≥3 "
                        "timepoints; set 1 for a 2-timepoint smoke run)")
    d.add_argument("--batched", action="store_true",
                   help="review/edit each task-queue batch's ranking before continuing")
    d.add_argument("--progress", action="store_true",
                   help="print timestamped per-condition/per-fold progress with ETA "
                        "(and per-session lines when --data is a raw directory)")
    d.set_defaults(func=_discover)

    f = sub.add_parser("freeze", help="freeze a discovery-selected model on 100%% of in-vitro data")
    f.add_argument("--run", required=True, help="a completed discovery run directory")
    f.add_argument("--data", required=True,
                   help="the in-vitro featureset (parquet/csv) or raw dir used for discovery")
    f.add_argument("--condition", default=None,
                   help="condition name to freeze (default: the top-ranked one)")
    f.add_argument("--stability-min", type=float, default=0.5, dest="stability_min",
                   help="min fold selection-frequency for a feature to enter the frozen subset")
    f.add_argument("--out", default="outputs/frozen_model")
    f.set_defaults(func=_freeze)

    p = sub.add_parser("deploy", help="recalibrate in-vivo data with a frozen model")
    p.add_argument("--model", required=True, help="path to a frozen_model/ bundle")
    p.add_argument("--data", default="", help="in-vivo featureset parquet/csv (omit if using --raw)")
    p.add_argument("--raw", default="", help="raw in-vivo directory (extract + per-session recalibrate)")
    p.add_argument("--out", default="outputs/deployments/recalibrated.parquet")
    p.add_argument("--flag-distance", type=float, default=None, dest="flag_distance",
                   help="optional CORAL distance threshold to flag EXTRAPOLATING (with --raw); "
                        "calibrate from in-vitro CV distances — omit to just read raw distances")
    p.set_defaults(func=_deploy)

    c = sub.add_parser("features", help="print the feature catalog (types, definitions, units, role)")
    c.set_defaults(func=_features)

    args = parser.parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
