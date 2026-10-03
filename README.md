# ElectroPyCal

Data processing and recalibration for electrochemical sensors (EIS / FSCV).

```bash
pip install electropycal
```

**ElectroPyCal** discovers interpretable PLSR recalibration models for drifting
electrochemical sensors under leakage-free, forward-chained nested cross-validation, and
deploys a frozen model to new data while quantifying in-vitro to in-vivo domain shift
(CORAL). It originates from a neurostring dopamine-sensor recalibration study but is
designed to generalize to electrochemical-sensor data processing broadly.

Pre-1.0: the public API is still moving. Pin a version if you depend on it.

## What it does

**Ingestion.** `data.pstrace` parses PSTrace UTF-16 exports. `features.extract` walks a raw
directory (`<YYYYMMDD>_<devicetype>_signal/<deviceid>_<signaltype>_<dose>.csv`, where `dose`
is a concentration or a protocol token such as `stabilization`), derives each device's
timepoints as elapsed days since its own first session, quality-gates, and builds a
featureset.

**Discovery.** EIS/FSCV feature extraction, D0/Z normalization, forward-chained nested CV
(Tracks 1/2/3), five architectures (linear, log, weighted, orthogonal, nonlinear PLSR), six
selectors (CARS, VIP, SR, sMC, MI, ICC), a decision-gated task-queue scheduler with optional
joblib fold-parallelism, and a checkpointed output layout (per-fold bundles,
`summary.parquet`, a ranked `report/`).

**Deployment.** Freeze a chosen model on 100% of in-vitro data, then recalibrate new in-vivo
data with frozen robust (median/IQR) scalers and a CORAL domain-shift confidence flag.
In-vivo raw ingestion (`paired` / `baseline` / `live` exports) feeds per-session
recalibration through `recalibrate_invivo` or `deploy --raw`. No pickle; bundles are portable
`.npz` plus a manifest.

**Diagnostics.** Variance partitioning across structural, drift and dose components, and
measurement-reliability estimates for the response.

Kernel, multi-block and multi-level PLSR, and the mRMR and permutation-`t_max` selectors, are
registered extension points that raise on use. See `docs/DESIGN.md`.

## Quick start

```bash
pip install electropycal

# Build a featureset. `demo` is a synthetic dataset, so this runs before you have
# data of your own; swap it for your own PSTrace export directory when you do.
electropycal extract --raw demo --out featureset.parquet

# Rank candidate recalibration models.
electropycal discover --data featureset.parquet --profile full

# Freeze the selected model on all of the in-vitro data.
electropycal freeze --run outputs/model_discovery_<timestamp> --data demo     --out outputs/frozen_model

# Recalibrate new in-vivo data with it.
electropycal deploy --model outputs/frozen_model --data invivo.parquet
```

`--raw demo` and `--data demo` resolve the bundled synthetic dataset, and synthesize an
equivalent one if this install has no copy of it, so the commands above run from nothing but
`pip install`. Note `--profile full` is a real run: budget a couple of hours. Use
`--profile fast` to see the pipeline work end to end in under a minute.

The repo also ships a Claude skill for in-chat analysis at
`.claude/skills/lab-analysis/`, which Claude Code picks up automatically in a clone: no
install step, just ask it for a stat or a figure against your data.

### Two things worth knowing up front

The CLI's EIS analysis `--band` defaults to `auto`, a data-driven upper bound just below the
tightest inductive onset, and the resolved band is printed so it is never a silent choice. The
**Python API deliberately has no `band` default**: `extract_dataset(root)` raises. `auto` is a
percentile over whatever corpus is present, so it cannot be reproduced over a staged subset.
Pass a `(lo, hi)` tuple, `band="auto"`, or `pin=` to reuse a previous run's band.

`discover` and `run_discovery` cap `min_train_times` to what the number of timepoints can
support, so a short series yields usable folds rather than silently empty ones.

## Notebooks

In `notebooks/`, descriptive names, runnable in any order. Each one runs as-is against the
shipped `demo/` tree, and synthesizes an equivalent tree if none is present (Colab).

| Notebook | Purpose |
|---|---|
| `raw_spectra_review` | inspect raw EIS/FSCV per device; recommends the analysis band. Run first. |
| `quality_filtering_dashboard` | per-device quality gating and channel-timepoint validity |
| `stabilization_review` | confirm FSCV stabilized, from the cycle-to-cycle drift |
| `diagnostics_review` | variance hierarchy, response reliability, QC-gate impact. Run between QC and discovery. |
| `discovery_checkpointed` | run discovery batch by batch with CI-ranked review between batches |
| `discovery_results_review` | review a completed run: ranking, CIs, per-fold spread, feature ranking |
| `deployment_domain_shift` | freeze a model, recalibrate drifted data, CORAL drift monitor |

## Docs

- **`docs/USAGE.md`**: the user guide. Installation and a full discovery walkthrough, start to finish.
- **`docs/REFERENCE.md`**: the lookup tables. Every default, the feature dictionary, directory layouts, and the API.
- **`docs/DESIGN.md`**: architecture, assumptions, and the reasoning behind the defaults.
- **`docs/RUNNING_AT_SCALE.md`**: extract once to parquet, then cluster workflows for large raw data.

## Layout

```
src/electropycal/{data,features,models,selection,evaluation,discovery,deployment,diagnostics}
notebooks/   # the seven review and workflow notebooks listed above
demo/        # synthetic in-vitro and in-vivo trees, input and output
docs/        # USAGE.md, REFERENCE.md, DESIGN.md, RUNNING_AT_SCALE.md
```

## License

MIT. See `LICENSE`.
