# Changelog

This project follows [semantic versioning](https://semver.org/). It is pre-1.0, which here
means what semver says it means: **the public API may change in a minor release.** Pin a
version if you depend on it. 1.0.0 will be a deliberate act, once the surface has stopped
moving, not a scheduled follow-up.

## 0.9.0

First public release.

### What ships

- **Ingestion.** `data.pstrace` parses PSTrace UTF-16 exports. `features.extract` walks a raw
  directory, derives each device's timepoints as elapsed days since its own first session,
  quality-gates, and builds a featureset.
- **Discovery.** EIS/FSCV feature extraction, D0/Z normalization, forward-chained nested CV
  across three tracks, five PLSR architectures, six feature selectors, and a decision-gated
  task-queue scheduler with optional fold-parallelism.
- **Deployment.** Freeze a model on the full in-vitro set, then recalibrate new in-vivo data
  with frozen robust scalers and a CORAL domain-shift confidence flag. Bundles are portable
  `.npz` plus a manifest, with no pickle anywhere.
- **Diagnostics.** Variance partitioning and measurement-reliability estimates.
- **A synthetic demo dataset** under `demo/`, so every command and all seven notebooks run
  before you have data of your own.
- **Seven notebooks**, each runnable as-is, which synthesize a demo tree when none is
  reachable (as on Colab).

`discovery` and `deployment` are included rather than held back for a later release. They are
import-load-bearing for the CLI, three of the seven notebooks depend on them, and holding them
back would have meant shipping a library that could not do the thing it exists to do.

### Deliberate design choices worth knowing

- **`extract_dataset` has no default for `band`** and raises without one. `"auto"` is a
  percentile over whatever corpus is present, so it cannot be reproduced over a subset. The
  CLI does default to `--band auto` and prints the band it resolved, so the choice is never
  silent.
- **Dependencies are bounded at both ends, and so is Python** (`>=3.11,<3.14`). A scikit-learn
  minor release can shift PLS numerics enough to move a reported RMSEP, and on 3.10 the
  declared ranges resolved to a stack years apart from the reference one.
- **Results are hypothesis-generating.** Every ranking carries a bootstrap interval, and the
  baseline queue is a one-factor-at-a-time design rather than a grid, because selecting among
  many conditions on few folds overfits the selection itself.
- **Extraction is reproducible through a pin.** `run_config.json` records every derived
  run-wide parameter, so a later extraction over a staged subset either reproduces the
  original rows or raises, rather than silently producing different ones.

### Known limits

- Plotting displays inline in notebooks but **writes no files**: there are no `savefig` calls,
  so running the review modules from a script or in CI completes and produces nothing.
- `Kd` is unidentifiable on non-saturating data, so the saturating targets (`sat_*`, `hill_*`)
  are off by default.
- Kernel, multi-block and multi-level PLSR, and the mRMR and permutation-`t_max` selectors,
  are registered extension points that raise on use.
