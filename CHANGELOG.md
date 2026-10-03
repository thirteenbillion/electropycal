# Changelog

This project follows [semantic versioning](https://semver.org/). It is pre-1.0, which here
means what semver says it means: **the public API may change in a minor release.** Pin a
version if you depend on it. 1.0.0 will be a deliberate act, once the surface has stopped
moving, not a scheduled follow-up.

## 0.10.0

A data-loss fix in ingestion, and the surface hardening that came out of running the
library from a bare install rather than from a checkout.

### Fixed: exports with double-quoted lines could not be read

**This is the reason to upgrade from 0.9.0.** An export is sometimes encountered with its
lines wrapped in double quotes. 0.9.0 could not read it.

The wrapping is selective *within* a file: the metadata lines, the EIS column header and
the data rows are wrapped, while the `CH N:` block header is bare. Because that block
header is bare, 0.9.0 did find the block, and then failed on the wrapped column header
with a bare `StopIteration` naming neither the file nor the column. So on a real export
this was a **loud, uninformative crash, not a silent drop**: reading a wrapped EIS export
under 0.9.0 aborts, and nothing on the ingestion path catches it.

Both styles now parse to bit-identical arrays, verified against a real wrapped export of
969 lines and 24 blocks: all 24 blocks and 888 frequency points agree byte-for-byte with
the same file unwrapped, in both encodings.

Three silences and one bare crash became named errors, each identifying the file:

- an EIS column not locatable, naming the column sought **and the header actually read**
  (this is the one a wrapped export actually hits)
- no FSCV or EIS blocks found at all, which is what a file wrapped on *every* line
  including the `CH` header would do
- FSCV blocks that recover zero samples, which is what a wrapped FSCV export would do.
  Wrapping has only been observed on an EIS export, so this guard is precautionary.

A stray byte-order mark inside the wrapper is also stripped. Real wrapped first lines read
`"<BOM>File date:,...`; had the same quirk landed on the `Date and time:` line it would
have dropped `export_date` with no error.

### Fixed: an empty session table reported `KeyError: 'device'`

`RawSpectraIndex.available_table()` is built from 0 nM FSCV backgrounds alone. A tree
holding EIS and dosed FSCV but no 0 nM background therefore cleared the "any signal file"
guard and produced no rows, and `pd.DataFrame([])` has no columns for `sort_values` to sort
on. The result named neither the tree nor the missing file.

New `rawspectra.NoSignalSessions` reports the tree, what was found by signal type, the FSCV
doses present, and that a `<deviceid>_fscv_0nM.csv` per session is what to add. An empty
tree is a different situation and still returns an empty frame with its columns intact,
rather than raising.

### Fixed: path errors reported from inside the wrong thing

`extract --raw <bad path>` used to fail with a bare `FileNotFoundError` from inside the
EIS band percentile, because `--band auto` runs a corpus pre-pass before anything checks
the path. Three cases now report separately, before any inference:

- the path does not exist, echoing what a relative path resolved to
- the path is a **URL**, named as such, with the Colab mount recipe for a Drive link
- the path exists but holds no session folders, and the message looks one level down *and*
  one level up, because being one directory off is the actual mistake

The CLI prints these as one sentence and exits 2 instead of printing a traceback.

### Fixed: figures never reached disk

48 `plt.show()` calls, no `savefig`, no `plt.close()`. Under a notebook kernel that
renders; on a headless backend, which is CI and any sandboxed code execution, `show()` is
a no-op and a plotting call reported success having produced nothing.

- New `viz.emit(name)`: saves PNG **and PDF**, closes the figure, and displays only where
  displaying can work. Every plotting function now ends in it.
- Filenames are stable and overwrite on re-run. Output defaults to `figures/`.
- `viz.ensure_style()` is applied at every plotting entry point, so the house style and
  the Okabe-Ito palette are actually in effect rather than merely available.
- Red-to-green and rainbow colormaps replaced with perceptually uniform, colour-blind-safe
  ones on the QC grid, the per-sensor traces and the feature-type panels.

### Added

- **`--raw demo` and `--data demo`** resolve the bundled demo dataset, synthesizing an
  equivalent when an install has no copy. The 0.9.0 README's first command used a literal
  `demo/in_vitro/input`, which cannot work without a checkout. Every command in the README
  and the user guide is now checked against a bare wheel install.
- **`pin_mode="extend"`** on `extract_dataset`. A pin previously meant "reproduce exactly",
  so adding a session to a corpus was an error and an incremental rebuild had to run
  unpinned, re-deriving the band, `device_d0` and the reference grid from whatever happened
  to be present. `extend` holds every pinned anchor and admits new sessions.
- `smc_significance`, and `selection` now offers VIP as a third PLS-importance filter.
- Path errors are importable: `data.paths.PathNotFound`, `NotAFilesystemPath`,
  `NoSessionFolders`. Ingestion errors: `data.pstrace.PSTraceFormatError`.

### Changed

- **`smc_scores` was computing the selectivity ratio**, not sMC. It regressed each feature
  on the target projection *with an intercept*, which recovers the loading and so
  reproduced `selectivity_ratio` to 8e-15 with rank correlation 1.000000. It now uses the
  normalized regression vector and returns the reference F statistic with (1, n-2) degrees
  of freedom. Both old conditions computed a legitimate selectivity ratio, so no published
  number was wrong; the designed contrast between them was.
- `select("smc", ...)` scores `-log10(p)`, so its threshold is a significance level and is
  portable across fold sizes. A raw F cutoff is not: at 2190 rows the 5% cutoff admits 136
  of 143 features. The baseline queue's sMC thresholds are `(2.0, 6.0)`.
- The baseline queue is **14 conditions, 12 CARS-free** (was 13 and 11), having gained VIP.
- A missing `stabreview` root raises `PathNotFound`, not `NoStabilizationFiles`. 0.9.0
  conflated "this tree has no stabilization sweeps" with "this tree does not exist"; they
  want different fixes. Both remain `FileNotFoundError`, so existing handlers still catch.
- `RunData.from_frame` names a path passed where a DataFrame belongs.

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
