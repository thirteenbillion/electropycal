# Changelog

This project follows [semantic versioning](https://semver.org/). It is pre-1.0, which here
means what semver says it means: **the public API may change in a minor release.** Pin a
version if you depend on it. 1.0.0 will be a deliberate act, once the surface has stopped
moving, not a scheduled follow-up.

## 0.11.1

A correction release: the documentation checked against the code, one crash fixed, and the
build made reproducible from a clone for as long as the release exists.

### Fixed: `electropycal discover --help` could crash on Windows

Help text containing a character outside the Windows ANSI codepage made `discover --help`
exit with `UnicodeEncodeError` whenever its output was piped or redirected. The CLI help is
now plain ASCII, and the command line replaces any character the output stream cannot
encode instead of failing, so no message can crash a command.

### Added: `python -m electropycal`

The same command line as the `electropycal` console script, for environments where the
script directory is not on `PATH`.

### Changed: plain punctuation in everything the library prints

Docstrings, comments, log lines, warnings, error messages, plot titles and the notebooks use
colons, commas, semicolons and parentheses where they used dashes. If you match a message's
text exactly, check it. One displayed value changed: `overview.conditions_table` shows
`none` in the selector column for a condition with no selector.

### Documentation

`REFERENCE.md`, `USAGE.md`, `DESIGN.md`, `RUNNING_AT_SCALE.md` and the README were checked
against the code. Corrected: signatures and defaults (`norm_ipeak`, `extract_dataset`,
`run_discovery`, `run_condition`, `freeze_model`, `freeze_top`, `recalibrate_invivo`,
`eis_global_features`), the CLI flags each subcommand takes, the run-directory and frozen-model
layouts, the extraction pin and `pin_mode`, the demo-dataset README, and which selectors are
implemented (mRMR and a permutation `t_max` filter are described, not implemented).
`REFERENCE.md` now says what the code does about the modelling target: `normipeak` is the
default, and `sensitivity` is the recommended recalibration target, passed explicitly.
`RUNNING_AT_SCALE.md` no longer claims a sweep resumes after an interruption; it does not.

### Build

The build backend is pinned exactly (`hatchling==1.32.4`), since the wheel records the
version that built it. A continuous-integration workflow runs the suite on Python 3.11 to
3.13, installs the published package on Linux, Windows and macOS and runs the quick start,
and on every release tag rebuilds the tree and checks it against the files PyPI serves.

## 0.11.0

### Changed: a condition writes two fold files, not four per fold

A discovery run wrote each outer fold's fitted model as its own directory,
`conditions/<name>/folds/<fold>/`, holding `model_arrays.npz`, `manifest.json`,
`hyperparams.json` and `metrics.json`. On a real featureset that is about 1,400 files per
condition and well over 100,000 per sweep, almost all of them a few hundred bytes. File count,
not computation, then decides how long a run takes on any filesystem with a per-file cost: a
network mount, a synced folder, an archive step.

The same content now goes into two files per condition:

- `fold_models.npz`: every fold's arrays, keyed `<fold>__<array>`, e.g. `ch3_t28__coef`.
- `folds.json`: `{"layout": 2, "folds": {<fold>: {"manifest", "hyperparams", "metrics"}}}`.

A whole condition is about six files. The per-fold files never acted as checkpoints, since a
condition's folds are written together when it finishes, so nothing is lost by merging them.
Arrays are stored uncompressed, as before, and every number a run reports is unchanged.

`discovery.folds` reads both: `fold_names(cond_dir)`, `fold_records(cond_dir)` for the
manifests, hyperparameters and metrics without touching any array, and
`load_fold_bundle(cond_dir, fold)`, which returns one fold's `(arrays, manifest)` for
`models.base.predict_from_bundle` and reads only that fold's members of the archive.

**Migrating.** Run directories written by 0.10.0 or earlier need no conversion: every reader,
including `deployment.freeze_top`, accepts the one-directory-per-fold layout too. Code that
globbed `conditions/<name>/folds/*/` itself should call `fold_records` or `load_fold_bundle`
instead, which work on both.

### Added: a figure log, and figures that say where their data came from

A figure drawn from a synthetic example run and the same figure drawn from real
measurements wrote to the same path, so the second silently replaced the first and the two
were afterwards indistinguishable. Separately, nothing recorded whether a figure showed an
instrument export or quality-gated features, which makes gated-out points look like missing
data.

`viz.emit` takes three new keyword arguments. `provenance` identifies the data source and is
slugified into the filename, so two sources no longer collide and a copied file still names
its source. `stage` names the processing state of the data. `params` records the choices
that shaped the plot, such as a filter, a pooling or an encoding. `provenance=None` keeps
the previous filenames exactly.

All three go to a plain-text log rather than into the image, so a saved figure carries
nothing but the plot. The output directory gains two files:

- `FIGURES_LOG.txt`, append-only: one block per save with the data source, stage, params,
  library version and, on an overwrite, which earlier version it replaced. Because filenames
  are stable and overwrite, this is the only record of an earlier version or a deleted file.
- `FIGURES.txt`, rebuilt on every save: each figure on disk now, newest first, with its
  entry, plus any image the log cannot vouch for. `viz.figure_index()` rebuilds it after
  files are added or deleted by hand.

`emit(..., footer=True)` or `configure_output(footer=True)` also draws stage and source into
the image, for a working figure that will travel without its log. It is off by default.

`discovery.review` fills `provenance` and `stage` in from the run's own manifest rather than
from its path, so a renamed directory still reports what produced it, and a run with no
manifest is marked unverified in its filename and its log entry.

### Fixed: the notebooks' setup cell could not upgrade an existing install

Every notebook began with `pip install -q electropycal`, and `pip install` is a no-op when
any version is already present: it does not upgrade. A hosted runtime carrying an older
version from an earlier session therefore kept it, while the cell reported success. The
symptom was a failure deep inside the EIS reader on an export the older version cannot
parse, with nothing pointing at the version as the cause.

The cell now uses `pip install -q --upgrade electropycal` and checks the version in place,
raising a readable error that names what is installed and says to upgrade and then restart
the runtime, since a module already imported stays in memory and an upgrade does not reach
it.

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
- `smc_significance`, and a VIP condition in the baseline queue, `pNproblem_3.1_VIP`, as a
  third PLS-importance filter beside SR and sMC.
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
- Kernel, multi-block and multi-level PLSR are registered extension points that raise on use.
  The mRMR and permutation-`t_max` selectors are documented extension points, not implemented.
