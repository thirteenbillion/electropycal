# electropycal user guide

Data processing and recalibration for electrochemical sensors (EIS / FSCV). This guide walks a
model-discovery run end to end, from the command line and from the Python API. Read it in
order the first time; after that, §7 says where to look things up.

Companion documents:

- **`docs/REFERENCE.md`** is the lookup table: every default, the feature dictionary, the
  directory layouts, and the function-level API.
- **`docs/DESIGN.md`** is the reasoning: the assumptions, why the defaults are what they are,
  and the known limits.
- **`docs/RUNNING_AT_SCALE.md`** covers large raw datasets, clusters, and what a full sweep
  actually costs.

---

## 1. Installation

```bash
pip install electropycal
```

For the notebooks, which need jupyter and matplotlib:

```bash
pip install "electropycal[notebooks]"
```

Python 3.11, 3.12 or 3.13. Core dependencies: numpy, scipy, pandas, scikit-learn, pyarrow,
joblib, all with upper bounds, because a scikit-learn minor release can shift PLS numerics
enough to move a reported RMSEP.

A synthetic demo dataset ships under `demo/`, so every command below runs before you have data
of your own.

---

## 2. Key concepts (read once)

- **Sensor identity is `(device, channel)`.** Channel 3 on device `2-2` is a different sensor
  than channel 3 on `2-3`. Cross-validation groups by this composite.
- **Timepoints are derived, not declared.** Each device's timepoints are elapsed days since
  its own first signal session, so staggered collection is handled automatically. Timepoints
  are therefore device-local: two devices' `D35` are different calendar dates.
- **CV tracks.** Track 1 is channel-specific (a per-sensor optimum), Track 2 is global (one
  shared model), Track 3 is universal (generalize to a held-out sensor).
- **Profiles trade speed for rigor.** `FAST` uses tiny grids and a small bootstrap; the
  default `Profile` is the real one. The same code runs both.
- **Outputs are hypothesis-generating, not a finalized model.** Rankings come with bootstrap
  confidence intervals. At small N, compare intervals rather than point estimates.

---

## 3. Three ways to run it

The pipeline itself runs through the **command line** or the **Python API**. The notebooks
wrap those same calls for inspection and for the batch-by-batch discovery workflow; they are
not a separate implementation.

| You want to | Use |
|---|---|
| Run discovery unattended | `electropycal discover ...`, or `run_discovery` from Python |
| Run it batch by batch, deciding between batches | `electropycal discover --batched`, or the `discovery_checkpointed` notebook |
| Look at raw spectra, QC, diagnostics or results | the notebooks |

### 3.1 On Google Colab

Every notebook runs on Colab with no local clone. The work here is CPU-bound, so use a
standard CPU runtime; a GPU runtime buys nothing.

Each notebook's first cell installs the package and mounts Drive, guarded so that it does
nothing off Colab:

```python
import sys
if "google.colab" in sys.modules:
    !pip install -q electropycal
    from google.colab import drive
    drive.mount("/content/drive")
```

Then point `ROOT` at your data, wherever it lives:

```python
ROOT = "/content/drive/MyDrive/<your-folder>"   # a directory of <date>_..._signal folders
```

Leave `ROOT = None`, the default, and the notebook finds the shipped `demo/` tree if one is
reachable and synthesizes an equivalent otherwise, so it runs before your own data is wired
in. Free Colab disconnects when the tab closes, which matters only for a long full-profile
discovery run; see `docs/RUNNING_AT_SCALE.md` for what those actually cost.

---

## 4. A discovery run, step by step

### 4.1 Organize the raw data (your convention)

The tree below shows the shape `extract` expects. `my_export/` stands for wherever your own
PSTrace export lives. A working example of exactly this shape ships with the package as
`demo/in_vitro/input/` in a checkout, or wherever `demo` resolves to from an install
(synthetic, ~15 s to rebuild with
`python scripts/build_demo_dataset.py`). **Every command in §4 and §5 leads with that demo
path and is runnable as-is**; `my_export/` is always a stand-in for your own directory.

```
my_export/
├── 20260708_neurostring_channeltest/     # preliminary (skipped by modeling)
│   └── 2-2_fscv_0nm.csv …
├── 20260715_neurostring_signal/
│   ├── 2-2_eis_0nm.csv                    # one EIS per device (background)
│   ├── 2-2_fscv_0nm.csv                   # 0nM = background reference
│   ├── 2-2_fscv_100nm.csv … 2-2_fscv_5000nm.csv
│   ├── 2-2_fscv_stabilization-full.csv    # optional: stabilization cycles
│   └── 2-3_eis_0nm.csv …
└── 20260722_neurostring_signal/ …        # later timepoints (staggering is fine)
```

**Filename convention:** `<deviceid>_<signaltype>_<dose>.csv`, where `signaltype` is `eis` or
`fscv`, and `dose` is either a concentration (`0nm`, `100nm`, and so on, parsed to a float in
nM, with `0nM` the background) or a non-numeric protocol token.

The one protocol token today is `stabilization`, matched as `*_fscv_stabilization*.csv` so
that suffixed variants are picked up: `-full` for a single continuous run, or the `-start`
and `-end` pair for the first and last rounds of a longer one. Each file holds several rounds
of consecutive FSCV cycles, stored as consecutive replicate blocks. Do not assume a fixed
number of cycles per round; the review code reads the round from each block's label and
counts cycles from the data. These files are not dose-response points: `extract` parses them
but excludes them from the featureset, and they feed the stabilization-review notebook
(§4.3) instead. They are entirely optional, and nothing in the modeling path reads them.

### 4.2 Extract to a featureset

```bash
# Runnable as-is from a bare `pip install`: `demo` resolves the bundled synthetic
# dataset, and synthesizes an equivalent if this install has no copy of it.
electropycal extract --raw demo --out featureset_extracted.parquet
# → "featureset written to: featureset_extracted.parquet  (N rows, M sensors, P columns)"

# Against your own data, substitute your PSTrace export directory.
electropycal extract --raw my_export --out featureset_extracted.parquet
```

**Defaults are `--peak-method direct --acceptance monotonic --band auto`**: the trio that
works on real broad-DA-peak data (`direct` = current at V_ox; `monotonic` = keep channels
whose `NormIpeak` rises with concentration; `auto` sets the EIS band from data, upper bound
below the tightest inductive onset so EIS.1, capacitive at *every* in-band frequency, stays
satisfiable). Pin the band with `--band 10,21544` if you want it fixed. The stricter
chord-height + SNR gate is still selectable (`--peak-method chord --acceptance snr`) and is the
right choice for sharp peaks, but on real broad-peak data the chord height under-counts and the
SNR gate can reject every row. See the FSCV peak-definition note in `docs/DESIGN.md`.

`--band auto` resolves through the same `recommended_band(ROOT)` the discovery notebook and
`channel_quality_report`'s QC verdicts use, so the dashboard, the featureset, and discovery agree
on the band. The CLI prints the band it resolved, so the choice is never silent.

**The Python API is deliberately different: `band` has no default.** `extract_dataset(root)`
raises `ValueError`. `"auto"` is a percentile over every EIS file in whatever corpus is present,
so the same call over a staged subset silently produces a *different* band, which is exactly the
class of bug the extraction pin exists to prevent. Pass a tuple, opt into `band="auto"`, or pass
`pin=<run_config.json>` to reuse a previous run's band verbatim.

### 4.3 Inspect the raw data first (optional but recommended)

First open **`notebooks/raw_spectra_review.ipynb`** and run it: `ROOT` defaults to the
shipped demo tree; point it at your own export to use real data:
per device it overlays the raw EIS spectra (Bode −phase / |Z| / Nyquist, with the
inductive-onset crossover) and FSCV I–V loops + `i − i_bg` dose surface across timepoints,
and recommends a `BAND`. Then open **`notebooks/quality_filtering_dashboard.ipynb`**, set
`ROOT` and the stringency parameters (incl. that `BAND`), and run it: the measurement
schedule, per-check (EIS.1–3, FSCV.1–2) pass/fail dropout, and the channel-timepoint
validity table that mirrors what discovery consumes.

To confirm the interface stabilized before dose collection, open
**`notebooks/stabilization_review.ipynb`** and set `DEVICE_DIR`/`DEVICE`: it ingests
the `<deviceid>_fscv_stabilization-start.csv` and `-end.csv` files, plots
`I(V_target)` per channel over the stabilization cycles for **start vs end** (with
round boundaries), and applies the objective plateau criterion
(`data.stabilization.check_converged`) so the start→end settling is visible per
channel. (A single legacy `..._stabilization.csv` is treated as the end file.)

### 4.4 Run discovery

Every run mode supports **both** a plain full-queue run **and** decision-gated,
*editable* batches, after each batch you can exclude unpromising conditions (or
keep auto-flagged ones anyway) before the next batch runs. Batches are defined by
the condition names in `baseline_queue()` (the integer before the dot:
`baselines_1.2…` → batch 1).

**Option A: terminal, one-shot (full queue):**
```bash
electropycal discover --data featureset_extracted.parquet --profile full --n-jobs 8 --out outputs
# prints the ranked discovery_summary.md when done
```
(You can also skip §4.2 and pass a raw directory straight to `--data`, `--data
demo` with the shipped demo, or your own export directory, and extraction
then uses `--band auto` too.)

**Short time series:** forward-chained CV needs `min_train_times + 1` distinct timepoints (default
`min_train_times=3` → 4 timepoints). With fewer, `discover` **auto-caps** `min_train_times` to
`n_timepoints − 1` and prints one warning, so a 3-timepoint dataset still produces folds instead of
silently returning empty/`nan` metrics. Override with `--min-train-times N`; the same cap is applied
by `run_discovery` / `run_condition` in the Python API and by the discovery notebook.

**Option B: terminal, editable batches (`--batched`):**
```bash
electropycal discover --data featureset_extracted.parquet --profile full --batched
```
Runs Batch 1, prints its CI-ranked table, lists the upcoming conditions with any
`[FLAGGED]` (advisory excludes computed from results so far), then prompts:
```
Actions: <Enter>=proceed (exclude flagged) | keep:<name> | drop:<name> | all | stop
```
`<Enter>` excludes the flagged conditions; `keep:<name>` re-includes a flagged one;
`drop:<name>` excludes another; `all` keeps everything; `stop` ends the run.

**Option C: Python script (programmatic gates or your own batch loop):**
```python
from electropycal.features.extract import extract_dataset
from electropycal.discovery.config import RunData, baseline_queue, Profile
from electropycal.discovery.scheduler import run_discovery, auto_flag_conditions

# band is required (no default); the shipped demo tree is a runnable example
from electropycal._demo import demo_input          # resolves or synthesizes the demo tree
data = RunData.from_frame(extract_dataset(demo_input(), band=(2.0, 2000.0)))

def batch_gate(batch_num, ranking, remaining):
    flags = auto_flag_conditions(ranking, remaining)     # advisory {name: reason}
    # edit the remaining queue: e.g. drop flagged conditions, keep the rest
    return [c for c in remaining if c.name not in flags]  # or return False to stop
run_dir = run_discovery(data, conditions=baseline_queue(), profile=Profile(n_jobs=8),
                        out_root="outputs", batch_gate=batch_gate)
```
`batch_gate` accepts `(batch_num, ranking)` or `(batch_num, ranking, remaining)` and
returns `True` (proceed), `False`/`None` (stop), or a **`list[Condition]` that
replaces the remaining queue**. That list *is* the include/exclude edit. A
per-condition `gate(condition, agg, ranking) -> bool` is also available. (For the
finest control, drive it yourself: loop over `run_condition` per batch, inspect,
and build the next batch, this is what the notebook does.)

**Option D: checkpointed notebook:** open **`notebooks/discovery_checkpointed.ipynb`**.
`ROOT = None` (the default) resolves the shipped demo tree from whatever directory the kernel
started in, and synthesizes an equivalent if there is no checkout at all (Colab); set `ROOT`
to your own PSTrace export directory to use real data. Run each batch cell,
reviewing the CI-ranked plot. Between batches, `advise(next_batch)` prints auto-flags
and `exclude("<name>", …)` edits the `queue` list before you run the next batch.

**Summary.** Full-queue: A / C (no gate) / D (run all cells). Editable, decision-gated
batches: **B** (terminal prompt), **C** (`batch_gate` returning an edited list), **D**
(edit `queue` between cells). Plain runs are never auto-pruned, gating is opt-in.

### 4.5 Read the results

```
outputs/model_discovery_<timestamp>/
├── run_config.json                 # target, queue, profile, seeds, cap flag, dataset size
├── summary.parquet                 # one row per (condition, fold): metrics + telemetry
├── report/
│   ├── discovery_summary.md         # ranked candidates with CIs; start here
│   ├── condition_ranking.parquet    # same, machine-readable
│   └── feature_ranking.parquet      # features by cross-condition selection stability
└── conditions/<name>/
    ├── aggregated_metrics.json       # pooled RMSEP + CI, macro RMSEP, Q²
    ├── feature_stability.parquet     # per-feature selection frequency
    └── folds/<sensor>_t<timepoint>/  # model_arrays.npz, hyperparams.json, metrics.json
```

Open `report/discovery_summary.md` for the ranked table, or run
**`notebooks/discovery_results_review.ipynb`** (set `RUN_DIR` to the run directory) to
plot the condition ranking with CIs, per-fold RMSEP spread, and the top features by
cross-condition selection stability.

> **Note on N:** forward-chained CV needs ≥ 3 prior timepoints, so meaningful
> discovery needs ≥ 4 timepoints per device. With fewer, conditions produce no
> evaluable folds (metrics are NaN), expected until your series fills in.

### 4.6 The terminal commands, explained

**`electropycal extract`**: parse a raw PSTrace directory into a featureset table.
- `--raw <dir>` *(required)*, root holding `<date>_<devicetype>_signal/` folders.
- `--out <path>`, output featureset (default `featureset_extracted.parquet`).

**`electropycal discover`**: run the discovery task queue; writes a timestamped
`outputs/model_discovery_<ts>/` run directory.
- `--data <src>`, `synthetic` (default), a featureset `.parquet`/`.csv`, or a raw
  PSTrace directory (auto-extracted).
- `--out <dir>`, output root (default `outputs`).
- `--profile {fast,full}`, `fast` = tiny grids/bootstrap (seconds; smoke tests);
  `full` = the default `Profile` (real run). Default `full`.
- `--n-jobs <int>`, joblib workers over CV folds within each condition (default 1);
  set to your core count for speed (BLAS is auto-pinned to avoid oversubscription).
- `--seed <int>`. RNG seed for CARS/bootstrap reproducibility (default 0).
- `--batched`, run batch-by-batch, printing each batch's ranking and prompting
  before continuing (the interactive decision-gate workflow).

**`electropycal freeze`**: freeze a discovery-selected model on 100% of in-vitro data.
- `--run <dir>` *(required)*, a completed discovery run directory.
- `--data <src>` *(required)*, the in-vitro featureset/raw dir used for discovery.
- `--condition <name>`, which condition to freeze (default: top-ranked).
- `--stability-min <float>`, min fold selection-frequency for a feature to be kept (0.5).
- `--out <dir>`, output bundle (default `outputs/frozen_model`).

**`electropycal deploy`**: recalibrate in-vivo data with a frozen model.
- `--model <dir>` *(required)*, a `frozen_model/` bundle from `freeze`/`freeze_model`.
- `--data <src>`, a pre-built in-vivo featureset `.parquet`/`.csv`, **or**
- `--raw <dir>`, a raw in-vivo directory (extract + per-session recalibrate). Pass one.
- `--flag-distance <float>`, *optional* CORAL threshold to flag `EXTRAPOLATING`
  (with `--raw`); calibrate from in-vitro CV distances, or omit for raw distances.
- `--out <path>`, output (default `outputs/deployments/recalibrated.parquet`).

---

## 5. Deployment (in-vitro → in-vivo)

**Freeze** the model you choose on 100% of in-vitro data, then **recalibrate** new
in-vivo data. The in-vivo raw directory uses the convention
`<deviceid>_<signaltype>_<period>.csv` with `signaltype ∈ {eis, fscv, paired}` and
`period ∈ {baseline, live}`, `paired` = time-sequential EIS+FSCV per channel; each
in-vivo row is one *time sample* (concentration is unlabeled).

**Step 0: freeze a model.** After a discovery run, freeze the chosen model on 100%
of in-vitro data. Turnkey (picks the top-ranked condition, refits it, writes the
bundle, no files copied from the run):
```bash
# Runnable against the shipped demo, using the run directory §4.4 produced:
electropycal freeze --run outputs/model_discovery_<ts> --data demo \
                    --out outputs/frozen_model
#   add --condition <name> to freeze a specific one instead of the top-ranked

# Against your own data, pass your PSTrace export directory, or the featureset built from it:
electropycal freeze --run outputs/model_discovery_<ts> --data my_export --out outputs/frozen_model
```
or in Python, `freeze_top` (from a run) or `freeze_model` (fully manual):
```python
from electropycal.deployment.deploy import freeze_top, freeze_model
freeze_top("outputs/model_discovery_<ts>", data, out_dir="outputs/frozen_model")   # top-ranked
# or choose everything yourself:
freeze_model(X_invitro, y_invitro, feature_names, "linear_plsr", k=3,
             feature_index=chosen_subset, out_dir="outputs/frozen_model")
```
> Discovery does **not** emit `frozen_model/`; its per-fold bundles are partial-train
> evaluation models. `freeze`/`freeze_top`/`freeze_model` create the deployable one.

**Step 1: recalibrate in-vivo:**
```python
from electropycal.deployment.deploy import recalibrate, recalibrate_invivo
# A) raw in-vivo directory → extract + D0-normalize (vs early in-vivo baseline) + per-session CORAL.
#    'out' saves the result; 'flag_distance' is optional (see note below).
res = recalibrate_invivo("outputs/frozen_model", "data/invivo",
                         out="outputs/deployments/recalibrated.parquet")
#   → DataFrame: timepoint, n, mean_norm_ipeak, domain_distance[, confidence]
# B) a pre-built in-vivo feature matrix (already D0-normalized, columns aligned):
out = recalibrate("outputs/frozen_model", X_invivo)   # {"norm_ipeak", "domain_distance"}
```
```bash
# default --out is outputs/deployments/recalibrated.parquet
electropycal deploy --model outputs/frozen_model --raw data/invivo
# or, from a pre-built featureset:
electropycal deploy --model outputs/frozen_model --data invivo_featureset.parquet
```
> **`flag_distance` / `--flag-distance` is optional and has no principled default.**
> CORAL distance is a Frobenius distance of aligned in-vitro/in-vivo moments; its
> scale depends on your features/normalization. Omit it to read the raw
> `domain_distance`, or set it from your own in-vitro cross-validation distances
> (flag in-vivo sessions that exceed that envelope).
Or open **`notebooks/deployment_domain_shift.ipynb`** and run it. Leave `ROOT` and `BUNDLE`
at `None`, the default, to use the shipped `demo/in_vivo/` tree, or set them to your own
in-vivo export directory and frozen-model directory at the top of the deployment cell. It
plots recalibrated NormIpeak and CORAL domain shift per session. `frozen_model/` = `model_arrays.npz` + `manifest.json` +
`feature_names.json` + `full_feature_names.json` + `d0_normalization.json` +
`scaler_center/scale.npy` (no pickle). Recalibration applies the frozen robust scaler,
predicts, back-transforms (if log), and flags CORAL domain-shift.

> In-vivo ingestion assumes the **same internal CSV format** as in-vitro exports and a
> **matching frequency grid** (so feature columns align to the frozen model).

---

## 6. Reproducible extraction, the pin

Extraction derives five parameters run-wide from *whatever input set is present*, and all five
change if you extract over a different subset:

| Derived parameter | Why it moves |
|---|---|
| `band` (when `"auto"`) | a percentile over every EIS file in the corpus |
| `device_d0` | `min(dates)` per device, the origin of every `timepoint` |
| `ref_grid` | the first quality-passing channel's EIS frequency grid, per device type |
| `d0_row` | each sensor's baseline = its earliest timepoint **in the input** |
| the feature column set | a feature that produced no values simply never appears |

A **pin** records them so a later extraction reuses them verbatim instead of re-deriving them.

```python
from electropycal.features.extract import extract_dataset

# 1. full extraction, writing the pin
full = extract_dataset(root, band=(2.0, 2000.0), pin_out="run_config.json")

# 2. later, over a staged subset: identical rows for the sessions present, or an error
subset = extract_dataset(staged_root, pin="run_config.json")
```

Without the pin, step 2 silently re-anchors, dropping a device's earliest session moves its
`device_d0`, so rows that were `timepoint=1.0` and `4.0` come back as `0.0` and `3.0`.

**New/changed API**

- `extract_dataset(..., pin=)`, a path to a `run_config.json` (or the parsed dict). `band`,
  `device_d0`, `ref_grid`, `d0_row` and the feature column set are taken verbatim and never
  recomputed; in particular `band="auto"` does **not** re-run its corpus pre-pass.
- `extract_dataset(..., pin_out=)`, write the parameters this run derived. The same record is
  always on the returned frame as `df.attrs["extraction_pin"]`, so the file is optional.
- `extract_dataset(..., device_types=)`, restrict to these device types (default `None` = no
  filter), recorded in the pin. Not a late row filter: it changes which channels feed the `band`
  percentile and which types get a `ref_grid` anchor.
- `extract_dataset(..., on_empty_session=)`, `"warn"` (default) emits
  `EmptySessionWarning` naming any `(device, timepoint)` that produced no rows and why;
  `"raise"` raises `EmptySessionError`; `"ignore"` is the old silence. A session absent from the
  featureset is otherwise indistinguishable from one never measured.
- `electropycal discover --seeds 0,1,2`, the seed **set** the stochastic selectors (CARS/MI) are
  repeated over and averaged across. Distinct from `--seed`, which is the single CV-fold seed.
  Deterministic conditions collapse to one seed internally; the *effective* set is recorded per
  condition in `condition_config.json` and as `n_seeds` on each summary row.

**What raises.** `electropycal.features.pin.PinMismatch` on: a device absent from the pin, a
session dated earlier than its pinned `device_d0`, a feature-column mismatch, an explicit `band`
or `device_types` conflicting with the pin, and, the one that catches a partial fetch, a
per-session row count that differs from the pin. A session short its EIS or its 0 nM background
yields zero rows without raising anywhere else, so a staged extraction would otherwise just be
quietly smaller.

**The record.** `run_config.json` carries `schema_version: 2` and an `extraction` block
(`band_hz`, `band_source`, `device_d0`, `ref_grid_hz` per device type, `feature_columns`,
`d0_rows`, `device_types`, per-session `{timepoint, n_rows}`, and the scalar `params`). A
discovery run embeds the same block, so a run states both the modelling knobs and the extraction
parameters behind its numbers. The public names are
`PIN_SCHEMA_VERSION`, `PinMismatch`, `load_pin`, `session_key`, `EmptySessionWarning` and
`EmptySessionError`; everything else in `features.pin` is an implementation detail.

---

## 7. Looking things up

This guide is a walkthrough. The lookup tables live in **`docs/REFERENCE.md`**, so that a
default is written down in exactly one place:

| You want | Where |
|---|---|
| every default, and whether it is settled | `docs/REFERENCE.md` §1 |
| how a parameter is surfaced (Python argument, CLI flag, or source only) | `docs/REFERENCE.md` §1.5 |
| what a feature means and how it is derived | `docs/REFERENCE.md` §2 |
| the 14 baseline conditions | `docs/REFERENCE.md` §4 |
| input and output directory layouts | `docs/REFERENCE.md` §5 |
| the function-level API, module by module | `docs/REFERENCE.md` §6 |
| why a default is what it is | `docs/DESIGN.md` |

Print the feature catalog at any time:

```bash
electropycal features
```

---

## 8. Notebooks

Names are descriptive rather than numbered. Use whichever fits your task, in any order.

| Notebook | Purpose |
|---|---|
| `raw_spectra_review` | inspect raw EIS/FSCV per device: Bode and Nyquist with the inductive onset, FSCV I-V loops and the `i - i_bg` dose surface. Recommends `BAND`. Run this first. |
| `quality_filtering_dashboard` | per-device quality gating: schedule, per-check (EIS.1-3, FSCV.1-2) dropout, channel-timepoint validity table. Run before discovery. |
| `stabilization_review` | confirm FSCV stabilized, from `I(V_target)` per channel across cycles plus a plateau check |
| `diagnostics_review` | variance hierarchy (structural, drift, dose), response reliability, QC-gate impact and a `monotonic_r_min` sweep. Answers whether recalibration is well-posed on this data. Run between QC and discovery. |
| `discovery_checkpointed` | run discovery batch by batch with CI-ranked review between batches |
| `discovery_results_review` | review a completed run: ranking with CIs, per-fold spread, feature ranking (set `RUN_DIR`) |
| `deployment_domain_shift` | freeze a model, recalibrate drifted data, CORAL drift monitor |

All of them run standalone. `ROOT` and `RUN_DIR` default to `None`, which locates the shipped
`demo/` tree from whatever directory the kernel started in, and synthesizes an equivalent
where no tree is reachable, as on Colab. Set them to point at your own data.

---

## 9. Applying this to your own data: a checklist

Whether a discovery run is worth trusting is an empirical question about your dataset, not a
property of the library. Work through these in order.

1. **Run `diagnostics_review` first. It is the go/no-go gate.** Read the variance hierarchy:
   how much of each feature's variance is within-temporal (drift) against between-sensor
   (structural, and removed by D0-normalization), and whether the drift slice sits above the
   replicate-noise floor. Read the QC-gate impact and the `monotonic_r_min` sweep alongside
   it. Proceed to modeling only if drift is comfortably above noise. If drift is close to
   noise, add timepoints or loosen gates first; no architecture recovers signal that is not
   there.

2. **Choose QC stringency from the sweep, not from a fixed constant.** FSCV dose-monotonicity
   is usually the dominant gate. Set `monotonic_r_min` and `acceptance` at the point where
   dropped channels' `r` merges into noise, so you do not discard the low-sensitivity tail
   that late-stage degradation produces.

3. **Run `discovery_checkpointed` twice**, once with `TARGET="normipeak"` and once with
   `TARGET="sensitivity"`, both on the full `Profile()`. Then use the target-framing
   comparison in `discovery_results_review` to choose between them head to head, on
   NormIpeak-RMSEP and on concentration recovery.

4. **Re-check D0-normalization as timepoints accumulate.** With few timepoints it can hurt,
   because it removes the large structural component and leaves only the small drift slice;
   with more it should help. Compare `extract_dataset(d0_normalize=True)` against `False` on
   the full set and keep whichever wins on your data.

The band and QC parameters are shared across notebooks through
`electropycal_analysis_config.json`, so set them once in `raw_spectra_review`'s save-config
cell and every later notebook picks them up. Treat every ranking as hypothesis-generating and
gate on the bootstrap intervals; the reason this matters more than usual here is in
`docs/DESIGN.md` under model selection at small N.
