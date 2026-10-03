# Running electropycal at scale

A raw PSTrace export can grow to many gigabytes, but the raw data is not in the training loop.
The extracted featureset is a few megabytes. So the shape of the work is: extract once into a
compact feature store, then iterate on that store indefinitely.

## Two different costs, with different fixes

**Extraction is I/O-bound.** `extract_dataset` and `electropycal extract` parse UTF-16 PSTrace
files, many small ones. The cost is dominated by file reads, not arithmetic. Over a network
or Drive mount this is the pathology: a measured 0.257 s per file, which over a large export
turns a one-hour job into an overnight one. Put the raw data on local or scratch disk before
extracting, and write run directories there too.

**Discovery is CPU-bound, and "fast" depends entirely on what you run.** The featureset is
small, so a single condition on a `fast` profile finishes in seconds. A full sweep does not.
Measured on a 2190 x 143 featureset, the full baseline queue costs **3.0 core-hours at three
seeds**, or **1.1 core-hours at one**, plus or minus about 15%. Serially that is roughly
**150 minutes of wall-clock** on an ordinary cloud runner. Plan for a sweep as an hours-long
job, not a coffee break, and scale from those figures by your own row count.

**The lever is the selector, not the core count.** CARS conditions are **86% of the sweep
cost**. Dropping CARS is worth about 7x; going from one seed to three costs about 2.7x. If a
sweep is too slow, change what you are running before you buy more cores.

## The workflow

```bash
# 1. Get the raw export onto local or scratch disk, once. Any transfer method is fine;
#    the point is that extraction does not run over a network mount.

# 2. Extract once into the compact artifact you will iterate on.
electropycal extract --raw ./raw --out featureset.parquet --band auto --n-jobs 8 --progress

# 3. All modeling reads the parquet. Never the raw again.
electropycal discover --data featureset.parquet --profile full --n-jobs -1
```

Re-extract only when new timepoints arrive, and transfer incrementally so only new
device-sessions move. Version `featureset.parquet` with a date or parameter hash in the
filename, or keep the run's `run_config.json`, whose extraction pin records every derived
parameter needed to rebuild it exactly.

Prefer `--n-jobs -1` over `--n-jobs "$(nproc)"`. `nproc` returns 1 inside some containers,
which silently runs the whole sweep serially while looking like it was parallelized.

The notebooks cache internally, so `diagnostics_review` writes
`<ROOT>/diagnostics_featureset__<hash>.parquet` and reloads it. For discovery at scale,
prefer the explicit extract-then-discover split above, so the raw pass happens exactly once.

## Where it runs

| | extraction | discovery |
|---|---|---|
| **Colab** | slow over a Drive mount. Do it once, or upload a pre-made parquet. | fine for a single condition or a `fast` profile; a full sweep will outlive a free session |
| **Local**, raw on SSD | fast | fast, all cores through joblib |
| **Cluster** | one node, raw on local scratch | split the queue across nodes, below |

## Two layers of parallelism

They apply at different stages, so do not try to stack them.

- **Per-session, during extraction.** `--n-jobs` on `electropycal extract`, or
  `extract_dataset(n_jobs=...)`, processes device-timepoint sessions in parallel worker
  processes. Output is byte-identical to serial. Each worker holds its own parsed data, so
  memory grows roughly linearly with worker count: prefer 4 to 8 over `-1` on a large export
  or a RAM-limited node. `channel_quality_report` and `replicate_feature_reliability` take
  the same argument.
- **Per-fold, during discovery.** `--n-jobs`, or `Profile(n_jobs=...)`, parallelizes the CV
  fold loop. Each worker is capped to a single BLAS thread, so the library will not
  oversubscribe cores even when called from a plain script with no environment setup.

## On a cluster

The task-queue scheduler writes each condition's outputs, fold models included, and is restartable: a
failed task re-runs only its own condition, so a long sweep survives a preemption.

There is no per-condition CLI index, so the straightforward split is by **target framing and
profile** across jobs, with joblib handling folds inside each job. The featureset is small
enough that one multi-core node running the full queue is usually the simplest thing that
works.

```bash
#!/bin/bash
#SBATCH --job-name=electropycal-discover
#SBATCH --cpus-per-task=16
#SBATCH --mem=8G
#SBATCH --time=04:00:00          # a three-seed sweep is ~3 core-hours; give it headroom

# Run directories go on node-local scratch, never a shared mount.
export TMPDIR=$SLURM_TMPDIR
electropycal discover --data "$FEATURESET" --profile full --n-jobs -1 \
                      --seeds 0,1,2 --out "$OUTDIR" --progress
```

To iterate faster, run a CARS-free queue first and add CARS only for the conditions that
survive. That is the 7x lever, and it costs nothing but ordering.

## Rules of thumb

- Never put raw data in the training loop. One extraction pass, then a frozen feature store.
- Keep run directories on local disk. A network mount costs 0.257 s per file, and a sweep
  writes on the order of 130,000 of them.
- Sync incrementally and extract incrementally. Only new timepoints cost anything.
- Change the selector before you change the hardware. CARS is 86% of the cost.
- For reproducibility, freeze `featureset.parquet` together with the
  `electropycal_analysis_config.json` used to build it, or keep the run's `run_config.json`.
  A run is then fully determined by those artifacts.
