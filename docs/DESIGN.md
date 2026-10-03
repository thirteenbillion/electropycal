# DESIGN, `electropycal`

The assumptions, data schema, notation, and the mapping from the analysis pipeline to code
modules, together with the reasoning behind the defaults. Read this to understand why the
library is shaped the way it is; `docs/REFERENCE.md` is the lookup table for what the values
actually are.

---

## 1. Goal, in one paragraph

Electrochemical sensors (here FSCV for dopamine, characterized by EIS) drift as
their sensor–tissue interface evolves in vivo. We want to *recalibrate* the FSCV
response using EIS as a proxy for the interface state, via an interpretable PLSR
model discovered in vitro and deployed in vivo. The library provides (a) a
**discovery** pipeline that searches model architectures / feature selectors / CV
tracks under leakage-free nested CV to find candidate recalibration models, and
(b) a **deployment** pipeline that applies a frozen model to new in vivo data and
quantifies in-vitro→in-vivo domain shift (CORAL). It is meant to generalize to
electrochemical sensor data processing broadly, not only this case study.

## 2. Assumptions

1. `s(t)` (sensitivity) can be represented by `y_EIS`.
2. Each `y_EIS` maps to one `s(t)` (identifiability).
3. `y_FSCV` is a linear function of `s(t) · true(t)`.
4. `y_EIS ~ f(s(t))` is the same transfer function in vitro as in vivo (the
   domain-transfer problem, outstanding, validated only in vivo).
5. **(statistical)** Predictors `X` and response `Y` are measured without
   appreciable error, bounded, not assumed, via a one-time measurement-
   reliability diagnostic.

### Prior art: background-current calibration (Roberts & Sombers, 2013)

The core physical rationale for using **background / interfacial features to predict sensitivity** is
the in-situ calibration strategy of Roberts, Lock & Sombers (*Anal. Chem.* 2013, "In Situ Electrode
Calibration Strategy for Voltammetric Measurements In Vivo"). Dopamine FSCV at carbon-fiber
microelectrodes is **adsorption-controlled**, so the faradaic peak current scales with the density of
active surface sites, the *same* quantity that sets the double-layer capacitance and hence the
non-faradaic **background charging current** (`i_bg ≈ C_dl·dV/dt`). One physical driver, two
observables:

```
active surface sites ─┬─▶ DA adsorption capacity ─▶ faradaic sensitivity (peak current per [DA])
                      └─▶ double-layer capacitance ─▶ background charging current
```

Roberts et al. used the **cumulative (integrated) background current**, a robust proxy for total
active area, to *estimate an electrode's sensitivity in situ*, then converted a measured peak current
to concentration **without** a per-electrode DA calibration. Note the shape of that argument: a
**dose-invariant, DA-free** quantity (background) predicts the **calibration slope**, and dose
resolution then comes from inverting the **measured** peak through that slope. That is exactly why
dose-invariant predictors are sufficient for **dose-resolved** recalibration (see §1).

ElectroPyCal generalizes this: their cumulative background → **`bg_charge`** (`∫ i_bg dV` over the V_ox
window ∝ `C_dl·window`; with `bg_cap`, `mean_Ibg`, and the whole EIS block measuring the same interface
across frequency); their single background→sensitivity regression → the **`TARGET="sensitivity"`**
discovery framing; their across-electrode variation → our within-electrode **temporal drift** (fouling
reduces active sites, lowering background and sensitivity together); their in-situ apply →
`deployment_domain_shift`. Roberts et al. is thus the single-feature, cross-electrode special case of
the multi-feature, over-time model here, and empirical grounding that the sensitivity/calibration-curve
target (not per-dose `NormIpeak`) is the deployable one.

## 3. Data schema

- **Observation unit:** one `(channel, timepoint, concentration)` row. Example
  in-vitro dataset: 8 unbroken/16 channels, 4 timepoints (D0, D1, D7, D20), 5
  concentrations (100/250/500/1000/5000 nM) → **89 valid rows** of 160 possible.
- **X: 157 features** = 2 FSCV (`mean_Vpeak`, `mean_Ibg`) + 143 frequency-
  dependent EIS (7 types × 20–21 log freqs, 10 Hz–100 kHz: `R_s`, `R_p`, `C_s`,
  `C_p`, `ideality_C`, `tau`, `local_n`) + 12 global EIS (`R_s_integral`,
  `R_p_integral`, `C_s_integral`, `C_p_integral`, `f_ideality_crossover`,
  `ideality_C_band`×3, `tau_ratio`, `n_band`×3).
- **Y: `NormIpeak`** = `peak_height(V_ox) / I_bgd(V_ox)` (background-subtracted,
  background-normalized FSCV oxidation peak).
- **EIS sign convention:** this library works in `Z = Z' + j·Im(Z)` with `Im(Z) < 0` for a
  capacitive interface. A PSTrace export's `Z'' / Ohm` column actually holds `-Im(Z)`, so
  `data.pstrace` negates it on ingest. See the sign-convention note below.

## 4. Pipeline → module map

| Pipeline stage | Module |
|---|---|
| Raw arrays, metadata, asset bundles (npy/parquet/json/npz) | `data.io` |
| EIS/FSCV quality checks A–D | `data.quality` |
| Dataclasses / schema | `data.schema` |
| FSCV features + `NormIpeak` | `features.fscv` |
| EIS features | `features.eis` |
| Per-replicate extraction (for uncertainty) | `features.extract` |
| D0-normalization (additive/mult/log) + in-loop Z-scoring | `features.normalize` |
| `PLSRegression` wrapper (`scale=False`) + VIP | `models.plsr` |
| log/weighted/orthogonal/nonlinear/kernel variants | `models.variants` |
| Model asset (de)serialization (npz + manifest, no pickle) | `models.base` |
| mRMR, permutation p-values, t_max | `selection.univariate` |
| VIP / sMC / SR | `selection.pseudo_multivariate` |
| CARS (seeded, time-box, multi-seed stability) | `selection.cars` |
| VCA/ICC pre-filter | `selection.icc` |
| LOTO-c / LOTO-c-AC forward-chaining, nested CV | `evaluation.cv` |
| pooled RMSEP, Q², macro, bootstrap CI | `evaluation.metrics` |
| Track 1/2/3 objectives + aggregation | `evaluation.tracks` |
| Task queue, decision gates, joblib fold parallelism, BLAS pinning | `discovery.scheduler` |
| Per-fold execution + serialization + directory layout | `discovery.runner` |
| Aggregation, rankings (`condition_ranking`/`feature_ranking`), `discovery_summary` | `discovery.scheduler._write_report` |
| Frozen-model load + apply + predict; raw in-vivo recalibrate | `deployment.deploy` |
| In-vivo raw ingestion (`paired`/`baseline`/`live` → featureset) | `features.extract.extract_invivo` |
| CORAL domain-shift quantification | `deployment.domain` |
| Variance partitioning and measurement reliability | `diagnostics.variance` |

## 5. CV tracks (evaluation contract)

Per fold, save `SSE_fold`, `N_test_fold`, `TSS_fold`. Pool as
`RMSEP = sqrt(ΣSSE/ΣN_test)`, `Q² = 1 − ΣSSE/ΣTSS` (micro-averaged). Also report
macro `mean_c(RMSEP_c)` and a bootstrap CI over folds.
- **Track 1** channel-specific: hyperparams optimized per channel.
- **Track 2** global: one hyperparam set minimizing the pooled objective over all
  channels (data-rich channels dominate; report macro too).
- **Track 3** universal: forward-chained LOCO, channel fully held out.

Splitters are **forward-chained** (`t_val < t_test`, train on `[0, t_val−1]` /
`[0, t_test−1]`). Nested CV: outer = evaluation, inner = hyperparameter + feature
selection (incl. CARS's own K-fold). Never tune on the outer test fold.

### 5.1 The baseline queue, why these 14 conditions

`baseline_queue()` is a **designed experiment, not a grid**: each batch changes exactly **one** factor
from a common baseline (`baselines_1.2` = linear PLSR, global, no selector), so every result is a clean
one-factor contrast. That discipline is forced by the regime. P ≫ N (~157 features, ~89 rows) and
interpretability *is* the deliverable, so a few well-motivated models beat a large blind sweep. All use
`k_grid=(2,3)` (few latent variables, more would overfit tiny N). The four batches answer the four
scientific questions the study poses:

- **B1.** Is one shared model enough, or does each sensor need its own?
- **B2.** Does feature selection help, given far more features than rows?
- **B3.** Does a model generalize to a sensor it never saw?
- **B4.** Which model architecture suits this data?

- **Batch 1: representation/architecture (global, no selector) → B4 "which architecture".** Varies only
  the model, spanning the choices that matter for this data: `linear_plsr` (**baseline**, the anchor),
  its `log` variant (`log(x/d0)` + log target, multiplicative/log-linear drift),
  `weighted_plsr` by concentration and by `repeatability_snr` (heteroscedasticity, the Assumption-5
  WLS response), `orthogonal_plsr` (O-PLS, strip Y-orthogonal nuisance like fabrication offset), and
  `nonlinear_plsr` (deg-2, curvature guard).
- **Batch 2: track → B1 "global vs channel-specific".** `channelspecific_2.1` is the *same* architecture
  as 1.2 with only `track="channel"`, so 2.1-vs-1.2 is a clean read of sensor heterogeneity.
- **Batch 3: feature selection (P ≫ N) → B2 "does selection help".** Holds linear/global fixed, varies
  only the selector: `sr`, `vip` and `smc` (PLS-importance filters) and `cars` (a Monte-Carlo
  *wrapper* that searches subsets). Each is a "beats the 1.2 baseline?" contrast. The three filters
  span the ways a PLS model can rank a feature: `sr` by how much of the feature the target-projected
  loading explains, `vip` by the feature's weighted share of explained Y variance, `smc` by whether
  the feature's association with the normalized regression vector is significant at all. They are
  measurably distinct on real data, and the section on them in §10 explains why that took a
  correction to achieve. `threshold_grid=(0.5,1.0)` sweeps the cutoff for the two ratio-valued
  filters; sMC is scored as `-log10(p)` and uses `(2.0,6.0)`, which is the same pair of significance
  levels at any fold size.
- **Batch 4: transfer → B3 "generalize to new channels".** `newchannels_4.1` is the `universal` (LOCO)
  track with CARS, the hardest test (held-out sensor); selection is what lets a P ≫ N model generalize
  rather than overfit the training sensors.

Every tool tracks a data property: collinear spectra + P ≫ N → the PLSR family (not OLS/ridge);
heteroscedastic → weighted PLSR; multiplicative → log; nonlinear → poly; structured nuisance → O-PLS;
P ≫ N → SR/VIP/sMC/CARS; new sensors → universal + selection. Kernel / multi-block / multi-level PLSR and
mRMR / permutation selectors are registered extension points that **raise on use**, they trade away the
interpretability that is the goal, and ~89 rows cannot support them. Because the queue is factorial-ish
around one baseline, a single run answers B1 to B4 at once, read off
`condition_ranking.parquet`.

## 6. Execution model

- Conditions: **sequential**, decision-gated task queue (cheap→expensive).
- Folds: **parallel** via joblib/loky, one single-threaded worker per outer fold.
- Inner loop + CARS generations + BLAS: **serial** within a worker; BLAS pinned
  to 1 thread (`OMP/OPENBLAS/MKL_NUM_THREADS=1` set before NumPy import) to avoid
  oversubscription. Reclaim idle cores via independent conditions-in-a-tier or
  multi-seed CARS chains in parallel.
- **CARS runtime/stall handling:** bounded by the `n_generations` cap **and** the
  `timebox_patience` valve (stop when the internal RMSEP stops improving) in
  `selection.cars`, so per-fold CARS runtime is finite and no worker hangs. Two further
  guards are **not** implemented: an adaptive pre-flight micro-profile (tuning
  `n_generations` from a timed single-fold probe) and a hard per-fold wall-clock timeout,
  because the generation cap already bounds runtime, making them optional rather than required.

## 7. Output layout

`outputs/model_discovery_<ts>/` → `run_config.json`, `logs/`, `data/` (shared
preprocessing, parquet/npy), `conditions/<name>/folds/<ch>_<t_test>/`
(`model_arrays.npz`, `hyperparams.json`, `metrics.json`),
`conditions/<name>/aggregated_metrics.json` + `feature_stability.parquet`,
top-level `summary.parquet`, `report/` (rankings + `discovery_summary.md`).

## 8. File formats

npy (raw arrays) · Parquet (feature tables, dtype/precision safe) · JSON
(config/metrics, cast numpy→python) · npz + manifest.json (model assets, **no
pickle**, since linear PLSR = coefficients + normalization scalars; nonlinear/
kernel add inner-map params, kernel also retains support data).

## 9. Deployment flow

1. Extract/quality-check in vivo EIS/FSCV → features.
2. D0-normalize against an **early in-vivo baseline** (not in-vitro D0).
3. Apply **frozen** scalers computed once on 100% of in-vitro data after final
   model selection (`scaler_*.npy`); prefer **robust (median/IQR)** scaling in
   vivo to resist biological/motion outliers.
4. Quantify domain shift each timepoint via **CORAL** (align mean+cov, Frobenius
   distance); chain across time for a drift path; flag Assumption-4 strain.
5. Apply frozen PLSR to predict recalibrated `NormIpeak`; back-transform if log.
6. Report estimate + domain-distance confidence flag; log drift trajectory.


## 10. Design notes and known limits

These are the decisions and caveats most likely to matter when you apply the library to your
own data.

**The `Z''` sign convention.** A PSTrace export stores the `Z'' / Ohm` column as `-Im(Z)`,
positive for a capacitive interface at low frequency, which is the opposite of what the column
name implies. `data.pstrace` negates it on read, so the rest of the library works in
`Z = Z' + j*Im(Z)` with `Im(Z) < 0` for a capacitive interface. This matters downstream:
`C_s = 1/(w*Z'')` comes out negative if the raw sign is used, so `features.eis` works from
`-Z''` and the test suite checks `C_s, C_p > 0` against a known RC circuit. This is a property
of the instrument software rather than of the physics, so verify it on a known-capacitive
channel of your own exports before trusting a first run.

**FSCV peak definition on broad peaks.** On real exports the dopamine oxidation feature is
often a broad hump spanning roughly 0.5 to 0.9 V rather than a sharp peak. A chord-baseline
`NormIpeak`, measured as deviation from a straight chord across the 0.6 to 0.8 V window,
under-counts a broad hump badly enough that the `3 * noise_floor` SNR gate then rejects every
row. `features.fscv.norm_ipeak(method="direct")` instead takes the background-subtracted
current at `V_ox` normalized by `I_bgd`, which recovers a clean dose response (log-dose r of
about 0.95). `"chord"` remains available and is the better choice for genuinely sharp peaks.

For the same reason `extract_dataset(acceptance="monotonic")` accepts a channel-timepoint when
`NormIpeak` rises with log-concentration (`r >= monotonic_r_min`), which is a better
early-timepoint criterion than a single-cycle SNR held hostage to background matching. The
recommended pairing on broad-peak data is `peak_method="direct", acceptance="monotonic"`, and
that is the default. An opt-in `detrend` removes a residual baseline slope but can eat
broad-onset signal on channels whose faradaic response starts below the baseline window, so it
is off by default. The switching-potential capacitive spike near 1.0 to 1.3 V is excluded by
restricting to the anodic sweep and the dopamine window.

**Selectivity ratio, VIP and sMC are three distinct filters, and keeping them distinct took
care.** All three are built on target projection, so it is easy to write two of them that
collapse onto one. `smc_scores` did: it regressed each feature on the target-projected score
*with an intercept*, which recovers the loading and so recomputes the selectivity ratio,
matching it to floating point (maximum relative difference 8e-15, rank correlation 1.000000)
across random data, wide P > N, and every latent-variable count.

The distinguishing vector is which one carries the decomposition. Selectivity ratio explains
each feature with the loading obtained by projecting the data onto the target-projected score.
sMC uses the **normalized regression vector** `b/||b||` directly, which is what makes it
sharper and noisier, since it mixes predictive with orthogonal variation. `smc_scores` now
forms the explained part as `outer(Xc @ b_hat, b_hat)` and returns the reference F statistic
with (1, n - 2) degrees of freedom. Rank correlation against the selectivity ratio on real
data is 0.39 at two components and 0.16 at three.

**sMC is thresholded on significance, not on the raw statistic.** An F cutoff is not portable
across folds, because F grows with residual degrees of freedom: at 2190 rows the 5% cutoff
F(1, 2188) = 3.85 admits 136 of 143 features. `select("smc", ...)` therefore scores
`-log10(p)` against F(1, n - 2), so a threshold of 2.0 means p <= 0.01 on a small fold and on
a large one alike. The condition's threshold grid is `(2.0, 6.0)` in those units rather than
the `(0.5, 1.0)` the ratio-valued filters use.

sMC still selects far more features than the other two. On the study featureset the three
filters keep, at their two grid cutoffs: selectivity ratio 41 and 31, VIP 35 and 28, sMC 135
and 132. That is the statistic behaving as defined rather than a fault: an F test asks whether
a feature associates with the target projection at all, and at this many rows nearly every one
does. Read the sMC condition as a mild filter and the other two as aggressive ones; that
contrast is part of what Batch 3 measures.

**Model selection at small N is the dominant risk.** Selecting among many conditions on few
folds overfits the selection itself, not just the model. This is why the outputs are framed as
hypothesis-generating, why every ranking carries a bootstrap interval, and why the baseline
queue is a one-factor-at-a-time design rather than a grid. Compare intervals; do not read a
ranking as a winner.

**Derivative features are noise-sensitive.** `local_n` and the ratio features come from
`d log|Z| / d log f`. Smooth the spectrum, or use a windowed slope, before differentiating,
and propagate replicate variance through the result.

**NIPALS Y-loading normalization.** Some write-ups of the algorithm normalize the Y-weight `c`
and then deflate with it, where canonical deflation uses the unnormalized
`q = Y'^T t / (t^T t)`. This library delegates to scikit-learn, which does the canonical thing,
so the discrepancy is a documentation trap rather than a behavioural difference.

**Deliberate simplifications, chosen rather than missing.** The ICC pre-filter uses the
SS-ANOVA variance split rather than a heavier VCA mixed-effects estimator. Weighted PLSR is
the square-root-weight WLS approximation. Kernel, multi-block and multi-level PLSR, and the
mRMR and permutation-`t_max` selectors, are registered extension points that raise on use:
they trade away the interpretability that is the point here, and the row counts involved
cannot support them. Promotion of the best model between batches is a manual review
checkpoint by design, supported by the checkpointed discovery notebook.

## 11. What the tests cover

The suite is structured around the places where a silent error would be most expensive:
leakage-safety of the forward-chained splitters, the metric-pooling identities, the EIS sign
and smoothing physics against a known RC circuit, serialization round-trips, the PSTrace
parser including the `Z''` convention, staggered per-device timepoint derivation, and
end-to-end runs on both synthetic featuresets and a real-format raw directory.

```bash
pytest                   # the whole suite
pytest -m "not slow"     # skips the one full-profile run
```
