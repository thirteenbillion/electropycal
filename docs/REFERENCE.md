# electropycal reference

The single lookup record: every default and how it is surfaced, the feature dictionary, the
cross-validation tracks, the baseline queue, the directory layouts, and the function-level API.

This tracks the source of truth in `features.catalog`, `discovery.config`, `analysis_config`,
`data.pstrace` and `features.extract`. If this file and the code disagree, the code wins and
this file is stale.

For a walkthrough rather than a lookup, see `docs/USAGE.md`. For why a default is what it is,
see `docs/DESIGN.md`.

The **Status** column below reads: *Locked*, a considered default that should not be changed
casually; *Provisional*, a default that may still move as more data accumulates; *Per-run*,
not a fixed default at all but something you set for the run in front of you.

---

## 1. Default parameters

### 1.1 `AnalysisConfig`, processing / QC knobs (`analysis_config.py`)
| Param | Default | Status | Meaning |
|-------|---------|--------|---------|
| `band` | `(2.0, 2000.0)` Hz | Provisional | EIS analysis band; `"auto"` = data-driven (`inventory.recommended_band`). Study default chosen by trading drift alignment against channel retention; `diagnostics_review` shows that trade-off on your own data. |
| `peak_method` | `"direct"` | Locked | FSCV oxidation-peak locator (`"direct"` \| `"chord"`). |
| `acceptance` | `"monotonic"` | Locked | Dose-response acceptance test for a channel-timepoint. |
| `mono_tol` | `0.10` | Locked | EIS.2 \|Z\|-monotonicity tolerance. |
| `min_norm_snr` | `3.0` | Locked | Per-dose reproducibility-SNR cutoff. |
| `monotonic_r_min` | `0.6` | Locked | Min dose–log-conc correlation (FSCV.1). |
| `mono_method` | `"pearson"` | Locked | Dose-response correlation type (`"pearson"` \| `"spearman"`). |
| `max_reps` | `3` | Locked | First N FSCV replicate cycles used. |
| `gate_on` | `{eis:True, monotonic:True, snr_all:False, peak_in_window:False}` | Locked | Which QC gates are active by default. |

### 1.2 `Profile`, runtime knobs (`discovery.config`)
| Param | Full default | `FAST` (tests) | Status | Meaning |
|-------|--------------|----------------|--------|---------|
| `seeds` | `(0)` | `(0)` | Per-run | RNG seeds; multi-seed only firms up stochastic selectors (CARS/MI). |
| `min_train_times` | `3` | `3` | Locked | Earliest-history requirement for a forward-chained fold (auto-capped by `effective_min_train_times`). |
| `n_boot` | `2000` | `200` | Locked | Bootstrap CI resamples. |
| `n_jobs` | `1` | `1` | Per-run | joblib workers over outer folds (raise for multi-core). |
| `cars` | `{k_max:5, n_generations:50, n_folds:5, timebox_patience:10}` | `{3,12,3,4}` | Locked | CARS selector search budget. |

`cap_min_train_times` is often listed with these but is **not** a `Profile` field: it is an
argument to `run_condition` and `run_discovery` (default `True`), which lowers
`min_train_times` on a short series so the run yields folds instead of nothing.

### 1.3 `extract_dataset` arguments outside `AnalysisConfig`

These are function arguments, not config fields, so they are passed per call rather than
saved into `electropycal_analysis_config.json`. A pinned run records the ones that affect the
featureset in `run_config.json` instead.

| Argument | Default | Meaning |
|---|---|---|
| `band` | **required** | no default on purpose; pass `(lo, hi)`, `"auto"`, or `pin=` |
| `detrend` | `False` | subtract a non-Faradaic baseline slope before peak and noise; can eat broad-onset signal |
| `baseline_window` | `(-0.1, 0.2)` V | window the detrend slope is fitted over |
| `smooth_window` | `11` | Savitzky-Golay window in samples, odd; `0` disables. Smooths signal and background before subtraction |
| `smooth_poly` | `2` | Savitzky-Golay polynomial order |
| `min_dose_response_range` | `None` | optional amplitude floor, dropping monotone-but-flat channel-timepoints |
| `drop_negative` | `True` | drop non-physical `NormIpeak < 0` rows |
| `nonfaradaic_window` | `None` | optional explicit non-Faradaic window |
| `peak_edge_tol` | `0.03` | how close to the sweep edge a peak may sit before it is flagged as edge-clipped |
| `require_interior_peak` | `False` | reject edge-clipped peaks outright rather than flagging them |
| `d0_normalize` | `True` | per-sensor drift-from-baseline normalization |
| `device_types` | `None` | restrict extraction to given device types |
| `on_empty_session` | `"warn"` | `"warn"` or `"raise"` when a session yields zero rows |
| `pin` / `pin_out` | `None` | read a previous run's derived parameters, or write this run's |
| `pin_mode` | `"reproduce"` | with `pin`: `"reproduce"` requires the input to be a subset the pin has seen; `"extend"` keeps the pinned anchors but admits new sessions, for an incremental rebuild |
| `n_jobs` | `1` | worker processes over device-timepoint sessions |
| `progress` | `False` | print per-session progress |

Fixed windows and constants: `PEAK_WINDOW = (0.4, 1.0) V` for the oxidation-peak search, and
EIS sub-bands `BANDS_HZ = LF (10-300), MF (300-3000), HF (3000-100000)` Hz. Note that the HF
sub-band is **empty under the default band (2, 2000)**, so `ideality_C_band_HF` and `n_band_HF`
come out all-NaN and are dropped at `RunData.from_frame`. Feature-schema version: 6.

### 1.4 Fixed modeling conventions
- **D0-normalization: ON**, per-sensor drift-from-baseline; **multiplicative** for magnitudes, **additive**
  for phase/bounded types (see feature dictionary D0 column). Locked by alignment (drift 0.62 ≫ absolute
  0.22), not by Q².
- **Leakage-safe predictors**: EIS + 0 nM-background FSCV only; faradaic peak features are RESERVED targets.
- **Negative-dose rows** (`NormIpeak < 0`) dropped as non-physical.
- **Saturating targets (`sat_*`, `hill_*`) OFF by default**, `Kd` unidentifiable on non-saturating data.
- **Default modeling target** (provisional): `sensitivity` (dose-response slope): locked by the A2 admissibility
  screen: identifiable, invertible, and it moves. Whether it is predictive on a given dataset
  is an empirical question that dataset has to answer.

### 1.5 How each parameter is surfaced

Where a value is defined and how a user changes it. "Python arg" means pass it when
calling the function; "CLI flag" means it is on the `electropycal` command; "edit source"
means change `baseline_queue()` or the default in the file; "fixed" means it is not
currently exposed.

| Stage | Parameter (default) | Defined in | Purpose | Surfaced via |
|---|---|---|---|---|
| **Ingestion** | `band` **required** (no default) / **(2,2000) shared-config default** / CLI `--band auto` | `features/extract.py` `extract_dataset`; `analysis_config.py` | EIS analysis band (gate + features). The `extract_dataset` primitive **requires** an explicit band: a tuple, `"auto"`, or a `pin`; the shared config default is (2,2000); the CLI defaults to `auto` and prints what it resolved | Python arg; **CLI `--band`** |
| Ingestion | `peak_method="direct"` | `features/extract.py` → `fscv.norm_ipeak` | FSCV peak height: `direct` (current at V_ox; **default**, robust to broad DA peaks) or `chord` (chord-baseline height) | Python arg; **CLI `--peak-method`** |
| Ingestion | `detrend=False` | `features/extract.py` → `fscv` | subtract non-Faradaic baseline slope before peak/noise (opt-in; can eat broad-onset signal) | Python arg; **CLI `--detrend`** |
| Ingestion | `acceptance="monotonic"` | `features/extract.py` | channel acceptance: `monotonic` (dose-response r ≥ `monotonic_r_min`; **default**) \| `snr` (per-dose gate) \| `none` | Python arg; **CLI `--acceptance`** |
| Ingestion | `min_norm_snr=3.0` | `features/extract.py` | SNR gate multiple (`acceptance=snr`): drop rows with `NormIpeak < 3·noise_floor` | Python arg; **CLI `--min-norm-snr`** |
| Ingestion | `monotonic_r_min=0.6` | `features/extract.py` | min log-dose correlation to accept a channel (`acceptance=monotonic`) | Python arg |
| Ingestion | `mono_method="pearson"` | `features/extract.py` → `fscv.dose_response_corr` | FSCV.1 correlation: `pearson` (linear-in-log; **default**) or `spearman` (rank) | Python arg |
| Ingestion | `mono_tol=0.10` | `features/extract.py` → `data/quality.py` | EIS \|Z\| monotonicity tolerance (check B); relaxed from a stricter 0.05 | Python arg; **CLI `--mono-tol`** |
| Ingestion | `max_reps=3` | `features/extract.py` → `_avg_fscv` | keep only the first N FSCV replicate cycles (intended `[0..N-1]` rounds; drops erroneous appended extras); `None` = all | Python arg; **CLI `--max-reps`** |
| Ingestion | `smooth_window=11` | `features/extract.py` → `fscv.smooth_current` | Savitzky-Golay window (samples, odd) applied to signal **and** background before subtraction; `0` disables | Python arg |
| Ingestion | `smooth_poly=2` | `features/extract.py` → `fscv.smooth_current` | Savitzky-Golay polynomial order | Python arg |
| Ingestion | `min_dose_response_range=None` | `features/extract.py` → `_accept` | **off by default**; amplitude floor: drop monotone-but-flat (near-dead) channel-timepoints whose NormIpeak range (max−min) is below this | Python arg; **CLI `--min-dose-response-range`** |
| Ingestion | `drop_negative=True` | `features/extract.py` | drop doses with `NormIpeak < 0` (non-physical) before acceptance | Python arg |
| Ingestion | `baseline_window=(-0.1,0.2)` V | `features/extract.py` → `fscv` | window for `detrend` baseline-slope fit (only used when `detrend=True`) | Python arg |
| Ingestion | `nonfaradaic_window=None` | `features/extract.py` → `fscv.noise_floor` | voltage window the RMS noise floor is measured over; `None` = `NONFARADAIC_WINDOW` (below the peak search) | Python arg |
| Ingestion | `peak_edge_tol=PEAK_EDGE_TOL` | `features/extract.py` → `fscv.peak_at_edge` | flag (non-gating) a dose whose `mean_Vpeak` lands within this of a `PEAK_WINDOW` bound | Python arg |
| Ingestion | `require_interior_peak=False` | `features/extract.py` → `fscv.norm_ipeak` | require the NormIpeak peak to be an interior maximum (drop edge-pinned as NaN); off = edge-pinning is reported but non-gating | Python arg |
| Ingestion | `n_jobs=1` | `features/extract.py`; `data/inventory.py` | worker processes for the session-parallel raw walks (extraction / QC / reliability); `-1` = all cores; byte-identical to serial, BLAS-capped per worker | Python arg; **CLI `--n-jobs`** |
| Feature | `smooth=True` | `features/eis.py` `eis_features` | Savitzky-Golay smooth `log\|Z\|` before `local_n` | fixed (True in extract) |
| Feature | `PEAK_WINDOW=(0.4,1.0)` V | `features/fscv.py` | oxidation-peak search window for `NormIpeak`/peak features | Python arg (`v_window`) |
| **Condition** | `architecture="linear_plsr"` | `discovery/config.py` `Condition` | model type per queue entry (`linear/log/weighted/orthogonal/nonlinear_plsr`) | edit `baseline_queue()` |
| Condition | `track="global"` | `config.py` | CV track: `channel` (Track 1) \| `global` (Track 2) \| `universal` (Track 3/LOCO) \| `random` (**diagnostic only**, random-split, not deployment-valid; compare to `global` to measure the forward-chaining cost) | edit `baseline_queue()` |
| Condition | `selector=None` | `config.py` | feature selector: `vip/sr/smc/mi/cars/icc/icc+cars` | edit `baseline_queue()` |
| Condition | `k_grid=(2)`, `threshold_grid=(0.0)` | `config.py` | inner-loop LV counts / selector thresholds swept | edit `baseline_queue()` |
| Condition | `weighted_by=None` | `config.py` | `'concentration'` → sample weights (weighted PLSR) | edit `baseline_queue()` |
| Architecture | `n_orth=1` (orthogonal), `degree=2` (nonlinear) | `models/variants.py` `build` | O-PLS orthogonal comps / poly degree | Python arg (fixed in queue) |
| **Profile (runtime)** | `n_jobs=1` | `config.py` `Profile` | joblib workers over CV folds | **CLI `--n-jobs`** |
| Profile | `seeds=(0)` | `config.py` | CARS seeds (multi-seed stability) | Python arg; **CLI `--seed`** sets the run seed |
| Profile | `min_train_times=3` | `config.py` | min prior timepoints before a `t_test` is used. **Nested CV needs ≥3 timepoints** (inner-train < t_val < t_test); with only 2 collected timepoints set this to 1 and expect 0 folds until a 3rd exists | Python arg |
| Profile | `n_boot=2000` | `config.py` | bootstrap RMSEP-CI resamples (`FAST`=200) | Python arg; **CLI `--profile`** |
| Profile | `cars={k_max:5, n_generations:50, n_folds:5, timebox_patience:10}` | `config.py` | CARS budget + stall/time-box valve | Python arg (`Profile(cars=…)`) |
| Profile | `FAST` preset | `config.py` | seconds-scale test config | **CLI `--profile fast`** |
| **Discovery run** | `conditions=baseline_queue()` | `discovery/scheduler.py` | which conditions run | edit `baseline_queue()` |
| Discovery run | `gate=None` | `scheduler.py` | per-condition programmatic prune | Python arg |
| Discovery run | `batch_gate=None` | `scheduler.py` | per-batch review/edit: return `bool` or a `list[Condition]` replacing the remaining queue (include/exclude) | Python arg; **CLI `--batched`** (interactive) |
| Discovery run | `out_root="outputs"`, `seed=0` | `scheduler.py` | output root / RNG seed | **CLI `--out` / `--seed`** |
| Discovery run | `cap_min_train_times=True` | `scheduler.py` | **safety valve:** auto-lower `min_train_times` on a short series so CV yields folds instead of 0/nan | Python arg; **CLI `--min-train-times`** overrides |
| **Target framing** | `TARGET` (`normipeak`) | `features/targets.py`; CLI `--target` | recalibration target: per-dose `normipeak` \| `sensitivity`/`_intercept`/`_curvature` \| `sat_imax`/`sat_kd`/`sat_logkd` \| `hill_imax`/`hill_kd`/`hill_n` | **CLI `--target`**; notebook `TARGET` / `TARGETS` |
| Target framing | `saturation=True`, `hill=True` | `features/targets.py` `sensitivity_featureset` | also fit Langmuir / Hill saturation curves (emit `sat_*` / `hill_*`) | Python arg |
| Target framing | `weight_col="repeatability_snr"` | `features/targets.py` | SNR-weight the per-sensor-timepoint curve fits (down-weight noisy low-dose points) | Python arg |
| Normalization | `d0_normalize=True` | `features/extract.py` | per-sensor drift-from-baseline. **Extraction-level, not per-condition**, toggling it means re-extracting the featureset, not editing the queue | Python arg; **CLI `--no-d0-normalize`** |
| Metrics | `alpha=0.05` | `evaluation/metrics.py` | CI level (95%) | fixed |
| **Deployment** | `architecture`, `k`, `feature_index=None` | `deployment/deploy.py` `freeze_model` | the model **you** freeze/deploy | Python arg |
| Deployment | `scaler="robust"` | `deploy.py` | in-vivo scaler: robust median/IQR vs `"zscore"` | Python arg |
| Deployment | `sample_weight=None`, `d0_kinds=None` | `deploy.py` | optional WLS weights / D0 kinds | Python arg |
| Deployment | `with_domain=True` | `deploy.py` `recalibrate` | attach CORAL domain-distance flag | Python arg |
| Deployment | `eps=1e-6`, `include_mean=True` | `deployment/domain.py` | CORAL regularization / mean term | Python arg |
| **Stabilization** (optional) | `tol=0.01`, `patience=10`, `smooth_window=5` | `data/stabilization.py` | convergence plateau criterion | Python arg; `stabilization_review` nb |
| Stabilization | `v_target=0.7`, `cycles_per_round=20` | `data/stabilization.py` `stabilization_traces` | oxidation potential tracked / cycles per round | Python arg; `stabilization_review` nb |

**Most common edits for a real run:** `--profile full --n-jobs <cores>` on the CLI;
and `baseline_queue()` in `discovery/config.py` to choose architectures/selectors/
tracks and their `k_grid`/`threshold_grid`. Everything else has a sensible default.

---

## 2. Feature dictionary

`Default` = library treatment (**role** · **D0 kind**). Role: **predictor** (feeds the model),
**RESERVED target** (never a predictor: DA-derived), **metadata** (describes the row). Per-frequency EIS
types are emitted as `<type>_fNN` across the in-band grid.

### 2.1 Predictors. EIS (impedance)
| Feature type | Default | Derivation | Purpose |
|--------------|---------|------------|---------|
| `R_s` | predictor · mult | series resistance from the Bode fit (Re path) | solution/access resistance |
| `R_p` | predictor · mult | parallel / charge-transfer resistance (Bode fit) | interfacial charge-transfer state |
| `C_s` | predictor · mult | series capacitance −1/(2πf·Im Z) | interfacial capacitance |
| `C_p` | predictor · mult | parallel capacitance (Im path) | interfacial capacitance (parallel model) |
| `ideality_C` | predictor · add | sin²(phase); 1=ideal cap, 0=resistor | how capacitive the interface is |
| `tau` | predictor · mult | local RC time constant R·C | interfacial time constant |
| `local_n` | predictor · add | CPE exponent −d log\|Z\|/d log f | surface dispersion/roughness |
| `R_s_integral` … `C_p_integral` | predictor · mult | ∫ feature d(log f) over band | whole-spectrum magnitude summary |
| `f_ideality_crossover` | predictor · add | freq where ideality_C crosses 0.5 | capacitive→resistive corner |
| `ideality_C_band_LF/MF/HF` | predictor · add | mean ideality_C per sub-band | banded ideality |
| `n_band_LF/MF/HF` | predictor · add | mean local_n per sub-band | banded CPE exponent |
| `tau_ratio` | predictor · mult | tau(low-f)/tau(high-f) | time-constant dispersion |
| `min_neg_phase` | predictor · add | most-inductive phase over FULL spectrum | inductive-artifact severity (band-independent) |
| `inductive_onset_hz` | predictor · mult | lowest freq with Im Z ≥ 0 (FULL spectrum) | inductive-onset location (band-independent) |

### 2.2 Predictors. FSCV background (0 nM cycle; DA-independent)
| Feature type | Default | Derivation | Purpose |
|--------------|---------|------------|---------|
| `mean_Vpeak` | predictor · add | mean V_ox of bg-subtracted FSCV | oxidation-peak potential |
| `mean_Ibg` | predictor · mult | mean background current at V_ox (0 nM) | NormIpeak denominator / baseline current |
| `bg_charge` | predictor · mult | ∫ i_bg dV over V_ox window | double-layer charge (∝ C_dl×window) |
| `bg_cap` | predictor · mult | mean \|i_anodic − i_cathodic\|/2 over window | double-layer-capacitance proxy |
| `bg_switch` | predictor · mult | bg current at anodic switching potential | electrode-window edge / fouling |

### 2.3 Predictors, temporal (experiment-only; added by notebook `add_temporal`, not `extract_dataset`)
| Feature type | Default | Derivation | Purpose |
|--------------|---------|------------|---------|
| `time_since_baseline` | predictor · mult | days since the sensor's first timepoint | aging covariate (known at deployment) |
| `<feature>__lag1` | predictor · mult | predictor's value at previous timepoint | short-term history |
| `<feature>__delta` | predictor · mult | predictor's first difference | bounded rate-of-change |

### 2.4 RESERVED targets (DA-derived; never predictors)
| Feature type | Default | Derivation | Purpose |
|--------------|---------|------------|---------|
| `NormIpeak` | target · mult | (I_sig − I_bg)/I_bg at V_ox | the primary response |
| `peak_height` | target · mult | bg-subtracted faradaic peak height | NormIpeak numerator |
| `peak_area` | target · mult | bg-subtracted peak area / charge | integrated response |
| `peak_fwhm` | target · mult | bg-subtracted peak FWHM | peak kinetics/shape |
| `sensitivity` | target · mult | dose-response slope (NormIpeak vs log10[DA]) | **default recalibration target** |
| `sensitivity_intercept` | target · mult | dose-response intercept (signed) | DC level (covariate) |
| `sensitivity_curvature` | target · mult | dose-response quadratic curvature (signed) | shape (covariate) |
| `power_beta` | target · mult | Freundlich exponent β (NormIpeak=a·Cᵝ) | non-saturating invertible framing |
| `power_a` | target · mult | Freundlich coefficient a | gain at unit concentration |
| `sat_imax` / `sat_kd` / `sat_logkd` | target · mult | Langmuir Imax·C/(Kd+C) fit | saturation model (OFF by default) |
| `hill_imax` / `hill_kd` / `hill_n` | target · mult | Hill Imax·Cⁿ/(Kdⁿ+Cⁿ) fit | cooperative saturation (OFF by default) |

### 2.5 Metadata / QC (RESERVED; describe the row)
`noise_floor`, `snr`, `rms_snr`, `repeatability_snr`, `rep_std`, `dose_response_r`, `peak_at_edge`,
`peak_area_clipped`, `n_conc`. SNR/quality bookkeeping used by gates and optional sample weighting.
`peak_area_clipped` flags a faradaic lobe truncated by the sweep/window edge (so `peak_area` is an
under-estimate, common at low dose; a hard voltage-sweep-span limit, not a filtering issue).

> Print the live catalog any time: `python -c "from electropycal import print_feature_catalog; print_feature_catalog()"`

### 2.6 `RESERVED_COLUMNS` (never predictors)
`device, channel, sensor_id, timepoint, concentration, NormIpeak, noise_floor, snr, rms_snr,
repeatability_snr, rep_std, peak_at_edge, dose_response_r, peak_height, peak_area, peak_fwhm, sensitivity,
sensitivity_intercept, sensitivity_curvature, n_conc, sat_imax, sat_kd, sat_logkd, hill_imax, hill_kd,
hill_n, power_a, power_beta`

---

## 3. Cross-validation tracks (`TRACK_CV`)
| Track | Outer mode | Inner split | Deployment-valid? | Use |
|-------|-----------|-------------|-------------------|-----|
| `channel` | `loto_c_ac` | channel | forward-chained | per-sensor (Track 1); few folds |
| `global` | `loto_c_ac` | global | forward-chained | pooled (Track 2); well-powered |
| `universal` | `loco` | global | forward-chained | LOCO transfer (Track 3) |
| `random` | `random` | random | No: diagnostic only | non-stationarity probe (E1) |

---

## 4. Baseline queue (`baseline_queue()`, 14 conditions)
| Condition | Arch | Track | Selector | k_grid | threshold | weighted_by | transform |
|-----------|------|-------|----------|--------|-----------|-------------|-----------|
| baselines_1.1_linearPLSR | linear_plsr | global |, | (2) |, |, | linear |
| baselines_1.2_linearPLSR | linear_plsr | global |, | (2,3) |, |, | linear |
| baselines_1.3_linearPLSR_log | linear_plsr | global |, | (2,3) |, |, | log |
| baselines_1.4_weightedPLSR | weighted_plsr | global |, | (2,3) |, | concentration | linear |
| baselines_1.4_weightedPLSR_log | weighted_plsr | global |, | (2,3) |, | concentration | log |
| baselines_1.5_orthogonalPLSR | orthogonal_plsr | global |, | (2,3) |, |, | linear |
| baselines_1.6_nonlinearPLSR | nonlinear_plsr | global |, | (2,3) |, |, | linear |
| baselines_1.7_snrWeightedPLSR | weighted_plsr | global |, | (2,3) |, | repeatability_snr | linear |
| channelspecific_2.1_linearPLSR | linear_plsr | channel |, | (2,3) |, |, | linear |
| pNproblem_3.1_SR | linear_plsr | global | sr | (2,3) | (0.5,1.0) |, | linear |
| pNproblem_3.1_VIP | linear_plsr | global | vip | (2,3) | (0.5,1.0) |, | linear |
| pNproblem_3.1_sMC | linear_plsr | global | smc | (2,3) | (2.0,6.0) |, | linear |
| pNproblem_3.2_CARS | linear_plsr | global | cars | (2,3) |, |, | linear |
| newchannels_4.1_CARS | linear_plsr | universal | cars | (2,3) |, |, | linear |

---

## 5. Directory structures

### 5.1 Input (both in vitro and in vivo)
```
<ROOT>/
  <YYYYMMDD>_<devicetype>_<testtype>/          session folder; e.g. signal, channeltest
    <deviceid>_<signaltype>_<dose|period>.csv   one PSTrace export per device/measurement
```
- **Folder** `parse_folder`: `<YYYYMMDD>_<devicetype>_<testtype>`, e.g. `20250115_neurostring_signal`.
  `testtype` is simply the trailing token, lowercased, so it is open-ended; only `signal`
  folders feed the featureset, and anything else (`channeltest`, for instance) is skipped.
- **File** `parse_filename`: `<deviceid>_<signaltype>_<last>.csv`.
  - `signaltype ∈ {eis, fscv, paired}`, `paired` = time-sequential EIS+FSCV in one file (in-vivo).
  - `last` = a dose (`0nm`, `100nM` → nM float) **or** a token: `stabilization` in vitro,
    optionally suffixed (`stabilization-full`, `-start`, `-end`); `baseline` / `live` in vivo.
    Doses are matched as a numeric predicate, not an enumerated list, so a new concentration
    needs no code change.
- **`timepoint`** = elapsed **days since that device's first signal session** (staggered series handled
  per device). Device baseline (D0) = the device's earliest date.
- **In vitro**: per-dose `fscv`/`eis` files across concentrations → dose-response per channel-timepoint.
- **In vivo**: `<deviceid>_paired_{baseline,live}.csv`, no concentration axis; the frozen model applies.

### 5.2 Output artifacts (written into `<ROOT>`)
| File | Writer | Contents |
|------|--------|----------|
| `electropycal_analysis_config.json` | `save_analysis_config` | the QC/extraction parameters used |
| `electropycal_qc_stats.json` | `save_qc_stats` | QC yields / per-gate dropout (the evidence) |
| `diagnostics_featureset__<hash>.parquet` | `diagnostics_review` cache | cached featureset, keyed by schema and parameters |

---

## 6. API surface

### 6.1 Module layout
```
data/         io, inventory, pstrace (ingestion), quality, schema (feature dict + RESERVED_COLUMNS),
              stabilization, synthetic
features/     extract (extractor), eis, fscv, normalize (D0), targets, catalog
models/       base, plsr, variants (linear/weighted/orthogonal/nonlinear PLSR)
selection/    cars, univariate (MI), icc, pseudo_multivariate (SR/sMC/VIP)
discovery/    config (Condition/RunData/Profile/baseline_queue), runner, baseline, scheduler,
              batch (BatchRunner), review (results-review plots)
evaluation/   cv (folds/tracks), metrics (+ fit_ridge / ALPHA_GRID inner-CV penalty), framing,
              multioutput (PLS2), classify, baselines (time-only / naive / feature-free
              channel-persistence controls), hierarchical, stratify, admissibility
deployment/   deploy, domain (domain-shift / LOCO transfer), plots (recalibration monitors)
diagnostics/  variance
rawspectra.py (raw_spectra_review API) · stabreview.py (stabilization_review API) ·
overview.py (discovery_checkpointed API: finalized-dataset plots) ·
qcdash.py (quality_filtering_dashboard API: QCDashboard) ·
diagreview.py (diagnostics_review API: DiagnosticsReview)
analysis_config.py · cli.py · viz.py (style + panel/color helpers) · _parallel.py · __init__.py
```
> **Review modules.** `rawspectra`, `stabreview`, `qcdash`, `diagreview`, `deployment/plots`,
> `overview`, `discovery/batch` and `discovery/review` hold the plotting and analysis that each
> notebook calls. The notebooks are thin wrappers over these, so the same figures and tables are
> reproducible from a plain script.

### 6.2 The functions, module by module

Signatures show the defaults the library itself uses. Where a default differs from the
shared `AnalysisConfig` value, §1.1 is the one a configured run actually gets.

#### `electropycal.data`
- `pstrace.read_pstrace(path, encoding="utf-16") -> PSTraceExport`, parse an
  export; `.fscv[(ch,rep)]`, `.eis[(ch,rep)]`. `parse_filename(name)` (dose →
  float nM, or a token like `"stabilization"`), `parse_folder(name)`.
- `quality.eis_quality(freq, z_real, z_imag, band=(10,1e5), mono_tol=0.05) -> dict` checks A (capacitive) / B (monotonic |Z|) / C (Z'>0) + `valid`.
- `inventory.index_raw(root)`, `channel_survival(idx, total_channels=16)`,
  `quality_filtering(idx)`, dashboard/QC tables.
- `synthetic.make_dataset(...)`, `make_invivo_drift(...)`,
  `write_synthetic_pstrace_dir(root, ...)`, synthetic featureset / raw tree.
- `stabilization.check_converged(drift, tol=0.01, patience=10, smooth_window=5)`,
  `cycle_drift`, `estimate_settle_tau`, `i_at_v_target(voltage, current, v_target=0.7)`,
  `stabilization_traces(export, v_target=0.7, cycles_per_round=20) -> DataFrame`.
- `io.*`, json/parquet/npz helpers. `schema.*`, feature-type metadata.

#### `electropycal.features`
- `eis.eis_features(freqs, z_real, z_imag, smooth=True) -> dict` (7 arrays);
  `eis_global_features(...) -> dict` (12 scalars).
- `fscv.norm_ipeak(signal, background, voltages, v_window=(0.6,0.8), method="chord",
  detrend=False, baseline_window=(-0.1,0.2))`, `method="direct"` = current at V_ox
  (recommended for broad DA peaks); `mean_vpeak`, `mean_ibg`, `noise_floor`
  (also takes `detrend`), `anodic_sweep`.
- `normalize.d0_normalize(X, names, d0_row, log=False)`, `fit_zscore/apply_zscore`,
  `fit_robust/apply_robust`.
- `extract.extract_dataset(root, band=(10,1e5), min_norm_snr=3.0, mono_tol=0.10,
  peak_method="direct", detrend=False, acceptance="monotonic", monotonic_r_min=0.6) ->
  DataFrame`. Defaults suit real broad-DA-peak data; pass `peak_method="chord",
  acceptance="snr"` for the stricter sharp-peak method. Acceptance also accepts
  `"monotonic+snr"` (monotonic channel gate + a lenient per-dose SNR cut).
- `extract.extract_invivo(root, band=(10,1e5), peak_method="direct", …) -> DataFrame` in-vivo featureset from a raw dir (`paired`/`baseline`/`live`); one row per time
  sample, no `concentration`.
- `normalize.d0_normalize_frame(df, feature_cols, group=("device","channel")) ->
  DataFrame`, per-sensor D0-normalization vs the earliest timepoint.

#### `electropycal.models`
- `variants.build(architecture, k, **kw) -> BaseArch`: architectures:
  `linear_plsr`, `log_plsr`, `weighted_plsr`, `orthogonal_plsr` (`n_orth`),
  `nonlinear_plsr` (`degree`). `.fit(X,y).predict(X).vip().to_arrays().manifest()`.
- `plsr.PLSRModel(k)`, the sklearn NIPALS core + VIP.
- `base.save_model_bundle / load_model_bundle / predict_from_bundle`.

#### `electropycal.selection`
- `cars.cars_select(X, y, k_max=5, n_generations=50, n_folds=5, sample_ratio=0.9,
  timebox_patience=10, random_state=0)`; `cars_select_multiseed(seeds=(0,1,2))`.
- `pseudo_multivariate.select(method, X, y, k, threshold)`, method ∈
  `{"vip","sr","smc"}`. VIP and SR are ratio-valued and thresholded near 1.0; `"smc"` scores
  `-log10(p)` against F(1, n-2), so its threshold is a significance level (2.0 means p <= 0.01).
  `vip_scores`, `selectivity_ratio`, `smc_scores` (the raw F) and `smc_significance` are also
  callable directly. `univariate.mi_select(X, y, threshold)`.
- `icc.icc_prefilter(X, channel, timepoint, concentration, threshold)`.

#### `electropycal.evaluation`
- `cv.outer_folds(channel, timepoint, mode="loto_c_ac", min_train_times=3)`,
  `inner_split(...)`.
- `metrics.FoldResult`, `pooled_rmsep`, `pooled_q2`, `macro_rmsep`,
  `bootstrap_rmsep_ci`, `aggregate`.
- `tracks.aggregate_by_track(folds, track, n_boot=2000)`. Returns `pooled_rmsep` and
  `n_folds` only when a condition yields no folds, so read the CI keys and `pooled_q2`
  with `.get(key, nan)`.
- `framing.compare_target_framings(df, value_col="NormIpeak", conc_col="concentration",
  k=3, min_train_times=3, include_interaction=True, include_quadratic=True, min_conc=3,
  min_conc_quad=4) -> DataFrame`, one row per target framing. Fits each framing under the
  same forward-chained folds so the comparison is like-for-like, and inverts the quadratic
  framing using the **training** dose grid, which is what a deployment has.

#### `electropycal.discovery`
- `config.Condition`, `Profile`, `RunData.from_frame(df)`, `FAST`, `baseline_queue()`.
- `scheduler.run_discovery(data, conditions=None, out_root="outputs",
  profile=None, seed=0, gate=None, batch_gate=None)`; `pin_blas_single_threaded()`.
- `runner.run_condition(condition, data, out_dir, profile, seed=0)`.
- `folds.fold_names(cond_dir)`, `fold_records(cond_dir) -> {fold: {"manifest",
  "hyperparams", "metrics"}}`, `load_fold_bundle(cond_dir, fold) -> (arrays, manifest)`: the
  per-fold models a condition writes to `fold_models.npz` + `folds.json`. Both read the
  one-directory-per-fold layout of run directories written before 0.11.0.
- `review.*`, plots over a finished run directory, each returning the frame it drew so the
  numbers are available without re-reading the run. All take `show=True`:
  `plot_condition_ranking(run_dir)` (pooled RMSEP with 95% CI per condition, best at
  bottom), `plot_fold_spread(run_dir)` (per-fold RMSEP by condition, ordered by median),
  `plot_feature_ranking(run_dir, top=15)` (features by cross-condition mean selection
  frequency), `plot_calibration_review(run_dir, condition=None)` (true against predicted by
  timepoint, parity, and a six-panel residual diagnostic for one condition), and
  `plot_target_framing_comparison(featureset, include_interaction=True)`.

#### `electropycal.deployment`
- `deploy.freeze_model(X, y, feature_names, architecture, k, out_dir,
  feature_index=None, sample_weight=None, scaler="robust", d0_kinds=None)`;
  `freeze_top(run_dir, data, condition=None, stability_min=0.5, out_dir=…)`, freeze a
  discovery-selected model on 100% of in-vitro data;
  `load_frozen_model`, `recalibrate(model, X_invivo, with_domain=True)`,
  `recalibrate_invivo(model, invivo_root, flag_distance=None)`.
- `domain.coral_distance / coral_transform / drift_path`.

#### `electropycal.viz`
- `set_pub_style()`, apply Nature-style matplotlib defaults (compact sans-serif,
  thin de-spined axes, frameless legends, Okabe-Ito CVD-safe cycle). `ensure_style()`
  applies it once per process and every plotting entry point calls it.
  `categorical(n)`, n colors in fixed order; `SEQUENTIAL` = `"viridis"`.
- `emit(name, fig=None, *, formats=None, provenance=None, stage=None, params=None,
  footer=None)`, save a figure as PNG and PDF, close it, and display it where that works.
  `provenance` is slugified into the filename; `provenance`, `stage` and `params` go to the
  output folder's `FIGURES_LOG.txt` (append-only) and `FIGURES.txt` (what is on disk now),
  not into the image unless `footer=True`. `configure_output(out_dir, formats, show,
  footer)` sets the defaults; `figure_index()` rebuilds `FIGURES.txt` after files are added
  or deleted by hand.

#### `electropycal.diagnostics`
- `variance.variance_hierarchy(...)` partitions each feature's variance into structural,
  drift and dose components; `measurement_reliability(...)` estimates the response
  reliability `R` against the replicate-noise floor.

---

## 7. Notebooks (`notebooks/`)

`raw_spectra_review` · `quality_filtering_dashboard` · `stabilization_review` ·
`diagnostics_review` · `discovery_checkpointed` · `discovery_results_review` ·
`deployment_domain_shift`

Each runs standalone against the shipped `demo/` tree, and synthesizes an equivalent where no
tree is reachable. See `docs/USAGE.md` §9 for what each one is for.
