"""Configuration objects for the discovery pipeline.

A ``Condition`` is one row of the task queue (architecture × feature selector × CV track +
its hyperparameter grids). A ``Profile`` holds the runtime knobs that let the *same code* run a
fast test config or a full scientific run — CI uses :data:`FAST`, the once-off real run uses the
defaults. ``RunData`` is the post-extraction featureset the runner consumes.
"""

from __future__ import annotations

from pathlib import Path

from dataclasses import dataclass, field

import numpy as np

# RESERVED_COLUMNS now lives in the data schema (it is a fact about the featureset, not about
# discovery) so feature extraction and the phase-1 review modules do not have to import the
# modeling package. Imported here for ``feature_columns`` below, which also keeps the old
# ``discovery.config.RESERVED_COLUMNS`` import path working for existing callers.
from ..data.schema import RESERVED_COLUMNS

# track name → (outer CV mode, inner-split track)
TRACK_CV = {"channel": ("loto_c_ac", "channel"),
            "global": ("loto_c_ac", "global"),
            "universal": ("loco", "global"),
            # DIAGNOSTIC only (not deployment-valid): random-split over (channel, timepoint) units,
            # ignoring time order — compare its Q² to "global" to measure the forward-chaining cost.
            "random": ("random", "random")}


@dataclass
class RunData:
    """Post-extraction featureset. Features are **D0-normalized per sensor** by
    ``extract_dataset`` (drift from each sensor's earliest timepoint); the
    in-fold Z-scoring is applied during CV (``runner.fit_zscore``). The target (``NormIpeak`` /
    ``sensitivity``) is not normalized."""

    X: np.ndarray
    y: np.ndarray
    channel: np.ndarray
    timepoint: np.ndarray
    concentration: np.ndarray
    feature_names: list[str]
    #: original "device:channel" (or "channel") label per factorized sensor id, so saved predictions
    #: can be split back to device / channel; ``None`` when identities weren't provided.
    sensor_labels: np.ndarray | None = None
    #: per-row reproducibility SNR (NormIpeak / std across cycles) for optional sample weighting
    #: (``weighted_by="repeatability_snr"``); ``None`` when the featureset didn't carry it.
    repeatability_snr: np.ndarray | None = None

    @classmethod
    def from_frame(cls, df, target: str = "NormIpeak") -> "RunData":
        # A path here is a common slip, and pandas' own error for it is unhelpful. Named
        # explicitly because the fix (read it yourself first) is not obvious from the type.
        if isinstance(df, (str, Path)):
            raise TypeError(
                f"from_frame takes a DataFrame, not a path ({df!r}). Read it first:"
                f"\n    import pandas as pd"
                f"\n    RunData.from_frame(pd.read_parquet({str(df)!r}))"
                f"\nOr use electropycal.cli._load_data, which accepts either.")
        """Build from a featureset DataFrame. Reserved columns: ``channel``,
        ``timepoint``, ``concentration``, ``NormIpeak`` (+ optional ``device``);
        every other column is a feature.

        ``target`` is the column used as ``y`` (default ``"NormIpeak"``; pass
        ``"sensitivity"`` for the dose-response-slope framing built by
        :func:`electropycal.features.targets.sensitivity_featureset`). The target column
        is always excluded from the feature matrix, and ``concentration`` is optional
        (the sensitivity featureset has one row per sensor-timepoint and no dose axis).

        When ``device`` is present, the CV "channel" (sensor) identity is the
        composite ``(device, channel)`` — channel 3 on device 2-2 is a different
        sensor than channel 3 on device 2-3."""
        import pandas as pd
        feats = [c for c in df.columns if c not in RESERVED_COLUMNS and c != target]
        X = np.array(df[feats].to_numpy(float), copy=True)
        X[~np.isfinite(X)] = np.nan
        # DROP columns that are entirely undefined (e.g. a band-averaged feature whose sub-band lies
        # outside the analysis band — ideality_C_band_HF / n_band_HF when band=(2,2000)). Imputing an
        # all-NaN column to a constant 0 poisons the PLSR-based selectors (SR/sMC/VIP) with a
        # zero-variance column (their per-fold PLS fit then produces NaN loadings). An empty feature
        # carries no signal, so remove it rather than keep a constant placeholder.
        _keep = ~np.all(np.isnan(X), axis=0)
        if not _keep.all():
            X = X[:, _keep]
            feats = [f for f, k in zip(feats, _keep) if k]
        # Sparse NaN/inf are left IN PLACE and imputed per fold from the training rows only
        # (features.normalize.fit_median_impute / apply_median_impute, applied in
        # discovery.runner). This used to median-impute here, over train and test together,
        # which let held-out rows set the values the model was fitted on. The evaluators
        # (hierarchical / stratify / classify) already z-scored with train-only nan-aware
        # statistics and zeroed the rest, so they read a NaN-carrying X correctly as-is.
        #
        # The all-NaN column drop above stays corpus-wide on purpose: a column is empty because
        # the feature is undefined for the chosen ``band`` (ideality_C_band_HF / n_band_HF when
        # band=(2,2000)) -- a property of a pinned run-wide parameter, not of row values. Dropping
        # a column also removes information rather than importing a test statistic, so it cannot
        # leak; and the surviving column set is recorded in the extraction pin, so it is
        # reproducible. A column that is empty only within one fold's training rows is handled by
        # fit_median_impute's 0.0 fallback.
        labels = None
        if "device" in df.columns:
            key = df["device"].astype(str) + ":" + df["channel"].astype(str)
            sensor, labels = pd.factorize(key)
            labels = np.asarray(labels)                      # sensor id -> "device:channel"
        else:
            sensor = df["channel"].to_numpy()                # unchanged: sensor == channel
        snr = (df["repeatability_snr"].to_numpy(float)
               if "repeatability_snr" in df.columns else None)
        conc = (df["concentration"].to_numpy(float) if "concentration" in df.columns
                else np.zeros(len(df)))                       # sensitivity rows have no dose axis
        return cls(
            X=X, y=df[target].to_numpy(float),
            channel=np.asarray(sensor), timepoint=df["timepoint"].to_numpy(float),
            concentration=conc, feature_names=feats,
            sensor_labels=labels, repeatability_snr=snr)


@dataclass
class Condition:
    """One testing condition (task-queue entry)."""

    name: str
    architecture: str = "linear_plsr"       # models.variants
    track: str = "global"                    # channel | global | universal
    selector: str | None = None              # vip | sr | mi | icc | cars | icc+cars
    k_grid: tuple[int, ...] = (2,)
    threshold_grid: tuple[float, ...] = (0.0,)   # used when the selector has a threshold
    weighted_by: str | None = None           # 'concentration' | 'repeatability_snr' → sample weights (weighted_plsr)
    transform: str = "linear"                # 'linear' | 'log' — feature+target representation.
    #: 'log' = the log model on ANY architecture: multiplicative features -> log(x/d0),
    #: additive features unchanged, target fit in log space and predictions exp'd back to NormIpeak.

    def cv_mode(self) -> str:
        return TRACK_CV[self.track][0]

    def inner_track(self) -> str:
        return TRACK_CV[self.track][1]

    def has_threshold(self) -> bool:
        return self.selector in ("vip", "sr", "smc", "mi", "icc", "icc+cars")

    @property
    def batch(self) -> int:
        """Report task-queue batch number parsed from the name (e.g.
        ``baselines_1.2_...`` -> 1); 0 if not encoded."""
        try:
            return int(self.name.split("_")[1].split(".")[0])
        except (IndexError, ValueError):
            return 0


@dataclass
class Profile:
    """Runtime knobs (the fast/full switch)."""

    seeds: tuple[int, ...] = (0,)
    min_train_times: int = 3
    n_boot: int = 2000                       # bootstrap CI resamples
    n_jobs: int = 1                          # joblib workers over outer folds
    cars: dict = field(default_factory=lambda: {
        "k_max": 5, "n_generations": 50, "n_folds": 5, "timebox_patience": 10})


FAST = Profile(seeds=(0,), n_boot=200, n_jobs=1,
               cars={"k_max": 3, "n_generations": 12, "n_folds": 3, "timebox_patience": 4})
"""Small profile for tests / smoke runs (seconds, not hours)."""


def effective_min_train_times(min_train_times: int, n_timepoints: int) -> int:
    """Cap ``min_train_times`` so forward-chained CV can form at least one fold.

    A test timepoint needs ``min_train_times`` earlier timepoints to train on, so the
    scheme needs ``>= min_train_times + 1`` distinct timepoints; with fewer, **every**
    fold is rejected and the run yields 0 folds / nan metrics silently. This caps the
    value to ``n_timepoints - 1`` (the most training history the data can offer) so the
    run produces folds instead. Returns the value unchanged when there is enough history,
    or when ``n_timepoints < 2`` (CV is impossible regardless — the caller should surface
    that separately). Applied by :func:`~electropycal.discovery.runner.run_condition` so the
    notebook, CLI, and scripts all get the same protection.
    """
    if n_timepoints < 2:
        return min_train_times
    return min(min_train_times, n_timepoints - 1)


def baseline_queue() -> list[Condition]:
    """The baseline task queue: the simplest architectures/selectors, run first as a reference."""
    return [
        # 1.1: the simplest baseline — fixed k=2, no k-optimization, no selection, Track 2
        # (global). 1.2 is the same with k swept over (2,3); comparing 1.1 vs 1.2 shows whether
        # k-optimization helps at all.
        Condition("baselines_1.1_linearPLSR", "linear_plsr", "global", None, k_grid=(2,)),
        Condition("baselines_1.2_linearPLSR", "linear_plsr", "global", None, k_grid=(2, 3)),
        # full log model: log(x/d0) features + log-target on the linear arch — replaces
        # the old target-only log_plsr. Any architecture can run its log version via transform="log".
        Condition("baselines_1.3_linearPLSR_log", "linear_plsr", "global", None,
                  k_grid=(2, 3), transform="log"),
        Condition("baselines_1.4_weightedPLSR", "weighted_plsr", "global", None,
                  k_grid=(2, 3), weighted_by="concentration"),
        Condition("baselines_1.4_weightedPLSR_log", "weighted_plsr", "global", None,
                  k_grid=(2, 3), weighted_by="concentration", transform="log"),
        Condition("baselines_1.5_orthogonalPLSR", "orthogonal_plsr", "global", None, k_grid=(2, 3)),
        Condition("baselines_1.6_nonlinearPLSR", "nonlinear_plsr", "global", None, k_grid=(2, 3)),
        Condition("baselines_1.7_snrWeightedPLSR", "weighted_plsr", "global", None,
                  k_grid=(2, 3), weighted_by="repeatability_snr"),
        Condition("channelspecific_2.1_linearPLSR", "linear_plsr", "channel", None, k_grid=(2, 3)),
        Condition("pNproblem_3.1_SR", "linear_plsr", "global", "sr",
                  k_grid=(2, 3), threshold_grid=(0.5, 1.0)),
        Condition("pNproblem_3.1_VIP", "linear_plsr", "global", "vip",
                  k_grid=(2, 3), threshold_grid=(0.5, 1.0)),
        # sMC is scored as -log10(p), so its thresholds are significance levels rather than
        # the 0.5/1.0 scale SR and VIP use: 2.0 is p <= 0.01 and 6.0 is p <= 1e-6. Expect it
        # to select most features on a large corpus, because the F test asks whether each
        # feature is associated with the target projection and almost all of them are.
        Condition("pNproblem_3.1_sMC", "linear_plsr", "global", "smc",
                  k_grid=(2, 3), threshold_grid=(2.0, 6.0)),
        Condition("pNproblem_3.2_CARS", "linear_plsr", "global", "cars", k_grid=(2, 3)),
        Condition("newchannels_4.1_CARS", "linear_plsr", "universal", "cars", k_grid=(2, 3)),
    ]
