"""Data schema for the observation unit and featureset (see DESIGN §3).

Feature *type* metadata lives here so normalization (`features.normalize`) and
diagnostics can dispatch on how each feature should be treated (additive vs
multiplicative D0-normalization, whether it is log-transformable, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass

# Frequency-dependent EIS feature types (evaluated per log frequency, 10 Hz–100 kHz).
EIS_FREQ_FEATURE_TYPES: tuple[str, ...] = (
    "R_s", "R_p", "C_s", "C_p", "ideality_C", "tau", "local_n",
)

# Global (whole-spectrum) EIS feature types.
EIS_GLOBAL_FEATURES: tuple[str, ...] = (
    "R_s_integral", "R_p_integral", "C_s_integral", "C_p_integral",
    "f_ideality_crossover", "ideality_C_band_LF", "ideality_C_band_MF",
    "ideality_C_band_HF", "tau_ratio", "n_band_LF", "n_band_MF", "n_band_HF",
)

# FSCV-derived predictor features.
FSCV_FEATURES: tuple[str, ...] = ("mean_Vpeak", "mean_Ibg")

# Frequency bands (Hz) for band-averaged features.
BANDS_HZ: dict[str, tuple[float, float]] = {
    "LF": (10.0, 300.0),
    "MF": (300.0, 3_000.0),
    "HF": (3_000.0, 100_000.0),
}

# Feature types that are naturally bounded / phase / log-slope: D0-normalize as an
# ADDITIVE shift (0.0-centered) and do NOT log-transform.
ADDITIVE_D0_TYPES: frozenset[str] = frozenset(
    {"mean_Vpeak", "ideality_C", "ideality_C_band", "local_n", "n_band",
     "f_ideality_crossover", "min_neg_phase"}
)
# min_neg_phase is a PHASE (degrees) that can sit near zero and goes negative when the interface
# turns inductive, so it must be additive: a multiplicative ratio to a near-zero / sign-changing
# baseline is unstable and meaningless. (inductive_onset_hz, by contrast, is a positive,
# log-distributed FREQUENCY and stays multiplicative like tau — its ratio to the baseline onset is
# an interpretable, log-transformable drift signal.)
#
# Everything else (magnitude/area-scaling: mean_Ibg, Bode R/C, tau, tau_ratio, inductive_onset_hz)
# is D0-normalized MULTIPLICATIVELY (divide by D0, 1.0-centered) and is log-transformable.


@dataclass(frozen=True)
class Observation:
    """One (channel, timepoint, concentration) measurement row."""

    device: str
    channel: int
    timepoint: object          # e.g. "D0", "D20"
    concentration_nM: float    # background encoded as 0.0

    def key(self) -> tuple:
        return (self.device, self.channel, self.timepoint, self.concentration_nM)


@dataclass
class FeatureSet:
    """Extracted featureset: X (N × P), Y (N,), aligned metadata, feature names."""

    X: "object"                # np.ndarray (N, P) — kept as object to avoid import at schema level
    y: "object"                # np.ndarray (N,)   — NormIpeak
    feature_names: list[str]
    observations: list[Observation]

    def __post_init__(self) -> None:
        n = len(self.observations)
        if self.X.shape[0] != n or self.y.shape[0] != n:
            raise ValueError("X, y, and observations must share the same N")
        if self.X.shape[1] != len(self.feature_names):
            raise ValueError("X columns must match feature_names")


_FREQ_SUFFIX = __import__("re").compile(r"_f\d+$")


def d0_normalization_kind(feature_name: str) -> str:
    """Return 'additive' or 'multiplicative' for a feature name."""
    base = _FREQ_SUFFIX.sub("", feature_name)  # strip per-frequency suffix: ideality_C_f03 -> ideality_C
    for t in ("ideality_C_band", "n_band"):    # collapse band suffixes to the base type
        if base.startswith(t):
            base = t
            break
    return "additive" if base in ADDITIVE_D0_TYPES else "multiplicative"


def is_log_transformable(feature_name: str) -> bool:
    """Naturally bounded features are left in raw units for D0-subtraction."""
    return d0_normalization_kind(feature_name) == "multiplicative"


#: Columns that are **never predictors**. Two kinds live here, and both must stay out of any feature
#: matrix: the *identifiers / QC bookkeeping* that describe a row rather than the electrode, and the
#: **RESERVED targets** — everything derived from the dopamine faradaic response. Using any of the
#: latter as an input would leak the quantity the model is supposed to predict from leakage-safe
#: electrode state (EIS + the 0 nM background), which is the core methodological guarantee of this
#: library. Note ``mean_Vpeak`` is deliberately *absent*: a peak *position* describes the electrode,
#: not the dopamine magnitude, so it is a legitimate predictor.
#:
#: This lives in the schema (not in the modeling code) because it is a fact about the featureset, so
#: feature extraction, diagnostics, and every evaluator can share one definition.
RESERVED_COLUMNS = ("device", "channel", "sensor_id", "timepoint", "concentration",
                    "NormIpeak", "noise_floor", "snr", "rms_snr", "repeatability_snr",
                    "rep_std", "peak_at_edge", "peak_area_clipped", "dose_response_r",
                    # faradaic peak-SHAPE (deformation-mode characterization / candidate targets;
                    # NOT predictors — derived from the DA signal, would leak the NormIpeak numerator):
                    "peak_height", "peak_area", "peak_fwhm",
                    # sensitivity-target framing (features.targets.sensitivity_featureset):
                    "sensitivity", "sensitivity_intercept", "sensitivity_curvature", "n_conc",
                    # Langmuir + Hill saturation-curve targets (monotone/invertible calibration model):
                    "sat_imax", "sat_kd", "sat_logkd", "hill_imax", "hill_kd", "hill_n",
                    "power_a", "power_beta")
