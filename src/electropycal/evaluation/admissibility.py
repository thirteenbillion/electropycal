"""Target-admissibility screen (study experiment A2) — pick the default target WITHOUT Q².

When no configuration achieves out-of-sample Q²>0, Q² cannot rank candidate targets (comparing noise to
noise). The default modeling target is instead locked by **admissibility** — four properties of the target
itself, each measurable with no model:

- **identifiability** — the fraction of sensor-timepoints where the target is estimable (a finite fit from
  the doses that sensor-timepoint actually has). An often-unidentifiable target (e.g. Langmuir ``Kd`` on
  non-saturating data) is disqualified here, no Q² needed.
- **invertibility** — whether the target, as a calibration model, is monotone and can be inverted to
  recover concentration (what recalibration ultimately needs). A dose-response *slope* or Freundlich
  *exponent* is invertible; a signed *curvature* or *intercept* alone is not (it is a shape/level
  covariate, not a standalone recalibration target).
- **dynamic range** — how much the target actually varies across sensor-timepoints (std and coefficient of
  variation). A target that barely moves is unpredictable *by definition*, independent of any model.
- **reliability** — a fit-quality proxy (median dose-response correlation ``dose_response_r``): is the
  target estimated from well-behaved dose-responses, or is it itself noise?

``target_admissibility`` returns one row per candidate with these columns and an ``admissible`` flag, so
the default target is chosen by an explicit, reproducible, Q²-free screen.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

# Whether a target, used as the calibration model, is monotone-invertible to concentration. This is a
# property of the framing (physics), not of any particular dataset. Slopes/exponents recover dose; signed
# shape/level parameters and unidentifiable-on-this-data saturation constants do not stand alone.
INVERTIBLE_TARGETS = {
    "sensitivity": True,          # NormIpeak = intercept + slope·log10(C) -> invert for C
    "power_beta": True,           # NormIpeak = a·C^beta -> invert for C
    "sensitivity_intercept": False,   # DC level; covariate, not a standalone recalibration target
    "sensitivity_curvature": False,   # signed shape; not invertible alone
    "power_a": False,             # gain coefficient; pairs with beta, not standalone
    "sat_imax": False, "hill_imax": False,
    "sat_kd": True, "sat_logkd": True, "hill_kd": True,   # half-saturation IS a monotone locator (if identifiable)
    "hill_n": False,
}

DEFAULT_CANDIDATES = ("sensitivity", "sensitivity_intercept", "sensitivity_curvature",
                      "power_a", "power_beta")


def target_admissibility(feat_df: pd.DataFrame, targets=None, min_estimable: float = 0.7,
                         min_cv: float = 0.15, featureset: pd.DataFrame | None = None) -> pd.DataFrame:
    """Q²-free admissibility screen over candidate targets. See module docstring.

    ``feat_df`` is the per-dose featureset; it is reframed to one row per ``(device, channel, timepoint)``
    via :func:`~electropycal.features.targets.sensitivity_featureset` (pass a pre-computed one as
    ``featureset`` to avoid recomputation). Returns a DataFrame sorted admissible-first then by dynamic
    range, with columns: ``target, identifiable (fraction estimable), n_estimable, invertible,
    target_std, target_cv (std/mean|value|), reliability (median dose_response_r), admissible``.
    ``admissible`` = identifiable ≥ ``min_estimable`` AND invertible AND ``target_cv`` ≥ ``min_cv``.
    """
    from ..features.targets import sensitivity_featureset
    S = featureset if featureset is not None else sensitivity_featureset(feat_df)
    cand = list(targets) if targets is not None else [t for t in DEFAULT_CANDIDATES if t in S.columns]
    n = len(S)
    r_all = S["dose_response_r"].to_numpy(float) if "dose_response_r" in S.columns else np.full(n, np.nan)

    rows = []
    for t in cand:
        if t not in S.columns:
            rows.append(dict(target=t, identifiable=0.0, n_estimable=0, invertible=INVERTIBLE_TARGETS.get(t, False),
                             target_std=float("nan"), target_cv=float("nan"), reliability=float("nan"),
                             admissible=False))
            continue
        v = S[t].to_numpy(float)
        fin = np.isfinite(v)
        vv = v[fin]
        ident = float(fin.mean()) if n else 0.0
        std = float(np.std(vv)) if vv.size else float("nan")
        mean_abs = float(np.mean(np.abs(vv))) if vv.size else float("nan")
        cv = (std / mean_abs) if (mean_abs and np.isfinite(mean_abs) and mean_abs > 0) else float("nan")
        rel = float(np.nanmedian(r_all[fin])) if fin.any() and np.isfinite(r_all[fin]).any() else float("nan")
        invert = INVERTIBLE_TARGETS.get(t, False)
        admissible = bool(ident >= min_estimable and invert and np.isfinite(cv) and cv >= min_cv)
        rows.append(dict(target=t, identifiable=round(ident, 3), n_estimable=int(fin.sum()),
                         invertible=invert, target_std=std, target_cv=cv, reliability=rel,
                         admissible=admissible))
    out = pd.DataFrame(rows)
    return out.sort_values(["admissible", "target_cv"], ascending=[False, False],
                           na_position="last").reset_index(drop=True)
