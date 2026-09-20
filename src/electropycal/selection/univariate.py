"""Univariate feature selection: rank each feature on its own against the target.

Mutual-information relevance filter (captures nonlinear/non-monotonic relations,
unlike Pearson/Spearman). mRMR's redundancy penalty and permutation/t_max p-values
are documented extension points; MI-threshold covers the univariate baseline.
"""

from __future__ import annotations

import numpy as np


def mi_scores(X: np.ndarray, y: np.ndarray, random_state: int | None = 0) -> np.ndarray:
    from sklearn.feature_selection import mutual_info_regression
    return mutual_info_regression(np.asarray(X, float), np.asarray(y, float).ravel(),
                                  random_state=random_state)


def mi_select(X: np.ndarray, y: np.ndarray, threshold: float,
              random_state: int | None = 0) -> np.ndarray:
    scores = mi_scores(X, y, random_state=random_state)
    idx = np.where(scores >= threshold)[0]
    return idx if idx.size else np.array([int(np.argmax(scores))])
