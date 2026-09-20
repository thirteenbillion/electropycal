"""Batch-by-batch discovery runner (folds discovery_checkpointed §3+).

A thin stateful driver over the discovery queue so the notebook runs one batch at a time with progress /
ETA, accumulates a cross-batch ranking for the between-batch auto-flag advisories, and plots each batch's
pooled RMSEP. Wraps ``run_condition`` + ``auto_flag_conditions``; holds no plotting logic the library
doesn't already own.
"""

from __future__ import annotations

import time

import pandas as pd

from .runner import run_condition
from .scheduler import auto_flag_conditions

BATCH_LABEL = {1: "1. Baselines", 2: "2. Channel-specific",
               3: "3. P>N feature selection", 4: "4. New-channel generalization"}


class BatchRunner:
    """Run discovery batches from a queue against one ``RunData``, checkpointing bundles to ``out_dir``.

    ``run(batch)`` evaluates that batch's conditions and returns a ranked frame; ``advise(next)`` prints
    auto-flag advisories for the next batch from results so far; ``exclude(*names)`` drops conditions;
    ``plot(df, title)`` draws the batch's pooled RMSEP (95% CI)."""

    def __init__(self, queue, data, out_dir, profile):
        self.queue = list(queue)
        self.data = data
        self.out_dir = str(out_dir)
        self.profile = profile
        self.ranking: list[dict] = []

    def print_settings(self) -> None:
        prof = "FAST" if getattr(self.profile, "n_boot", None) == 200 else "full"
        print(f"BatchRunner: {len(self.queue)} conditions | out_dir = {self.out_dir} | profile = {prof} "
              f"(min_train_times={self.profile.min_train_times}, seeds={self.profile.seeds})")
        print("  run(batch_num[, only=...]) -> ranked frame; advise(next); exclude(*names); plot(df, title)")

    def run(self, batch_num: int, only=None) -> pd.DataFrame:
        conds = [c for c in self.queue if c.batch == batch_num and (only is None or c.name in only)]
        if not conds:
            print(f"batch {batch_num}: no conditions selected"); return pd.DataFrame()
        rows, t0 = [], time.time()
        for i, c in enumerate(conds, 1):
            _, agg = run_condition(c, self.data, self.out_dir, self.profile, seed=0, progress=True)
            # A condition that yields NO folds gets the short early return from
            # ``aggregate_by_track`` -- {"pooled_rmsep": nan, "n_folds": 0} -- with no CI keys
            # and no ``pooled_q2``. That is reachable on small corpora: on the 96x143 demo
            # featureset (4 timepoints) ``baselines_1.6_nonlinearPLSR`` produces 0 folds, so
            # subscripting here killed the whole batch with a bare KeyError. Read defensively,
            # the way ``runner.py`` already logs these, and let the NaN row fall out downstream
            # (``plot`` drops non-finite rmsep; ``auto_flag_conditions`` filters on isfinite).
            nan = float("nan")
            self.ranking.append(dict(condition=c.name, architecture=c.architecture,
                                     pooled_rmsep=agg.get("pooled_rmsep", nan),
                                     rmsep_ci_lo=agg.get("rmsep_ci_lo", nan),
                                     rmsep_ci_hi=agg.get("rmsep_ci_hi", nan)))
            rows.append(dict(condition=c.name, track=c.track, selector=c.selector,
                             rmsep=agg.get("pooled_rmsep", nan),
                             ci_lo=agg.get("rmsep_ci_lo", nan), ci_hi=agg.get("rmsep_ci_hi", nan),
                             q2=agg.get("pooled_q2", nan), folds=agg.get("n_folds", 0)))
            el = time.time() - t0; eta = el / i * (len(conds) - i)
            print(f"  [batch {batch_num}] {i}/{len(conds)} conditions done — {el:.0f}s elapsed, ETA {eta:.0f}s")
        return pd.DataFrame(rows).sort_values("rmsep", na_position="last").reset_index(drop=True)

    def advise(self, next_batch: int) -> dict:
        upcoming = [c for c in self.queue if c.batch == next_batch]
        flags = auto_flag_conditions(self.ranking, upcoming)
        for c in upcoming:
            print(f"  {c.name}" + (f"   [FLAGGED: {flags[c.name]}]" if c.name in flags else ""))
        return flags

    def exclude(self, *names) -> None:
        self.queue = [c for c in self.queue if c.name not in names]
        print("queue now:", [c.name for c in self.queue])

    def plot(self, df: pd.DataFrame, title: str) -> pd.DataFrame:
        import matplotlib.pyplot as plt
        if not len(df):
            return df
        fig, ax = plt.subplots(figsize=(8, 3.2))
        m = df.dropna(subset=["rmsep"])
        if len(m):
            err = [m.rmsep - m.ci_lo, m.ci_hi - m.rmsep]
            ax.bar(m.condition, m.rmsep, yerr=err, capsize=4, color="#4C72B0")
        ax.set_ylabel("pooled RMSEP (95% CI)"); ax.set_title(title)
        plt.xticks(rotation=25, ha="right"); plt.tight_layout(); plt.show()
        return df
