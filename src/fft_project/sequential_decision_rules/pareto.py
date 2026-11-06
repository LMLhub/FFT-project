import numpy as np
import pandas as pd

def pareto_frontier(metrics: pd.DataFrame) -> pd.DataFrame:
    """
    Return features that are nondominated in
    the (accuracy, coverage) plane.

    Features with undefined accuracy are excluded.
    Features with identical metrics are all retained.
    """

    valid = metrics.dropna(subset=["accuracy"]).copy()

    accuracy = valid["accuracy"].to_numpy()
    coverage = valid["coverage"].to_numpy()

    n = len(valid)
    dominated = np.zeros(n, dtype=bool)

    for j in range(n):
        for k in range(n):
            if j == k:
                continue

            if (
                accuracy[k] >= accuracy[j]
                and coverage[k] >= coverage[j]
                and (
                    accuracy[k] > accuracy[j]
                    or coverage[k] > coverage[j]
                )
            ):
                dominated[j] = True
                break

    return (
        valid.loc[~dominated]
        .sort_values("coverage")
        .reset_index(drop=True)
    )
