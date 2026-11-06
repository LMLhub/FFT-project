import numpy as np
import pandas as pd
from fft_project.sequential_decision_rules.data import Dataset

def evaluate_features(
    dataset: Dataset,
    remaining: np.ndarray
) -> pd.DataFrame:
    """
    Evaluate all features on the remaining instances.

    Returns a DataFrame with one row per feature.
    """

    X = dataset.X[remaining]
    Y = dataset.Y[remaining]

    n_remaining = len(Y)

    classified = X != 0
    correct = X == Y[:, None]

    n_classified = classified.sum(axis=0)
    n_correct = correct.sum(axis=0)
    n_incorrect = n_classified - n_correct

    coverage = np.divide(
        n_classified,
        n_remaining,
        out=np.zeros(dataset.M, dtype=float),
        where=n_remaining > 0
    )

    accuracy = np.divide(
        n_correct,
        n_classified,
        out=np.full(dataset.M, np.nan),
        where=n_classified > 0
    )

    return pd.DataFrame({
        "feature": np.arange(dataset.M),
        "correct": n_correct,
        "incorrect": n_incorrect,
        "classified": n_classified,
        "coverage": coverage,
        "accuracy": accuracy
    })
