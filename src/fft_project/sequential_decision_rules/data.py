from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    predictions: np.ndarray
    labels: np.ndarray
    feature_names: list[str]
