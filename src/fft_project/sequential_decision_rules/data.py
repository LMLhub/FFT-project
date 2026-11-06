from dataclasses import dataclass
import numpy as np

@dataclass
class Dataset:
    X: np.ndarray
    Y: np.ndarray

    def __post_init__(self):
        assert self.X.ndim == 2
        assert self.Y.ndim == 1
        assert self.X.shape[0] == self.Y.shape[0]

        assert np.isin(self.X, [-1, 0, 1]).all()
        assert np.isin(self.Y, [-1, 1]).all()

    @property
    def N(self):
        return self.X.shape[0]

    @property
    def M(self):
        return self.X.shape[1]
