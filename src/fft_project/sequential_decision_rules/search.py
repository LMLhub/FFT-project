from dataclasses import dataclass
import numpy as np
from fft_project.sequential_decision_rules.data import Dataset

@dataclass
class SearchState:
    sequence: tuple[int, ...]
    remaining: np.ndarray
    correct: int = 0
    incorrect: int = 0


def validate_state(state: SearchState, dataset: Dataset):
    assert state.remaining.shape == (dataset.N,), \
        "Remaining mask has incorrect shape"

    assert state.remaining.dtype == np.bool_, \
        "Remaining mask must be Boolean"

    assert (
        state.correct
        + state.incorrect
        + np.count_nonzero(state.remaining)
        == dataset.N
    ), "Classification counts do not sum to N"
