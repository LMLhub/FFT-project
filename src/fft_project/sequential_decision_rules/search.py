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

def extend_state(
    state: SearchState,
    feature: int,
    dataset: Dataset
) -> SearchState:
    """
    Extend a search state by applying a decision rule.

    The rule is applied only to instances that remain
    unclassified. The original state is not modified.
    """

    if feature < 0 or feature >= dataset.M:
        raise ValueError("Invalid feature index")

    if feature in state.sequence:
        raise ValueError("Feature already used in sequence")

    predictions = dataset.X[:, feature]

    # Instances classified by this feature
    classified = state.remaining & (predictions != 0)

    # Count correct and incorrect classifications
    n_classified = np.count_nonzero(classified)

    n_correct = np.count_nonzero(
        classified & (predictions == dataset.Y)
    )

    n_incorrect = n_classified - n_correct

    # Update residual dataset
    new_remaining = state.remaining & (predictions == 0)

    return SearchState(
        sequence=state.sequence + (feature,),
        remaining=new_remaining,
        correct=state.correct + n_correct,
        incorrect=state.incorrect + n_incorrect,
    )
