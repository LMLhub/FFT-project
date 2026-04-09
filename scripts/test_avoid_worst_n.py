#!/usr/bin/env python3
"""
Test script for the avoid_worst_n_ranks feature function.

Runs a set of hand-checked cases to verify that the signed cue and
DataFrame evaluation behave as expected.
"""
import sys
from pathlib import Path

# Allow running from the scripts/ directory without installing the package
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fft_project.feature_avoid_worst_n import avoid_worst_n_ranks, signed_cue, evaluate_df


# ---------------------------------------------------------------------------
# Unit tests for avoid_worst_n_ranks (1-sided feature)
# ---------------------------------------------------------------------------

def test_avoid_worst_n_ranks():
    # With n=1, only rank 0 is "worst"
    assert avoid_worst_n_ranks(1, 2, 3, 4, n=1) == True,  "g1=(1,2) avoids rank 0"
    assert avoid_worst_n_ranks(0, 2, 3, 4, n=1) == False, "g1=(0,2) contains rank 0"
    assert avoid_worst_n_ranks(0, 0, 3, 4, n=1) == False, "g1=(0,0) contains rank 0"

    # With n=3, ranks 0,1,2 are "worst"
    assert avoid_worst_n_ranks(3, 4, 5, 6, n=3) == True,  "g1=(3,4) avoids ranks 0-2"
    assert avoid_worst_n_ranks(2, 4, 5, 6, n=3) == False, "g1=(2,4) contains rank 2"
    assert avoid_worst_n_ranks(1, 3, 5, 6, n=3) == False, "g1=(1,3) contains rank 1"

    # g2 values are ignored
    assert avoid_worst_n_ranks(4, 5, 0, 0, n=3) == True,  "g2 having worst ranks is irrelevant"

    print("avoid_worst_n_ranks: all tests passed.")


# ---------------------------------------------------------------------------
# Unit tests for signed_cue
# ---------------------------------------------------------------------------

def test_signed_cue():
    # Left avoids worst-1, right does not → F = 1
    assert signed_cue(1, 2, 0, 3, n=1) == 1,  "left clear, right has rank 0 → prefer left"

    # Right avoids worst-1, left does not → F = -1
    assert signed_cue(0, 2, 1, 3, n=1) == -1, "left has rank 0, right clear → prefer right"

    # Both contain a worst-1 fractal → F = 0 (undecided)
    assert signed_cue(0, 2, 0, 3, n=1) == 0,  "both contain rank 0 → undecided"

    # Neither contains a worst-1 fractal → F = 0 (undecided)
    assert signed_cue(1, 2, 3, 4, n=1) == 0,  "neither contains rank 0 → undecided"

    print("signed_cue: all tests passed.")


# ---------------------------------------------------------------------------
# Tests for evaluate_df
# ---------------------------------------------------------------------------

def test_evaluate_df():
    df = pd.DataFrame({
        'fractal_left_up':    [1, 0, 0, 1],
        'fractal_left_down':  [2, 2, 2, 2],
        'fractal_right_up':   [0, 1, 0, 1],
        'fractal_right_down': [3, 3, 3, 3],
    })

    result = evaluate_df(df, n=1)

    expected_values = [1, -1, 0, 0]
    expected_sides  = ['left', 'right', None, None]

    assert list(result['avoid_worst_1_value'])       == expected_values, \
        f"Values mismatch: {list(result['avoid_worst_1_value'])}"
    assert list(result['avoid_worst_1_side_if_true']) == expected_sides,  \
        f"Sides mismatch: {list(result['avoid_worst_1_side_if_true'])}"

    # Original df should not be modified
    assert 'avoid_worst_1_value' not in df.columns, "evaluate_df should not modify the input df"

    print("evaluate_df: all tests passed.")


def test_evaluate_df_missing_column():
    df = pd.DataFrame({'fractal_left_up': [1], 'fractal_left_down': [2]})
    try:
        evaluate_df(df, n=1)
        assert False, "Should have raised ValueError for missing columns"
    except ValueError:
        pass
    print("evaluate_df missing column: correctly raised ValueError.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_avoid_worst_n_ranks()
    test_signed_cue()
    test_evaluate_df()
    test_evaluate_df_missing_column()
    print("\nAll tests passed.")
