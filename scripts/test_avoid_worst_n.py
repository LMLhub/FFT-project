#!/usr/bin/env python3
"""
Test script for the avoid_worst_n_ranks cue (issue #25).

Tests the feature function directly and as a properly instantiated Cue object,
using the same Cue class and patterns as cue-test.py.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_n_ranks


# ---------------------------------------------------------------------------
# Unit tests for the feature function directly
# ---------------------------------------------------------------------------

def test_feature_function():
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

    print("feature function: all tests passed.")


# ---------------------------------------------------------------------------
# Tests for the Cue class with avoid_worst_n_ranks
# ---------------------------------------------------------------------------

def test_cue_evaluate():
    cue = Cue(
        id          = "avoid_worst_1",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the single worst fractal rank.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1},
        required_args = ["fractal_left_up", "fractal_left_down",
                         "fractal_right_up", "fractal_right_down"],
    )

    # Left avoids worst-1, right does not -> prefer left
    val, side = cue.evaluate(1, 2, 0, 3)
    assert side == "left",  f"Expected 'left', got {side!r}"

    # Right avoids worst-1, left does not -> prefer right
    val, side = cue.evaluate(0, 2, 1, 3)
    assert side == "right", f"Expected 'right', got {side!r}"

    # Both contain rank 0 -> undecided
    val, side = cue.evaluate(0, 2, 0, 3)
    assert side is None,    f"Expected None, got {side!r}"

    # Neither contains rank 0 -> undecided
    val, side = cue.evaluate(1, 2, 3, 4)
    assert side is None,    f"Expected None, got {side!r}"

    print("Cue.evaluate: all tests passed.")


def test_cue_evaluate_df():
    cue = Cue(
        id          = "avoid_worst_1_df",
        name        = "Avoid worst 1 rank (df)",
        description = "DataFrame version of the avoid-worst-1-rank cue.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1},
        required_args = ["fractal_left_up", "fractal_left_down",
                         "fractal_right_up", "fractal_right_down"],
    )

    df = pd.DataFrame({
        "fractal_left_up":    [1, 0, 0, 1],
        "fractal_left_down":  [2, 2, 2, 2],
        "fractal_right_up":   [0, 1, 0, 1],
        "fractal_right_down": [3, 3, 3, 3],
    })

    result = cue.evaluate_df(df)

    expected_sides = ["left", "right", None, None]
    actual_sides   = list(result["avoid_worst_1_df_side_if_true"])

    assert actual_sides == expected_sides, \
        f"Sides mismatch: {actual_sides}"

    print("Cue.evaluate_df: all tests passed.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_cue_evaluate_df()
    print("\nAll tests passed.")
