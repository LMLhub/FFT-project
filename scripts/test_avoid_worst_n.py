#!/usr/bin/env python3
#Test script for the avoid_worst_n_ranks cue (issue #25).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_n_ranks

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 avoids the worst fractal values.
    #Tests that the function returns False if g1 contains a worst fractal value.
    #Tests that g2 values do not affect the result.

    # n=1: only -407.0 is the worst
    assert avoid_worst_n_ranks(50.0, 108.5, -407.0, -305.5, n=1, fractal_values=FRACTAL_VALUES) == True
    assert avoid_worst_n_ranks(-407.0, 108.5, 50.0, 309.5, n=1, fractal_values=FRACTAL_VALUES) == False
    assert avoid_worst_n_ranks(-407.0, -305.5, 50.0, 309.5, n=1, fractal_values=FRACTAL_VALUES) == False

    # n=3: -407.0, -305.5 and -241.5 are the worst
    assert avoid_worst_n_ranks(50.0, 108.5, -407.0, -305.5, n=3, fractal_values=FRACTAL_VALUES) == True
    assert avoid_worst_n_ranks(-241.5, 108.5, 50.0, 309.5, n=3, fractal_values=FRACTAL_VALUES) == False

    # g2 values should not affect the result
    assert avoid_worst_n_ranks(50.0, 108.5, -407.0, -407.0, n=3, fractal_values=FRACTAL_VALUES) == True

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue = Cue(
        id          = "avoid_worst_1",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    val, side = cue.evaluate(50.0, 108.5, -407.0, -305.5)
    assert side == "left"

    val, side = cue.evaluate(-407.0, 108.5, 50.0, 309.5)
    assert side == "right"

    val, side = cue.evaluate(-407.0, 108.5, -407.0, 309.5)
    assert side is None

    val, side = cue.evaluate(50.0, 108.5, 309.5, 440.5)
    assert side is None

    print("Cue.evaluate: all tests passed.")


def test_cue_evaluate_df():
    #Tests that the Cue object returns the correct preference for each row in a DataFrame.
    #Adds two new columns to the DataFrame: the cue value and the preferred side.
    cue = Cue(
        id          = "avoid_worst_1_df",
        name        = "Avoid worst 1 rank (df)",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    df = pd.DataFrame({
        "gamma_left_up":    [50.0,   -407.0, -407.0, 50.0],
        "gamma_left_down":  [108.5,   108.5,  108.5, 108.5],
        "gamma_right_up":   [-407.0,   50.0, -407.0,  50.0],
        "gamma_right_down": [309.5,   309.5,  309.5, 309.5],
    })

    result = cue.evaluate_df(df)

    expected_sides = ["left", "right", None, None]
    actual_sides   = list(result["avoid_worst_1_df_side_if_true"])

    for i, (actual, expected) in enumerate(zip(actual_sides, expected_sides)):
        if expected is None:
            assert pd.isna(actual) or actual is None, f"Row {i}: expected None, got {actual!r}"
        else:
            assert actual == expected, f"Row {i}: expected {expected!r}, got {actual!r}"

    print("Cue.evaluate_df: all tests passed.")


if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_cue_evaluate_df()
    print("\nAll tests passed.")
