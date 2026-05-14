#!/usr/bin/env python3
#Test script for the avoid_worst_n_ranks cue (issue #25).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 does not contain the worst fractal value.
    #Tests that the function returns False if g1 contains the worst fractal value.
    #Tests that g2 values affect the result.

    assert avoid_worst(50.0, 108.5, -407.0, -305.5) == True
    assert avoid_worst(-407.0, 108.5, 50.0, 309.5) == False
    assert avoid_worst(-407.0, -305.5, -407.0, 309.5) == False

    # g2 values should affect the result
    assert avoid_worst(50.0, 108.5, -407.0, -407.0) == True
    assert avoid_worst(50.0, 108.5, 440.5, 108.5) == False


    print("feature function: all tests passed.")


def test_cue_evaluate(cue):
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal value.
    #Returns right if only the right gamble avoids the worst fractal value.
    #Returns None if both gambles contain the worst fractal value.


    val, side = cue.evaluate(50.0, 108.5, -407.0, -305.5)
    assert side == "left"

    val, side = cue.evaluate(-407.0, 108.5, 50.0, 309.5)
    assert side == "right"

    val, side = cue.evaluate(-407.0, 108.5, -407.0, 309.5)
    assert side is None

    val, side = cue.evaluate(50.0, 108.5, 309.5, 440.5)
    assert side is "right"

    print("Cue.evaluate: all tests passed.")


def test_cue_evaluate_df(cue):
    #Tests that the Cue object returns the correct preference for each row in a DataFrame.
    #Adds two new columns to the DataFrame: the cue value and the preferred side.

    df = pd.DataFrame({
        "gamma_left_up":    [50.0,   -407.0, -407.0, 50.0],
        "gamma_left_down":  [108.5,   108.5,  108.5, 108.5],
        "gamma_right_up":   [-407.0,   50.0, -407.0,  50.0],
        "gamma_right_down": [309.5,   309.5,  309.5, 309.5],
    })

    result = cue.evaluate_df(df)

    expected_sides = ["left", "right", None, None]
    actual_sides   = list(result["avoid_worst_side_if_true"])

    for i, (actual, expected) in enumerate(zip(actual_sides, expected_sides)):
        if expected is None:
            assert pd.isna(actual) or actual is None, f"Row {i}: expected None, got {actual!r}"
        else:
            assert actual == expected, f"Row {i}: expected {expected!r}, got {actual!r}"

    print("Cue.evaluate_df: all tests passed.")




if __name__ == "__main__":
    cue = Cue(
        id          = "avoid_worst",
        name        = "Avoid worst",
        description = "Prefers the gamble that does not contain the worst of the 4 fractal values in the presented gamble pair.",
        feature     = avoid_worst,
        type        = "boolean",
        params      = {},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )
    test_feature_function()
    test_cue_evaluate(cue)
    test_cue_evaluate_df(cue)
    print("\nAll tests passed.")
