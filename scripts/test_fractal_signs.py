#!/usr/bin/env python3
#Test script for the fractal_signs cue (issue #23).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import fractal_signs

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]


def test_feature_function():
    #Tests that the function counts the number of positive fractal values in gamble 1.
    #Returns 2 if both fractal values are positive.
    #Returns 1 if only one fractal value is positive.
    #Returns 0 if both fractal values are negative.
    #Tests that g2 values do not affect the result.

    # g1 has two positive fractals (50.0, 108.5), so the function returns 2
    assert fractal_signs(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES) == 2
    # g1 has one positive (50.0) and one negative (-407.0), so the function returns 1
    assert fractal_signs(50.0, -407.0, -305.5, 108.5, fractal_values=FRACTAL_VALUES) == 1
    # g1 has two negative fractals (-407.0, -305.5), so the function returns 0
    assert fractal_signs(-407.0, -305.5, 50.0, 108.5, fractal_values=FRACTAL_VALUES) == 0
    # g2 has two negative fractals but g1 has two positive, so the function still returns 2
    assert fractal_signs(50.0, 108.5, -407.0, -407.0, fractal_values=FRACTAL_VALUES) == 2

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if left has more positive fractals than right.
    #Returns right if right has more positive fractals than left.
    #Returns None if both gambles have the same number of positive fractals.
    cue = Cue(
        id          = "fractal_signs",
        name        = "Fractal signs",
        description = "Prefers the gamble with more positive fractal values.",
        feature     = fractal_signs,
        type        = "numerical",
        threshold   = 0,
        params      = {"fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    # left gamble has two positive fractals (50.0, 108.5), so its score is 2
    # right gamble has two negative fractals (-407.0, -305.5), so its score is 0
    # left has a higher score, so the cue prefers left
    val, side = cue.evaluate(50.0, 108.5, -407.0, -305.5)
    assert side == "left"

    # left gamble has two negative fractals (-407.0, -305.5), so its score is 0
    # right gamble has two positive fractals (50.0, 108.5), so its score is 2
    # right has a higher score, so the cue prefers right
    val, side = cue.evaluate(-407.0, -305.5, 50.0, 108.5)
    assert side == "right"

    # left gamble has one positive (50.0) and one negative (-407.0), so its score is 1
    # right gamble has one positive (108.5) and one negative (-305.5), so its score is 1
    # both gambles have the same score, so the cue cannot decide
    val, side = cue.evaluate(50.0, -407.0, 108.5, -305.5)
    assert side is None

    # left gamble has two positive fractals (50.0, 108.5), so its score is 2
    # right gamble has one positive (309.5) and one negative (-407.0), so its score is 1
    # left has a higher score, so the cue prefers left
    val, side = cue.evaluate(50.0, 108.5, 309.5, -407.0)
    assert side == "left"

    print("Cue.evaluate: all tests passed.")


def test_cue_evaluate_df():
    #Tests that the Cue object returns the correct preference for each row in a DataFrame.
    #Adds two new columns to the DataFrame: the cue value and the preferred side.
    cue = Cue(
        id          = "fractal_signs_df",
        name        = "Fractal signs (df)",
        description = "Prefers the gamble with more positive fractal values.",
        feature     = fractal_signs,
        type        = "numerical",
        threshold   = 0,
        params      = {"fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    # Row 1: left has two positive fractals (50.0, 108.5), so its score is 2
    #         right has two negative fractals (-407.0, -305.5), so its score is 0
    #         left has a higher score, so the cue prefers left
    # Row 2: left has two negative fractals (-407.0, -305.5), so its score is 0
    #         right has two positive fractals (50.0, 108.5), so its score is 2
    #         right has a higher score, so the cue prefers right
    # Row 3: left has one positive (50.0) and one negative (-407.0), so its score is 1
    #         right has one positive (108.5) and one negative (-305.5), so its score is 1
    #         both gambles have the same score, so the cue cannot decide
    # Row 4: left has two positive fractals (50.0, 108.5), so its score is 2
    #         right has one positive (309.5) and one negative (-407.0), so its score is 1
    #         left has a higher score, so the cue prefers left
    df = pd.DataFrame({
        "gamma_left_up":    [50.0,   -407.0,  50.0,   50.0],
        "gamma_left_down":  [108.5,  -305.5, -407.0,  108.5],
        "gamma_right_up":   [-407.0,  50.0,   108.5,  309.5],
        "gamma_right_down": [-305.5,  108.5, -305.5, -407.0],
    })

    result = cue.evaluate_df(df)

    expected_sides = ["left", "right", None, "left"]
    actual_sides   = list(result["fractal_signs_df_side_if_true"])

    for i, (actual, expected) in enumerate(zip(actual_sides, expected_sides)):
        if expected is None:
            assert pd.isna(actual) or actual is None, f"Row {i}: expected None, got {actual!r}"
        else:
            assert actual == expected, f"Row {i}: expected {expected!r}, got {actual!r}"

    print("Cue.evaluate_df: all tests passed.")


def test_invalid_fractal_value():
    #Tests that a ValueError is raised if a gamble value is not in fractal_values.
    try:
        fractal_signs(999.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES)
        assert False, "Should have raised ValueError"
    except ValueError:
        pass
    print("invalid fractal value: correctly raised ValueError.")


if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_cue_evaluate_df()
    test_invalid_fractal_value()
    print("\nAll tests passed.")
