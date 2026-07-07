#!/usr/bin/env python3
#Test script for the avoid_worst_n_ranks cue (issue #25).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import minimum_gains

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 avoids the worst fractal values.
    #Tests that the function returns False if g1 contains a worst fractal value.
    #Tests that g2 values do not affect the result.

    assert minimum_gains(1001, 0, 100, 100, tol = 0.1) == False
    assert minimum_gains(999, 0, 100, 100, tol = 0.1) == False
    assert minimum_gains(100, 100, 999, 0, tol=0.1 ) == True
    assert minimum_gains(100, 100, 1001, 0, tol=0.1 ) == False
    assert minimum_gains(-400, 1002, 999, 0, tol=0.1 ) == False
    assert minimum_gains(999, 0, -400, 1002, tol=0.1 ) == True
    assert minimum_gains(-400, -1, -300, -1, tol=0.1 ) == False
    assert minimum_gains(-300, -1, -400, -1, tol=0.1 ) == True
    
    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue = Cue(
        id          = "min_01",
        name        = "Minimum gains difference 10 percent",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 10 percent of the maximum gain",
        feature     = minimum_gains,
        type        = "boolean",
        params      = {"tol": 0.1},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    val, side = cue.evaluate(1001, 0, 100, 100)
    assert side is None

    val, side = cue.evaluate(100, 100, 1001, 0)
    assert side is None

    val, side = cue.evaluate(100, 100, 999, 0)
    assert side == "left"

    val, side = cue.evaluate(999, 0, 100, 100)
    assert side == "right"

    val, side = cue.evaluate(-10000, 1002, 999, 0)
    assert side == "right"

    val, side = cue.evaluate(-400, -300, -400, -350)
    assert side is None

    val, side = cue.evaluate(-1002, 0, -100, -100)
    assert side is "right" #correct assert since -1002 is a too large loss to accept

    val, side = cue.evaluate(-999, 0, -100, -100)
    assert side is None # Because the largest loss is not large enough to decide.

    print("Cue.evaluate: all tests passed.")




if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    print("\nAll tests passed.")
