import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import ph_simple_1, ph_simple_3
from fft_project.decision_class import FFT

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 avoids the worst fractal values.
    #Tests that the function returns False if g1 contains a worst fractal value.
    #Tests that g2 values do not affect the result.
    
    # Minimum gain of g1 is greater than minimum gain of g2
    assert ph_simple_1(50.0, 108.5, -407.0, 108.5, fractal_values=FRACTAL_VALUES, tol = 1) is True

    # Minimum gain of g1 is less than minimum gain of g2
    assert ph_simple_1(-407.0, 108.5, 50.0, 309.5, fractal_values=FRACTAL_VALUES, tol = 1) is False

    # Minimum gain of g1 is equal to minimum gain of g2
    assert ph_simple_1(440.5, 50.0, 50.0, 309.5, fractal_values=FRACTAL_VALUES, tol = 1) is False

    # Maximum gain of g1 is greater than maximum gain of g2
    assert ph_simple_3(440.5, -407.0, 50.0, 309.5) is True

    # Maximum gain of g2 is greater than maximum gain of g1
    assert ph_simple_3(309.5, -407.0, 440.5, 50.0) is False

    # Maximum gain of g1 is equal to maximum gain of g2
    assert ph_simple_3(-49.0, 309.5, 309.5, 50.0) is False

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue1 = Cue(
        id          = "ph-simple-1",
        name        = f"simplified ph cue 1",
        description = f"Checks if the difference in the rank of the minimum gains is greater than the threshold.",
        feature     = ph_simple_1,
        type        = "boolean",
        threshold   = 0,
        params      = {},
        required_args = ["gamma_left_up", "gamma_left_down",
                                 "gamma_right_up", "gamma_right_down", "fractal_values", "tol"]
        )

    cue2 = Cue(
            id          = "ph-simple-3",
            name        = f"simplified ph cue 3",
            description = f"returns the gamble with the highest maximum gain.",
            feature     = ph_simple_3,
            type        = "boolean",
            threshold   = 0,
            params      = {},
            required_args = ["gamma_left_up", "gamma_left_down",
                                     "gamma_right_up", "gamma_right_down"]
            )
        
    val, side = cue1.evaluate(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES, tol = 1)
    assert side == "left"

    val, side = cue1.evaluate(-407.0, 108.5, 50.0, 309.5, fractal_values=FRACTAL_VALUES, tol = 1)
    assert side == "right"

    val, side = cue1.evaluate(50.0, 108.5, 50.0, 440.5, fractal_values=FRACTAL_VALUES, tol = 1)
    assert side is None

    val, side = cue2.evaluate(50.0, 108.5, -407.0, -305.5)
    assert side == "left"

    val, side = cue2.evaluate(-407.0, 108.5, 50.0, 309.5)
    assert side == "right"

    val, side = cue2.evaluate(50.0, 108.5, -49.0, 108.5)
    assert side is None

    print("Cue.evaluate: all tests passed.")


def test_fft():
    #Tests that the FFT object returns the correct preference for a single gamble pair.
    cue1 = Cue.cue_registry["ph-simple-1"]
    cue2 = Cue.cue_registry["ph-simple-3"]
    fft = FFT(id="fft1",
              name="Simplified priority heuristic",
              description="An example FFT with simplified priority heuristic.",
              cues=[cue1, cue2])
    
    # Test fft when the minima differ by more than tol
    cue_values, side, i = fft.decide(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES, tol = 1)
    assert side == "left"
    assert i == 1

    # Test fft when minima differ by less than tol, but maxima differ
    cue_values, side, i = fft.decide(50.0, -407.0, 108.5, -305.5, fractal_values=FRACTAL_VALUES, tol = 3)
    assert side == "right"
    assert i == 2

    # Test when minima differ by less than the tol, and maxima are equal
    cue_values, side, i = fft.decide(108.5, -407.0, 108.5, -305.5, fractal_values=FRACTAL_VALUES, tol = 3)
    assert i == 3

    print("FFT.decide: all tests passed.")
    

if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    #test_fft()
    print("\nAll tests passed.")
