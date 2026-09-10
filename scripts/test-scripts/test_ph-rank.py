import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import ph_rank_1, ph_rank_3
from fft_project.decision_class import FFT

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 avoids the worst fractal values.
    #Tests that the function returns False if g1 contains a worst fractal value.
    #Tests that g2 values do not affect the result.
    
    assert priority_step1(1001, 0, 100, 100, tol = 0.1, dynamic="additive") == False
    assert priority_step1(999, 0, 100, 100, tol = 0.1, dynamic="additive") == False
    assert priority_step1(100, 100, 999, 0, tol=0.1, dynamic="additive" ) == True
    assert priority_step1(100, 100, 1001, 0, tol=0.1, dynamic="additive" ) == False
    assert priority_step1(-400, 1002, 999, 0, tol=0.1, dynamic="additive" ) == False
    assert priority_step1(999, 0, -400, 1002, tol=0.1, dynamic="additive" ) == True
    assert priority_step1(-400, -1, -300, -1, tol=0.1, dynamic="additive" ) == False
    assert priority_step1(-300, -1, -400, -1, tol=0.1, dynamic="additive" ) == False
    assert priority_step1(-1001, 0, -100, -100, tol = 0.1, dynamic="additive") == False
    assert priority_step1(-100, -100, -1001, 0, tol = 0.1, dynamic="additive") == False
    assert priority_step1(-999, 0, -100, -100, tol = 0.1, dynamic="additive") == True #correct?
    assert priority_step1(-100, -100, -999, 0, tol=0.1, dynamic="additive" ) == False
    assert priority_step1(-100, -100, -1001, 0, tol=0.1, dynamic="additive" ) == False

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue1 = Cue(
        id          = "ph-rank-1",
        name        = f"Rank-based ph cue 1",
        description = f"Checks if the difference in the rank of the minimum gains is greater than the threshold, but unlike the simplified version, it switches sign if the gambles are mainly loosing.",
        feature     = ph_rank_1,
        type        = "boolean",
        threshold   = 0,
        params      = {},
        required_args = ["gamma_left_up", "gamma_left_down",
                                 "gamma_right_up", "gamma_right_down", "fractal_values", "tol"]
        )

    cue2 = Cue(
            id          = "ph-rank-3",
            name        = f"simplified ph cue 3",
            description = f"returns the gamble with the highest maximum gain.",
            feature     = ph_rank_3,
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
    cue1 = Cue.cue_registry["ph-rank-1"]
    cue2 = Cue.cue_registry["ph-rank-3"]
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
