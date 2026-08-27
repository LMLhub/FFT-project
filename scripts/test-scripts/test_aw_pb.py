import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_or_prefer_best
from fft_project.decision_class import FFT

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():
    #Tests that the function returns True if g1 avoids the worst fractal values.
    #Tests that the function returns False if g1 contains a worst fractal value.
    #Tests that g2 values do not affect the result.
    
    # no worst or best fractal values present in g1
    assert avoid_worst_or_prefer_best(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES) == 0

    # g1 contains the worst fractal value
    assert avoid_worst_or_prefer_best(-407.0, 108.5, 50.0, 309.5, fractal_values=FRACTAL_VALUES) == -1

    # g1 contains the best fractal value
    assert avoid_worst_or_prefer_best(440.5, -305.5, 50.0, 309.5, fractal_values=FRACTAL_VALUES) == 1

    # g1 contains both the worst and best fractal values
    assert avoid_worst_or_prefer_best(440.5, -407.0, 50.0, 309.5, fractal_values=FRACTAL_VALUES) == 0

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue = Cue(
        id          = "AW-PB",
        name        = f"Avoid worst or prefer best",
        description = f"Checks whether the best or the worst fractal is present. Choses only if the best is present but not the worst or vice versa",
        feature     = avoid_worst_or_prefer_best,
        type        = "boolean",
        threshold   = 0,
        params      = {},
        required_args = ["gamma_left_up", "gamma_left_down",
                                 "gamma_right_up", "gamma_right_down", "fractal_values"]
        )
    
    val, side = cue.evaluate(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES)
    assert side == "left"

    val, side = cue.evaluate(-407.0, 108.5, 50.0, 309.5, fractal_values=FRACTAL_VALUES)
    assert side == "right"

    val, side = cue.evaluate(50.0, 108.5, 309.5, 440.5, fractal_values=FRACTAL_VALUES)
    assert side == "right"

    val, side = cue.evaluate(50.0, 440.5, 108.5, 309.5, fractal_values=FRACTAL_VALUES)
    assert side == "left"

    val, side = cue.evaluate(-407.0, 440.5, 108.5, 309.5, fractal_values=FRACTAL_VALUES)
    assert side is None
    
    val, side = cue.evaluate(108.5, 309.5,-407.0, 440.5, fractal_values=FRACTAL_VALUES)
    assert side is None

    val, side = cue.evaluate( -241.5, 108.5, -49.0, 50.0, fractal_values=FRACTAL_VALUES)
    assert side is None

    print("Cue.evaluate: all tests passed.")


def test_fft():
    #Tests that the FFT object returns the correct preference for a single gamble pair.
    cue = Cue.cue_registry["AW-PB"]
    fft = FFT(id="fft1",
              name="Avoid worst or prefer best",
              description="An example FFT with the avoid worst or prefer best cue.",
              cues=[cue])
    
    # Test fft when worst fractal value is present in right gamble
    cue_values, side, i = fft.decide(50.0, 108.5, -407.0, -305.5, fractal_values=FRACTAL_VALUES)
    assert side == "left"
    assert i == 1

    # Test fft when worst fractal value is present in left gamble
    cue_values, side, i = fft.decide(50.0, -407.0, 108.5, -305.5, fractal_values=FRACTAL_VALUES)
    assert side == "right"
    assert i == 1

    # Test fft when best fractal value is present in right gamble
    cue_values, side, i = fft.decide(50.0, 108.5, 440.5, -305.5, fractal_values=FRACTAL_VALUES)
    assert side == "right"
    assert i == 1

    # Test fft when worst fractal value is present in left gamble
    cue_values, side, i = fft.decide(50.0, 440.5, 108.5, -305.5, fractal_values=FRACTAL_VALUES)
    assert side == "left"
    assert i == 1

    # Test fft when worst and best fractal values are present
    cue_values, side, i = fft.decide( 440.5, -407.0, 108.5, -305.5, fractal_values=FRACTAL_VALUES)
    assert i == 2

    # Test fft when neither worst nor best fractal values are present
    cue_values, side, i = fft.decide( 50.0, 108.5, 108.5, -305.5, fractal_values=FRACTAL_VALUES)
    assert i == 2

    # Test fft when best and worst is on opposite sides
    cue_values, side, i = fft.decide( -305.5, 440.5, -407.0, 50.0, fractal_values=FRACTAL_VALUES)
    assert side == "left"
    assert i == 1

    print("FFT.decide: all tests passed.")
    

if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_fft()
    #test_invalid_n()
    #test_invalid_fractal_value()
    print("\nAll tests passed.")
