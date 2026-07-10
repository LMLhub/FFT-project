#!/usr/bin/env python3
#Test script for the avoid_worst_n_ranks cue (issue #25).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.cue_features import priority_step1, priority_step3

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

def test_feature_function():

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

    assert priority_step3(1001, 0, 100, 100, dynamic="additive") == True
    assert priority_step3(999, 0, 100, 100, dynamic="additive") == True
    assert priority_step3(100, 100, 999, 0, dynamic="additive") == False
    assert priority_step3(100, 100, 1001, 0, dynamic="additive") == False
    assert priority_step3(-400, 1002, 999, 0, dynamic="additive") == True
    assert priority_step3(999, 0, -400, 1002, dynamic="additive") == False
    assert priority_step3(-400, -1, -300, -1, dynamic="additive") == False
    assert priority_step3(-300, -1, -400, -1, dynamic="additive") == True
    assert priority_step3(-1001, 0, -100, -100, dynamic="additive") == False
    assert priority_step3(-999, 0, -100, -100, dynamic="additive") == False
    assert priority_step3(-100, -100, -999, 0, dynamic="additive") == True
    assert priority_step3(-100, -100, -1001, 0, dynamic="additive") == True

    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue = Cue(
        id          = "pri_1_01_a",
        name        = "Minimum gains difference 10 percent (additive)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 10 percent of the maximum gain",
        feature     = priority_step1,
        type        = "boolean",
        params      = {"tol": 0.1, "dynamic": "additive"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    cue2 = Cue(
        id          = "pri_3_a",
        name        = "Maximum gains (additive)",
        description = "Give preference to the gamble with the highest maximum gains.",
        feature     = priority_step3,
        type        = "boolean",
        params      = {"dynamic": "additive"},
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
    assert side is None

    val, side = cue.evaluate(-400, -300, -400, -350)
    assert side == "left"

    val, side = cue.evaluate(-1002, 0, -100, -100)
    assert side is None 

    val, side = cue.evaluate(-999, 0, -100, -100)
    assert side == "left" 

    print("step 1: Cue.evaluate: all tests passed.")

    val, side = cue2.evaluate(1001, 0, 100, 100)
    assert side == "left"

    val, side = cue2.evaluate(100, 100, 1001, 0)
    assert side == "right"

    val, side = cue2.evaluate(100, 100, 999, 0)
    assert side == "right"

    val, side = cue2.evaluate(999, 0, 100, 100)
    assert side == "left"

    val, side = cue2.evaluate(-10000, 1002, 999, 0)
    assert side == "right"

    val, side = cue2.evaluate(-400, -300, -400, -350)
    assert side is None

    val, side = cue2.evaluate(-1002, 0, -100, -100)
    assert side == "right" 

    val, side = cue2.evaluate(-999, 0, -100, -100)
    assert side == "right"


    print("step 3: Cue.evaluate: all tests passed.")

def test_priority_heuristic():
    cue1 = Cue.cue_registry["pri_1_01_a"]
    cue2 = Cue.cue_registry["pri_3_a"]

    fft = FFT(id="fft1",
            name="Priority heuristic",
            description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006).",
            cues=[cue1, cue2])
    
    cue_values, side, i = fft.decide(100, 100, 0, 999, wealth=1000)
    assert side == "left" 
    assert i == 1     

    cue_values, side, i = fft.decide(100, 100, 0, 1001, wealth=1000)
    assert side == "right"     
    assert i == 2

    cue_values, side, i = fft.decide(-100, -100, 0, -1001, wealth=2000)
    assert side == "left"     
    assert i == 2

    cue_values, side, i = fft.decide(-100, -100, 0, -999, wealth=2000)
    assert i == 1
    assert side == "right" 

    print("FFT.decide: all tests passed.")


if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_priority_heuristic()
    print("\nAll tests passed.")
