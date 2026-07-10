#!/usr/bin/env python3
#Test script for the avoid_worst_n_ranks cue (issue #25).
#Tests the feature function directly and as a Cue object.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import numpy as np
from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.cue_features import priority_step1, priority_step3

def test_feature_function():
    high_high = np.log(2.1) #corresponds to a gain just above the threshold
    high_low = np.log(1.9) #corresponds to a gain just below the threshold
    low = np.log(11/10)

    loss_high_high = np.log(9.1/10) #corresponds to a loss just above the threshold
    loss_high_low = np.log(8.9/10) #corresponds to a loss just below the threshold
    loss_low = np.log(99/100)

    assert priority_step1(high_high, 0, low, low, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3(high_high, 0, low, low, dynamic="multiplicative") == True
    
    assert priority_step1(high_low, 0, low, low, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3(high_low, 0, low, low, dynamic="multiplicative") == True

    assert priority_step1(low, low, high_low, 0, tol = 0.1, dynamic="multiplicative") == True
    assert priority_step3(low, low, high_low, 0, dynamic="multiplicative") == False
    
    assert priority_step1( low, low, high_high, 0, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3( low, low, high_high, 0, dynamic="multiplicative") == False

    assert priority_step1(loss_high_low, 0, loss_low, loss_low, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3(loss_high_low, 0, loss_low, loss_low, dynamic="multiplicative") == False

    assert priority_step1(loss_high_high, 0, loss_low, loss_low, tol = 0.1, dynamic="multiplicative") == True
    assert priority_step3(loss_high_high, 0, loss_low, loss_low, dynamic="multiplicative") == False

    assert priority_step1( loss_low, loss_low, loss_high_high, 0, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3( loss_low, loss_low, loss_high_high, 0, dynamic="multiplicative") == True

    assert priority_step1( loss_low, loss_low, loss_high_low, 0, tol = 0.1, dynamic="multiplicative") == False
    assert priority_step3( loss_low, loss_low, loss_high_low, 0, dynamic="multiplicative") == True
    
    print("feature function: all tests passed.")


def test_cue_evaluate():
    #Tests that the Cue object returns the correct preference for a single gamble pair.
    #Returns left if only the left gamble avoids the worst fractal values.
    #Returns right if only the right gamble avoids the worst fractal values.
    #Returns None if both or neither gamble contains a worst fractal value.
    cue = Cue(
        id          = "pri_1_01_m",
        name        = "Minimum gains difference 10 percent (multiplicative)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 10 percent of the maximum gain",
        feature     = priority_step1,
        type        = "boolean",
        params      = {"tol": 0.1, "dynamic": "multiplicative"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    cue2 = Cue(
        id          = "pri_3_m",
        name        = "Maximum gains (multiplicative)",
        description = "Give preference to the gamble with the highest maximum gains.",
        feature     = priority_step3,
        type        = "boolean",
        params      = {"dynamic": "multiplicative"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    high_high = np.log(2.1) #corresponds to a gain just above the threshold
    high_low = np.log(1.9) #corresponds to a gain just below the threshold
    low = np.log(11/10)

    loss_high_low = np.log(9.1/10) #corresponds to a loss just above the threshold
    loss_high_high = np.log(8.9/10) #corresponds to a loss just below the threshold
    loss_low = np.log(99/100)

    val, side = cue.evaluate(high_high, 0, low, low)
    assert side is None

    val, side = cue.evaluate(low, low, high_high, 0)
    assert side is None

    val, side = cue.evaluate(low, low, high_low, 0)
    assert side == "left"

    val, side = cue.evaluate(high_low, 0, low, low)
    assert side == "right"

    val, side = cue.evaluate(loss_high_high, 0, loss_low, loss_low)
    assert side is None 

    val, side = cue.evaluate(loss_high_low, 0, loss_low, loss_low)
    assert side == "left" 

    print("step 1: Cue.evaluate: all tests passed.")

    val, side = cue2.evaluate(high_high, 0, high_low, high_low)
    assert side == "left"

    val, side = cue2.evaluate(low, low, high_high, 0)
    assert side == "right"

    val, side = cue2.evaluate(low, low, high_low, 0)
    assert side == "right"

    val, side = cue2.evaluate(high_low, 0, low, low)
    assert side == "left"

    val, side = cue2.evaluate(loss_high_high, 0, loss_low, loss_low)
    assert side == "right"

    val, side = cue2.evaluate(loss_high_low, 0, loss_low, loss_low)
    assert side == "right"

    print("step 3: Cue.evaluate: all tests passed.")

def test_priority_heuristic():
    high_high = np.log(2.1) #corresponds to a gain just above the threshold
    high_low = np.log(1.9) #corresponds to a gain just below the threshold
    low = np.log(11/10)

    loss_high_low = np.log(9.1/10) #corresponds to a loss just above the threshold
    loss_high_high = np.log(8.9/10) #corresponds to a loss just below the threshold
    loss_low = np.log(99/100)

    cue1 = Cue.cue_registry["pri_1_01_m"]
    cue2 = Cue.cue_registry["pri_3_m"]

    fft = FFT(id="fft1",
            name="Priority heuristic",
            description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006).",
            cues=[cue1, cue2])
    
    cue_values, side, i = fft.decide(low, low, 0, high_low, wealth=1000)
    assert side == "left" 
    assert i == 1     

    cue_values, side, i = fft.decide(low, low, 0, high_high, wealth=1000)
    assert side == "right"     
    assert i == 2

    cue_values, side, i = fft.decide(loss_low, loss_low, 0, loss_high_high, wealth=2000)
    assert side == "left"     
    assert i == 2

    cue_values, side, i = fft.decide(loss_low, loss_low, 0, loss_high_low, wealth=2000)
    assert i == 1
    assert side == "right" 

    print("FFT.decide: all tests passed.")


if __name__ == "__main__":
    test_feature_function()
    test_cue_evaluate()
    test_priority_heuristic()
    print("\nAll tests passed.")
