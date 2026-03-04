#This file contains code implenting cues as instances of the Cue class.
import numpy as np
import pandas as pd 
from fft_project.cue_class import Cue

def growth_rate(g1_up, g1_down, g2_up, g2_down, dynamic):
    #This function calculates the growth rate of the g1 given the dynamic.
    if dynamic == "multiplicative":
        return (np.log(g1_up) + np.log(g1_down))/2
    elif dynamic == "additive":
        return (g1_up + g1_down)/2
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")

def expected_isoelastic_utility(g1_up, g1_down, g2_up, g2_down, wealth, dynamic, eta):
    #This function calculates the expected isoelastic utility of the g1 given the eta parameter.
    if dynamic == "multiplicative":
        x11 = g1_up*wealth
        x12 = g1_down*wealth
    elif dynamic == "additive":
        x11 = g1_up + wealth
        x12 = g1_down + wealth
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")
    if eta == 1:
        return (np.log(x11) + np.log(x12))/2
    elif eta != 1:
        return (np.power(x11, 1-eta) + np.power(x12, 1-eta))/(2*(1-eta))

