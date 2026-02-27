#This file contains code implenting cues as instances of the Cue class.
import numpy as np
import pandas as pd 
from fft_project.cue_class import Cue

def growth_rate(g1_up, g1_down, g2_up, g2_down, wealth, dynamic):
    #This function calculates the growth rate of the g1 given the dynamic.
    if dynamic == "multiplicative":
        return (np.log(g1_up*wealth) + np.log(g1_down*wealth))/2
    elif dynamic == "additive":
        return (g1_up + g1_down)/2
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")


