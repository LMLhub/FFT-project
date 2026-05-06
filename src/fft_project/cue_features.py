#This file contains code implenting cues as instances of the Cue class.
import numpy as np

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

def avoid_worst_n_ranks(g1_up, g1_down, g2_up, g2_down, n, fractal_values):
    #Checks if gamble 1 contains any of the n worst fractal values.
    #The n worst fractal values are the n smallest values in fractal_values.
    #Returns True if neither g1_up nor g1_down is among the n worst fractal values.
    #Returns False if g1_up or g1_down is among the n worst fractal values.
    #g2_up and g2_down are not used here, but the Cue class always passes both gambles.
    worst_values = sorted(fractal_values)[:n]
    if g1_up in worst_values or g1_down in worst_values:
        return False
    else:
        return True

