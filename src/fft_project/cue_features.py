#This file contains code implenting cues as instances of the Cue class.
import numpy as np
import logging
logger = logging.getLogger(__name__)

def growth_rate(g1_up, g1_down, g2_up, g2_down):
    #This function calculates the growth rate of the g1 given the dynamic.
    # Fractal values g must be gamma values, such that the growth rate is 
    # the expected value of the fractal values for both dynamics.
    return (g1_up + g1_down)/2

def expected_isoelastic_utility(g1_up, g1_down, g2_up, g2_down, wealth, dynamic, eta):
    #This function calculates the expected isoelastic utility of the g1 given the eta parameter.
    if dynamic == "multiplicative":
        x11 = np.exp(g1_up) * wealth
        x12 = np.exp(g1_down) * wealth
    elif dynamic == "additive":
        x11 = g1_up + wealth
        x12 = g1_down + wealth
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")
    tol = 10**(-15)
    if abs(eta - 1) < tol:
        return (np.log(x11) + np.log(x12))/2
    if abs(eta) <= tol:
        return (x11 + x12) / 2

    elif eta != 1:
        return (np.power(x11, 1-eta) + np.power(x12, 1-eta))/(2*(1-eta))

def avoid_worst_n_ranks(g1_up, g1_down, g2_up, g2_down, n, fractal_values):
    #Checks if gamble 1 contains any of the n worst fractal values.
    #The n worst fractal values are the n smallest values in fractal_values.
    #Returns True if neither g1_up nor g1_down is among the n worst fractal values.
    #Returns False if g1_up or g1_down is among the n worst fractal values.
    #g2_up and g2_down are not used here, but the Cue class always passes both gambles.
    if n >= len(fractal_values):
        logger.error(f"n ({n}) must be smaller than the number of fractals ({len(fractal_values)}).")
        raise ValueError(f"n ({n}) must be smaller than the number of fractals ({len(fractal_values)}).")
    if g1_up not in fractal_values or g1_down not in fractal_values:
        logger.error(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
        raise ValueError(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
    worst_values = sorted(fractal_values)[:n]
    if g1_up in worst_values or g1_down in worst_values:
        return False
    else:
        return True

def fractal_signs(g1_up, g1_down, g2_up, g2_down, fractal_values):
    #Counts the number of positive fractal values in gamble 1.
    #Returns 2 if both g1_up and g1_down are positive.
    #Returns 1 if only one of g1_up or g1_down is positive.
    #Returns 0 if neither g1_up nor g1_down is positive.
    #g2_up and g2_down are not used here, but the Cue class always passes both gambles.
    if g1_up not in fractal_values or g1_down not in fractal_values:
        logger.error(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
        raise ValueError(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
    count = 0
    if g1_up > 0:
        count += 1
    if g1_down > 0:
        count += 1
    return count

def signs(g1_up, g1_down, g2_up, g2_down):
    #Counts the number of fractal values in gamble 1 that leads to an increase in wealth.
    #Returns 2 if both g1_up and g1_down lead to an increase.
    #Returns 1 if only one of g1_up or g1_down leads to an increase.
    #Returns 0 if neither g1_up nor g1_down leads to an increase.
    #g2_up and g2_down are not used here, but the Cue class always passes both gambles.
    count = 0
    if g1_up > 0:
       count += 1
    if g1_down > 0:
       count += 1
    return count

def prefere_best_n_ranks(g1_up, g1_down, g2_up, g2_down, n, fractal_values):
    #Checks if gamble 1 contains any of the n best fractal values.
    #The n best fractal values are the n biggest values in fractal_values.
    #Returns True if neither g1_up nor g1_down is among the n best fractal values.
    #Returns False if g1_up or g1_down is among the n best fractal values.
    #g2_up and g2_down are not used here, but the Cue class always passes both gambles.
    if n >= len(fractal_values):
        logger.error(f"n ({n}) must be smaller than the number of fractals ({len(fractal_values)}).")
        raise ValueError(f"n ({n}) must be smaller than the number of fractals ({len(fractal_values)}).")
    if g1_up not in fractal_values or g1_down not in fractal_values:
        logger.error(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
        raise ValueError(f"Gamble values {g1_up}, {g1_down} are not in fractal_values.")
    best_values = sorted(fractal_values)[-n:]
    if g1_up in best_values or g1_down in best_values:
        return True
    else:
        return False