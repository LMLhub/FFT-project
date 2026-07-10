#This file contains code implenting cues as instances of the Cue class.
import numpy as np
import logging
logger = logging.getLogger(__name__)

def growth_rate(g1_up, g1_down, g2_up, g2_down):
    #This function calculates the growth rate of the g1 given the dynamic.
    # Fractal values g must be gamma values, such that the growth rate is 
    # the expected value of the fractal values for both dynamics.
    return (g1_up + g1_down)/2

def growth_rate_min(g1_up, g1_down, g2_up, g2_down):
    #This function calculates the growth rate of the g1 given the dynamic.
    # Fractal values g must be gamma values, such that the growth rate is 
    # the expected value of the fractal values for both dynamics.
    return -(g1_up + g1_down)/2

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

def prefer_best_n_ranks(g1_up, g1_down, g2_up, g2_down, n, fractal_values):
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
    
def priority_step1(g1_up, g1_down, g2_up, g2_down, tol, dynamic):
    # This function is to be used for the first step of the priority heuristics
    # as suggested by the 2008 paper. It is asymmetrical, which means that losses and gains 
    # are treated differently (risk seeking for losses and risk aversion for gains).
    # Since the paper does not specify what to do for mixed gambles (with both gains and losses) we 
    # switch based on the average outcome. If the choice is deemed a 'loss' problem, we change sign of all values.

    # Checks if the difference between the minimum of gamble 1 and minimum of gamble 2
    # is greater than the tolerance multiplied by the maximimum gain of gamble 2.
    # Returns true if the minimum gain is above the the threshold, indicating preference of the less risky choice of g1.
    
    
    if dynamic == "multiplicative":
        g1_up = np.exp(g1_up)-1
        g1_down = np.exp(g1_down)-1
        g2_up = np.exp(g2_up)-1
        g2_down = np.exp(g2_down)-1

    # This variable is used to keep track of whether the signs of the gamble values have been reversed.
    reversed = False
    
    #Check if the average outcome is negative, if so, switch signs of all values
    if ((g1_up + g1_down + g2_up + g2_down)/4 < 0):
        reversed = True

    if reversed:
        #calculate the minimum loss
        g1_min = np.min([-g1_up, -g1_down])
        g2_min = np.min([-g2_up, -g2_down])
        #calculate the maximum loss
        g1_max = np.max([-g1_up, -g1_down])
        g2_max = np.max([-g2_up, -g2_down])
    else:
        #calculate the minimum gain
        g1_min = np.min([g1_up, g1_down])
        g2_min = np.min([g2_up, g2_down])
        #calculate the maximum gain
        g1_max = np.max([g1_up, g1_down])
        g2_max = np.max([g2_up, g2_down])

    min_difference = g1_min - g2_min
    #print("reversed: ", reversed)
    #print("g1_up: ", g1_up, " g1_down: ", g1_down, " g2_up: ", g2_up, " g2_down: ", g2_down)
    #print("minimum difference: ", min_difference, " g1_min: ", g1_min, " g2_min: ", g2_min)
    #print("tolerance * maximum gain: ", tol * np.max([np.abs(g1_max), np.abs(g2_max)]))
    
    
    #If the minimum gain/loss differ by tol (or more) of the maximum gain/loss
    if np.abs(min_difference) > tol * np.max([g1_max, g2_max]):#it seems like the sign should change.
    
        if reversed:
            # then choose the gamble with the lowest mimimum loss
            if g1_min < g2_min:
                return True
        
        if not reversed:
            # then choose the gamble with the highest minumum gain
            if g1_min > g2_min:
                return True
        
    return False

def priority_step3(g1_up, g1_down, g2_up, g2_down, dynamic):
    # This is the third step of the priority heurstic (step 2 is about probabilities,
    # and since they are all 0.5 in the experiment, we skip this step).
    # The third step is about the maximum gain, and it is symmetrical, which means that losses and gains are treated the same way.)'
    
    if dynamic == "multiplicative":
        g1_up = np.exp(g1_up)-1
        g1_down = np.exp(g1_down)-1
        g2_up = np.exp(g2_up)-1
        g2_down = np.exp(g2_down)-1
    
    # This variable is used to keep track of whether the signs of the gamble values have been reversed.
    reversed = False

    #Check if the average outcome is negative, if so, switch signs of all values
    if (g1_up + g1_down + g2_up + g2_down)/4 < 0:
        reversed = True

    if reversed:
        #calculate the maximum loss
        g1_max = np.max([-g1_up, -g1_down])
        g2_max = np.max([-g2_up, -g2_down])

    else:
        #calculate the maximum gain
        g1_max = np.max([g1_up, g1_down])
        g2_max = np.max([g2_up, g2_down])

    if reversed:
        #pick the one with the lowest maximum loss:
        if g1_max < g2_max:
            return True
        
    if not reversed:
        #pick the one with the highest maximum gain:
        if g1_max > g2_max:
            return True
        
    return False

def priority_step1_no_loss(g1_up, g1_down, g2_up, g2_down, tol, dynamic):
    # This is a version of priority_step1 that does not consider losses.
    # This is done by by checking if any values are negative, and if so, 
    # adding the absolute value of the minimum value to all values, so that they are all positive.

    if dynamic == "multiplicative":
        g1_up = np.exp(g1_up)-1
        g1_down = np.exp(g1_down)-1
        g2_up = np.exp(g2_up)-1
        g2_down = np.exp(g2_down)-1
    '''
    if min(g1_up, g1_down, g2_up, g2_down) < 0:
        min_value = min(g1_up, g1_down, g2_up, g2_down)
        g1_up += abs(min_value)
        g1_down += abs(min_value)
        g2_up += abs(min_value)
        g2_down += abs(min_value)
    '''
    min_value = min(g1_up, g1_down, g2_up, g2_down)
    g1_up -= min_value
    g1_down -= min_value
    g2_up -= min_value
    g2_down -= min_value

    return priority_step1(g1_up, g1_down, g2_up, g2_down, tol, "additive")

def priority_step3_no_loss(g1_up, g1_down, g2_up, g2_down, dynamic):
    # This is a version of priority_step3 that does not consider losses.
    # This is done by by checking if any values are negative, and if so, 
    # adding the absolute value of the minimum value to all values, so that they are all positive.

    if dynamic == "multiplicative":
        g1_up = np.exp(g1_up)-1
        g1_down = np.exp(g1_down)-1
        g2_up = np.exp(g2_up)-1
        g2_down = np.exp(g2_down)-1
    '''
    if min(g1_up, g1_down, g2_up, g2_down) < 0:
        min_value = min(g1_up, g1_down, g2_up, g2_down)
        g1_up += abs(min_value)
        g1_down += abs(min_value)
        g2_up += abs(min_value)
        g2_down += abs(min_value)
    '''
    min_value = min(g1_up, g1_down, g2_up, g2_down)
    g1_up -= min_value
    g1_down -= min_value
    g2_up -= min_value
    g2_down -= min_value

    return priority_step3(g1_up, g1_down, g2_up, g2_down, "additive")

    