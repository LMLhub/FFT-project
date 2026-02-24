#This module defines the Cue class, which represents a single cue in a decision-making task.

class Cue:
    "Cue class representing a single cue in a decision-making task."
    "Cue can be either numerical or boolean (0,1), and has an associated evaluator function that determines how the cue is evaluated for a given gamble pair."
    
    def __init__(self, name: str, description: str, side_if_true: str, evaluator, params=None):
        self.name = name #Unique identifier for the cue
        self.description = description #Text description of the cue
        self.type = "boolean" #"numerical" or "boolean"
        self.evaluator = evaluator #function that takes a fractal pair and returns a function value
        self.params = params #additional parameters for the evaluator function, if needed

    def evaluate(self, x):
        '''
        Parameters:
        - x: a pd gamble row with fractal values. 
        
        Returns:
        - cue_value: value of the cue for the given gamble pair.
        - side_if_true: the side cue favours is evaluated positively
        '''


        value_left, value_right = self.evaluator(x_left, **self.params), self.evaluator(x_right, **self.params)
        cue_difference = value_left - value_right
        cue_value = abs(cue_difference)

        if self.type == "boolean":
            if cue_difference > 0:
                side_if_true = "left"
            elif cue_difference < 0:
                side_if_true = "right"
            else:
                side_if_true = "na"
        
        if self.type == "numerical":
            side_if_true = "left" if cue_value >  else "right" if cue_value < 0 else "na"

        return cue_value, side_if_true