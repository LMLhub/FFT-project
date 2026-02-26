#This module defines the Cue class, which represents a single cue in a decision-making task.
import pandas as pd
import numpy as np
import inspect
import yaml

class Cue:
    '''
    Cue class representing a single cue in a decision-making task.
    Cue can be either numerical or boolean (0,1), and has an associated feature
    function (f) that checks a certain feature of the gamble pair.
    The difference in f(G1,G2) and f(G2,G1) determines the value of the signed cue (F).
    The signed cue determines the preferred side for a given gamble pair if the cue value exceeds
    a certain threshold. For boolean cues, the threshold is 0, meaning that any positive F value
    favours the left gamble and any negative F value favours the right gamble. For numerical cues,
    the threshold can be set to a specific value."
    '''
    cue_registry = {}

    def __init__(self,id: str, name: str, description: str, feature, type: "boolean", threshold=None, params=None):
        self.id = id #Unique identifier for the cue
        self.name = name #Short name of the cue
        self.description = description #Text description of the cue
        self.type = type #"numerical" or "boolean"
        self.feature = feature #feature function that takes a gamble pair and returns a value. Higher value favours first input gamble compared to second.
        self.params = params or {} #additional parameters for the feature function. Must include "threshold" for numerical cues.
        self.threshold = threshold

        if self.type not in ["boolean", "numerical"]:
            raise ValueError("Cue type must be 'boolean' or 'numerical'.")
        
        # Boolean cues always use threshold = 0
        if self.type == "boolean":
            self.threshold = 0

        # Numerical cues must provide threshold
        if self.type == "numerical":
            if self.threshold is None:
                raise ValueError("Numerical cues require a threshold.")
            if not isinstance(self.threshold, (int, float)):
                raise TypeError("Threshold must be numeric.")   

        # Check that feature is a callable function
        if not callable(self.feature):
            raise ValueError("Feature must be a callable function.")

        # Check that feature parameters are provided as a dictionary
        if self.params is not None and not isinstance(self.params, dict):
            raise TypeError("Feature parameters must be a dictionary.")

        # Check that feature function signature is as expected
        sig = inspect.signature(self.feature)
        param_names = list(sig.parameters.keys())
        if len(param_names) < 4:
            raise ValueError("Feature function must accept at least 4 arguments (fractal values).")

        # Check that all feature parameters are provided in params
        expected_params = set(param_names[4:])
        provided_params = set(self.params.keys())

        if expected_params != provided_params:
            raise ValueError(
                f"Feature parameters mismatch.\n"
                f"Expected: {expected_params}\n"
                f"Provided: {provided_params}"
            )
        
        #Check that the feature function can be executed with the expected arguments.
        dummy = pd.Series([1, 2])

        try:
            test_output = self.feature(
                dummy, dummy, dummy, dummy,**self.params)

        except Exception as e:
            raise ValueError(
                f"Feature function for cue '{self.id}' could not be executed "
                f"during initialization: {e}"
            )

        test_output = pd.Series(test_output)

        #Check that the output of the feature function is numeric or boolean and matches the declared cue type.
        if not (pd.api.types.is_numeric_dtype(test_output) and self.type == "numerical"):
            if not (pd.api.types.is_bool_dtype(test_output) and self.type == "boolean"):
                raise ValueError(
                    f"Feature function for cue '{self.id}' does not match declared type '{self.type}'."
                )
        # Register the cue in the class-level registry
        if self.id in Cue.cue_registry:
            raise ValueError(f"Cue with id '{self.id}' already exists. IDs must be unique.")

        Cue.cue_registry[self.id] = self

    def evaluate(self, gamble_data: pd.DataFrame) -> pd.DataFrame:
        '''
        Parameters:
        - gamble_data: a pd gamble row with fractal values. 
        
        Returns:
        - cue_value: value of the cue for the given gamble pair. The value is the
          absolute value of the difference between the cue values for the left and right gambles.
        - side_if_true: the side cue favours is evaluated positively
        '''
        
        # --- Basic checks ----
        # Gamble data must be a dataframe
        if not isinstance(gamble_data, pd.DataFrame):
            raise ValueError("Input gamble_data must be a dataframe with gamble pairs.")
        
        #Gamble data must not be empty        
        if len(gamble_data) == 0:
            raise ValueError("Input gamble_data must not be empty.")
        
        #Gamble data must contain fractal value columns
        required_cols = [
        "gamma_left_up",
        "gamma_left_down",
        "gamma_right_up",
        "gamma_right_down"
        ]

        missing_cols = [col for col in required_cols if col not in gamble_data.columns]
        if missing_cols:
            raise ValueError(f"Missing required columns: {missing_cols}")
        
        #-------------------
        # Calculate feature values f(g1,g2) and f(g2,g1) for the left and right gambles, respectively.       
        
        f_LR = self.feature(gamble_data["gamma_left_up"],
                            gamble_data["gamma_left_down"],
                            gamble_data["gamma_right_up"],
                            gamble_data["gamma_right_down"],
                              **self.params)
        
        f_RL = self.feature(gamble_data["gamma_right_up"],
                            gamble_data["gamma_right_down"],
                            gamble_data["gamma_left_up"],
                            gamble_data["gamma_left_down"], 
                            **self.params)
        
        #Convert to float if boolean cue to ensure correct comparison with threshold
        if self.type == "boolean":
            f_LR = f_LR.astype(float)
            f_RL = f_RL.astype(float)

        F_value = f_LR - f_RL
        cue_value = F_value.abs()
        
        # choose the side that the cue favours based on the F_value and the threshold
        side_if_true = pd.Series(
            np.where(
                cue_value > self.threshold,
                np.where(F_value > self.threshold, "left", "right"),
                None),
            dtype="string"
        )
        
        #Save the cue value and the side it favours in the gamble_data dataframe
        gamble_data[self.id + "_cue_value"] = cue_value
        gamble_data[self.id + "_side_if_true"] = side_if_true
        
        return gamble_data
    
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "threshold": self.threshold,
            "params": self.params,
            "feature_name": self.feature.__name__
        }
    
    @classmethod
    def save_registry(cls, filepath):
        # This method saves the cue registry to a YAML file.
        registry_dict = {
            cue_id: cue.to_dict()
            for cue_id, cue in cls.cue_registry.items()
        }

        with open(filepath, "w") as f:
            yaml.dump(registry_dict, f, sort_keys=False)