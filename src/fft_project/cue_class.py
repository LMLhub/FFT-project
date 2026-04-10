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
    the threshold can be set to a specific value.
    '''
    cue_registry = {}

    def __init__(self,id: str, name: str, description: str, feature, type: str,
                  threshold=None, params=None, required_args = list):
        self.id = id #Unique identifier for the cue
        self.name = name #Short name of the cue
        self.description = description #Text description of the cue
        self.type = type #"numerical" or "boolean"
        self.feature = feature #feature function that takes a gamble pair and returns a value. Higher value favours first input gamble compared to second.
        self.params = params or {} #additional parameters for the feature function.
        self.threshold = threshold #threshold for numerical cues. For boolean cues, threshold is set to 0.
        self.required_args = required_args #List of required input for the feature function. These first four are fractal values (outcome or rank) that are passed as the first four arguments to the feature function. The order of these columns must be specified in the same order as they are passed to the feature function (i.e. x_left_up, x_left_down, x_right_up, x_right_down, wealth, ect).

        # Check that feature is a callable function
        if not callable(self.feature):
            raise ValueError("Feature must be a callable function.")

        # Check that feature parameters are provided as a dictionary
        if self.params is not None and not isinstance(self.params, dict):
            raise TypeError("Feature parameters must be a dictionary.")

        # Check that feature function signature is as expected
        sig = inspect.signature(self.feature)
        param_names = list(sig.parameters.keys())
        if len(param_names) != len(self.params) + len(self.required_args) :
            raise ValueError(f"Feature function input mismatch: declared arguments ({len(param_names)}) and parameters ({len(self.params)}) do not add up to the total input to the feature function ({len(param_names)}).")
                               
        # Check that all feature parameters are provided in params
        expected_params = set(param_names[len(self.required_args):])
        provided_params = set(self.params.keys())

        if expected_params != provided_params:
            raise ValueError(
                f"Feature parameters mismatch.\n"
                f"Expected: {expected_params}\n"
                f"Provided: {provided_params}"
            )

        # Boolean cues always use threshold = 0
        if self.type == "boolean":
            self.threshold = 0

        # Numerical cues must provide threshold
        if self.type == "numerical":
            if self.threshold is None:
                raise ValueError("Numerical cues require a threshold.")
            if not isinstance(self.threshold, (int, float)):
                raise TypeError("Threshold must be numeric.")   
            
        # Register the cue in the class-level registry
        if self.id in Cue.cue_registry:
            raise ValueError(f"Cue with id '{self.id}' already exists. IDs must be unique.")

        Cue.cue_registry[self.id] = self


    def evaluate(self, x_left_up, x_left_down, x_right_up, x_right_down, **extra_args):
        '''
        Evaluates the cue for a given gamble pair based on the feature function and the threshold.
        Input:
        - four fractal values, either outcome or rank.
        - additional arguments for the feature function if needed, which can be passed as columns in the gamble_data dataframe when using the evaluate_pd method. These are specified in the required_columns attribute of the Cue object.
        Returns:
        - cue_value: value of the cue for the given gamble pair. The value is the
          absolute value of the difference between the feature values for the left and right gambles.
        - side_if_true: the side the cue favours if evaluated positively
        '''
        f_LR = self.feature(x_left_up, x_left_down, x_right_up, x_right_down, **extra_args, **self.params)
        f_RL = self.feature(x_right_up, x_right_down, x_left_up, x_left_down, **extra_args, **self.params)

        if self.type == "boolean":
            f_LR = float(f_LR)
            f_RL = float(f_RL)

        F_value = f_LR - f_RL
        cue_value = abs(F_value)

        if cue_value > self.threshold:
            side_if_true = "left" if F_value > self.threshold else "right"
        else:
            side_if_true = None

        return cue_value, side_if_true
            
    import pandas as pd

    def evaluate_df(self, gamble_data: pd.DataFrame) -> pd.DataFrame:
        """
        Evaluates the cue for each row in a pandas DataFrame.

        The first four names in self.required_args must correspond to:
        x_left_up, x_left_down, x_right_up, x_right_down.

        Any additional names in required_args are passed as extra arguments
        to the feature function.

        Parameters
        ----------
        gamble_data : pd.DataFrame
            DataFrame containing the required columns.

        Returns
        -------
        pd.DataFrame
            Copy of gamble_data with two new columns:
            - '<cue_id>_value'
            - '<cue_id>_side_if_true'
        """

        # Ensure all required columns exist
        missing_cols = set(self.required_args) - set(gamble_data.columns)
        if missing_cols:
            raise ValueError(f"Missing required columns in DataFrame: {missing_cols}")

        # Split required arguments
        base_args = self.required_args[:4]
        extra_arg_names = self.required_args[4:]

        def evaluate_row(row):
            # This function evaluates the cue for a single row of the DataFrame. 
            # It extracts the required arguments from the row and passes them to the evaluate method.

            # Extract the four mandatory gamble values
            x_left_up = row[base_args[0]]
            x_left_down = row[base_args[1]]
            x_right_up = row[base_args[2]]
            x_right_down = row[base_args[3]]

            # Extract any additional required arguments
            extra_args = {arg: row[arg] for arg in extra_arg_names}

            return self.evaluate(
                x_left_up,
                x_left_down,
                x_right_up,
                x_right_down,
                **extra_args
            )

        # Evaluate cue row-wise
        results = gamble_data.apply(evaluate_row, axis=1)

        # Split tuple output into two columns
        gamble_data[f"{self.id}_value"] = results.apply(lambda x: x[0])
        gamble_data[f"{self.id}_side_if_true"] = results.apply(lambda x: x[1])

        return gamble_data
        
    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "type": self.type,
            "threshold": self.threshold,
            "params": self.params,
            "feature_name": self.feature.__name__,
            "required arguments": self.required_args
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