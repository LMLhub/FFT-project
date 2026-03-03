#This module defines the Decision class, which represents a fast-and-frugal tree (FFT) for decision-making tasks. The FFT consists of a sequence of cues that are evaluated in order to make a decision between two options (e.g., left vs right gamble). The Decision class includes methods for evaluating the cues on a given input and making a decision based on the cue values and thresholds.

import pandas as pd
import numpy as np
import yaml


class FFT: 
    '''
    FFT class representing a fast-and-frugal tree for decision-making tasks.
    The FFT consists of a sequence of cues that are evaluated in order to make a decision
    between two options (e.g., left vs right gamble).
    The Decision class includes methods for evaluating the cues on a given input and making
    a decision based on the cue values and thresholds.
    '''
    FFT_registry = {}

    def __init__(self, id, name, description, cues):
        self.id = id
        self.name = name
        self.description = description
        self.cues = cues
        self.tree_length = len(cues)
        
        #Add FFT to registry
        if self.id in FFT.FFT_registry:
            raise ValueError(f"FFT with id '{self.id}' already exists. IDs must be unique.")
        FFT.FFT_registry[self.id] = self

    def to_dict(self):
        return {
            "id": self.id,
            "name": self.name,
            "description": self.description,
            "cues": [{"id": cue.id, "name": cue.name} for cue in self.cues],
            "tree length": self.tree_length,
        }
    
    @classmethod
    def save_registry(cls, filepath):
        # This method saves the FFT registry to a YAML file.
        registry_dict = {
            cue_id: fft.to_dict()
            for cue_id, fft in cls.FFT_registry.items()
        }

        with open(filepath, "w") as f:
            yaml.dump(registry_dict, f, sort_keys=False)

    def decide(self, x_left_up, x_left_down, x_right_up, x_right_down, **kwargs):
        # This method evaluates the cues in the FFT on the given input and makes a decision
        # based on the cue values and thresholds. It returns the decision ("left" or "right") and the number of cues used.
        cue_values = []

        for i, cue in enumerate(self.cues, start=1):
            # Determine which extra arguments this cue needs
            extra_arg_names = cue.required_args[4:]

            # Filter kwargs to only those required by this cue
            filtered_kwargs = {
                key: kwargs[key]
                for key in extra_arg_names
                if key in kwargs
            }

            #Evalue each cue and get the cue value and the side it favors if the cue is true
            cue_value, side = cue.evaluate(
                x_left_up,
                x_left_down,
                x_right_up,
                x_right_down,
                **filtered_kwargs)

            #Append cue value to list of cue values
            cue_values.append(cue_value)

            # If this cue makes a decision, stop
            if side is not None:
                return cue_values, side, i

        # If no cue makes a decision, return random choice and the number of cues used +1
        i = len(cue_values) + 1 
        side = np.random.choice(["left", "right"])
        
        return cue_values, side, i


    def decide_df(self, gamble_data: pd.DataFrame, required_args = list ) -> pd.DataFrame:
        # This method evaluates the cues in the FFT on the given gamble_data and makes a decision
        # It returns the gamble_data dataframe with additional columns for each cue's value and the 
        # side it favors if the cue is true, as well as the final decision and number of cues used.
        # required_args must contain four fractal values + extra arguments used by cues.
        
        # Prepare storage for final decision and number of cues used        
        df = gamble_data.copy()
        df["fft_decision"] = None
        df["fft_cues_used"] = 0

        # Process row by row
        for idx, row in df.iterrows():
            # Extract the four main fractals
            x_left_up = row[required_args[0]]
            x_left_down = row[required_args[1]]
            x_right_up = row[required_args[2]]
            x_right_down = row[required_args[3]]

            # Extract any additional required arguments
            extra_arg_names = required_args[4:]
            extra_args = {arg: row[arg] for arg in extra_arg_names}

            cue_values, side, cues_used = self.decide(
                x_left_up,
                x_left_down,
                x_right_up,
                x_right_down,
                **extra_args
            )
            
            #Save cue values in df
            for i in range(len(cue_values)):
                df.loc[idx, f"{self.cues[i].id}_cue_value"] = cue_values[i]

            df.loc[idx, f"{self.id}_decision"] = side
            df.loc[idx, f"{self.id}_cues_used"] = cues_used
        
        return df
