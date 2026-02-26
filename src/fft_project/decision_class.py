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
        

    def decide(self, gamble_data: pd.DataFrame) -> pd.DataFrame:
        # This method evaluates the cues in the FFT on the given gamble_data and makes a decision
        # based on the cue values and thresholds. It returns the gamble_data dataframe with
        # additional columns for each cue's value and the side it favors if the cue is true.
        
        df = gamble_data.copy()

        n = len(df)

        # Store final decision per row
        decision = pd.Series([None] * n, index=df.index, dtype="object")

        # Store how many cues were evaluated per row
        cues_used = pd.Series(0, index=df.index)

        # Mask for rows that are still undecided
        undecided = pd.Series(True, index=df.index)

        for i, cue in enumerate(self.cues, start=1):

            # Evaluate cue (adds cue columns to df)
            df = cue.evaluate(df)

            side_col = cue.id + "_side_if_true"

            # Rows where this cue gives a decision
            decides_here = undecided & df[side_col].notna()

            # Assign decision
            decision.loc[decides_here] = df.loc[decides_here, side_col]

            # Count cue usage for rows still undecided
            cues_used.loc[undecided] += 1
            
            # For rows already decided BEFORE this cue,
            # mark this cue column as "-"
            df.loc[~undecided, side_col] = "-"
            
            # Update undecided mask
            undecided = undecided & df[side_col].isna()

        # Handle rows never decided
        if undecided.any():
            # Example fallback: random
            random_choice = np.random.choice(["left", "right"], size=undecided.sum())
            decision.loc[undecided] = random_choice
            cues_used.loc[undecided] = self.tree_length+1  # All cues were used - plus one for the random decision

        df["fft_decision"] = decision
        df["fft_cues_used"] = cues_used

        return df