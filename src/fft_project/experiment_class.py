import pandas as pd
import numpy as np
import yaml
import logging
logger = logging.getLogger(__name__)

from .cue_class import Cue
from .decision_class import FFT



class Experiment:
    '''
    Experiment class representing a decision-making experiment.
    An experiment consists of one decision rule (FFT) that is evaluated on a gamble_data dataframe.
    The Experiment class includes methods for running the experiment on a given gamble_data and
    analyzing the results.
    '''
    experiment_registry = {}

    def __init__(self, id, name, description, fft, dynamic = None, gamble_data = None, initial_wealth = None):
        self.id = id                    #Unique identifier for the experiment
        self.name = name                #Short name of the experiment
        self.description = description  #Text description of the experiment
        self.fft = fft                  #FFT decision rule to be evaluated in the experiment
        self.dynamic = dynamic          #"multiplicative" or "additive" dynamic for the wealth trajectory.
        self.gamble_data = gamble_data  #DataFrame containing the gamble pairs and any additional required arguments for the cues.
        self.initial_wealth = initial_wealth  #Initial wealth for the experiment.
        self.random_seeds= []           #List to store the random seeds used in the experiment, which can be useful for tracking and reproducibility purposes.
        self.runs = 0                      #Counter for the number of times the experiment has been run.

        #Check that fft is an instance of the FFT class
        if not isinstance(self.fft, FFT):
            logger.error("FFT must be an instance of the FFT class.")
            raise ValueError("FFT must be an instance of the FFT class.")
        
        #Check that dynamic is either "multiplicative" or "additive"
        if self.dynamic not in ["multiplicative", "additive"]: 
            logger.error("Dynamic must be either 'multiplicative' or 'additive'.")
            raise ValueError("Dynamic must be either 'multiplicative' or 'additive'.")
        
        # Check that initial_wealth is provided        
        if initial_wealth is None:
            logger.error("initial_wealth must be provided to calculate wealth trajectory.")
            raise ValueError("initial_wealth must be provided to calculate wealth trajectory.")

        #Retrieve the required arguments for the cues in the FFT.
        self.required_args = self.fft.retrieve_required_args()
        
        # Check that required arguments are present in the gamble_data dataframe
        if self.gamble_data is not None:
            missing_args = [arg for arg in self.required_args if arg not in self.gamble_data.columns]
            if missing_args:
                logger.error(f"Gamble data is missing required arguments: {missing_args}")
                raise ValueError(f"Gamble data is missing required arguments: {missing_args}")
        else:
            logger.error("Gamble data must be provided for the experiment.")
            raise ValueError("Gamble data must be provided for the experiment.")
        
        #Add Experiment to registry
        if self.id in Experiment.experiment_registry:
            logger.error(f"Experiment with id '{self.id}' already exists. IDs must be unique.")
            raise ValueError(f"Experiment with id '{self.id}' already exists. IDs must be unique.")
        Experiment.experiment_registry[self.id] = self

    def run_experiment(self,
                          initial_wealth: float = None,
                          random_seed: int = None
                          ) -> pd.DataFrame:
        # This method evaluates the cues in the FFT on the given gamble_data and makes a decision.
        # Then it flips a coin to determine the outcome of the gamble and updates the wealth
        # trajectory based on the decision and the outcome.
        # It returns the gamble_data dataframe with additional columns for each cue's value and the 
        # side it favors if the cue is true, as well as the final decision and number of cues used.
        # Additionally, it includes a column for the wealth trajectory over time.
        # OBS: the "wealth" column is the initial wealth, and the "wealth_final" columns
        # is the wealth after the gamble outcome has been realized.
        
        # Set random seed for reproducibility if provided        
        if random_seed is not None:
            np.random.seed(random_seed)
        else:
            random_seed = np.random.randint(0, 1_000_000)
            np.random.seed(random_seed)
        
        # adds the random seed to the list of random seeds used in the experiment, which can be useful for tracking and reproducibility purposes.
        self.random_seeds.append(random_seed)

        # keep tract of the number of runs of the experiment.
        self.runs += 1

        # set initial wealth for the experiment if not provided as an argument to the method
        if initial_wealth is None:
            initial_wealth = self.initial_wealth
                
        # Prepare storage for final decision and number of cues used        
        df = self.gamble_data.copy()
        df["time_step"] = df.index
        df.loc[0, "wealth"] = initial_wealth

        # Process row by row
        for idx, row in df.iterrows():
            # Extract the four main fractals
            fractal_values = {
                "x_left_up": row[self.required_args[0]],
                "x_left_down": row[self.required_args[1]],
                "x_right_up": row[self.required_args[2]],
                "x_right_down": row[self.required_args[3]]
            }

            # Extract any additional required arguments
            extra_arg_names = self.required_args[4:]
            extra_args = {arg: row[arg] for arg in extra_arg_names}

            cue_values, side, cues_used = self.fft.decide(
                fractal_values["x_left_up"],
                fractal_values["x_left_down"],
                fractal_values["x_right_up"],
                fractal_values["x_right_down"],
                **extra_args
            )
            
            #Save cue values in df - OBS: this seems not to belong to the the Experiment class, but to the FFT class. We can move it there later if we want to keep the Experiment class.
            #for i in range(len(cue_values)):
            #    df.loc[idx, f"{self.fft.cues[i].id}_cue_value"] = cue_values[i]

            df.loc[idx, f"{self.fft.id}_decision_{self.runs}"] = side
            df.loc[idx, f"{self.fft.id}_cues_used_{self.runs}"] = cues_used

            # coin flip to determine outcome of gamble
            outcome = np.random.choice(["up", "down"])
            outcome_value = fractal_values[f"x_{side}_{outcome}"]

            # Update wealth trajectory based on decision and outcome
            if self.dynamic == "multiplicative":
                df.loc[idx, f"{self.fft.id}_wealth_{self.runs}"] = df.loc[idx, "wealth"] * np.exp(outcome_value)
            elif self.dynamic == "additive":
                df.loc[idx, f"{self.fft.id}_wealth_{self.runs}"] = df.loc[idx, "wealth"] + outcome_value

            if idx < len(df) - 1:
                df.loc[idx + 1, "wealth"] = df.loc[idx, f"{self.fft.id}_wealth_{self.runs}"]
            
            self.gamble_data = df

        return df

def accuracy(self, decision: str=None, reference_decisions: str=None):
    # This method calculates the accuracy of the FFT's decisions at a given run compared to the
    # optimal decisions based on the gamble_data.
    
    # Check that the decision column for the given run exists in the gamble_data dataframe
    if decision is None:
        logger.error("Column name of the decision must be provided to calculate accuracy.")
        raise ValueError("Column name of the decision must be provided to calculate accuracy.")
    if decision not in self.gamble_data.columns:
        logger.error(f"Decision column '{decision}' not found in gamble_data.")
        raise ValueError(f"Decision column '{decision}' not found in gamble_data.")
    
    # Check that the reference_decisions column exists in the gamble_data dataframe
    if reference_decisions is None:
        logger.error("Column name of the reference decisions must be provided to calculate accuracy.")
        raise ValueError("Column name of the reference decisions must be provided to calculate accuracy.")
    
    # Check that the reference_decisions column exists in the gamble_data dataframe
    if reference_decisions not in self.gamble_data.columns:
        logger.error(f"Reference decisions column '{reference_decisions}' not found in gamble_data.")
        raise ValueError(f"Reference decisions column '{reference_decisions}' not found in gamble_data.")
    
    # Calculate accuracy as the proportion of decisions that match the reference decisions
    correct_decisions = self.gamble_data[decision] == self.gamble_data[reference_decisions]
    accuracy = correct_decisions.mean()
    
    return accuracy
