import pandas as pd
import numpy as np
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

    def __init__(self, id, name, description, ffts, dynamic = None, gamble_data = None, initial_wealth = None):
        self.id = id                    #Unique identifier for the experiment
        self.name = name                #Short name of the experiment
        self.description = description  #Text description of the experiment
        self.ffts = ffts                #List of FFT decision rule to be evaluated in the experiment
        self.dynamic = dynamic          #"multiplicative" or "additive" dynamic for the wealth trajectory.
        self.gamble_data = gamble_data  #DataFrame containing the gamble pairs and any additional required arguments for the cues.
        self.initial_wealth = initial_wealth  #Initial wealth for the experiment.
        self.random_seeds= []           #List to store the random seeds used in the experiment, which can be useful for tracking and reproducibility purposes.
        self.runs = 0                      #Counter for the number of times the experiment has been run.
        self.results = None   # accumulates result_df across runs

        #Check that ffts is an list of the FFT class
        if not isinstance(self.ffts, list):
            logger.error("FFTs must be a list of FFT instances.")
            raise ValueError("FFTs must be a list of FFT instances.")
        else:
            for fft in self.ffts:
                if not isinstance(fft, FFT):
                    logger.error("All items in ffts must be instances of the FFT class.")
                    raise ValueError("All items in ffts must be instances of the FFT class.")

        #Check that dynamic is either "multiplicative" or "additive"
        if self.dynamic not in ["multiplicative", "additive"]: 
            logger.error("Dynamic must be either 'multiplicative' or 'additive'.")
            raise ValueError("Dynamic must be either 'multiplicative' or 'additive'.")
        
        # Check that initial_wealth is provided        
        if initial_wealth is None:
            logger.error("initial_wealth must be provided to calculate wealth trajectory.")
            raise ValueError("initial_wealth must be provided to calculate wealth trajectory.")

        #Retrieve the required arguments for the cues in the FFT.
        self.required_args = []
        for fft in self.ffts:
            self.required_args.extend(fft.retrieve_required_args())

        # Check that required arguments are present in the gamble_data dataframe.
        # wealth is an exception because it may not be a column in the gamble_data
        # but is updated on the way.
        if self.gamble_data is not None:
            missing_args = [arg for arg in self.required_args if arg not in self.gamble_data.columns]
            if "wealth" in missing_args:
                missing_args.remove("wealth")
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
                    random_seed: int = None,
                    wealth_update: str = None #Choose "process" "constant" "data"
                    ) -> pd.DataFrame:
        """
        For each FFT, walk through every gamble in sequence:
        1. Ask the FFT to make a decision (left or right).
        2. Flip a coin to reveal the outcome (up or down).
        3. Update wealth based on what was decided and what happened and wealth_update method.
            - If wealth_update is "process", update wealth based on the experiment dynamic (multiply or add).
            - If wealth_update is "constant", keep wealth constant across all gambles.
            - If wealth_update is "data", update wealth based on the gamble data (if a column "wealth" is provided, otherwise raise an error).
            
        Returns a multi-index DataFrame indexed by (fft_id, run, metric),
        with gamble_data index values as columns, accumulated across all runs
        in self.results.
        """
        # ── Initial checks ─────────────────────────────────────────────────────────────────
        if self.gamble_data is None:
            logger.error("Gamble data must be provided to run the experiment.")
            raise ValueError("Gamble data must be provided to run the experiment.")
        
        if self.dynamic not in ["multiplicative", "additive"]:
            logger.error("Dynamic must be either 'multiplicative' or 'additive'.")
            raise ValueError("Dynamic must be either 'multiplicative' or 'additive'.")
        
        if initial_wealth is None and self.initial_wealth is None and (wealth_update != "data"):
            logger.error("Initial wealth must be provided either as an argument to run_experiment or as an attribute of the Experiment.")
            raise ValueError("Initial wealth must be provided either as an argument to run_experiment or as an attribute of the Experiment.")
        
        if wealth_update not in [None, "process", "constant", "data"]:
            logger.error("wealth_update must be one of 'process', 'constant', 'data', or None.")
            raise ValueError("wealth_update must be one of 'process', 'constant', 'data', or None.")
        
        if (wealth_update == "data") and (initial_wealth is not None):
            logger.warning("WARNING: Both initial wealth and data passed - data is used")
            
        if wealth_update == "data" and "wealth" not in self.gamble_data.columns:
            logger.error("wealth_update indicates data as wealth source, but there is no 'wealth' column in data")
            raise ValueError("wealth_update indicates data as wealth source, but there is no 'wealth' column in data")

        # ── Setup ─────────────────────────────────────────────────────────────────

        # If wealth_update is None, set to default "process".
        if wealth_update is None:
            wealth_update = "process"

        # Set the random seed (for reproduction).
        if random_seed is None:
            random_seed = np.random.randint(0, 1_000_000)
        np.random.seed(random_seed)
        self.random_seeds.append(random_seed)

        # Count this as a new run.
        self.runs += 1
        run = self.runs

        # ── Run every FFT through every gamble ────────────────────────────────────
        # Initialize a dictionary to collect results for all FFTs and runs.
        collected = {}

        #Run each FFT through the gamble data
        for fft in self.ffts:

            # Unless data is chosen each FFT starts the run with the same initial wealth.
            if wealth_update != "data":
                # If no initial wealth is provided, use default.
                if initial_wealth is None:
                    initial_wealth = self.initial_wealth
                wealth = initial_wealth

            # Prepare empty result lists for this FFT / run combination.
            decisions = []   # which side the FFT chose  ("left" or "right")
            cues_used = []   # how many cues were needed to reach a decision
            outcomes  = []   # what the coin flip revealed ("up" or "down")
            wealths_pre   = []   # wealth before the gamble is resolved
            wealths_post = []   # wealth after the gamble is resolved
            realised_gammas = [] # the actual growth rate of the chosen option
            average_gammas = [] # the time-average growth rate of the chosen option (averaged across the two possible outcomes, given gammas are ergodic measures)

            for _, gamble in self.gamble_data.iterrows():

                # ── Step 1: the FFT inspects the gamble and makes a decision ──────

                # Pull the four fractal payoff values that describe this gamble.
                x_left_up    = gamble[self.required_args[0]]
                x_left_down  = gamble[self.required_args[1]]
                x_right_up   = gamble[self.required_args[2]]
                x_right_down = gamble[self.required_args[3]]

                fractal_values = {
                    "x_left_up":    x_left_up,
                    "x_left_down":  x_left_down,
                    "x_right_up":   x_right_up,
                    "x_right_down": x_right_down,
                }

                # If wealth_update is data, then collect the wealth from dataset:
                if wealth_update == "data":
                    wealth = gamble["wealth"]
                
                # Save starting wealth:
                wealths_pre.append(wealth)

                # Pass any extra arguments the FFT might need (beyond the four fractals).
                # If wealth is one of the required arguments, make sure to pass the current wealth value.
                extra_args = {}
                for arg in self.required_args[4:]:
                    if arg == "wealth":
                        extra_args["wealth"] = wealth
                    else:
                        extra_args[arg] = gamble[arg]
        
                # Decide which side to choose based on the FFT's decision rule
                _, side, n_cues = fft.decide(
                    x_left_up, x_left_down,
                    x_right_up, x_right_down,
                    **extra_args
                )

                # ── Step 2: flip a coin to reveal what actually happened ──────────
                coin = np.random.choice(["up", "down"])

                # The payoff is the fractal value for whichever side was chosen
                # and whatever the coin showed.
                payoff = fractal_values[f"x_{side}_{coin}"]

                # ── Step 3: update wealth based on the dynamic (multiply or add) ──
                if self.dynamic == "multiplicative":
                    final_wealth = wealth * np.exp(payoff)
                elif self.dynamic == "additive":
                    final_wealth = wealth + payoff
                
                # Update inital wealth for next round
                if wealth_update == "constant":
                    pass # wealth stays the same

                elif wealth_update == "process":
                    wealth = final_wealth #next round's wealth is final wealth of this round 

                # ── Record what happened this time step ───────────────────────────
                average_gamma = (fractal_values[f"x_{side}_up"] + fractal_values[f"x_{side}_down"]) / 2
                realised_gamma = payoff

                decisions.append(side)
                cues_used.append(n_cues)
                outcomes.append(coin)
                wealths_post.append(final_wealth)
                average_gammas.append(average_gamma)
                realised_gammas.append(realised_gamma)

            # Store all four metric series for this FFT and run.
            for metric, values in [("decision",  decisions),
                                    ("cues_used", cues_used),
                                    ("outcome",   outcomes),
                                    ("wealth_pre",    wealths_pre),
                                    ("wealth_post",    wealths_post),
                                    ("average_gamma", average_gammas),
                                    ("realised_gamma", realised_gammas)]:
                collected[(fft.id, run, metric)] = values

        # ── Assemble the multi-index DataFrame ────────────────────────────────────

        multi_index = pd.MultiIndex.from_tuples(
            collected.keys(),
            names=["fft_id", "run", "metric"]
        )

        # Save the collected results into a DataFrame with a multi-index (fft_id, run, metric) and columns corresponding to the gamble_data index values. 
        # Transpose so that gambles are rows and metrics are columns.
        result_df = pd.DataFrame(
            list(collected.values()),
            index=multi_index,
            columns=self.gamble_data.index
        ).T
        
        # ── Update results of experiment object ─────────────────────────────
        if self.results is None:
            self.results = result_df
        else:
            self.results = pd.concat([self.results, result_df], axis=1)

        # Return the result dataframe for all experiments runs so far.
        return self.results
    
    def accuracy(self, fft_id: str, reference_id: str, run_no: int = None) -> float:
    # This method calculates the accuracy of the FFT's decisions compared to the
    # optimal decisions of a reference FFT, across one or more runs.
    # Determine which runs to evaluate.
        if run_no is None:
            runs = range(1, self.runs + 1) #if no run number is provided, evaluate all runs
        elif isinstance(run_no, int):
            runs = [run_no]
        else:
            runs = run_no

        correct_decisions = 0
        number_of_decisions = 0

        for run in runs:

            # Check that the decision series exists for the FFT and the reference.
            if (fft_id, run, "decision") not in self.results.columns:
                logger.error(f"Decision column for FFT '{fft_id}' run {run} not found in results.")
                raise ValueError(f"Decision column for FFT '{fft_id}' run {run} not found in results.")

            if (reference_id, run, "decision") not in self.results.columns:
                logger.error(f"Decision column for reference FFT '{reference_id}' run {run} not found in results.")
                raise ValueError(f"Decision column for reference FFT '{reference_id}' run {run} not found in results.")

            # Pull the decision series for this run.
            fft_decisions       = self.results[(fft_id,       run, "decision")]
            reference_decisions = self.results[(reference_id, run, "decision")]

            # Count the number of correct decisions (where the FFT's decision matches the reference)
            # and the total number of decisions.
            correct_decisions   += (fft_decisions == reference_decisions).sum()
            number_of_decisions += len(fft_decisions)
        
        # Return accuracy as the proportion of decisions that match the reference.
        return correct_decisions / number_of_decisions
    
    def frugality(self, fft_id: str, run_no: int = None) -> float:
        # This method calculates the frugality of the FFT's decisions across one or more runs,
        # defined as the average number of cues used to make a decision.

        # Determine which runs to evaluate.
        if run_no is None:
            runs = range(1, self.runs + 1)
        elif isinstance(run_no, int):
            runs = [run_no]
        else:
            runs = run_no

        total_cues_used = 0
        total_decisions = 0

        for run in runs:

            # Check that the cues_used series exists for this FFT and run.
            if (fft_id, run, "cues_used") not in self.results.columns:
                logger.error(f"Cues used column for FFT '{fft_id}' run {run} not found in results.")
                raise ValueError(f"Cues used column for FFT '{fft_id}' run {run} not found in results.")

            cues_used = self.results[(fft_id, run, "cues_used")]

            total_cues_used += cues_used.sum()
            total_decisions += len(cues_used)

        if total_decisions == 0:
            logger.warning(f"No decisions found for FFT '{fft_id}' in the specified runs.")
            raise ValueError(f"No decisions found for FFT '{fft_id}' in the specified runs.")

        return total_cues_used / total_decisions
    
