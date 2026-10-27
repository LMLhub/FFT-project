from pathlib import Path
import sys
import logging
import copy
import numpy as np
import pandas as pd


from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders, save_config
from fft_project.cue_features import expected_isoelastic_utility
from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues2
from fft_project.cue_class import Cue
from fft_project.utilities import confusion_rates, classification_accuracy, Gini_impurity_partition


import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

def decide_using_single_cue(gamble_data, cue, cue_args):
  df = gamble_data.copy()
  df["choice"] = 1
  df["random_decision"] = False

  for idx, gamble_row in gamble_data.iterrows():
      # Get the gamble parameters
      g_l_up = gamble_row["gamma_left_up"]
      g_l_down = gamble_row["gamma_left_down"]
      g_r_up = gamble_row["gamma_right_up"]
      g_r_down = gamble_row["gamma_right_down"]
      wealth = gamble_row["wealth"]

      extra_arg_names = cue.required_args[4:] if getattr(cue, "required_args", None) else []
      # Check that all the required additional arguments are defined (either
      # as columns in gamble_data or in the cue_args dict) and assemble them
      # the extra_args dict.
      extra_args = {}
      for name in extra_arg_names:
        if name in gamble_row.index:
          extra_args[name] = gamble_row[name]
        elif name in cue_args.keys():
          extra_args[name] = cue_args[name]
        else:
          raise ValueError(f"Missing required argument '{name}' for cue '{cue.id}' in row {idx}.")

      _, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)
      # Decide which side is preferred and record the random choices for later
      # evaluation
      label = {"left" : 1, "right": -1}
      if side_if_true is not None:
          df.loc[idx, "choice"] = label[side_if_true]
      else:
          df.loc[idx, "choice"] = np.random.choice([1, -1])  # Randomly choose left or right if side_if_true is None
          df.loc[idx, "random_decision"] = True

  return df

def find_non_dominated_cues():
  return 0

def calculate_potential_splits():
  return 0

def main():
  # Parse command-line arguments
  args = parse_args()
  # Set up logging based on the provided log level
  setup_logging(loglevel=args.loglevel)
  logging.info("Starting Workstream 1 of the FFT Project...")
  logging.info(f"Using configuration file: {args.config}")

 # Load environment variables from .env file
  REMOTE_DRIVE = read_dotenv()["REMOTE_DRIVE"]
  logging.info(f"Local location of shared remote drive: {REMOTE_DRIVE}")

  # Read config file
  config = read_config_file(args.config)
  logging.info(f"Configuration loaded: {config}")

  # Set up run ID
  config["run_id"] = setup_run_id(args.run_id)

  # Set up output folders
  setup_output_folders(REMOTE_DRIVE, config["run_id"])

  # Save config file to the run's input folder
  save_config(args.config, REMOTE_DRIVE, config["run_id"])


  # Simulate gamble data
  if config["gamble_simulation"]["dynamic"] == "multiplicative":
    dynamic = "multiplicative"
    fractal_values = config["gamble_simulation"]["fractals_mul"]
    priority_heuristic_tolerance = 0.35
    priority_heuristic_tolerance_no_loss = 0.2
    gamble_data = simulate_gamble_data(100, config["gamble_simulation"]["fractals_mul"], random_seed = 42)
  elif config["gamble_simulation"]["dynamic"] == "additive":
    dynamic = "additive"
    fractal_values = config["gamble_simulation"]["fractals_add"]
    gamble_data = simulate_gamble_data(100, config["gamble_simulation"]["fractals_add"], random_seed = 42)
    priority_heuristic_tolerance = 0.5
    priority_heuristic_tolerance_no_loss = 0.4
  else:
    raise ValueError(f"Invalid dynamic type: {config['gamble_simulation']['dynamic']}. Must be 'multiplicative' or 'additive'.")
   # add a wealth column to gamble_data for testing purposes
  gamble_data["wealth"] = np.random.randint(410, 810, size=len(gamble_data))


# This is where all the additional arguments required by some Cue objects
# are set
  cue_args = {"dynamic": dynamic,
              "eta": 0.5,
              "fractal_values": fractal_values,
              "n": 2,
              "tol" : priority_heuristic_tolerance
              }



  print(f"Simulated gamble data with {len(gamble_data)} rows.")
  #print(gamble_data.describe())

  # Set up cues and set up the cue registry
  create_cues2(config)

  # Copy the cue registry to a dict
  all_cues = Cue.cue_registry


  # Add column with reference choices :  e.g. GR-max or experimental
  df = decide_using_single_cue(gamble_data, all_cues["GR-max"], cue_args)
  gamble_data["Reference-choice"] = df["choice"]



  # Create the complement of each cue and add it to the list for e
  # subsequent evaluation
  complement_cues={}
  for (id, cue) in all_cues.items():
    new_id = id+"-minus"
    complement_cues[new_id] = copy.deepcopy(cue)
    complement_cues[new_id].id = new_id
    complement_cues[new_id].direction = - cue.direction
  for (id, cue) in complement_cues.items():
    all_cues[id] = cue

  # Identify cues that are better than random and only keep those that are
  for (k, cue) in all_cues.items():
    logging.info(f"Cue ID: {cue.id}, Name: {cue.name}")
    if cue.id in ["PH-1","PH-3"]:
      cue_args["tol"] = priority_heuristic_tolerance
    if cue.id in ["PH-1-NL", "PH-3-NL"]:
      cue_args["tol"] = priority_heuristic_tolerance_no_loss
    df = decide_using_single_cue(gamble_data, cue, cue_args)
    print(df.columns)
    TPR, FPR, _, _ = confusion_rates(df, "choice","Reference-choice", 1, verbose=False)
    logger.info(f"{cue.id} {TPR} {FPR}")
  exit()


if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
