import sys
from pathlib import Path
import sys
import logging
import copy
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders, save_config
from fft_project.cue_features import expected_isoelastic_utility
from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues2
from fft_project.cue_class import Cue
from fft_project.sequential_decision_rules.data import Dataset


logger = logging.getLogger(__name__)


def main():
  print("Running sequential decision rules test script")
  args = parse_args()
  # Set up logging based on the provided log level
  setup_logging(loglevel=args.loglevel)
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
  N=config["gamble_simulation"]["n"]
  if config["gamble_simulation"]["dynamic"] == "multiplicative":
    dynamic = "multiplicative"
    fractal_values = config["gamble_simulation"]["fractals_mul"]
    priority_heuristic_tolerance = 0.35
    priority_heuristic_tolerance_no_loss = 0.2
    gamble_data = simulate_gamble_data(N, config["gamble_simulation"]["fractals_mul"], random_seed = 42)
  elif config["gamble_simulation"]["dynamic"] == "additive":
    dynamic = "additive"
    fractal_values = config["gamble_simulation"]["fractals_add"]
    gamble_data = simulate_gamble_data(N, config["gamble_simulation"]["fractals_add"], random_seed = 42)
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

  # Create prediction matrix and labels
  cue_labels=list(all_cues.keys())
  # Remove GR-max from the cues to be considered since this will be used to
  # generate the labels
  cue_labels.remove("GR-max")
  N = len(gamble_data)
  M = len(cue_labels)
  logger.info(f"Number of training instances = {N}")
  logger.info(f"Number of candidate cues = {M}")
  Y = np.zeros(N, dtype=int)
  X = np.zeros((N,M), dtype=int)
  for i in range(0,M):
    logger.info(f"i = {i}, cue = {cue_labels[i]}, {Y[i]}")

  for i, row in gamble_data.iterrows():
    # Get the gamble parameters
    g_l_up = row["gamma_left_up"]
    g_l_down = row["gamma_left_down"]
    g_r_up = row["gamma_right_up"]
    g_r_down = row["gamma_right_down"]
    wealth = row["wealth"]
    cue = all_cues["GR-max"]
    extra_args={}
    val, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)
    if side_if_true == "left":
      Y[i]=1
    elif side_if_true == "right":
      Y[i] = -1
    else:
      logger.error(f"Reference cue {cue.id} can't fail to assign a choice")
      exit()
  print(Y)
  return 0


if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
