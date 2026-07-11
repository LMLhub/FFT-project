from pathlib import Path
import sys
import logging

from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders, save_config
from fft_project.cue_features import expected_isoelastic_utility
from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues2
from fft_project.cue_class import Cue

import matplotlib.pyplot as plt



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
    priority_heuristic_tolerance = 0.1
    gamble_data = simulate_gamble_data(100, config["gamble_simulation"]["fractals_mul"], random_seed = 42)
  elif config["gamble_simulation"]["dynamic"] == "additive":
    dynamic = "additive"
    fractal_values = config["gamble_simulation"]["fractals_add"]
    gamble_data = simulate_gamble_data(100, config["gamble_simulation"]["fractals_add"], random_seed = 42)
    priority_heuristic_tolerance = 0.4
  else:
    raise ValueError(f"Invalid dynamic type: {config['gamble_simulation']['dynamic']}. Must be 'multiplicative' or 'additive'.")

  cue_args = {"dynamic": dynamic,
                 "eta": 0.5,
                 "fractal_values": fractal_values,
                 "n": 2,
                 "tol" : priority_heuristic_tolerance
                 }
  # add a wealth column for testing purposes
  import numpy as np
  gamble_data["wealth"] = np.random.randint(410, 810, size=len(gamble_data))

  print(f"Simulated gamble data with {len(gamble_data)} rows.")
  #print(gamble_data.describe())

  # Set up cues and set up the cue registry
  create_cues2(config)

  for k, cue in Cue.cue_registry.items():
    logging.info(f"Cue ID: {cue.id}, Name: {cue.name}")
    class_column_name = f"{cue.id}_choice"
    random_decision_column_name = f"{cue.id}_random_decision"
    label = {"left" : 1, "right": -1}

    # Iterate over the rows in gamble_data and apply the cue's feature function to each row
    for idx, gamble_row in gamble_data.iterrows():
      #logging.info(f"Row {idx}: columns '{gamble_row.index.tolist()}'")
      g_l_up = gamble_row["gamma_left_up"]
      g_l_down = gamble_row["gamma_left_down"]
      g_r_up = gamble_row["gamma_right_up"]
      g_r_down = gamble_row["gamma_right_down"]
      wealth = gamble_row["wealth"]
      extra_arg_names = cue.required_args[4:] if getattr(cue, "required_args", None) else []
      extra_args = {}
      for name in extra_arg_names:
        if name in gamble_row.index:
          extra_args[name] = gamble_row[name]
        elif name in cue_args.keys():
          extra_args[name] = cue_args[name]
        else:
          raise ValueError(f"Missing required argument '{name}' for cue '{cue.id}' in row {idx}.")

      cue_value, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)
      if side_if_true is not None:
          gamble_data.loc[idx, class_column_name] = label[side_if_true]
          #print(f"Index: {idx}, Cue Value: {cue_value}, Side if True: {side_if_true}")
      else:
          gamble_data.loc[idx, class_column_name] = np.random.choice([1, -1])  # Randomly choose left or right if side_if_true is None
          gamble_data.loc[idx, random_decision_column_name] = True

  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
