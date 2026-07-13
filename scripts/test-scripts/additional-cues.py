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
from fft_project.utilities import confusion_rates, Gini_impurity_partition


import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)

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

  # Choose preferred gamble using each cue in isolation. When we use a single
  # cue in isolation, it might be absent for a given pair of gambles or might
  # not be above threshold. When this happens, preference is assigned randomly.
  # For each cue, we add two columns to the gamble_data DataFrame:
  #  -  cue.id + "_choice" : the choice made for each gamble if only the current
  #                          cue in isolation is used
  #  -  cue.id+"_random".  : whether the is choice is random or not

  for (k, cue) in all_cues.items():
    logging.info(f"Cue ID: {cue.id}, Name: {cue.name}")
    # Add new columns to gamble_data to contain information about choices made
    # using current cue
    class_column_name = f"{cue.id}_choice"
    gamble_data[class_column_name]=0
    random_decision_column_name = f"{cue.id}_random"
    gamble_data[random_decision_column_name] = False
    label = {"left" : 1, "right": -1}

    # Iterate over the rows in gamble_data and evaluate the current cue for
    # the gamble pair contained in each row.
    for idx, gamble_row in gamble_data.iterrows():
      # Get the gamble parameters
      g_l_up = gamble_row["gamma_left_up"]
      g_l_down = gamble_row["gamma_left_down"]
      g_r_up = gamble_row["gamma_right_up"]
      g_r_down = gamble_row["gamma_right_down"]
      wealth = gamble_row["wealth"]
      # Read what additional arguments the current cue requires
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

      # Now we have everything we need to evaluate the current cue on the current
      # gamble
      _, side_if_true = cue.evaluate(g_l_up, g_l_down, g_r_up, g_r_down, **extra_args)

      # Decide which side is preferred and record the random choices for later
      # evaluation
      if side_if_true is not None:
          gamble_data.loc[idx, class_column_name] = label[side_if_true]
      else:
          gamble_data.loc[idx, class_column_name] = np.random.choice([1, -1])  # Randomly choose left or right if side_if_true is None
          gamble_data.loc[idx, random_decision_column_name] = True

  # Now measure the performance of the cues
  evaluation = pd.DataFrame(index=all_cues.keys(), columns = ["TPR", "FPR", "non-random", "purity" ])
  for (id, cue) in all_cues.items():
    print(f"Evaluating cue {id}")
    # Proportion of non-random choices
    nonrandom_choices = len(gamble_data[gamble_data[f"{id}_random"] == False])
    total_choices = len(gamble_data)
    evaluation.loc[id, "non-random"] = nonrandom_choices/total_choices

    # TPR and FPR
    TPR, FPR, _, _ = confusion_rates(gamble_data, f"{id}_choice","GR-max_choice", 1, verbose=False)
    evaluation.loc[id, "TPR"] = TPR
    evaluation.loc[id, "FPR"] = FPR

    # Gini purity of instances that were not classified randomly
    nonrandom_df = gamble_data[gamble_data[f"{id}_random"] == False]
    Gini_purity = 1 - Gini_impurity_partition(nonrandom_df, f"{id}_choice", "GR-max_choice", [1, -1], verbose=False)
    evaluation.loc[id, "purity"] = Gini_purity

  print(evaluation)

  # Select the cues which have TPR > FPR
  predictive_cues = evaluation[evaluation["TPR"] > evaluation["FPR"]]
  # Drop  cues that are duplicates
  predictive_cues = predictive_cues.drop(index=["GR-min-minus", "AW-a-minus", "PB-a"])
  print(f"Predictive cues (TPR > FPR): {len(predictive_cues)}")
  # Sort the predictive cues by TPR
  predictive_cues = predictive_cues.sort_values(by="TPR", ascending=False)
  print(predictive_cues)

  # Create a plot of TPR vs FPR for the predictive cues
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(predictive_cues["FPR"], predictive_cues["TPR"], color='red', label='Predictive Cues')
  ax.plot([0, 1], [0, 1], linestyle='--', color='gray', label='TPR = FPR')
  # Label each point with the cue id (DataFrame index)
  for cue_id, row in predictive_cues.iterrows():
    ax.annotate(
        str(cue_id),
        (row["FPR"], row["TPR"]),
        textcoords="offset points",
        xytext=(5, 5),   # slight offset so text is not on top of marker
        fontsize=8
    )
  ax.set_xlim(0, 1)
  ax.set_ylim(0, 1)
  ax.set_xlabel("FPR")
  ax.set_ylabel("TPR")
  ax.set_title("TPR vs FPR for Cues")
  ax.legend()
  #plt.show()  # This remains the same as it is a global function to display the plot
  # save plot to output folder
  fig.savefig(Path(REMOTE_DRIVE) / Path(f"{config['run_id']}/4-visualizations/TPR_vs_FPR_{dynamic}.png"))

  # Plot Gini purity vs non-random proportion for the predictive cues
  fig, ax = plt.subplots(figsize=(6, 6))
  ax.scatter(predictive_cues["non-random"], predictive_cues["purity"], color='blue', label='Predictive Cues')
  # Label each point with the cue id (DataFrame index)
  for cue_id, row in predictive_cues.iterrows():
    ax.annotate(
        str(cue_id),
        (row["non-random"], row["purity"]),
        textcoords="offset points",
        xytext=(5, 5),   # slight offset so text is not on top of marker
        fontsize=8
    )
  ax.set_xlim(0, 1.25)
  ax.set_ylim(0, 1.25)
  ax.set_xlabel("Proportion of Non-Random Choices")
  ax.set_ylabel("Gini Purity")
  ax.set_title("Gini Purity vs Non-Random Proportion for Cues")
  ax.legend()
  fig.savefig(Path(REMOTE_DRIVE) / Path(f"{config['run_id']}/4-visualizations/Gini_purity_vs_nonrandom_{dynamic}.png"))
  #plt.show()  # This remains the same as it is a global function to display the

  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
