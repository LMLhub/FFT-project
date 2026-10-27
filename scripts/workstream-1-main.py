#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
import sys
import logging
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders, save_config
from fft_project.simulation_gamble_data import simulate_gamble_data

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment

from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues_ffts
from fft_project.prepare_experimental_data import prepare_experimental_data
from fft_project.analysis_compare import plot_etas_compare, plot_accuracy_gamma_scatter

def main():
  """
  Main function for workstream 1 of the FFT project.
  """

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

  # Load gamble data
  GAMBLE_SIMULATION_CONFIG = config["gamble_simulation"]
  FRACTAL_VALUES = GAMBLE_SIMULATION_CONFIG["fractals_add"]
  FRACTAL_VALUES_MULTI = GAMBLE_SIMULATION_CONFIG["fractals_mul"]

  # Select decision rules to be explored
  FFT_NAMES_A = ["fft_gr", "fft_aw_1_a", "fft_aw_2_a", "fft_pb_1_a","fft_pb_1_aw_1_a", "fft_fs", "fft_fs_aw_1_a", "pri_a", "pri_nl_a", "fft-ph-rank" ]
  FFT_NAMES_M = ["fft_gr", "fft_aw_1_m", "fft_aw_2_m", "fft_pb_1_m","fft_pb_1_aw_1_m", "fft_fs", "fft_fs_aw_1_m", "pri_m", "pri_nl_m", "fft-ph-rank" ]

  def prepare_data_from_experiment():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "FFT-project/data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results
    experimental_results_additive.reset_index()
    experimental_results_multiplicative.reset_index()
    
    gamble_data_additive["fractal_values"] = [FRACTAL_VALUES.copy() for _ in range(len(gamble_data_additive))]
    gamble_data_additive["tol_relative"] = 2
    gamble_data_additive["dynamic"] = "additive"
        
    gamble_data_multiplicative["fractal_values"] = [FRACTAL_VALUES_MULTI.copy() for _ in range(len(gamble_data_multiplicative))]
    gamble_data_multiplicative["tol_relative"] = 2
    gamble_data_multiplicative["dynamic"] = "multiplicative"

    return (gamble_data_additive, gamble_data_multiplicative), (experimental_results_additive, experimental_results_multiplicative)

  def generate_simulated_data(config, REMOTE_DRIVE):
    generated_data_folder_path = f"{REMOTE_DRIVE}/{config['run_id']}/2-generated-data"
    df = simulate_gamble_data(config["gamble_simulation"]["n"],
                            config["gamble_simulation"]["fractals_mul"],
                            config["gamble_simulation"]["exclude_nobrainer"],
                            config["gamble_simulation"]["mirror_gambles"],
                            config["gamble_simulation"]["random_draw"],
                            generated_data_folder_path,
                            config["gamble_simulation"]["random_seed"])
    logging.info(f"Simulated gamble data with {len(df)} rows.")
    
    return df
  
  def create_and_run_additive_experiment(name):
    # Additive
    fft_names_a = FFT_NAMES_A.copy()
    experiment_a = Experiment(id=name,
                          name="Example Experiment",
                          dynamic="additive",
                          description="Experiment using the experimental gamble data.",
                          gamble_data=gamble_data_additive,
                          ffts=[FFT.FFT_registry[name] for name in fft_names_a]
                          )
    results_a = experiment_a.run_experiment(wealth_update="data", random_seed=42)  
    fft_names_a.append("experiment_a")
    results_a = pd.concat([results_a, experimental_results_additive], axis=1)  
    return(fft_names_a, results_a, experiment_a)

  def create_and_run_multiplicative_experiment(name):
    #Multiplicative
    fft_names_m = FFT_NAMES_M.copy()
    experiment_m = Experiment(id=name,
                          name="Example Experiment",
                          dynamic="multiplicative",
                          description="An example experiment using the simulated gamble data and the example FFT.",
                          gamble_data=gamble_data_multiplicative,
                          ffts=[FFT.FFT_registry[name] for name in fft_names_m]
                          )
  
    results_m = experiment_m.run_experiment(wealth_update="data", random_seed=42)  
    fft_names_m.append("experiment_m")
    results_m = pd.concat([results_m, experimental_results_multiplicative], axis=1)
    
    return(fft_names_m, results_m, experiment_m)
  
  ###########################################

  # Simulate gamble data
  # df = generate_simulated_data(config, REMOTE_DRIVE)

  # Create all cues and ffts
  create_cues_ffts(f"{REMOTE_DRIVE}/{config['run_id']}")

  # Prepare data from experiment
  gamble_data, experimental_results = prepare_data_from_experiment()
  gamble_data_additive, gamble_data_multiplicative = gamble_data
  experimental_results_additive, experimental_results_multiplicative = experimental_results

  #---------- Set up and run experiments ----------------------------
  fft_names_a, results_a, experiment_a = create_and_run_additive_experiment("exp-a")
  fft_names_m, results_m, experiment_m = create_and_run_multiplicative_experiment("exp-m")
    
  #----------------- Fig 4 --------------------------
  
  # This code creates the eta profile plot

  fig, ax = plt.subplots(1,2, figsize=(10,4))  
  etas = np.linspace(-1,5,31)
    
  plot_etas_compare(
      gamble_data_additive,
      results_a,
      fft_names_a,
      etas,
      "additive",
      ax=ax[0],
      runs=1,
      random_seed=42,
      )
  
  plot_etas_compare(
      gamble_data_multiplicative,
      results_m,
      fft_names_m,
      etas,
      "multiplicative",
      ax=ax[1],
      runs=1,
      random_seed=42,
      )
    
  fig.savefig(f"{REMOTE_DRIVE}/{config['run_id']}/4-visualizations/fig4.png", dpi=300, bbox_inches="tight")
  
  return 0



if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
