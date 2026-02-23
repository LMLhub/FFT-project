#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
import sys
import logging
from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging
from fft_project.config import setup_run_id, setup_output_folders
from fft_project.simulation_gamble_data import simulate_gamble_data

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

  # Save the config file to the output folder for record-keeping

  # Simulate gamble data
  gamble_data_folder_path = f"{REMOTE_DRIVE}/{config['run_id']}/2-generated-data"
  df = simulate_gamble_data(config["gamble_simulation"]["f"],
                            config["gamble_simulation"]["exclude_nobrainer"],
                            config["gamble_simulation"]["mirror_gambles"],
                            gamble_data_folder_path)
  logging.info(f"Simulated gamble data with {len(df)} rows.")

  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
