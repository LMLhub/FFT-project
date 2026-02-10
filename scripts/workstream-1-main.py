#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
import sys
import logging
from fft_project.config import read_dotenv, parse_args, read_config_file, setup_logging



def main():
  """
  Main function for workstream 1 of the FFT project.
  """



  # Parse command-line arguments
  args = parse_args()
  setup_logging(args.loglevel)
  logging.info("Starting Workstream 1 of the FFT Project...")
  logging.info(f"Using configuration file: {args.config}")

 # Load environment variables from .env file
  REMOTE_DRIVE = read_dotenv()["REMOTE_DRIVE"]
  logging.info(f"Local location of shared remote drive: {REMOTE_DRIVE}")

  # Read config file
  config = read_config_file(args.config)
  logging.info(f"Configuration loaded: {config}")
  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
