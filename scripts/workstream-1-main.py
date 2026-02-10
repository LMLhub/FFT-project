#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
from fft_project.config import read_dotenv, parse_args, read_config_file
import sys


def main():
  """
  Main function for workstream 1 of the FFT project.
  """
  print("Starting Workstream 1 of the FFT Project...")
  # Load environment variables from .env file
  REMOTE_DRIVE = read_dotenv()["REMOTE_DRIVE"]

  print(f"Local location of shared remote drive: {REMOTE_DRIVE}")

  # Parse command-line arguments
  args = parse_args()
  print(f"Using configuration file: {args.config}")

  # Read config file
  config = read_config_file(args.config)
  print(f"Configuration loaded: {config}")

  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
