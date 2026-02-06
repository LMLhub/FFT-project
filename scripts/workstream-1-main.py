#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
from fft_project.config import read_dotenv, parse_args
import sys


def main():
  """
  Main function for workstream 1 of the FFT project.
  """
  print("Starting Workstream 1 of the FFT Project...")
  # Load environment variables from .env file
  REMOTE_DRIVE = read_dotenv()

  print(f"Local location of shared remote drive: {REMOTE_DRIVE}")
  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
