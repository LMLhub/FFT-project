#!/usr/bin/env python3
"""
FFT Project - Workstream 1 Main Script
"""
import sys
from dotenv import load_dotenv
import os

def main():
  """
  Main function for workstream 1 of the FFT project.
  """
  print("Starting Workstream 1 of the FFT Project...")
  # Load environment variables from .env file
  load_dotenv()
  # Example: Access an environment variable
  REMOTE_DRIVE = os.getenv("REMOTE_DRIVE")
  print(f"Local location of shared remote drive: {REMOTE_DRIVE}")
  return 0

if __name__ == "__main__":
  returncode = main()
  sys.exit(returncode)
