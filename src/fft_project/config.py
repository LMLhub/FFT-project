#src/fft_project/config.py
import logging

logger = logging.getLogger(__name__)

def read_dotenv():
    from dotenv import load_dotenv
    import os

    load_dotenv()
    return {
        "REMOTE_DRIVE": os.getenv("REMOTE_DRIVE")
    }

def parse_args():
    import argparse

    parser = argparse.ArgumentParser(description="FFT Project Configuration")
    parser.add_argument("--config", type=str, default="config.yaml", help="Configuration file path")
    parser.add_argument("--loglevel", type=str,
                        default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)")
    parser.add_argument("--run-id", type=str, default=None, help="Unique identifier for this run (defaults to a timestamp)")
    return parser.parse_args()

def read_config_file(config_path):
    import yaml
    logger.info(f"Reading configuration file: {config_path}")
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config

def setup_logging(loglevel="INFO"):
    logging.basicConfig(
        level=getattr(logging, loglevel),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler()
        ]
    )

def setup_run_id(run_id=None):
    import datetime
    if run_id is None:
        run_id = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    logger.info(f"Run ID set to: {run_id}")
    return run_id

def setup_output_folders(remote_drive, run_id,):
    from pathlib import Path
    # Check that remote_drive is not None or empty and log an error if it is
    if not remote_drive:
        logger.error("REMOTE_DRIVE environment variable is not set or is empty.")
        raise ValueError("REMOTE_DRIVE environment variable is not set or is empty.")

    # Check that the path to the remote drive exists before trying to create the output folder
    remote_drive_path = Path(remote_drive)
    if not remote_drive_path.exists():
        logger.error(f"Remote drive path does not exist: {remote_drive}")
        raise FileNotFoundError(f"Remote drive path does not exist: {remote_drive}")

    # Create string for output folder path
    output_folder = f"{remote_drive}/{run_id}"
    logger.info(f"Setting up output folder: {output_folder}")

    # Check if folder already exists
    output_path = Path(output_folder)
    if output_path.exists():
        logger.warning(f"Output folder already exists: {output_folder}")
    else:
        # Create output folder if it doesn't exist
        output_path.mkdir(parents=True, exist_ok=True)
        logger.info(f"Successfully created output folder: {output_folder}")
    # create subfolders
    subfolders = ["1-inputs", "2-generated-data", "3-analysis", "4-visualizations"]
    for subfolder in subfolders:
        subfolder_path = output_path / subfolder
        if not subfolder_path.exists():
            subfolder_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Successfully created subfolder: {subfolder_path}")
        else:
            logger.warning(f"Subfolder already exists: {subfolder_path}")

    return
