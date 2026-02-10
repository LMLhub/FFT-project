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
