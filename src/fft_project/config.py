#src/fft_project/config.py

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
    return parser.parse_args()

def read_config_file(config_path):
    import yaml
    with open(config_path, 'r') as file:
        config = yaml.safe_load(file)
    return config
