import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility, signs
from fft_project.config import read_config_file
import pandas as pd

from fft_project.decision_class import FFT

CONFIG = read_config_file(PROJECT_ROOT / "config.yaml")
GAMBLE_SIMULATION_CONFIG = CONFIG["gamble_simulation"]
FRACTAL_VALUES = GAMBLE_SIMULATION_CONFIG["fractals_add"]
FRACTAL_VALUES_MULTI = GAMBLE_SIMULATION_CONFIG["fractals_mul"]

def create_cues():
    # This script creates cues and saves the cue registry to a yaml file. It can be run once and then deleted.
    # Example cue definition
    
    Cue(
        id="gr",
        name="Maximising time-average growth rate",
        description="This cue compares the growth rates of the gambles and picks the side with the highest time-average growth rate. Works for both additive and multiplicative dynamics.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down"]
    )

    Cue(
        id="eu_1_5_a",
        name="Expected Isoelastic Utility - eta=1.5, additive",
        description="This cue that evaluates the expected isoelastic utility of the first gamble with eta=1.5 and picks a side if the cue value is greater than 2.",
        feature= expected_isoelastic_utility,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive",
                "eta": 1.5},
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth"]
    )

    Cue(
        id="eu_1_5_m",
        name="Expected Isoelastic Utility - eta=1.5, multiplicative",
        description="This cue that evaluates the expected isoelastic utility of the first gamble with eta=1.5 and picks a side if the cue value is greater than 2.",
        feature= expected_isoelastic_utility,
        type="numerical",
        threshold=0,
        params={"dynamic": "multiplicative",
                "eta": 1.5},
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth"]
    )

    Cue(
        id          = "aw_1_a",
        name        = "Avoid worst 1 of all - additive",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "aw_1_m",
        name        = "Avoid worst 1 of all - multiplicative",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES_MULTI},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "aw_2_a",
        name        = "Avoid worst 2 of all - additive",
        description = "Prefers the gamble that does not contain the two worst fractal values.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 2, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "aw_2_m",
        name        = "Avoid worst 2 of all - multiplicative",
        description = "Prefers the gamble that does not contain the two worst fractal values.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 2, "fractal_values": FRACTAL_VALUES_MULTI},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "fs",
        name        = "Positive fractal signs",
        description = "Give preference to the gamble with the most fractals that leads to an increase in wealth",
        feature     = signs,
        type        = "boolean",
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

def create_ffts():
    # This script creates FFTs and saves the FFT registry to a yaml file. It can be run once and then deleted.
    FFT(id="fft_aw_1_a",
        name="Avoid the worst or random - additive",
        description="An example FFT avoid the worst.",
        cues=[Cue.cue_registry["aw_1_a"]])

    FFT(id="fft_aw_1_m",
        name="Avoid the worst - multiplicative",
        description="An example FFT with avoid the worst cue.",
        cues=[Cue.cue_registry["aw_1_m"]])
    
    FFT(id="fft_gr",
        name="Growth rate maximisation",
        description="An example FFT with the growth rate cue.",
        cues=[Cue.cue_registry["gr"]])

    FFT(id="fft_eu_1_5_a",
        name="Expected Isoelastic Utility - eta=1.5, additive",
        description="An example FFT with the expected isoelastic utility cue.",
        cues=[Cue.cue_registry["eu_1_5_a"]])
    
    FFT(id="fft_eu_1_5_m",
        name="Expected Isoelastic Utility - eta=1.5, multiplicative",
        description="An example FFT with the expected isoelastic utility cue.",
        cues=[Cue.cue_registry["eu_1_5_m"]])
    
    FFT(id="fft_aw_2_a",
         name="Avoid the worst - additive",
         description="An example FFT with the avoid the worst cue.",
         cues=[Cue.cue_registry["aw_2_a"]])
    
    FFT(id="fft_aw_2_m",
        name="Avoid the worst - multiplicative",
        description="An example FFT with the avoid the worst cue.",
        cues=[Cue.cue_registry["aw_2_m"]])  
    
    FFT(id="fft_fs",
        name="Positive fractal signs",
        description="An example FFT with the positive fractal signs cue. Works for both dynamics.",
        cues=[Cue.cue_registry["fs"]])
    
    FFT(id="fft_aw_1_fs_a",
        name="Avoid the worst then positive fractal signs - additive",
        description="FFT with the avoid the worst first, then choose based on positive fractal signs. Additive dynamic.",
        cues=[Cue.cue_registry["aw_1_a"], Cue.cue_registry["fs"]])

    FFT(id="fft_fs_aw_1_a",
        name="Positive fractal signs then avoid the worst then p- additive",
        description="FFT with positive fractal signs then the avoid the worst first. Additive dynamic.",
        cues=[ Cue.cue_registry["fs"], Cue.cue_registry["aw_1_a"]])

def create_cues_ffts():
    create_cues()
    create_ffts()
    Cue.save_registry("cue_registry.yaml")
    FFT.save_registry("fft_registry.yaml")
    print("Cues and FFTs created and registries saved.")

if __name__ == "__main__":
    create_cues()
    create_ffts()
    Cue.save_registry("cue_registry.yaml")
    FFT.save_registry("fft_registry.yaml")
    print("Cues and FFTs created and registries saved.")
