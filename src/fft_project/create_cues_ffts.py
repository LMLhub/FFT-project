import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility, signs
import pandas as pd

from fft_project.decision_class import FFT

FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]
FRACTAL_VALUES_MULTI = [-0.850, -0.5395, -0.433, -0.1735, 0.006, 0.1685, 0.369, 0.5535, 0.772]

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
        id          = "fs_m",
        name        = "Positive fractal signs - multiplicative",
        description = "Give preference to the gamble with the most fractals that leads to an increase in wealth - multiplicative",
        feature     = signs,
        type        = "boolean",
        params      = {"dynamic": "multiplicative",},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "fs_a",
        name        = "Positive fractal signs - additive",
        description = "Give preference to the gamble with the most fractals that leads to an increase in wealth - additive",
        feature     = signs,
        type        = "boolean",
        params      = {"dynamic": "additive",},
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
        cues=[Cue.cue_registry["eu_1_5_add"]])
    
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
    
    FFT(id="fft_fs_m",
        name="Positive fractal signs - multiplicative",
        description="An example FFT with the positive fractal signs cue.",
        cues=[Cue.cue_registry["fs_m"]])
    
    FFT(id="fft_fs_a",
        name="Positive fractal signs - additive",
        description="An example FFT with the positive fractal signs cue.",
        cues=[Cue.cue_registry["fs_a"]])
    
if __name__ == "__main__":
    create_cues()
    create_ffts()
    Cue.save_registry("cue_registry.yaml")
    FFT.save_registry("fft_registry.yaml")
    print("Cues and FFTs created and registries saved.")
