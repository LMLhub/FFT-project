import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fft_project.cue_class import Cue
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility, signs, prefer_best_n_ranks, priority_step1, priority_step3, priority_step1_no_loss, priority_step3_no_loss, growth_rate_min
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
        id="gr_min",
        name="Minimising time-average growth rate",
        description="This cue compares the growth rates of the gambles and picks the side with the lowest time-average growth rate. Works for both additive and multiplicative dynamics.",
        feature= growth_rate_min,
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

    Cue(
        id          = "pb_1_a",
        name        = "Prefer best 1 of all - additive",
        description = "Give preference to the gamble that has the best of all fractals",
        feature     = prefer_best_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pb_1_m",
        name        = "Prefer best 1 of all - multiplicative",
        description = "Give preference to the gamble that has the best of all fractals",
        feature     = prefer_best_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES_MULTI},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_1_50_a",
        name        = "Minimum gains difference 50 percent (additive)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 50 percent of the maximum gain",
        feature     = priority_step1,
        type        = "boolean",
        params      = {"tol": 0.5, "dynamic": "additive"}, #0.5 seems to be a good value
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_3_a",
        name        = "Maximum gains (additive)",
        description = "Give preference to the gamble with the highest maximum gains.",
        feature     = priority_step3,
        type        = "boolean",
        params      = {"dynamic": "additive"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_1_35_m",
        name        = "Minimum gains difference 35 percent (multiplicative)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 35 percent of the maximum gain",
        feature     = priority_step1,
        type        = "boolean",
        params      = {"tol": 0.35, "dynamic": "multiplicative"},#0.35 seems to be a good value
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_3_m",
        name        = "Maximum gains (multiplicative)",
        description = "Give preference to the gamble with the highest maximum gains.",
        feature     = priority_step3,
        type        = "boolean",
        params      = {"dynamic": "multiplicative"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_nl_1_40_a",
        name        = "Variation No Loss of priority heuristic step 1 with 40 percent tolerance(additive)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 40 percent of the maximum gain. Before evaluating, all outcomes are moved such that they are positive",
        feature     = priority_step1_no_loss,
        type        = "boolean",
        params      = {"tol": 0.4, "dynamic": "additive"}, #0.4 seem to be a good value
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_nl_3_a",
        name        = "Variation No Loss of step 3 of the priority heuristic (additive)",
        description = "Give preference to the gamble with the highest maximum gains. Before evaluating, all outcomes are moved such that they are positive",
        feature     = priority_step3_no_loss,
        type        = "boolean",
        params      = {"dynamic": "additive"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_nl_1_20_m",
        name        = "Variation No Loss of priority heuristic step 1 with 20 percent tolerance (multiplicative)",
        description = "Give preference to the gamble with the highest minimum gains if the minimum gains differs by 20 percent of the maximum gain. Before evaluating, all outcomes are moved such that they are positive",
        feature     = priority_step1_no_loss,
        type        = "boolean",
        params      = {"tol": 0.2, "dynamic": "multiplicative"}, #0.2 seems to be a good value
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "pri_nl_3_m",
        name        = "Variation No Loss of step 3 of the priority heuristic (multiplicative)",
        description = "Give preference to the gamble with the highest maximum gains. Before evaluating, all outcomes are moved such that they are positive",
        feature     = priority_step3_no_loss,
        type        = "boolean",
        params      = {"dynamic": "multiplicative"},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

def create_ffts():
    # This script creates FFTs and saves the FFT registry to a yaml file. It can be run once and then deleted.
    FFT(id="fft_aw_1_a",
        name="Avoid the worst",
        description="An example FFT avoid the worst - additive.",
        cues=[Cue.cue_registry["aw_1_a"]])

    FFT(id="fft_aw_1_m",
        name="Avoid the worst",
        description="An example FFT with avoid the worst cue - multiplicative.",
        cues=[Cue.cue_registry["aw_1_m"]])
    
    FFT(id="fft_gr",
        name="Growth rate maximisation",
        description="FFT with the growth rate cue",
        cues=[Cue.cue_registry["gr"]])

    FFT(id="fft_eu_1_5_a",
        name="Expected Isoelastic Utility - eta=1.5",
        description="An example FFT with the expected isoelastic utility cue - additive.",
        cues=[Cue.cue_registry["eu_1_5_a"]])
    
    FFT(id="fft_eu_1_5_m",
        name="Expected Isoelastic Utility - eta=1.5",
        description="An example FFT with the expected isoelastic utility cue - multiplicative.",
        cues=[Cue.cue_registry["eu_1_5_m"]])
    
    FFT(id="fft_aw_2_a",
         name="Avoid the two worst",
         description="An example FFT with the avoid the worst cue.",
         cues=[Cue.cue_registry["aw_2_a"]])
    
    FFT(id="fft_aw_2_m",
        name="Avoid the two worst",
        description="An example FFT with the avoid the worst cue - multiplicative",
        cues=[Cue.cue_registry["aw_2_m"]])  
    
    FFT(id="fft_fs",
        name="Positive fractal signs",
        description="FFT with the positive fractal signs cue. Works for both dynamics.",
        cues=[Cue.cue_registry["fs"]])
    
    FFT(id="fft_aw_1_fs_a",
        name="Avoid the worst then positive fractal signs",
        description="FFT with the avoid the worst first, then choose based on positive fractal signs. Additive dynamic.",
        cues=[Cue.cue_registry["aw_1_a"], Cue.cue_registry["fs"]])

    FFT(id="fft_fs_aw_1_a",
        name="Positive fractal signs then avoid the worst",
        description="FFT with positive fractal signs then the avoid the worst first. Additive dynamic.",
        cues=[ Cue.cue_registry["fs"], Cue.cue_registry["aw_1_a"]])
    
    FFT(id="fft_pb_1_a",
        name="Prefer the best",
        description="FFT that prefers the best if present. Additive dynamic.",
        cues=[ Cue.cue_registry["pb_1_a"]])
   
    FFT(id="fft_aw_1_pb_1_a",
        name="Avoid the worst then prefer the best",
        description="FFT with avoid the worst then prefer the best. Additive dynamic.",
        cues=[ Cue.cue_registry["aw_1_a"], Cue.cue_registry["pb_1_a"]])
    
    FFT(id="fft_pb_1_aw_1_a",
        name="Avoid the worst then prefer the best",
        description="FFT with avoid the worst then prefer the best. Additive dynamic.",
        cues=[ Cue.cue_registry["pb_1_a"], Cue.cue_registry["aw_1_a"]])

    FFT(id="fft_pb_1_m",
        name="Prefer the best",
        description="FFT that prefers the best if present. Additive dynamic.",
        cues=[ Cue.cue_registry["pb_1_m"]])
   
    FFT(id="fft_aw_1_pb_1_m",
        name="Avoid the worst then prefer the best",
        description="FFT with avoid the worst then prefer the best. Additive dynamic.",
        cues=[ Cue.cue_registry["aw_1_m"], Cue.cue_registry["pb_1_m"]])
    
    FFT(id="fft_pb_1_aw_1_m",
        name="Avoid the worst then prefer the best",
        description="FFT with avoid the worst then prefer the best. Additive dynamic.",
        cues=[ Cue.cue_registry["pb_1_m"], Cue.cue_registry["aw_1_m"]])

    FFT(id="fft_fs_aw_1_m",
        name="Positive fractal signs then avoid the worst",
        description="FFT with positive fractal signs then the avoid the worst first. Additive dynamic.",
        cues=[ Cue.cue_registry["fs"], Cue.cue_registry["aw_1_m"]])
    
    FFT(id="pri_a",
        name="Priority heuristic",
        description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006). For additive dynamics",
        cues=[ Cue.cue_registry["pri_1_50_a"], Cue.cue_registry["pri_3_a"]])

    FFT(id="pri_m",
        name="Priority heuristic",
        description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006) but with different tolerance. For multiplicative dynamics",
        cues=[ Cue.cue_registry["pri_1_35_m"], Cue.cue_registry["pri_3_m"]])
    
    FFT(id="pri_nl_a",
        name="Priority heuristic (no loss)",
        description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006) without the loss option and with different tolerance. For additive dynamics",
        cues=[ Cue.cue_registry["pri_nl_1_40_a"], Cue.cue_registry["pri_nl_3_a"]])

    FFT(id="pri_nl_m",
        name="Priority heuristic (no loss)",
        description="The priority heuristic as described by Brandstätter, Gigerenzer, and Hertwig (2006) without the loss option and with different tolerance. For multiplicative dynamics",
        cues=[ Cue.cue_registry["pri_nl_1_20_m"], Cue.cue_registry["pri_nl_3_m"]])

    FFT(id="fft_gr_min",
        name="Growth rate minimisation",
        description="This fft chooses the opposite of the growth rate maximisation",
        cues=[ Cue.cue_registry["gr_min"]])


    
def create_cues_ffts(filepath=None):
    create_cues()
    create_ffts()

    if filepath is None:
        registry_dir = Path(".")
    else:
        registry_dir = Path(filepath)

    registry_dir.mkdir(parents=True, exist_ok=True)

    Cue.save_registry(registry_dir / "cue_registry.yaml")
    FFT.save_registry(registry_dir / "fft_registry.yaml")
    print(f"Cues and FFTs created and registries saved to {registry_dir}.")

if __name__ == "__main__":
    create_cues_ffts()
