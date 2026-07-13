import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import priority_step1

#from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues_ffts
from fft_project.prepare_experimental_data import prepare_experimental_data
from fft_project.analysis_compare import plot_etas_compare, plot_accuracy_gamma_scatter


from fft_project.config import read_config_file
CONFIG = read_config_file(PROJECT_ROOT / "config.yaml")
GAMBLE_SIMULATION_CONFIG = CONFIG["gamble_simulation"]
FRACTAL_VALUES = GAMBLE_SIMULATION_CONFIG["fractals_add"]
FRACTAL_VALUES_MULTI = GAMBLE_SIMULATION_CONFIG["fractals_mul"]

FFT_NAMES_A = ["fft_gr", "fft_aw_1_a", "fft_aw_2_a", "fft_pb_1_a","fft_pb_1_aw_1_a", "fft_fs", "fft_fs_aw_1_a", "pri_a", "pri_nl_a" ]
FFT_NAMES_M = ["fft_gr", "fft_aw_1_m", "fft_aw_2_m", "fft_pb_1_m","fft_pb_1_aw_1_m", "fft_fs", "fft_fs_aw_1_m", "pri_m", "pri_nl_m"]

def initialise():
    create_cues_ffts()


def ph_with_different_tolerances():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results
    experimental_results_additive.reset_index()
    
    cue1 = Cue(
        id="ph_tolerance",
        name="Step 1 of the priority heuristic with variable tolerance",
        description="This cue evaluates the first step of the priority heuristic with the tolerance as an input argument.",
        type = "boolean",
        feature = priority_step1,
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "tol", "dynamic"],
    )
    cue2 = Cue.cue_registry["pri_3_a"]

    PH_tol = FFT(
        id="ph1",
        name="Priority Heuristic Step 1",
        description="First step of the priority heuristic with variable tolerance",
        cues=[cue1, cue2],
    )

    print("Cue and fft created successfully.")
    
    gamble_data_additive["dynamic"] = "additive"
    gamble_data_additive["tol"] = 0.1
    
    experiment_a = Experiment(id="exp_ph_tolerance_a",
                            name="Priority Heuristic Tolerance Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_additive,
                            ffts=[PH_tol])
    
    experiment_gr = Experiment(id="exp_ph_tolerance_gr_a",
                            name="Growth Rate Experiment",
                            dynamic="additive",
                            description="this experiment evaluates the growth rate optimal choices for the experimental gamble data.",
                            gamble_data=gamble_data_additive,
                            ffts=[FFT.FFT_registry["fft_gr"]])
    
    results_gr = experiment_gr.run_experiment(wealth_update="data", random_seed=42)
    experiment_gr.results = pd.concat([results_gr, experimental_results_additive], axis=1)

    experiment_a.results = experiment_gr.results.copy()  # Initialize with the growth rate results

    tolerances = np.arange(0.2, 0.9, 0.05)  # Tolerances from 0.1 to 1.0 in steps of 0.1
    
    accuracies = []
    accuracy_grs = []

    for i, tol in enumerate(tolerances):
        #Set the tolerance for the current iteration
        gamble_data_additive["tol"] = tol

        #update the gamble data in the experiment and run it
        experiment_a.gamble_data = gamble_data_additive
        experiment_a.run_experiment(wealth_update="data", random_seed=42)

        print(i, "Tolerance:", tol)
        accuracy = experiment_a.accuracy("ph1", "experiment_a", i+1, 1)
        accuracy_gr = experiment_a.accuracy("ph1", "fft_gr", i+1, 1)
        
        print(f"Accuracy (experiment) for tolerance {tol}:", accuracy)
        print(f"Accuracy (growth rate)for tolerance {tol}:", accuracy_gr)
        accuracies.append(accuracy)
        accuracy_grs.append(accuracy_gr)


    gamble_data_multiplicative["dynamic"] = "multiplicative"
    gamble_data_multiplicative["tol"] = 0.1
    
    experiment_m = Experiment(id="exp_ph_tolerance_m",
                            name="Priority Heuristic Tolerance Experiment",
                            dynamic="multiplicative",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_multiplicative,
                            ffts=[PH_tol])
    
    experiment_gr_m = Experiment(id="exp_ph_tolerance_gr_m",
                            name="Growth Rate Experiment",
                            dynamic="multiplicative",
                            description="This experiment evaluates the growth rate optimal choices for the experimental gamble data.",
                            gamble_data=gamble_data_multiplicative,
                            ffts=[FFT.FFT_registry["fft_gr"]])
    
    results_gr = experiment_gr_m.run_experiment(wealth_update="data", random_seed=42)
    experiment_gr_m.results = pd.concat([results_gr, experimental_results_multiplicative], axis=1)

    experiment_m.results = experiment_gr_m.results.copy()  # Initialize with the growth rate results

    tolerances_m = np.arange(0.05, 0.8, 0.05)  # Tolerances from 0.1 to 1.0 in steps of 0.1
    
    accuracies_m = []
    accuracy_grs_m = []

    for i, tol in enumerate(tolerances_m):
        #Set the tolerance for the current iteration
        gamble_data_multiplicative["tol"] = tol

        #update the gamble data in the experiment and run it
        experiment_m.gamble_data = gamble_data_multiplicative
        experiment_m.run_experiment(wealth_update="data", random_seed=42)

        print(i, "Tolerance:", tol)
        accuracy_m = experiment_m.accuracy("ph1", "experiment_m", i+1, 1)
        accuracy_gr_m = experiment_m.accuracy("ph1", "fft_gr", i+1, 1)
        
        print(f"Accuracy (experiment) for tolerance {tol}:", accuracy_m)
        print(f"Accuracy (growth rate)for tolerance {tol}:", accuracy_gr_m)
        accuracies_m.append(accuracy_m)
        accuracy_grs_m.append(accuracy_gr_m)
    

    fig, ax = plt.subplots(1,2,figsize=(10, 5))
    ax[0].plot(accuracies, accuracy_grs, marker='o')

    for tolerance, accuracy, accuracy_gr in zip(tolerances, accuracies, accuracy_grs):
        ax[0].annotate(
            f"{tolerance:.2f}",
            (accuracy, accuracy_gr),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    
    ax[0].set_xlabel("Accuracy against Experimental Data", fontsize=12)
    ax[0].set_ylabel("Accuracy against Growth Rate Maximisation", fontsize=12)
    ax[0].set_title("Additive dynamics", fontsize=14)
    
    ax[1].plot(accuracies_m, accuracy_grs_m, marker='o')

    for tolerance, accuracy, accuracy_gr in zip(tolerances_m, accuracies_m, accuracy_grs_m):
        ax[1].annotate(
            f"{tolerance:.2f}",
            (accuracy, accuracy_gr),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )
    
    ax[1].set_xlabel("Accuracy against Experimental Data", fontsize=12)
#    ax[1].set_ylabel("Accuracy against Growth Rate Maximisation", fontsize=12)
    ax[1].set_title("Multiplicative dynamics", fontsize=14)
    
    fig.savefig("tolerance_priority.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    initialise()
    ph_with_different_tolerances()

