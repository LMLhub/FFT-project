import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from fft_project.create_cues_ffts import create_cues_ffts
from fft_project.cue_class import Cue
from fft_project.cue_features import priority_step1, priority_step1_no_loss
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.prepare_experimental_data import prepare_experimental_data



def initialise():
    create_cues_ffts()


def plot_tolerance_sweep(
    ax,
    gamble_data,
    experimental_results,
    dynamic,
    tolerances,
    fft_name,
    feature_function,
    final_cue_name,
    experimental_reference_name,
    comparison_fft_name="fft_gr",
):
    """Run one tolerance experiment and draw the results on the given axis."""
    gamble_data = gamble_data.copy()
    gamble_data["dynamic"] = dynamic
    gamble_data["tol"] = 0

    # Cue used for tolerance sweep"
    tolerance_cue = Cue(
        id=f"{fft_name}_tolerance_cue",
        name="Step 1 of the priority heuristic with variable tolerance",
        description=(
            "This cue evaluates the first step of the priority heuristic "
            "with the tolerance as an input argument."
        ),
        type="boolean",
        feature=feature_function,
        required_args=[
            "gamma_left_up",
            "gamma_left_down",
            "gamma_right_up",
            "gamma_right_down",
            "tol",
            "dynamic",
        ],
    )

    # FFT using the cue with tolerance sweep and the final PH cue
    tolerance_fft = FFT(
        id=fft_name,
        name="Priority Heuristic Step 1",
        description="First step of the priority heuristic with variable tolerance",
        cues=[tolerance_cue, Cue.cue_registry[final_cue_name]],
    )

    # Experiment used to calculate the accuracy for different tolerances
    tolerance_experiment = Experiment(
        id=f"exp_{fft_name}",
        name="Priority Heuristic Tolerance Experiment",
        dynamic=dynamic,
        description="Evaluate the priority heuristic over a range of tolerances.",
        gamble_data=gamble_data,
        ffts=[tolerance_fft, FFT.FFT_registry[comparison_fft_name]],
    )

    '''# Experiment for the 
    comparison_experiment = Experiment(
        id=f"exp_{comparison_fft_name}_{dynamic}",
        name="Comparison FFT Experiment",
        dynamic=dynamic,
        description="Evaluate the comparison FFT for the experimental gamble data.",
        gamble_data=gamble_data,
        ffts=[FFT.FFT_registry[comparison_fft_name]],
    )
    '''
    comparison_results = tolerance_experiment.run_experiment(
        wealth_update="data", random_seed=42, fft_id=comparison_fft_name
    )
    tolerance_experiment.results = pd.concat(
        [comparison_results, experimental_results], axis=1
    )

    comparison_accuracy_against_data = tolerance_experiment.accuracy(
        comparison_fft_name, experimental_reference_name, 1, 1
    )

    print(
        f"{comparison_fft_name} accuracy against data ({dynamic}):",
        comparison_accuracy_against_data,
    )

    # Start with the comparison and experimental results, then append one run
    # of the tolerance FFT for every tolerance value.
    
    # tolerance_experiment.results = tolerance_experiment.results.copy()

    accuracies = []
    comparison_accuracies = []
    
    # Reset run number, so it starts from 1.
    tolerance_experiment.runs = 0
    
    for run_number, tolerance in enumerate(tolerances, start=1):
        gamble_data["tol"] = tolerance
        tolerance_experiment.run_experiment(
            wealth_update="data", random_seed=42, fft_id=fft_name
        )

        accuracies.append(
            tolerance_experiment.accuracy(
                fft_name,
                experimental_reference_name,
                run_number,
                1,
            )
        )
        comparison_accuracies.append(
            tolerance_experiment.accuracy(
                fft_name,
                comparison_fft_name,
                run_number,
                1,
            )
        )

    ax.plot(accuracies, comparison_accuracies, marker="o")
    ax.vlines(
        x=comparison_accuracy_against_data,
        ymin=np.min(comparison_accuracies) - 0.1,
        ymax=np.max(comparison_accuracies) + 0.1,
        color="grey",
        linestyle="--",
        label=f"{comparison_fft_name} accuracy",
    )

    for tolerance, accuracy, comparison_accuracy in zip(
        tolerances, accuracies, comparison_accuracies
    ):
        ax.annotate(
            f"{tolerance:.2f}",
            (accuracy, comparison_accuracy),
            xytext=(5, 5),
            textcoords="offset points",
            fontsize=9,
        )

    ax.set_xlabel("Accuracy against Experimental Data", fontsize=12)
    ax.set_ylabel(f"Accuracy against {comparison_fft_name}", fontsize=12)
    ax.set_title(f"{dynamic.capitalize()} dynamics", fontsize=14)

    return accuracies, comparison_accuracies


def ph_with_different_tolerances():
    gamble_data, experimental_results = prepare_experimental_data(
        PROJECT_ROOT / "data/all_active_phase_data.csv"
    )
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = (
        experimental_results
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))

    plot_tolerance_sweep(
        ax=ax[0],
        gamble_data=gamble_data_additive,
        experimental_results=experimental_results_additive,
        dynamic="additive",
        tolerances=np.arange(0.2, 0.9, 0.05),
        fft_name="ph_a",
        feature_function=priority_step1,
        final_cue_name="pri_3_a",
        experimental_reference_name="experiment_a",
    )

    plot_tolerance_sweep(
        ax=ax[1],
        gamble_data=gamble_data_multiplicative,
        experimental_results=experimental_results_multiplicative,
        dynamic="multiplicative",
        tolerances=np.arange(0.05, 0.8, 0.05),
        fft_name="ph_m",
        feature_function=priority_step1,
        final_cue_name="pri_3_m",
        experimental_reference_name="experiment_m",
    )

    fig.tight_layout()
    fig.savefig("tolerance_priority.png", dpi=300, bbox_inches="tight")

def ph_nl_with_different_tolerances():
    gamble_data, experimental_results = prepare_experimental_data(
        PROJECT_ROOT / "data/all_active_phase_data.csv"
    )
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = (
        experimental_results
    )

    fig, ax = plt.subplots(1, 2, figsize=(10, 5))
    plot_tolerance_sweep(
        ax=ax[0],
        gamble_data=gamble_data_additive,
        experimental_results=experimental_results_additive,
        dynamic="additive",
        tolerances=np.arange(0.2, 0.9, 0.05),
        fft_name="ph_nl_a",
        feature_function=priority_step1_no_loss,
        final_cue_name="pri_nl_3_a",
        experimental_reference_name="experiment_a",
    )

    plot_tolerance_sweep(
        ax=ax[1],
        gamble_data=gamble_data_multiplicative,
        experimental_results=experimental_results_multiplicative,
        dynamic="multiplicative",
        tolerances=np.arange(0.05, 0.8, 0.05),
        fft_name="ph_nl_m",
        feature_function=priority_step1_no_loss,
        final_cue_name="pri_nl_3_m",
        experimental_reference_name="experiment_m",
    )

    fig.tight_layout()
    fig.savefig("tolerance_priority_no_loss.png", dpi=300, bbox_inches="tight")

if __name__ == "__main__":
    initialise()
    #ph_with_different_tolerances()
    ph_nl_with_different_tolerances()