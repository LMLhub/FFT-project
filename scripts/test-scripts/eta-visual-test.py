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
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility

from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues_ffts
from fft_project.prepare_experimental_data import prepare_experimental_data
from fft_project.analysis_compare import (
    plot_etas_compare,
    plot_accuracy_gamma_scatter,
    plot_participant_accuracy_gamma_scatter,
    plot_participant_accuracy_comparison,
    participant_rank_statistics,
    plot_participant_rank_fractions,
    plot_rank_fraction_accuracy,
    run_dynamic_experiment,
)


from fft_project.config import read_config_file
CONFIG = read_config_file(PROJECT_ROOT / "config.yaml")
GAMBLE_SIMULATION_CONFIG = CONFIG["gamble_simulation"]
FRACTAL_VALUES = GAMBLE_SIMULATION_CONFIG["fractals_add"]
FRACTAL_VALUES_MULTI = GAMBLE_SIMULATION_CONFIG["fractals_mul"]

FFT_NAMES_A = ["fft_gr", "fft_aw_1_a", "fft_aw_2_a", "fft_pb_1_a","fft_pb_1_aw_1_a", "fft_fs", "fft_fs_aw_1_a", "pri_a", "pri_nl_a" ]
FFT_NAMES_M = ["fft_gr", "fft_aw_1_m", "fft_aw_2_m", "fft_pb_1_m","fft_pb_1_aw_1_m", "fft_fs", "fft_fs_aw_1_m", "pri_m", "pri_nl_m"]

def initialise():
    create_cues_ffts()

def test():
    
    # Simulate some gamble data
    gamble_data = simulate_gamble_data(100, FRACTAL_VALUES, random_seed = 42)

    # add a wealth column for testing purposes
    import numpy as np
    gamble_data["wealth"] = np.random.randint(100, 2001, size=len(gamble_data)) 
    
    # Create an experiment instance
    experiment = Experiment(id="exp1",
                            name="Example Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data,
                            initial_wealth=1000,
                            ffts=[FFT.FFT_registry[name] for name in FFT_NAMES_A]
    )

    # Evaluate the experiment
    results = experiment.run_experiment(wealth_update="constant", random_seed=42)
    results = experiment.run_experiment(wealth_update="constant", random_seed=43)

    print("Accuracy fs vs gr:",experiment.accuracy("fft_fs", "fft_gr"))
    print("Accuracy aw_1 vs gr:",experiment.accuracy("fft_aw_1_a", "fft_gr"))
    print("Accuracy aw_2 vs gr:",experiment.accuracy("fft_aw_2_a", "fft_gr"))
    print("Accuracy aw_1_fs vs gr:",experiment.accuracy("fft_aw_1_fs_a", "fft_gr"))
    print("Accuracy fs_aw_1 vs gr:",experiment.accuracy("fft_fs_aw_1_a", "fft_gr"))
    print("Accuracy fft_pb_1_a vs gr:",experiment.accuracy("fft_aw_1_pf_1_a", "fft_gr"))
    print("Accuracy fft_aw_1_pb_1_a vs gr:",experiment.accuracy("fft_aw_1_pb_1_a", "fft_gr"))
    print("Accuracy fft_pb_1_aw_1_a vs gr:",experiment.accuracy("fft_pf_1_aw_1_a", "fft_gr"))
    print("Accuracy pri_a vs gr:",experiment.accuracy("pri_a", "fft_gr"))
    print("Accuracy pri_nl_a vs gr:",experiment.accuracy("pri_nl_a", "fft_gr"))

def plot_gamma_match():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results
    experimental_results_additive.reset_index()
    
    #print(experimental_results_additive.head())
    #print("Gamble Data:")
    #print(gamble_data_additive.head())
    #print("\nExperimental Results:")
    #print(experimental_results_additive.head())

    #fft_names = ["fft_gr", "fft_aw_1_a", "fft_aw_2_a", "fft_pb_1_a","fft_pb_1_aw_1_a", "fft_fs", "fft_fs_aw_1_a", "pri_a" ]
    fft_names = FFT_NAMES_A.copy()

    experiment_a = Experiment(id="exp2_a",
                            name="Example Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_additive,
                            ffts=[FFT.FFT_registry[name] for name in fft_names]
    )

    results_a = experiment_a.run_experiment(wealth_update="data", random_seed=42)
    
    fft_names.append("experiment_a")
    results_a = pd.concat([results_a, experimental_results_additive], axis=1)
        
    fig, ax = plt.subplots(1,2, figsize=(10,4))
    
    etas = np.linspace(-1,5,31)
    
    plot_etas_compare(
        gamble_data_additive,
        results_a,
        fft_names,
        etas,
        "additive",
        ax=ax[0],
        runs=1,
        random_seed=42,
    )
    print('Done with additive - starting multiplicative')
    
    #Do the same for multiplicative:
    fft_names = FFT_NAMES_M.copy()
   
    experiment_m = Experiment(id="exp2_m",
                            name="Example Experiment",
                            dynamic="multiplicative",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_multiplicative,
                            ffts=[FFT.FFT_registry[name] for name in fft_names]
    )

    results_m = experiment_m.run_experiment(wealth_update="data", random_seed=42)
    
    fft_names.append("experiment_m")
    results_m = pd.concat([results_m, experimental_results_multiplicative], axis=1)

    plot_etas_compare(
        gamble_data_multiplicative,
        results_m,
        fft_names,
        etas,
        "multiplicative",
        ax=ax[1],
        runs=1,
        random_seed=42,
    )
    
    fig.savefig("eta_accuracy.png", dpi=300, bbox_inches="tight")

def plot_growth_rate_match():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results
    experimental_results_additive.reset_index()
    
    fig, ax = plt.subplots(1,2, figsize=(10,4))    
    
    #fft_names = ["fft_gr", "fft_aw_1_a", "fft_aw_2_a", "fft_pb_1_a","fft_pb_1_aw_1_a", "fft_fs", "fft_fs_aw_1_a", "pri_a" ]
    fft_names = FFT_NAMES_A.copy()

    experiment_a = Experiment(id="exp3_a",
                            name="Example Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_additive,
                            ffts=[FFT.FFT_registry[name] for name in fft_names])

    results_a = experiment_a.run_experiment(wealth_update="data", random_seed=42)
    
    fft_names.append("experiment_a")
    results_a = pd.concat([results_a, experimental_results_additive], axis=1)
 
    plot_accuracy_gamma_scatter(results_a, fft_names,
                                 "fft_gr", runs=1, ax=ax[0],
                                 title = "Additive dynamics")
    
    fft_names = FFT_NAMES_M.copy()
    
    experiment_m = Experiment(id="exp3_m",
                            name="Example Experiment",
                            dynamic="multiplicative",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_multiplicative,
                            ffts=[FFT.FFT_registry[name] for name in fft_names]
    )

    results_m = experiment_m.run_experiment(wealth_update="data", random_seed=42)
    
    fft_names.append("experiment_m")
    results_m = pd.concat([results_m, experimental_results_multiplicative], axis=1)
    
    plot_accuracy_gamma_scatter(results_m, fft_names, "fft_gr",
                                runs=1, ax=ax[1],
                                title = "Multiplicative dynamics")
    
    fig.subplots_adjust(wspace=0.4)
    fig.savefig("growth_rate_compare.png", dpi=300, bbox_inches="tight")

def plot_experiment_match():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results
    experimental_results_additive.reset_index()
    
    fig, ax = plt.subplots(1,2, figsize=(10,4))    
    
    fft_names = FFT_NAMES_A.copy()
    
    for name in fft_names:
        print(FFT.FFT_registry[name].id)

    experiment_a = Experiment(id= "exp4_a",
                            name= "Example Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data = gamble_data_additive,
                            ffts= [FFT.FFT_registry[name] for name in fft_names])

    results_a = experiment_a.run_experiment(wealth_update="data", random_seed=42)
    fft_names.append("experiment_a")
    results_a = pd.concat([results_a, experimental_results_additive], axis=1)

    plot_accuracy_gamma_scatter(results_a, fft_names,
                                 "experiment_a", runs=1, ax=ax[0],
                                 title = "Additive dynamics")
    
    fft_names = FFT_NAMES_M.copy()
    for name in fft_names:
        print(FFT.FFT_registry[name].id)

    experiment_m = Experiment(id="exp4_m",
                            name="Example Experiment",
                            dynamic="multiplicative",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data = gamble_data_multiplicative,
                            ffts=[FFT.FFT_registry[name] for name in fft_names])

    results_m = experiment_m.run_experiment(wealth_update="data", random_seed=42)
    
    fft_names.append("experiment_m")
    results_m = pd.concat([results_m, experimental_results_multiplicative], axis=1)
    
    #print the priority heuristic results for the multiplicative dynamics:
    print("Accuracy pri_m vs gr:",experiment_m.accuracy("pri_m", "fft_gr"))

    plot_accuracy_gamma_scatter(results_m, fft_names, "experiment_m",
                                runs=1, ax=ax[1],
                                title = "Multiplicative dynamics")
    
    fig.subplots_adjust(wspace=0.4)
    fig.savefig("choices_compare.png", dpi=300, bbox_inches="tight")

def plot_participant_matches():
    """
    Recreate the growth-rate and experimental-choice match plots with results
    calculated separately for every participant.

    Each marker represents one participant/FFT combination. Marker colour
    identifies the FFT, and all participants use the same marker shape.
    """
    # Load the gamble inputs and observed choices for both dynamics.
    gamble_data, experimental_results = prepare_experimental_data(
        PROJECT_ROOT / "data/all_active_phase_data.csv"
    )

    # Prepare the additive and multiplicative results once for both figures.
    dynamic_data = [
        ("additive", gamble_data[0], experimental_results[0]),
        ("multiplicative", gamble_data[1], experimental_results[1]),
    ]
    prepared_results = [
        (dynamic,)
        + run_dynamic_experiment(
            dynamic,
            gambles,
            results,
            FFT_NAMES_A if dynamic == "additive" else FFT_NAMES_M,
        )
        for dynamic, gambles, results in dynamic_data
    ]

    # Create comparisons against growth rate and observed choices.
    figures = []
    for reference_kind, output_name in [
        ("growth_rate", "growth_rate_compare_by_participant.png"),
        ("experiment", "choices_compare_by_participant.png"),
    ]:
        fig, axes = plt.subplots(1, 2, figsize=(12, 4))
        for ax, (dynamic, results, fft_names, experiment_id) in zip(
            axes, prepared_results
        ):
            reference_id = "fft_gr" if reference_kind == "growth_rate" else experiment_id
            plot_participant_accuracy_gamma_scatter(
                results,
                fft_names,
                experiment_id,
                reference_id,
                ax,
                title=f"{dynamic.capitalize()} dynamics",
            )
        fig.subplots_adjust(wspace=0.4, bottom=0.28)

        # Save the completed figure and return it for optional reuse.
        fig.savefig(output_name, dpi=300, bbox_inches="tight")
        figures.append(fig)

    # Compare experimental-choice accuracy directly with growth-rate accuracy.
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, (dynamic, results, fft_names, experiment_id) in zip(
        axes, prepared_results
    ):
        plot_participant_accuracy_comparison(
            results,
            fft_names,
            experiment_id,
            experiment_id,
            "fft_gr",
            ax=ax,
            title=f"{dynamic.capitalize()} dynamics",
            x_label="Accuracy against experimental data",
            y_label="Accuracy against time-average growth-rate maximisation",
        )

    # Save and return the additional accuracy-against-accuracy figure.
    fig.subplots_adjust(wspace=0.4, bottom=0.28)
    fig.savefig(
        "experiment_vs_growth_rate_accuracy_by_participant.png",
        dpi=300,
        bbox_inches="tight",
    )
    figures.append(fig)

    # Count how often each FFT is the most accurate for an individual.
    # Tied FFTs split that participant's first-place credit equally, keeping
    # the fractions in each panel summing to one.
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    rank_accuracy_plot_data = []
    for ax, (dynamic, results, fft_names, experiment_id) in zip(
        axes, prepared_results
    ):
        candidate_ffts = [name for name in fft_names if name != experiment_id]
        rank_statistics = participant_rank_statistics(
            results,
            candidate_ffts,
            experiment_id,
            rank_cutoffs=(1, 2, 3),
        )
        rank_accuracy_plot_data.append((dynamic, rank_statistics))
        plot_participant_rank_fractions(
            rank_statistics,
            ax=ax,
            title=f"{dynamic.capitalize()} dynamics",
        )

    fig.subplots_adjust(wspace=0.3, bottom=0.35)
    fig.savefig(
        "fft_first_place_fraction_by_participant.png",
        dpi=300,
        bbox_inches="tight",
    )
    figures.append(fig)

    # Relate each plotted rank fraction to the FFT's average accuracy among
    # the participant contributions included in that fraction.
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for ax, (dynamic, rank_statistics) in zip(
        axes, rank_accuracy_plot_data
    ):
        plot_rank_fraction_accuracy(
            rank_statistics,
            ax=ax,
            title=f"{dynamic.capitalize()} dynamics",
            y_limits=(0.55, 0.80),
        )

    fig.subplots_adjust(wspace=0.3, bottom=0.42)
    fig.savefig(
        "fft_rank_fraction_vs_accuracy_by_participant.png",
        dpi=300,
        bbox_inches="tight",
    )
    figures.append(fig)

    return figures

if __name__ == "__main__":
    initialise()
    # test()
    plot_gamma_match()
    plot_growth_rate_match()
    plot_experiment_match()
    plot_participant_matches()
