import sys
from pathlib import Path

from scipy import fft

from scipy.fft import fft

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility
import pandas as pd

from fft_project.simulation_gamble_data import simulate_gamble_data
from fft_project.create_cues_ffts import create_cues, create_ffts
from fft_project.prepare_experimental_data import prepare_experimental_data

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]
FRACTAL_VALUES_MULTI = [-0.850, -0.5395, -0.433, -0.1735, 0.006, 0.1685, 0.369, 0.5535, 0.772]

def main():
    create_cues()
    create_ffts()

    # Simulate some gamble data
    gamble_data = simulate_gamble_data(10, FRACTAL_VALUES, random_seed = 42)

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
                            ffts=[FFT.FFT_registry["fft_eu_1_5_a"],
                                  FFT.FFT_registry["fft_gr"],
                                  FFT.FFT_registry["fft_aw_1_a"],
                                  FFT.FFT_registry["fft_aw_2_a"],
                                  FFT.FFT_registry["fft_fs"],
                                  FFT.FFT_registry["fft_aw_1_fs_a"],
                                  FFT.FFT_registry["fft_fs_aw_1_a"]]
    )

    # Evaluate the experiment
    results = experiment.run_experiment(wealth_update="constant", random_seed=42)
    results = experiment.run_experiment(wealth_update="constant", random_seed=43)

    print("Accuracy fs vs gr:",experiment.accuracy("fft_fs", "fft_gr"))
    print("Accuracy aw_1 vs gr:",experiment.accuracy("fft_aw_1_a", "fft_gr"))
    print("Accuracy aw_2 vs gr:",experiment.accuracy("fft_aw_2_a", "fft_gr"))
    print("Accuracy aw_1_fs vs gr:",experiment.accuracy("fft_aw_1_fs_a", "fft_gr"))
    print("Accuracy fs_aw_1 vs gr:",experiment.accuracy("fft_fs_aw_1_a", "fft_gr"))

def main_2():
    gamble_data, experimental_results = prepare_experimental_data(PROJECT_ROOT / "data/all_active_phase_data.csv")
    gamble_data_additive, gamble_data_multiplicative = gamble_data
    experimental_results_additive, experimental_results_multiplicative = experimental_results

    print("Gamble Data:")
    print(gamble_data_additive.head())
    print("\nExperimental Results:")
    print(experimental_results_additive.head())

    experiment = Experiment(id="exp2",
                            name="Example Experiment",
                            dynamic="additive",
                            description="An example experiment using the simulated gamble data and the example FFT.",
                            gamble_data=gamble_data_additive,
                            ffts=[FFT.FFT_registry["fft_gr"],
                                  FFT.FFT_registry["fft_fs"],]
    )
    results = experiment.run_experiment(wealth_update="data", random_seed=42)

    print("results:")
    print(results.head())

if __name__ == "__main__":
    main()
    main_2()