import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import (
    avoid_worst_n_ranks,
    expected_isoelastic_utility,
    growth_rate,
    signs,
)
import numpy as np
import pandas as pd

from fft_project.simulation_gamble_data import simulate_gamble_data

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]
FRACTAL_VALUES_MULTI = [-0.850, -0.5395, -0.433, -0.1735, 0.006, 0.1685, 0.369, 0.5535, 0.772]

def create_cues():
     # Example cue definition
    Cue(
        id="c01",
        name="Expected Isoelastic Utility - eta=1.5, additive",
        description="This cue that evaluates the expected isoelastic utility of the first gamble with eta=1.8 and picks a side if the cue value is greater than 2.",
        feature= expected_isoelastic_utility,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive",
                "eta": 1.5},
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth"]
    )

    Cue(
        id="c02",
        name="Growth Rate",
        description="This cue compares the growth rates of the gambles and picks the side with the highest rate. Works for both additive and multiplicative dynamics.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down"]
    )

    Cue(
        id          = "c03",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id          = "c04",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES_MULTI},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

    Cue(
        id="c05",
        name="Fractal signs",
        description="Prefers the gamble with more positive fractal values.",
        feature=signs,
        type="numerical",
        threshold=0,
        required_args=["gamma_left_up", "gamma_left_down",
                       "gamma_right_up", "gamma_right_down"],
    )
def create_ffts():
    fft = FFT(id="fft2",
              name="Avoid the worst or random - additive",
              description="An example FFT avoid the worst.",
              cues=[Cue.cue_registry["c03"]])

    growth_rate_fft = FFT(id="fft3",
              name="Growth rate maximisation",
              description="An example FFT with the growth rate cue.",
              cues=[Cue.cue_registry["c02"]])

    growth_rate_multi_fft = FFT(id="fft4",
              name="Avoid the worst - multiplicative",
              description="An example FFT with avoid the worst cue.",
              cues=[Cue.cue_registry["c04"]])

    fractal_sign_fft = FFT(
        id="fft5",
        name="Fractal sign",
        description="An FFT that prefers the gamble with more positive outcomes.",
        cues=[Cue.cue_registry["c05"]],
    )

def test_wealth_trajectory():
    # Test wealth trajectory method
    # Create a sample gamble_data dataframe with additive data
    
    gamble_data = pd.DataFrame({
        "gamma_left_up": [-407.0, -305.5],
        "gamma_left_down": [50.0, 440.5],
        "gamma_right_up": [-241.5, 108.5],
        "gamma_right_down": [ 210.5, 309.5]
    }) #GRM: choose right gamble in both cases.

    Experiment(
        id="exp1",
        name="Test Experiment - Additive",
        description="An experiment to test the methods of the Experiment class.",
        ffts=[FFT.FFT_registry["fft2"], FFT.FFT_registry["fft3"]],
        gamble_data=gamble_data,
        initial_wealth=1000,
        dynamic="additive"
    )

    #test running the experiment and getting the trajectory
    trajectory_result = Experiment.experiment_registry['exp1'].run_experiment()
    print(trajectory_result)
    
    trajectory_result = Experiment.experiment_registry['exp1'].run_experiment()
    print(trajectory_result)
    
def test_accuracy():
    accuracy_result = Experiment.experiment_registry['exp1'].accuracy(fft_id="fft2", reference_id="fft3", run_no=1)    
    print(f"Accuracy of fft2 at run 1: {accuracy_result:.2f}")

    accuracy_result = Experiment.experiment_registry['exp1'].accuracy(fft_id="fft2", reference_id="fft3")    
    print(f"Accuracy of fft2 at run 1 and 2: {accuracy_result:.2f}")

def test_importance_weighted_accuracy_simulated_data():
    """Check importance-weighted accuracy against a manual calculation."""
    gamble_data = simulate_gamble_data(
        100,
        FRACTAL_VALUES,
        random_seed=42,
    )
    experiment = Experiment(
        id="exp_importance_weighted_accuracy",
        name="Importance-weighted accuracy test",
        description="Test weighted accuracy using simulated additive gambles.",
        ffts=[
            FFT.FFT_registry["fft2"],
            FFT.FFT_registry["fft3"],
            FFT.FFT_registry["fft5"],
        ],
        gamble_data=gamble_data,
        initial_wealth=1000,
        dynamic="additive",
    )
    experiment.run_experiment(wealth_update="constant", random_seed=42)

    default_accuracy = experiment.accuracy("fft5", "fft3", run_no=1)
    explicit_unweighted_accuracy = experiment.accuracy(
        "fft5",
        "fft3",
        run_no=1,
        importance_weighted=False,
    )
    weighted_accuracy = experiment.accuracy(
        "fft5",
        "fft3",
        run_no=1,
        importance_weighted=True,
    )
    perfect_weighted_accuracy = experiment.accuracy(
        "fft5",
        "fft5",
        run_no=1,
        importance_weighted=True,
    )

    left_expected_gamma = gamble_data[
        ["gamma_left_up", "gamma_left_down"]
    ].mean(axis=1)
    right_expected_gamma = gamble_data[
        ["gamma_right_up", "gamma_right_down"]
    ].mean(axis=1)
    distances = (left_expected_gamma - right_expected_gamma).abs()
    weights = distances / distances.max()

    fft_decisions = experiment.results[("fft5", 1, "selected_side")]
    reference_decisions = experiment.results[("fft3", 1, "selected_side")]
    signed_matches = np.where(fft_decisions == reference_decisions, 1.0, -1.0)
    expected_weighted_accuracy = 0.5 * (
        1.0 + (signed_matches * weights).sum() / weights.sum()
    )

    assert np.isclose(default_accuracy, explicit_unweighted_accuracy)
    assert 0.0 <= weighted_accuracy <= 1.0
    assert np.isclose(weighted_accuracy, expected_weighted_accuracy)
    assert np.isclose(perfect_weighted_accuracy, 1.0)

    print(f"Fractal-sign unweighted accuracy: {default_accuracy:.3f}")
    print(f"Fractal-sign importance-weighted accuracy: {weighted_accuracy:.3f}")

def test_frugality():
    #test frugality method
    frugality_result = Experiment.experiment_registry['exp1'].frugality(fft_id="fft2", run_no = 1)
    print(f"Frugality of fft2 at run 1: {frugality_result:.2f}")
    
    frugality_result = Experiment.experiment_registry['exp1'].frugality(fft_id="fft2")
    print(f"Frugality of fft2 at run 1 and 2: {frugality_result:.2f}")

def test_multi_wealth_trajectory():
    # Test wealth trajectory method
    # Create a sample gamble_data dataframe with additive data
    
    gamble_data = pd.DataFrame({
        "gamma_left_up": [-0.850, -0.5395],
        "gamma_left_down": [0.006, 0.772],
        "gamma_right_up": [-0.433, 0.5535],
        "gamma_right_down": [ 0.369, 0.772]
    }) #GRM: choose right gamble in both cases.

    Experiment(
        id="exp2",
        name="Test Experiment - multiplicative",
        description="An experiment to test the methods of the Experiment class.",
        ffts=[FFT.FFT_registry["fft3"], FFT.FFT_registry["fft4"]],
        gamble_data=gamble_data,
        initial_wealth=1000,
        dynamic="multiplicative"
    )

    #test running the experiment and getting the trajectory
    trajectory_result = Experiment.experiment_registry['exp2'].run_experiment()
    print(trajectory_result)
    
    trajectory_result = Experiment.experiment_registry['exp2'].run_experiment()
    print(trajectory_result)

def test_multi_wealth_trajectory_sim_data():
    # Create a sample gamble_data dataframe with multiplicative data
    gamble_data_multi = simulate_gamble_data(10,
                              FRACTAL_VALUES_MULTI,
                              True,
                              True,
                              True,
                              None,
                              42)
    print(gamble_data_multi.head())

    import numpy as np
    gamble_data_multi["wealth"] = np.ones(len(gamble_data_multi))*100

    Experiment(
        id="exp3",
        name="Test Experiment - Multiplicative",
        description="An experiment to test the methods of the Experiment class with multiplicative data.",
        ffts=[FFT.FFT_registry["fft3"]],
        gamble_data=gamble_data_multi,
        initial_wealth=1000,
        dynamic="multiplicative"
    )

    trajectory_result = Experiment.experiment_registry['exp3'].run_experiment(initial_wealth = 1000, wealth_update = "data")
    print(trajectory_result)

def main():
    create_cues()
    create_ffts()
    test_wealth_trajectory()
    test_accuracy()
    test_frugality()
    test_multi_wealth_trajectory()
    test_multi_wealth_trajectory_sim_data()
    test_importance_weighted_accuracy_simulated_data()

if __name__ == "__main__":
    main() 
