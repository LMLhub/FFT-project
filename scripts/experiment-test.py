from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility
import pandas as pd

from fft_project.simulation_gamble_data import simulate_gamble_data

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]
FRACTAL_VALUES_MULTI = [0.427, 0.583, 0.649, 0.841, 1.006, 1.184, 1.446, 1.739, 2.164]

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
        name="Growth Rate - additive",
        description="This cue compares the additive growth rates of the gambles and picks the side with the highest rate.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive"},
        required_args=["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down"]
    )

    Cue(
        id="c04",
        name="Growth Rate - multiplicative",
        description="This cue compares the multiplicative growth rates of the gambles and picks the side with the highest rate.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        params={"dynamic": "multiplicative"},
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
def create_ffts():
    fft = FFT(id="fft2",
              name="Avoid the worst or random",
              description="An example FFT with two cues.",
              cues=[Cue.cue_registry["c03"]])

    growth_rate_fft = FFT(id="fft3",
              name="Growth Rate - additive",
              description="An example FFT with the growth rate cue.",
              cues=[Cue.cue_registry["c02"]])

    growth_rate_multi = FFT(id="fft4",
              name="Growth Rate - multiplicative",
              description="An example FFT with the multiplicative growth rate cue.",
              cues=[Cue.cue_registry["c04"]])

def test_wealth_trajectory():
    # Test wealth trajectory method
    # Create a sample gamble_data dataframe with additive data
    
    gamble_data = pd.DataFrame({
        "gamma_left_up": [-407.0, -305.5],
        "gamma_left_down": [50.0, 440.5],
        "gamma_right_up": [-241.5, 108.5],
        "gamma_right_down": [ 210.5, 309.5]
    })

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

def test_frugality():
    #test frugality method
    frugality_result = Experiment.experiment_registry['exp1'].frugality(fft_id="fft2", run_no = 1)
    print(f"Frugality of fft2 at run 1: {frugality_result:.2f}")
    
    frugality_result = Experiment.experiment_registry['exp1'].frugality(fft_id="fft2")
    print(f"Frugality of fft2 at run 1 and 2: {frugality_result:.2f}")

def test_multi_wealth_trajectory():
    # Create a sample gamble_data dataframe with multiplicative data
    gamble_data_multi = simulate_gamble_data(100,
                              FRACTAL_VALUES_MULTI,
                              True,
                              True,
                              True,
                              None,
                              42)
    print(gamble_data_multi.head())

    Experiment(
        id="exp2",
        name="Test Experiment - Multiplicative",
        description="An experiment to test the methods of the Experiment class with multiplicative data.",
        ffts=[FFT.FFT_registry["fft4"]],
        gamble_data=gamble_data_multi,
        initial_wealth=1,
        dynamic="multiplicative"
    )

    trajectory_result = Experiment.experiment_registry['exp2'].run_experiment()
    print(trajectory_result)

def main():
    create_cues()
    create_ffts()
    # test_wealth_trajectory()
    # test_accuracy()
    # test_frugality()
    test_multi_wealth_trajectory()
    
if __name__ == "__main__":
    main() 
