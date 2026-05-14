from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.experiment_class import Experiment
from fft_project.cue_features import avoid_worst_n_ranks, growth_rate, expected_isoelastic_utility
import pandas as pd

#Fractal values for the additive dynamic from the experiment.
FRACTAL_VALUES = [-407.0, -305.5, -241.5, -49.0, 50.0, 108.5, 210.5, 309.5, 440.5]

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
        id          = "c03",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

def test_wealth_trajectory():
    # Test wealth trajectory method
    # Create a sample gamble_data dataframe with additive data
    cue1 = Cue.cue_registry["c03"]

    fft = FFT(id="fft2",
              name="Avoid the worst or random",
              description="An example FFT with two cues.",
              cues=[cue1])

    gamble_data = pd.DataFrame({
        "gamma_left_up": [-407.0, -305.5],
        "gamma_left_down": [50.0, 440.5],
        "gamma_right_up": [-241.5, 108.5],
        "gamma_right_down": [ 210.5, 309.5]
    })

    experiment =Experiment(
        id="exp1",
        name="Test Experiment",
        description="An experiment to test the wealth trajectory method.",
        fft=fft,
        gamble_data=gamble_data,
        initial_wealth=1000,
        random_seed=42,
        dynamic="additive"
    )

    trajectory_result = experiment.wealth_trajectory(initial_wealth=1000, random_seed=42)
    
    print(trajectory_result[["time_step", "wealth","fft2_cues_used","fft2_decision","fft2_wealth"]])

def main():
    create_cues()
    test_wealth_trajectory()

if __name__ == "__main__":
    main() 
