# This is a test script for the FFT class.
# It creates an instance of the Cue class and FFT class, evaluates it on some sample gamble data,
# and prints the results.
# This script can be deleted after the FFT class testing is complete.

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
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

    cue = Cue(
        id          = "c03",
        name        = "Avoid worst 1 rank",
        description = "Prefers the gamble that does not contain the worst fractal value.",
        feature     = avoid_worst_n_ranks,
        type        = "boolean",
        params      = {"n": 1, "fractal_values": FRACTAL_VALUES},
        required_args = ["gamma_left_up", "gamma_left_down",
                         "gamma_right_up", "gamma_right_down"],
    )

def test_fft_evaluate():
        # Call cue objects from registry
    cue1 = Cue.cue_registry["c01"]
    cue2 = Cue.cue_registry["c02"]
    
    # Create an FFT instance using the cues
    fft = FFT(id="fft1",
              name="Example FFT",
              description="An example FFT with two cues.",
              cues=[cue2, cue1])
    
    # Create a sample gamble_data dataframe
    gamble_data = pd.DataFrame({
        "wealth": [1000, 2000],
        "gamma_left_up": [100, 200],
        "gamma_left_down": [110, 150],
        "gamma_right_up": [120, 180],
        "gamma_right_down": [80, 140]
    })
    
    # Test evaluate method on the first row of gamble_data
    x1_1, x1_2, x2_1, x2_2, wealth = gamble_data.loc[0, ["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth"]]
        
    # Evaluate the cue on the gamble_data
    result = fft.decide(x1_1, x1_2, x2_1, x2_2, wealth=wealth)
    
    result = fft.decide_df(gamble_data, ["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth"])
    print(result)
    print("result columns:", result.columns)
    print("FFT evaluation successful.")
    print("FFT registry:", FFT.FFT_registry)
    #FFT.save_registry("fft_registry.yaml")

def main():
    create_cues()
    test_fft_evaluate()

if __name__ == "__main__":
    main() 
