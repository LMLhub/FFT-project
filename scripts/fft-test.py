# This is a test script for the Cue class.
# It creates an instance of the Cue class, evaluates it on some sample gamble data,
# and prints the results.
# This script can be deleted after the cue class testing is complete.

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
from fft_project.cue_features import growth_rate, expected_isoelastic_utility
import pandas as pd

def main():
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
        required_args=["output_left_up", "output_left_down", "output_right_up", "output_right_down", "wealth"]
    )

    Cue(
        id="c02",
        name="Growth Rate - additive",
        description="This cue compares the additive growth rates of the gambles and picks the side with the highest rate.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive"},
        required_args=["output_left_up", "output_left_down", "output_right_up", "output_right_down"]
    )

    # Create a sample gamble_data dataframe
    gamble_data = pd.DataFrame({
        "wealth": [1000, 2000],
        "output_left_up": [100, 200],
        "output_left_down": [100, 150],
        "output_right_up": [120, 180],
        "output_right_down": [80, 140]
    })
    
    # Test evaluate method on the first row of gamble_data
    x1_1, x1_2, x2_1, x2_2, wealth = gamble_data.loc[0, ["output_left_up", "output_left_down", "output_right_up", "output_right_down", "wealth"]]
    
    # Call cue objects from registry
    cue1 = Cue.cue_registry["c01"]
    cue2 = Cue.cue_registry["c02"]
    
    # Create an FFT instance using the cues
    fft = FFT(id="fft1", name="Example FFT", description="An example FFT with two cues.", cues=[cue2, cue1])
    
    # Evaluate the cue on the gamble_data
    result = fft.decide(x1_1, x1_2, x2_1, x2_2, wealth=wealth)
    
    result = fft.decide_df(gamble_data, ["output_left_up", "output_left_down", "output_right_up", "output_right_down", "wealth"])
    #result = fft.decide_df_01(gamble_data)
    print(result)
    #print(result[["fft1_cues_used", "c01_value", "c01_side_if_true", "c02_value", "c02_side_if_true","fft1_decision"]])
    print("result columns:", result.columns)
    print("FFT evaluation successful.")
    print("FFT registry:", FFT.FFT_registry)
    #FFT.save_registry("fft_registry.yaml")


if __name__ == "__main__":
    main() 
