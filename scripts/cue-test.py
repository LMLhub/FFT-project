# This is a test script for the Cue class.
# It creates an instance of the Cue class, evaluates it on some sample gamble data,
# and prints the results.
# This script can be deleted after the cue class testing is complete.

from fft_project.cue_class import Cue
from fft_project.cue_features import growth_rate, expected_isoelastic_utility
import pandas as pd

def main():
    # Example cue definition
    c01 = Cue(
        id="c01",
        name="Expected Isoelastic Utility - eta=1.5, additive",
        description="This cue that evaluates the expected isoelastic utility of the first gamble with eta=1.5 and picks a side if the cue value is greater than 0.",
        feature= expected_isoelastic_utility,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive",
                "eta": 1.5},
        required_args=["output_left_up", "output_left_down", "output_right_up", "output_right_down", "wealth"]
    )

    c02 = Cue(
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
    
    cue_value_1, side_if_true_1 = c01.evaluate(x1_1, x1_2, x2_1, x2_2, wealth=wealth)
    cue_value_2, side_if_true_2 = c02.evaluate(x1_1, x1_2, x2_1, x2_2)
    
    # Test evaluate_pd method on the gamble_data
    result_1 = c01.evaluate_df(gamble_data)
    result_2 = c02.evaluate_df(gamble_data)
    
    print(f'Cue value 1: {cue_value_1}, side_if_true 1: {side_if_true_1}')
    print(f'Cue value 2: {cue_value_2}, side_if_true 2: {side_if_true_2}')
    print(result_1)
    print(result_2)
    print("Cue evaluation successful.")
    print("Cue registry:", Cue.cue_registry)
    #Cue.save_registry("cue_registry.yaml")

if __name__ == "__main__":
    main() 
