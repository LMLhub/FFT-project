# This is a test script for the Cue class.
# It creates an instance of the Cue class, evaluates it on some sample gamble data,
# and prints the results.
# This script can be deleted after the cue class testing is complete.

from fft_project.cue_class import Cue
from fft_project.cue_features import growth_rate
import pandas as pd

def main():
    # Example feature function that calculates the expected value of the first gamble multiplied by some parameter
    def example_feature(gamma_left_up, gamma_left_down, gamma_right_up, gamma_right_down, multiplier=1):
        ev_left = (gamma_left_up + gamma_left_down) / 2
        return ev_left * multiplier

    # Create a Cue instance
    '''cue1 = Cue(id="ex1", name="Example Cue, multiplier 2",
              description="A cue that evaluates expected values of the first gamble.",
              feature=example_feature, type="numerical", threshold=1, params={"multiplier": 2})
    '''
    
    c01 = Cue(
        id="c01",
        name="Growth Rate - multiplicative dynamic",
        description="A cue that evaluates the multiplicative growth rate of the left gamble.",
        feature= growth_rate,
        type="numerical",
        threshold=10,
        params={"dynamic": "multiplicative"},
        required_columns=["wealth"]
    )

    c02 = Cue(
        id="c02",
        name="Growth Rate - additive dynamic",
        description="A cue that evaluates the additive growth rate of the left gamble.",
        feature= growth_rate,
        type="numerical",
        threshold=0,
        params={"dynamic": "additive"},
        required_columns=["wealth"]
    )

# Create a sample gamble_data dataframe
    gamble_data = pd.DataFrame({
        "wealth": [100, 200],
        "gamma_left_up": [10, 20],
        "gamma_left_down": [10, 15],
        "gamma_right_up": [12, 18],
        "gamma_right_down": [8, 14]
    })

    # Evaluate the cue on the gamble_data
    result = c01.evaluate(gamble_data)
    result = c02.evaluate(result)
    print(result)
    print("Cue evaluation successful.")
    print("Cue registry:", Cue.cue_registry)
    #Cue.save_registry("cue_registry.yaml")

if __name__ == "__main__":
    main() 
