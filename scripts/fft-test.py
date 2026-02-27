# This is a test script for the Cue class.
# It creates an instance of the Cue class, evaluates it on some sample gamble data,
# and prints the results.
# This script can be deleted after the cue class testing is complete.

from fft_project.cue_class import Cue
from fft_project.decision_class import FFT
import pandas as pd

def main():
    # Example feature function that calculates the expected value of the first gamble multiplied by some parameter
    def example_feature(gamma_left_up, gamma_left_down, gamma_right_up, gamma_right_down, multiplier=1):
        ev_left = (gamma_left_up + gamma_left_down) / 2
        return ev_left * multiplier

    # Create a Cue instance
    cue1 = Cue(id="ex1", name="Example Cue, multiplier 2",
              description="A cue that evaluates expected values of the first gamble.",
              feature=example_feature, type="numerical", threshold=1, params={"multiplier": 2})

    cue2 = Cue(id="ex2", name="Example Cue, multiplier 5",
              description="A cue that evaluates expected values of the first gamble.",
              feature=example_feature, type="numerical", threshold=0, params={"multiplier": 5})

    
    fft = FFT(id="fft1", name="Example FFT", description="An example FFT with two cues.", cues=[cue1, cue2])
    
    # Create a sample gamble_data dataframe
    gamble_data = pd.DataFrame({
        "gamma_left_up": [10, 20],
        "gamma_left_down": [10, 15],
        "gamma_right_up": [12, 18],
        "gamma_right_down": [8, 14]
    })

    # Evaluate the cue on the gamble_data
    result = fft.decide(gamble_data)
    print(result)
    print("FFT evaluation successful.")
    print("FFT registry:", FFT.FFT_registry)
    #FFT.save_registry("fft_registry.yaml")


if __name__ == "__main__":
    main() 
