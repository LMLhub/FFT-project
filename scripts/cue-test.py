from fft_project.cue_class import Cue
import pandas as pd

def main():
    # Example feature function that calculates the expected value of the first gamble multiplied by some parameter
    def example_feature(gamma_left_up, gamma_left_down, gamma_right_up, gamma_right_down, multiplier=5):
        ev_left = (gamma_left_up + gamma_left_down) / 2
        return ev_left*multiplier

    # Create a Cue instance
    cue = Cue(id="ex1", name="Example Cue",
              description="A cue that evaluates expected values of the first gamble.",
              feature=example_feature, type="boolean", params={"multiplier": 5})

    # Create a sample gamble_data dataframe
    gamble_data = pd.DataFrame({
        "gamma_left_up": [10, 20],
        "gamma_left_down": [10, 15],
        "gamma_right_up": [12, 18],
        "gamma_right_down": [8, 14]
    })

    # Evaluate the cue on the gamble_data
    result = cue.evaluate(gamble_data)
    print(result)

if __name__ == "__main__":
    main() 
