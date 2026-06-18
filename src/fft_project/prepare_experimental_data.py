import numpy as np
import pandas as pd

def prepare_experimental_data(file_name):
    # Load the data
    data = pd.read_csv(file_name, sep=None, engine="python")

    #Update colummn names to match the expected format
    data = data.rename(columns={
        "eta": "dynamic",
        "trial": "trial_number",
        "realized_gamma": "realised_gamma",
    })

    data["dynamic"] = data["dynamic"].apply(lambda x: "additive" if x == 0 else "multiplicative")
    data["wealth_pre"] = data["wealth"] # calculate wealth before the gamble is resolved
    data["wealth_post"] = data["wealth"] + data["delta_wealth"] # calculate wealth after the gamble is resolved
    
    # Select only the relevant columns for our analysis
    data = data[["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", 
                 "dynamic",
                 "wealth",
                 "selected_side",
                 "gamble_up",
                 "chosen_expected_gamma",
                 "realised_gamma",
                 "participant_id",
                 "trial_number",
                 "wealth_pre",
                 "wealth_post"]]

    #Need to divide on dynamic!!!!
    data_additive = data[data["dynamic"] == "additive"].copy()
    data_multiplicative = data[data["dynamic"] == "multiplicative"].copy()

    # Organise the results columns in the same way as the output of our experiments, to make comparison easier.
    # First separate gamble and output data
    gamble_data_additive = data_additive[["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth", "participant_id", "trial_number"]]
    gamble_data_multiplicative = data_multiplicative[["gamma_left_up", "gamma_left_down", "gamma_right_up", "gamma_right_down", "wealth", "participant_id", "trial_number"]]

    result_metrics = ["trial_number", "participant_id","selected_side", "gamble_up", "wealth_pre", "wealth_post", "chosen_expected_gamma", "realised_gamma"]
    
    results_data_additive = data_additive[result_metrics].copy()
    results_data_additive.columns = pd.MultiIndex.from_tuples(
        [("experiment_a", 1, metric) for metric in result_metrics],
        names=["fft_id", "run", "metric"]
    )
    results_data_multiplicative = data_multiplicative[result_metrics].copy()
    results_data_multiplicative.columns = pd.MultiIndex.from_tuples(
        [("experiment_m", 1, metric) for metric in result_metrics],
        names=["fft_id", "run", "metric"]   )
    
    
    # Save the processed data to a new CSV file
    print(f"Experimental data loaded and processed.")

    return [gamble_data_additive, gamble_data_multiplicative], [results_data_additive, results_data_multiplicative] 
