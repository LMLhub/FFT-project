import logging
import numpy as np
from fft_project.cue_features import expected_isoelastic_utility
import random
import yaml
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

logger = logging.getLogger(__name__)

GAMBLE_COLUMNS = [
    "gamma_left_up",
    "gamma_left_down",
    "gamma_right_up",
    "gamma_right_down",
]

def eta_choice(
    eta,
    dynamic,
    wealth,
    x_left_up,
    x_left_down,
    x_right_up,
    x_right_down,
    rng=None,
):
    """
    Return the side selected by expected isoelastic utility for one gamble.

    If the eta cue does not make a decision, choose randomly like FFT.decide().
    """
    #If wealth is less than 0 and dynamics is additive, reset wealth to 1000:
    if wealth + np.min([x_left_up,x_left_down,x_right_up,x_right_down]) <= 0 and dynamic == "additive":
        wealth = 1000

    left_utility = expected_isoelastic_utility(
        x_left_up,
        x_left_down,
        x_right_up,
        x_right_down,
        wealth,
        dynamic,
        eta,
    )
    right_utility = expected_isoelastic_utility(
        x_right_up,
        x_right_down,
        x_left_up,
        x_left_down,
        wealth,
        dynamic,
        eta,
    )

    utility_difference = left_utility - right_utility
    cue_value = abs(utility_difference)
    
    if cue_value > 0 and utility_difference > 0:
        return "left"
    if cue_value > 0:
        return "right"
    
    elif abs(cue_value) < 10**(-20) :
        print(f"Random choice: No difference in utility for eta {eta} for {x_left_up, x_left_down, x_right_up, x_right_down, wealth}")
        return random.choice(["left", "right"])
    #Retrun error if no choice could be made
    
    logger.error("Utility couldn't be calculated")
    raise ValueError("Utility couldn't be calculated")


def match_eta(
    eta,
    choice, #Reference choice to which EUT is compared
    dynamic, # additive or multiplicative
    wealth, # Wealth before gamble is chosen and realised
    x_left_up, 
    x_left_down,
    x_right_up,
    x_right_down,
    rng=None,
):
    """
    Compare a given choice with the choice implied by EUT for eta.
    Returns 1 for a match and 0 otherwise.
    """
    eta_side = eta_choice(
        eta,
        dynamic,
        wealth,
        x_left_up,
        x_left_down,
        x_right_up,
        x_right_down,
        rng,
    )
    return int(choice == eta_side)


def _validate_gamble_data(gamble_data):
    # Checks if gamma values exists in gamble data
    missing_columns = [column for column in GAMBLE_COLUMNS if column not in gamble_data.columns]
    if missing_columns:
        logger.error(f"Gamble data is missing required columns: {missing_columns}")
        raise ValueError(f"Gamble data is missing required columns: {missing_columns}")


def _latest_run(results, fft_id):
    # If no run value is given,
    # this function finds the last experimental run and returns it.
    
    # Checks if runs are in the results
    if not hasattr(results.columns, "names") or "run" not in results.columns.names:
        logger.error("results must have MultiIndex columns with a 'run' level.")
        raise ValueError("results must have MultiIndex columns with a 'run' level.")

    fft_columns = results.loc[:, results.columns.get_level_values("fft_id") == fft_id]
    
    # Checks if fft_id is in the results columns
    if fft_columns.empty:
        logger.error(f"No results found for fft_id '{fft_id}'.")
        raise ValueError(f"No results found for fft_id '{fft_id}'.")

    #Return last run
    return max(fft_columns.columns.get_level_values("run"))

def _result_series(results, fft_id, run, metric):
    # Retrieve the relevant data from the results data
    column = (fft_id, run, metric)
    if column not in results.columns:
        logger.error(f"Result column {column} not found.")
        raise ValueError(f"Result column {column} not found.")

    return results[column]


def match_eta_data(gamble_data, choices, wealth, eta, dynamic, rng=None):
    """
    Calculate eta match values for a full data set of gambles.
    choices and wealth should have the same order as gamble_data.
    """
    _validate_gamble_data(gamble_data)
    
    choices = list(choices)
    wealth = list(wealth)

    # Check that different data series has the same length
    if len(gamble_data) != len(choices) or len(gamble_data) != len(wealth):
        logger.error(f"gamble_data ({len(gamble_data)}), choices ({len(choices)}), and wealth ({len(wealth)}) must have the same length.")
        raise ValueError(f"gamble_data ({len(gamble_data)}), choices ({len(choices)}), and wealth ({len(wealth)}) must have the same length.")

    # Set random seed if no seed is provided:
    if rng is None:
        rng = np.random

    # Initialise list of matches between EUT(eta) and choices
    matches = []

    # Calculate the choice for each gamble given eta
    for (_, gamble), choice, starting_wealth in zip(
        gamble_data.iterrows(),
        choices,
        wealth,
    ):
        # Calculate match for each row and append it to matches
        matches.append(
            match_eta(
                eta,
                choice,
                dynamic,
                starting_wealth,
                gamble["gamma_left_up"],
                gamble["gamma_left_down"],
                gamble["gamma_right_up"],
                gamble["gamma_right_down"],
                rng,
            )
        )
        rng.choice(["up", "down"])

    return matches


def match_etas_data(gamble_data, choices, wealth, etas, dynamic, random_seed=None):
    """
    Calculate eta accuracies for a list of eta values.
    Returns two lists: eta values and their accuracies.
    """
    #Initialise the list of accuracies for each eta
    accuracies = []

    #For each value of eta, find the matches and calculate the accuracy
    for eta in etas:
        if random_seed is None:
            rng = np.random
        else:
            rng = np.random.RandomState(random_seed)

        matches = match_eta_data(gamble_data, choices, wealth, eta, dynamic, rng)
        if len(matches) == 0:
            raise ValueError("Cannot calculate eta matches for empty data.")
        accuracies.append(sum(matches) / len(matches))

    return list(etas), accuracies


def etas_compare(gamble_data, results, etas, fft_id, dynamic, runs=None, random_seed=None):
    """
    Compare stored FFT decisions with deterministic EUT decisions for a list of etas.
    The wealth used for the eta decision is the stored wealth_pre for
    the selected fft_id and run.
    """
    _validate_gamble_data(gamble_data)

    # Get the latest run if no run is given
    if runs is None:
        runs = _latest_run(results, fft_id)

    # Get the reference choices and wealth data from the experiment
    choices = _result_series(results, fft_id, runs, "selected_side")
    wealth = _result_series(results, fft_id, runs, "wealth_pre")

    # Calculate the match accuracy for each eta.
    etas, accuracies = match_etas_data(gamble_data, choices, wealth, etas, dynamic, random_seed)
    
    return etas, accuracies


def eta_match(gamble_data, results, fft_id, runs=None, eta_values=None, dynamic="additive", random_seed=None):
    """
    Backwards-compatible wrapper for comparing one FFT to a list of eta values.
    """
    if eta_values is None:
        eta_values = np.arange(-2, 5, 0.25)

    return etas_compare(
        gamble_data,
        results,
        eta_values,
        fft_id,
        dynamic,
        runs=runs,
        random_seed=random_seed,
    )


def plot_etas_compare(
    gamble_data,
    results,
    fft_names,
    etas,
    dynamic,
    ax=None,
    runs=1,
    random_seed=None,
    title=None,
):
    """
    Plot eta-match accuracies for a list of FFT ids.

    This is the reusable version of each subplot loop in eta-visual-test.py.
    Pass an existing matplotlib axis to draw into a subplot, or omit ax to create
    a standalone figure and axis.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    with open("fft_registry.yaml", "r") as f:
        fft_registry = yaml.safe_load(f)
    
    for fft_name in fft_names:
        etas_result, fft_accuracy = etas_compare(
            gamble_data,
            results,
            etas,
            fft_name,
            dynamic,
            runs=runs,
            random_seed=random_seed,
        )
        
        if "experiment" in fft_name:
            plot_name = "Experimental data"
            style = "--"
        else:
            plot_name = fft_registry[fft_name]["name"]
            style = "-"
            
        ax.plot(etas_result, fft_accuracy, label=plot_name, linestyle = style)

    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=8,
    )
    ax.set_xlabel("Eta")
    ax.set_ylabel("Accuracy")

    ax.set_title(title or f"{dynamic.capitalize()} dynamics")

    return ax

def average_chosen_expected_gamma(results, fft_id, runs=None):
    """
    Return the average chosen_expected_gamma for a given FFT and run.

    If runs is None, use the latest run available for the FFT.
    """
    if runs is None:
        runs = _latest_run(results, fft_id)

    chosen_expected_gamma = _result_series(
        results,
        fft_id,
        runs,
        "chosen_expected_gamma",
    )

    return chosen_expected_gamma.mean()


def accuracy_against_reference(results, fft_id, reference_id="fft_gr", runs=None):
    """
    Return the decision accuracy of one FFT against a reference FFT.

    If runs is None, use the latest run available for fft_id.
    """
    if runs is None:
        runs = _latest_run(results, fft_id)

    fft_decisions = _result_series(results, fft_id, runs, "selected_side")
    reference_decisions = _result_series(results, reference_id, runs, "selected_side")

    if len(fft_decisions) != len(reference_decisions):
        raise ValueError("FFT decisions and reference decisions must have the same length.")

    return (fft_decisions == reference_decisions).mean()


def plot_accuracy_gamma_scatter(
    results,
    fft_ids=None,
    reference_id="fft_gr",
    runs=None,
    ax=None,
    title=None,
):
    """
    Scatter plot FFT accuracy against a reference and average chosen expected gamma.

    x-axis: accuracy against reference_id.
    y-axis: average chosen_expected_gamma.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()

    if fft_ids is None:
        fft_ids = list(dict.fromkeys(results.columns.get_level_values("fft_id")))

    with open("fft_registry.yaml", "r") as f:
        fft_registry = yaml.safe_load(f)

    if "experiment" in reference_id:
        reference_name = "Experimental data"
    else:
        reference_name = fft_registry.get(reference_id, {}).get("name", reference_id)

    for fft_id in fft_ids:
        run = _latest_run(results, fft_id) if runs is None else runs
        accuracy = accuracy_against_reference(results, fft_id, reference_id, run)
        avg_gamma = average_chosen_expected_gamma(results, fft_id, run)

        if "experiment" in fft_id:
            plot_name = "Experimental data"
        else:
            plot_name = fft_registry.get(fft_id, {}).get("name", fft_id)

        ax.scatter(accuracy, avg_gamma)

        if accuracy > 0.8:
            label_offset = (-6, 0)
            horizontal_alignment = "right"
            vertical_alignment = "center"
        elif plot_name == "Positive fractal signs then avoid the worst":
            plot_name = "Positive fractal signs\nthen avoid the worst"
            label_offset = (6, 0)
            horizontal_alignment = "left"
            vertical_alignment = "bottom"
        else:
            label_offset = (6, 0)
            horizontal_alignment = "left"
            vertical_alignment = "center"

        ax.annotate(
            plot_name,
            (accuracy, avg_gamma),
            xytext=label_offset,
            textcoords="offset points",
            va=vertical_alignment,
            ha=horizontal_alignment,
        )

    ax.set_xlabel(f"Accuracy against {reference_name}")
    ax.set_ylabel("Time-average growth rate")
    ax.set_title(title or "Accuracy vs time-average growth rate")

    return ax
