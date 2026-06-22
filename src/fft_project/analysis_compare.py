import logging

import numpy as np

from fft_project.cue_features import expected_isoelastic_utility

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
    if rng is None:
        rng = np.random

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
    return rng.choice(["left", "right"])


def match_eta(
    eta,
    choice,
    dynamic,
    wealth,
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
    missing_columns = [column for column in GAMBLE_COLUMNS if column not in gamble_data.columns]
    if missing_columns:
        logger.error(f"Gamble data is missing required columns: {missing_columns}")
        raise ValueError(f"Gamble data is missing required columns: {missing_columns}")


def _latest_run(results, fft_id):
    if not hasattr(results.columns, "names") or "run" not in results.columns.names:
        raise ValueError("results must have MultiIndex columns with a 'run' level.")

    fft_columns = results.loc[:, results.columns.get_level_values("fft_id") == fft_id]
    if fft_columns.empty:
        raise ValueError(f"No results found for fft_id '{fft_id}'.")

    return max(fft_columns.columns.get_level_values("run"))


def _result_series(results, fft_id, run, metric):
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

    if len(gamble_data) != len(choices) or len(gamble_data) != len(wealth):
        raise ValueError("gamble_data, choices, and wealth must have the same length.")

    if rng is None:
        rng = np.random

    matches = []
    for (_, gamble), choice, starting_wealth in zip(
        gamble_data.iterrows(),
        choices,
        wealth,
    ):
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
    accuracies = []

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


def eta_compare(gamble_data, results, eta, fft_id, runs=None, dynamic="additive", random_seed=None):
    """
    Compare stored FFT decisions with deterministic EUT decisions for one eta.

    This mirrors Experiment.eta_compare without creating Cue, FFT, or Experiment
    objects. The wealth used for the eta decision is the stored wealth_pre for
    the selected fft_id and run.
    """
    _validate_gamble_data(gamble_data)

    if runs is None:
        runs = _latest_run(results, fft_id)

    if random_seed is None:
        rng = np.random
    else:
        rng = np.random.RandomState(random_seed)

    choices = _result_series(results, fft_id, runs, "selected_side")
    wealth = _result_series(results, fft_id, runs, "wealth_pre")

    matches = match_eta_data(gamble_data, choices, wealth, eta, dynamic, rng)
    if len(matches) == 0:
        raise ValueError("Cannot calculate eta comparison for empty data.")

    return sum(matches) / len(matches)


def eta_match(
    gamble_data,
    results,
    fft_id,
    runs=None,
    eta_values=None,
    dynamic="additive",
    random_seed=None,
):
    """
    Compare stored FFT decisions with EUT decisions across eta values.

    Returns two lists: eta values and accuracies.
    """
    if eta_values is None:
        eta_values = np.arange(-2, 5, 0.25)

    accuracies = []
    for eta in eta_values:
        accuracy = eta_compare(gamble_data, results, eta, fft_id, runs, dynamic, random_seed)
        accuracies.append(accuracy)

    return list(eta_values), accuracies
