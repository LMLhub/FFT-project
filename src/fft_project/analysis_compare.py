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


def run_dynamic_experiment(
    dynamic,
    gamble_data,
    experimental_results,
    fft_ids,
    random_seed=42,
    wealth_update="data",
):
    """
    Run a collection of registered FFTs and append participant observations.

    Return the combined results, the FFT IDs including the experimental result
    ID, and that experimental result ID. FFT registration remains the caller's
    responsibility so this helper can be reused with different registries.
    """
    import pandas as pd

    from fft_project.decision_class import FFT
    from fft_project.experiment_class import Experiment

    suffix = "a" if dynamic == "additive" else "m"
    fft_ids = list(fft_ids)
    experiment = Experiment(
        id=f"exp_participant_{suffix}",
        name="Participant-level comparison",
        dynamic=dynamic,
        description="Compare FFT and experimental results by participant.",
        gamble_data=gamble_data,
        ffts=[FFT.FFT_registry[fft_id] for fft_id in fft_ids],
    )
    results = experiment.run_experiment(
        wealth_update=wealth_update,
        random_seed=random_seed,
    )

    participant_result_id = f"experiment_{suffix}"
    results = pd.concat([results, experimental_results], axis=1)
    return results, fft_ids + [participant_result_id], participant_result_id

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


def plot_participant_accuracy_gamma_scatter(
    results,
    fft_ids,
    participant_result_id,
    reference_id,
    ax=None,
    title=None,
    runs=1,
):
    """
    Plot accuracy and average chosen growth rate separately by participant.

    Each point represents one participant/FFT combination. FFTs retain a
    consistent colour, and the observed participant IDs are read from the
    participant result columns.
    """
    if ax is None:
        import matplotlib.pyplot as plt

        _, ax = plt.subplots()
    else:
        import matplotlib.pyplot as plt

    participant_ids = results[
        (participant_result_id, runs, "participant_id")
    ]
    participants = participant_ids.dropna().unique()
    colours = plt.get_cmap("tab10")

    for fft_number, fft_id in enumerate(fft_ids):
        for participant_number, participant_id in enumerate(participants):
            participant_results = results.loc[
                participant_ids == participant_id
            ]
            accuracy = accuracy_against_reference(
                participant_results,
                fft_id,
                reference_id,
                runs=runs,
            )
            average_gamma = average_chosen_expected_gamma(
                participant_results,
                fft_id,
                runs=runs,
            )
            ax.scatter(
                accuracy,
                average_gamma,
                color=colours(fft_number % 10),
                marker=".",
                alpha=0.75,
                label=fft_id if participant_number == 0 else None,
            )

    ax.set_xlabel(f"Accuracy against {reference_id}")
    ax.set_ylabel("Time-average growth rate")
    ax.set_title(title or "Participant accuracy vs time-average growth rate")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
        ncol=2,
    )

    return ax


def plot_participant_accuracy_comparison(
    results,
    fft_ids,
    participant_result_id,
    x_reference_id,
    y_reference_id,
    ax=None,
    title=None,
    runs=1,
    x_label=None,
    y_label=None,
):
    """Plot two decision accuracies for every participant and FFT."""
    import matplotlib.pyplot as plt

    if ax is None:
        _, ax = plt.subplots()

    participant_ids = results[(participant_result_id, runs, "participant_id")]
    participants = participant_ids.dropna().unique()
    colours = plt.get_cmap("tab10")

    for fft_number, fft_id in enumerate(fft_ids):
        for participant_number, participant_id in enumerate(participants):
            participant_results = results.loc[participant_ids == participant_id]
            fft_choices = participant_results[(fft_id, runs, "selected_side")]

            accuracies = []
            for reference_id in (x_reference_id, y_reference_id):
                reference_choices = participant_results[
                    (reference_id, runs, "selected_side")
                ]
                valid_rows = fft_choices.notna() & reference_choices.notna()
                accuracies.append(
                    (fft_choices[valid_rows] == reference_choices[valid_rows]).mean()
                )

            ax.scatter(
                accuracies[0],
                accuracies[1],
                color=colours(fft_number % 10),
                marker=".",
                alpha=0.75,
                label=fft_id if participant_number == 0 else None,
            )

    ax.set_xlabel(x_label or f"Accuracy against {x_reference_id}")
    ax.set_ylabel(y_label or f"Accuracy against {y_reference_id}")
    ax.set_title(title or "Participant accuracy comparison")
    ax.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
        ncol=2,
    )
    return ax


def _rank_cutoff_label(cutoff):
    labels = {
        1: "1st",
        2: "1st or 2nd",
        3: "1st, 2nd or 3rd",
    }
    return labels.get(cutoff, f"Top {cutoff}")


def participant_rank_statistics(
    results,
    fft_ids,
    participant_result_id,
    rank_cutoffs=(1, 2, 3),
    runs=1,
):
    """
    Calculate FFT prevalence and accuracy within participant-level rankings.

    Ranking is based on agreement with each participant's experimental choices.
    Tied FFTs split the available rank slots evenly. Fractions are normalized
    within each cutoff and therefore sum to one.
    """
    import pandas as pd

    fft_ids = list(fft_ids)
    rank_cutoffs = tuple(sorted(set(rank_cutoffs)))
    if not rank_cutoffs or rank_cutoffs[0] < 1:
        raise ValueError("rank_cutoffs must contain positive integers")

    credits = {
        cutoff: pd.Series(0.0, index=fft_ids) for cutoff in rank_cutoffs
    }
    accuracy_sums = {
        cutoff: pd.Series(0.0, index=fft_ids) for cutoff in rank_cutoffs
    }
    participant_ids = results[(participant_result_id, runs, "participant_id")]
    participant_count = 0

    for participant_id in participant_ids.dropna().unique():
        participant_results = results.loc[participant_ids == participant_id]
        reference_choices = participant_results[
            (participant_result_id, runs, "selected_side")
        ]
        participant_accuracies = {}
        for fft_id in fft_ids:
            fft_choices = participant_results[(fft_id, runs, "selected_side")]
            valid_rows = fft_choices.notna() & reference_choices.notna()
            participant_accuracies[fft_id] = (
                fft_choices[valid_rows] == reference_choices[valid_rows]
            ).mean()
        accuracies = pd.Series(participant_accuracies, dtype=float).dropna()
        if accuracies.empty:
            continue

        participant_count += 1
        accuracy_tiers = sorted(accuracies.unique(), reverse=True)
        for cutoff in rank_cutoffs:
            remaining_slots = min(float(cutoff), float(len(accuracies)))
            for accuracy_level in accuracy_tiers:
                tied_ffts = accuracies.index[
                    np.isclose(accuracies, accuracy_level)
                ]
                tier_slots = min(remaining_slots, float(len(tied_ffts)))
                tier_credit = tier_slots / len(tied_ffts)
                credits[cutoff].loc[tied_ffts] += tier_credit
                accuracy_sums[cutoff].loc[tied_ffts] += (
                    accuracies.loc[tied_ffts] * tier_credit
                )
                remaining_slots -= tier_slots
                if remaining_slots == 0:
                    break

    fractions = {}
    average_accuracies = {}
    first_cutoff = rank_cutoffs[0]
    for cutoff in rank_cutoffs:
        total_credit = credits[cutoff].sum()
        fractions[cutoff] = (
            credits[cutoff] / total_credit if total_credit else credits[cutoff].copy()
        )
        average_accuracies[cutoff] = accuracy_sums[cutoff].div(
            credits[cutoff].replace(0, np.nan)
        )

    stacking_order = fractions[first_cutoff].sort_values(
        ascending=False,
        kind="stable",
    ).index
    fractions = {
        cutoff: values.reindex(stacking_order)
        for cutoff, values in fractions.items()
    }

    return {
        "fft_ids": fft_ids,
        "rank_cutoffs": rank_cutoffs,
        "fractions": fractions,
        "average_accuracies": average_accuracies,
        "credits": credits,
        "participant_count": participant_count,
    }


def plot_participant_rank_fractions(
    rank_statistics,
    ax=None,
    title=None,
    width=0.9,
):
    """Plot normalized FFT shares as stacked bars for each rank cutoff."""
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    if ax is None:
        _, ax = plt.subplots()

    fft_ids = rank_statistics["fft_ids"]
    colours = plt.get_cmap("tab10")
    fft_colours = {
        fft_id: colours(fft_number % 10)
        for fft_number, fft_id in enumerate(fft_ids)
    }

    for cutoff in rank_statistics["rank_cutoffs"]:
        bottom = 0.0
        for fft_id, fraction in rank_statistics["fractions"][cutoff].items():
            ax.bar(
                _rank_cutoff_label(cutoff),
                fraction,
                bottom=bottom,
                color=fft_colours[fft_id],
                width=width,
            )
            bottom += fraction

    ax.set_ylim(0, 1)
    ax.set_ylabel("Fraction of individuals")
    ax.set_title(title or "Participant FFT rankings")
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))
    ax.legend(
        handles=[
            Patch(facecolor=fft_colours[fft_id], label=fft_id)
            for fft_id in fft_ids
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.15),
        fontsize=7,
        ncol=2,
    )
    return ax


def plot_rank_fraction_accuracy(
    rank_statistics,
    ax=None,
    title=None,
    y_limits=(0.55, 0.80),
    x_label="Relative prevalence among individuals",
):
    """Plot rank prevalence against weighted participant accuracy for each FFT."""
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from matplotlib.ticker import PercentFormatter

    if ax is None:
        _, ax = plt.subplots()

    fft_ids = rank_statistics["fft_ids"]
    rank_cutoffs = rank_statistics["rank_cutoffs"]
    marker_options = ("o", "s", "^", "D", "v", "P", "X")
    rank_markers = {
        cutoff: marker_options[number % len(marker_options)]
        for number, cutoff in enumerate(rank_cutoffs)
    }
    colours = plt.get_cmap("tab10")
    fft_colours = {
        fft_id: colours(fft_number % 10)
        for fft_number, fft_id in enumerate(fft_ids)
    }

    for cutoff in rank_cutoffs:
        for fft_id in fft_ids:
            accuracy = rank_statistics["average_accuracies"][cutoff].loc[fft_id]
            if np.isnan(accuracy):
                continue
            ax.scatter(
                rank_statistics["fractions"][cutoff].loc[fft_id],
                accuracy,
                color=fft_colours[fft_id],
                marker=rank_markers[cutoff],
                alpha=0.8,
            )

    for fft_id in fft_ids:
        points = [
            (
                rank_statistics["fractions"][cutoff].loc[fft_id],
                rank_statistics["average_accuracies"][cutoff].loc[fft_id],
            )
            for cutoff in rank_cutoffs
            if not np.isnan(
                rank_statistics["average_accuracies"][cutoff].loc[fft_id]
            )
        ]
        if len(points) > 1:
            x_values, y_values = zip(*points)
            ax.plot(
                x_values,
                y_values,
                color=fft_colours[fft_id],
                linewidth=1,
                alpha=0.6,
            )

    ax.set_xlim(left=0)
    ax.set_ylim(*y_limits)
    ax.set_box_aspect(1)
    ax.set_xlabel(x_label)
    ax.set_ylabel("Average accuracy against experimental data")
    ax.set_title(title or "Rank prevalence vs participant accuracy")
    ax.xaxis.set_major_formatter(PercentFormatter(1.0))
    ax.yaxis.set_major_formatter(PercentFormatter(1.0))

    fft_legend = ax.legend(
        handles=[
            Patch(facecolor=fft_colours[fft_id], label=fft_id)
            for fft_id in fft_ids
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.18),
        fontsize=7,
        ncol=2,
    )
    ax.add_artist(fft_legend)
    ax.legend(
        handles=[
            Line2D(
                [0],
                [0],
                marker=rank_markers[cutoff],
                color="black",
                linestyle="None",
                label=_rank_cutoff_label(cutoff),
            )
            for cutoff in rank_cutoffs
        ],
        loc="upper center",
        bbox_to_anchor=(0.5, -0.43),
        fontsize=7,
        ncol=len(rank_cutoffs),
    )
    return ax
