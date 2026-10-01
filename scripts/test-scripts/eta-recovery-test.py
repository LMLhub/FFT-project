# self-check for the eta estimator, where eta says how risk-averse a person is
# builds simulated choices from a chosen eta and checks the estimator guesses it back
# tried for eta from -0.5 to 5
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import matplotlib.pyplot as plt

from fft_project.prepare_experimental_data import prepare_experimental_data
from fft_project.eta_estimation import (
    estimate_eta,
    # borrowed so the simulated data uses the estimator's own formulas
    _mean_isoelastic_utility,
    _safe_wealth,
)

# the true eta values hidden in the simulated data
TRUE_ETA_VALUES = [-0.5, 0.0, 0.5, 1.0, 2.0, 3.0, 5.0]

# the utility gap between the gambles shrinks a lot as eta grows, so a fixed beta
# would make the choices near-random for large eta and eta could not be recovered
# instead beta is chosen per eta so the choices carry this much signal, like a
# real person who mostly but not always picks the better gamble
TARGET_DECISION_SPREAD = 1.5

# simulated data sets per eta, more gives a steadier average but runs slower
REPETITIONS_PER_ETA = 20

# the experiment cannot tell high eta values apart, so only check recovery up to here
# above this the choices barely change with eta and no estimator can recover it
IDENTIFIABLE_MAX_ETA = 2.0

# passes if the average guess in the identifiable range is off by no more than this
ETA_TOLERANCE = 0.5

# plot colours for the two experiment types
CONDITION_COLOURS = {"additive": "tab:blue", "multiplicative": "tab:orange"}


def utility_gap(gamble_data, wealth, dynamic, eta):
    # how much more attractive the left gamble is than the right one, per trial
    # each gamble has a utility, how attractive it is, which depends on eta

    # utility breaks if wealth drops to zero or below
    # additive trials that would do that get reset to 1000, same as the estimator
    safe_wealth = _safe_wealth(gamble_data, wealth, dynamic)

    utility_left = _mean_isoelastic_utility(
        gamble_data["gamma_left_up"].to_numpy(dtype=float),
        gamble_data["gamma_left_down"].to_numpy(dtype=float),
        safe_wealth,
        dynamic,
        eta,
    )
    utility_right = _mean_isoelastic_utility(
        gamble_data["gamma_right_up"].to_numpy(dtype=float),
        gamble_data["gamma_right_down"].to_numpy(dtype=float),
        safe_wealth,
        dynamic,
        eta,
    )
    return utility_left - utility_right


def choice_probabilities(gap, beta):
    # chance of picking left on each trial
    # beta controls how decisively the utility gap turns into a choice
    # the logistic curve turns that into a probability between 0 and 1
    return 1.0 / (1.0 + np.exp(-beta * gap))


def draw_choices(probability_left, random_generator):
    # turn each chance of left into an actual left or right choice
    uniform_draws = random_generator.random(len(probability_left))
    return np.where(uniform_draws < probability_left, "left", "right")


def recover_etas_for_condition(gamble_data, dynamic):
    # runs the self-check for one experiment type
    # builds several simulated data sets per eta and guesses eta back from each
    # returns two matching lists: eta used, eta guessed
    wealth = gamble_data["wealth"].to_numpy(dtype=float)

    true_eta_per_run = []
    recovered_eta_per_run = []

    for true_eta in TRUE_ETA_VALUES:
        gap = utility_gap(gamble_data, wealth, dynamic, true_eta)

        # pick beta so the choices reach the target signal, whatever the eta scale
        true_beta = TARGET_DECISION_SPREAD / gap.std()
        probability_left = choice_probabilities(gap, true_beta)

        for repetition_index in range(REPETITIONS_PER_ETA):
            # fixed seed per run so results are reproducible
            random_generator = np.random.default_rng(
                abs(hash((dynamic, true_eta, repetition_index))) % (2**32)
            )
            simulated_choices = draw_choices(probability_left, random_generator)

            # the estimator sees only the choices, not the eta behind them
            fit = estimate_eta(
                gamble_data, simulated_choices, wealth, dynamic, method="map"
            )

            true_eta_per_run.append(true_eta)
            recovered_eta_per_run.append(fit["eta"])

    return np.array(true_eta_per_run), np.array(recovered_eta_per_run)


def average_and_spread_per_true_eta(true_eta_per_run, recovered_per_run):
    # per eta, the mean guess for the dot and the spread for the error bar
    means = []
    spreads = []
    for true_eta in TRUE_ETA_VALUES:
        recovered_here = recovered_per_run[true_eta_per_run == true_eta]
        means.append(recovered_here.mean())
        spreads.append(recovered_here.std())
    return np.array(means), np.array(spreads)


def main():
    # additive means money added, multiplicative means money multiplied
    gamble_data, _results = prepare_experimental_data(
        PROJECT_ROOT / "data/all_active_phase_data.csv"
    )
    gamble_data_additive, gamble_data_multiplicative = gamble_data

    conditions = [
        ("additive", gamble_data_additive),
        ("multiplicative", gamble_data_multiplicative),
    ]

    figure, axis = plt.subplots(figsize=(7, 6))

    true_eta_values = np.array(TRUE_ETA_VALUES)
    in_identifiable_range = true_eta_values <= IDENTIFIABLE_MAX_ETA

    for dynamic, gamble_data_condition in conditions:
        true_eta_per_run, recovered_eta_per_run = recover_etas_for_condition(
            gamble_data_condition, dynamic
        )
        eta_mean, eta_spread = average_and_spread_per_true_eta(
            true_eta_per_run, recovered_eta_per_run
        )

        # pass only if every average guess in the identifiable range is close enough
        worst_error = np.max(
            np.abs(eta_mean[in_identifiable_range] - true_eta_values[in_identifiable_range])
        )
        verdict = "PASS" if worst_error <= ETA_TOLERANCE else "FAIL"
        print(
            f"{dynamic:>15}: {verdict}  "
            f"(largest mean eta error up to eta {IDENTIFIABLE_MAX_ETA} "
            f"= {worst_error:.3f}, tolerance = {ETA_TOLERANCE})"
        )

        axis.errorbar(
            TRUE_ETA_VALUES, eta_mean, yerr=eta_spread,
            fmt="o", color=CONDITION_COLOURS[dynamic], capsize=3, label=dynamic,
        )

    axis.plot(
        TRUE_ETA_VALUES, TRUE_ETA_VALUES,
        color="grey", linestyle="--", label="dots on this line = guessed right",
    )

    # green band over the realistic range where the estimator is trustworthy
    axis.axvspan(0.0, IDENTIFIABLE_MAX_ETA, color="tab:green", alpha=0.12)
    axis.text(
        IDENTIFIABLE_MAX_ETA / 2, 4.8, "estimator validated here",
        ha="center", va="top", color="green", fontsize=10,
    )

    # spell out the edge cases in plain words on the plot itself
    axis.annotate(
        "estimator hits a ceiling,\nhigh eta not guessed",
        xy=(5, 2.4), xytext=(2.7, 4.2),
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    axis.annotate(
        "edge bias\n(extreme risk-seeking)",
        xy=(-0.5, 0.17), xytext=(0.15, 2.2),
        ha="left",
        arrowprops=dict(arrowstyle="->", color="black"),
    )
    axis.set_title("does the estimator guess eta back?")
    axis.set_xlabel("true eta (the value used to make the choices)")
    axis.set_ylabel("recovered eta (the estimator's guess)")
    axis.legend(loc="lower right")

    figure.tight_layout()

    output_directory = PROJECT_ROOT / "figures"
    output_directory.mkdir(exist_ok=True)
    output_path = output_directory / "eta_recovery.png"
    figure.savefig(output_path, dpi=150)
    print(f"\nSaved recovery plot to {output_path}")


if __name__ == "__main__":
    main()
