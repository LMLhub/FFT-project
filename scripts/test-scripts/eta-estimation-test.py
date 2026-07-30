# Estimate the risk aversion parameter eta from the experimental choice data,
# using the simple maximum likelihood / MAP method (no MCMC).
#
# It prints:
#   1. a pooled eta per condition (all choices treated as one participant), and
#   2. a per-participant eta for the additive and multiplicative conditions,
#      with a short summary.
#
# Sanity check: the mean/median eta should be markedly higher under the
# multiplicative condition than under the additive one -- the central finding
# of the paper.
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

import numpy as np
import pandas as pd

from fft_project.prepare_experimental_data import prepare_experimental_data
from fft_project.eta_estimation import estimate_eta, estimate_eta_per_participant

SELECTED_SIDE = ("experiment_a", 1, "selected_side")  # column label pattern


def _choices_and_wealth(gamble_data, results, fft_id):
    # Pull the observed choices and pre-choice wealth for a condition, aligned
    # with the gamble_data rows.
    choices = results[(fft_id, 1, "selected_side")].to_numpy()
    wealth = gamble_data["wealth"].to_numpy(dtype=float)
    return choices, wealth


def main():
    gamble_data, results = prepare_experimental_data(
        PROJECT_ROOT / "data/all_active_phase_data.csv"
    )
    gamble_add, gamble_mul = gamble_data
    results_add, results_mul = results

    conditions = [
        ("additive", gamble_add, results_add, "experiment_a"),
        ("multiplicative", gamble_mul, results_mul, "experiment_m"),
    ]

    # 1. Pooled estimate per condition.
    print("=" * 60)
    print("Pooled estimate (all choices together)")
    print("=" * 60)
    for dynamic, gamble, res, fft_id in conditions:
        choices, wealth = _choices_and_wealth(gamble, res, fft_id)
        fit = estimate_eta(gamble, choices, wealth, dynamic, method="map")
        print(
            f"{dynamic:>15}: eta = {fit['eta']:+.3f}   "
            f"beta = {fit['beta']:.3f}   "
            f"n = {fit['n_trials']}"
        )

    # 2. Per-participant estimates.
    print()
    print("=" * 60)
    print("Per-participant estimate")
    print("=" * 60)
    per_participant = {}
    for dynamic, gamble, res, fft_id in conditions:
        choices, wealth = _choices_and_wealth(gamble, res, fft_id)
        df = estimate_eta_per_participant(
            gamble,
            choices,
            wealth,
            gamble["participant_id"].to_numpy(),
            dynamic,
            method="map",
        )
        per_participant[dynamic] = df
        print(
            f"{dynamic:>15}: median eta = {df['eta'].median():+.3f}   "
            f"mean eta = {df['eta'].mean():+.3f}   "
            f"participants = {len(df)}"
        )

    # 3. Compare the two conditions per participant.
    merged = per_participant["additive"].merge(
        per_participant["multiplicative"],
        on="participant_id",
        suffixes=("_add", "_mul"),
    )
    higher = (merged["eta_mul"] > merged["eta_add"]).mean()
    print()
    print(
        f"Participants with eta_mul > eta_add: {higher:.0%} "
        f"(expected: a clear majority)"
    )
    print()
    print("First 10 participants (eta per condition):")
    print(
        merged[["participant_id", "eta_add", "eta_mul"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
