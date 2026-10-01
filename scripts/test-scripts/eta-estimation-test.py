# estimates eta from the real experimental choices with the ML / MAP method, no MCMC
# prints a pooled eta per condition and a per-participant eta for both conditions
# sanity check: eta should be clearly higher in the multiplicative condition than in
# the additive one, which is the central finding of the paper
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
    # the observed choices and the wealth before each choice, lined up with gamble_data
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

    # pooled estimate, all choices treated as one big participant
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

    # per-participant estimate
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

    # compare the two conditions for each participant
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
