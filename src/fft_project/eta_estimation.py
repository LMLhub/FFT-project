# estimates the risk aversion parameter eta from a person's observed choices
# eta says how risk-averse someone is, a higher eta means more cautious
# the chance of choosing the left gamble is a logistic function of how much more
# attractive the left gamble is than the right one
# the best (eta, beta) pair is found with a numerical optimiser, no MCMC
# the full method is described in docs/docs/eta-estimation.md
import logging
import numpy as np
from scipy.optimize import minimize

logger = logging.getLogger(__name__)

GAMBLE_COLUMNS = [
    "gamma_left_up",
    "gamma_left_down",
    "gamma_right_up",
    "gamma_right_down",
]

# weak prior beliefs for the MAP estimate, they only rule out absurd values
# eta follows a normal distribution and log(beta) follows a normal distribution
# so beta itself is log-normal, matching the paper
DEFAULT_PRIORS = {
    "eta_mean": 0.0,
    "eta_sd": 2.5,
    "log_beta_mean": 0.0,
    "log_beta_sd": 1.5,
}

# keeps the optimiser in a sensible range for eta and log_beta
# also avoids overflow when turning log_beta back into beta
_BOUNDS = [(-4.0, 6.0), (-6.0, 8.0)]


def _mean_isoelastic_utility(up, down, wealth, dynamic, eta):
    # average attractiveness (utility) of one gamble with two equally likely outcomes
    # how much a given amount of money is worth to the person depends on eta
    # same meaning as cue_features.expected_isoelastic_utility
    if dynamic == "multiplicative":
        outcome_up = np.exp(up) * wealth
        outcome_down = np.exp(down) * wealth
    elif dynamic == "additive":
        outcome_up = up + wealth
        outcome_down = down + wealth
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")

    # eta close to 1 and eta close to 0 need their own formulas, the general one
    # would divide by zero there
    tolerance = 1e-15
    if abs(eta - 1) < tolerance:
        return (np.log(outcome_up) + np.log(outcome_down)) / 2
    if abs(eta) < tolerance:
        return (outcome_up + outcome_down) / 2
    return (np.power(outcome_up, 1 - eta) + np.power(outcome_down, 1 - eta)) / (2 * (1 - eta))


def _safe_wealth(gamble_data, wealth, dynamic):
    # in the additive experiment a bad outcome can push wealth to zero or below
    # the utility formula is undefined there, so those trials get wealth reset to 1000
    # same safeguard as analysis_compare.eta_choice
    wealth = np.asarray(wealth, dtype=float).copy()
    if dynamic == "additive":
        min_outcome = gamble_data[GAMBLE_COLUMNS].to_numpy().min(axis=1)
        wealth[wealth + min_outcome <= 0] = 1000.0
    return wealth


def _validate_gamble_data(gamble_data):
    missing = [c for c in GAMBLE_COLUMNS if c not in gamble_data.columns]
    if missing:
        logger.error(f"Gamble data is missing required columns: {missing}")
        raise ValueError(f"Gamble data is missing required columns: {missing}")


def _neg_log_posterior(params, utility_gap_of_eta, choices_left, priors):
    # how badly a given (eta, log_beta) explains the choices, smaller is better
    # returns the negative log-posterior, or the negative log-likelihood if priors is None
    # log_beta is optimised instead of beta so beta always stays positive
    eta, log_beta = params
    beta = np.exp(log_beta)

    # how strongly the model leans towards left on each trial
    decision_value = beta * utility_gap_of_eta(eta)

    # log-likelihood of the observed choices, written in a numerically stable way
    # so very confident predictions do not overflow
    log_likelihood = np.sum(
        choices_left * (-np.logaddexp(0.0, -decision_value))
        + (1.0 - choices_left) * (-np.logaddexp(0.0, decision_value))
    )

    if priors is not None:
        # add the prior beliefs on eta and log_beta for the MAP estimate
        log_likelihood += -0.5 * ((eta - priors["eta_mean"]) / priors["eta_sd"]) ** 2
        log_likelihood += -0.5 * ((log_beta - priors["log_beta_mean"]) / priors["log_beta_sd"]) ** 2

    return -log_likelihood


def estimate_eta(
    gamble_data,
    choices,
    wealth,
    dynamic,
    method="map",
    priors=None,
    n_starts=5,
    random_seed=0,
):
    """
    Estimate (eta, beta) for one set of choices by fitting the logistic choice model.

    Parameters
    ----------
    gamble_data : DataFrame with the four GAMBLE_COLUMNS (gamma values).
    choices : sequence of "left"/"right", aligned with gamble_data rows.
    wealth : sequence of wealth-before-choice values, aligned with gamble_data.
    dynamic : "additive" or "multiplicative".
    method : "map" (default, weak priors) or "mle" (no priors).
    priors : dict overriding DEFAULT_PRIORS, or None.
    n_starts : number of random restarts for the optimiser.
    random_seed : seed for the restart draws.

    Returns
    -------
    dict with keys: eta, beta, n_trials, loglik, success, method.
    """
    _validate_gamble_data(gamble_data)

    # 1.0 where the person chose left, 0.0 where they chose right
    choices_left = (np.asarray(choices) == "left").astype(float)
    if len(choices_left) != len(gamble_data) or len(choices_left) != len(wealth):
        raise ValueError("gamble_data, choices and wealth must have the same length.")
    if len(choices_left) == 0:
        raise ValueError("Cannot estimate eta from empty data.")

    safe_wealth = _safe_wealth(gamble_data, wealth, dynamic)

    # pull the four gamma columns once, only eta changes during optimisation
    left_up = gamble_data["gamma_left_up"].to_numpy(dtype=float)
    left_down = gamble_data["gamma_left_down"].to_numpy(dtype=float)
    right_up = gamble_data["gamma_right_up"].to_numpy(dtype=float)
    right_down = gamble_data["gamma_right_down"].to_numpy(dtype=float)

    def utility_gap(eta):
        # how much more attractive the left gamble is than the right one
        # the term for current wealth cancels out between left and right
        return (
            _mean_isoelastic_utility(left_up, left_down, safe_wealth, dynamic, eta)
            - _mean_isoelastic_utility(right_up, right_down, safe_wealth, dynamic, eta)
        )

    if method == "map":
        used_priors = {**DEFAULT_PRIORS, **(priors or {})}
    elif method == "mle":
        used_priors = None
    else:
        raise ValueError("method must be 'map' or 'mle'.")

    # several starting points so the optimiser does not get stuck in a poor spot
    random_generator = np.random.default_rng(random_seed)
    start_points = [np.array([0.5, 0.0])]
    for _ in range(max(0, n_starts - 1)):
        start_points.append(np.array([random_generator.uniform(-2.0, 3.0), random_generator.uniform(-2.0, 2.0)]))

    # keep the best fit across all starting points
    best_result = None
    for start_point in start_points:
        result = minimize(
            _neg_log_posterior,
            start_point,
            args=(utility_gap, choices_left, used_priors),
            method="Nelder-Mead",
            bounds=_BOUNDS,
        )
        if best_result is None or result.fun < best_result.fun:
            best_result = result

    estimated_eta, estimated_log_beta = best_result.x
    # report the plain log-likelihood without the priors, so fits are comparable
    loglik = -_neg_log_posterior(best_result.x, utility_gap, choices_left, None)

    return {
        "eta": float(estimated_eta),
        "beta": float(np.exp(estimated_log_beta)),
        "n_trials": int(len(choices_left)),
        "loglik": float(loglik),
        "success": bool(best_result.success),
        "method": method,
    }


def estimate_eta_per_participant(
    gamble_data,
    choices,
    wealth,
    participant_ids,
    dynamic,
    method="map",
    priors=None,
    n_starts=5,
    random_seed=0,
):
    """
    Run estimate_eta separately for each participant.

    Returns a DataFrame with one row per participant: participant_id, eta,
    beta, n_trials, loglik, success.
    """
    import pandas as pd

    participant_ids = np.asarray(participant_ids)
    choices = np.asarray(choices)
    wealth = np.asarray(wealth, dtype=float)
    gamble_data = gamble_data.reset_index(drop=True)

    results = []
    for participant_id in pd.unique(participant_ids):
        belongs_to_participant = participant_ids == participant_id
        result = estimate_eta(
            gamble_data.loc[belongs_to_participant],
            choices[belongs_to_participant],
            wealth[belongs_to_participant],
            dynamic,
            method=method,
            priors=priors,
            n_starts=n_starts,
            random_seed=random_seed,
        )
        result["participant_id"] = participant_id
        results.append(result)

    columns = ["participant_id", "eta", "beta", "n_trials", "loglik", "success"]
    return pd.DataFrame(results)[columns]
