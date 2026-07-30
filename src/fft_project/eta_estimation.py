# This module estimates the risk aversion parameter eta from observed choices.
#
# It implements the simple maximum likelihood / MAP version of the model
# described in docs/docs/eta-estimation.md: the probability of choosing the
# left gamble is a logistic function of the difference in expected isoelastic
# utility between the two gambles, and we find the (eta, beta) pair that best
# explains the observed choices with a numerical optimiser (no MCMC).
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

# Weakly informative priors used for the MAP estimate. They only rule out
# absurd values; within a broad range the data decides. eta is normal, and
# log(beta) is normal (so beta itself is log-normal), mirroring the paper.
DEFAULT_PRIORS = {
    "eta_mean": 0.0,
    "eta_sd": 2.5,
    "log_beta_mean": 0.0,
    "log_beta_sd": 1.5,
}

# Bounds keep the optimiser in a sensible region and prevent overflow in
# exp(log_beta) for near-separable data.
_BOUNDS = [(-4.0, 6.0), (-6.0, 8.0)]  # (eta, log_beta)


def _mean_isoelastic_utility(up, down, wealth, dynamic, eta):
    # Mean isoelastic utility of a single gamble (two equally likely outcomes).
    # Vectorised over trials; identical in meaning to
    # cue_features.expected_isoelastic_utility.
    if dynamic == "multiplicative":
        x1 = np.exp(up) * wealth
        x2 = np.exp(down) * wealth
    elif dynamic == "additive":
        x1 = up + wealth
        x2 = down + wealth
    else:
        raise ValueError("Invalid dynamic. Must be 'multiplicative' or 'additive'.")

    tol = 1e-15
    if abs(eta - 1) < tol:
        return (np.log(x1) + np.log(x2)) / 2
    if abs(eta) < tol:
        return (x1 + x2) / 2
    return (np.power(x1, 1 - eta) + np.power(x2, 1 - eta)) / (2 * (1 - eta))


def _safe_wealth(gamble_data, wealth, dynamic):
    # Under additive dynamics an outcome can push wealth to zero or below,
    # which makes the isoelastic utility undefined. Mirror the safeguard in
    # analysis_compare.eta_choice: reset such trials to a wealth of 1000.
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


def _neg_log_posterior(params, delta_f_of_eta, y, priors):
    # Negative log-posterior (or negative log-likelihood if priors is None).
    # params = (eta, log_beta). We optimise log_beta so beta stays positive.
    eta, log_beta = params
    beta = np.exp(log_beta)

    z = beta * delta_f_of_eta(eta)  # = beta * Delta<delta f_eta>

    # Numerically stable Bernoulli log-likelihood:
    #   log theta       = -log(1 + e^-z) = -logaddexp(0, -z)
    #   log(1 - theta)  = -log(1 + e^+z) = -logaddexp(0, +z)
    log_lik = np.sum(
        y * (-np.logaddexp(0.0, -z)) + (1.0 - y) * (-np.logaddexp(0.0, z))
    )

    if priors is not None:
        log_lik += -0.5 * ((eta - priors["eta_mean"]) / priors["eta_sd"]) ** 2
        log_lik += -0.5 * ((log_beta - priors["log_beta_mean"]) / priors["log_beta_sd"]) ** 2

    return -log_lik


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
    Estimate (eta, beta) for one set of choices by maximising the log-likelihood.

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

    y = (np.asarray(choices) == "left").astype(float)
    if len(y) != len(gamble_data) or len(y) != len(wealth):
        raise ValueError("gamble_data, choices and wealth must have the same length.")
    if len(y) == 0:
        raise ValueError("Cannot estimate eta from empty data.")

    w = _safe_wealth(gamble_data, wealth, dynamic)

    # Precompute the gamma arrays once; only eta changes during optimisation.
    l_up = gamble_data["gamma_left_up"].to_numpy(dtype=float)
    l_down = gamble_data["gamma_left_down"].to_numpy(dtype=float)
    r_up = gamble_data["gamma_right_up"].to_numpy(dtype=float)
    r_down = gamble_data["gamma_right_down"].to_numpy(dtype=float)

    def delta_f_of_eta(eta):
        # Delta<delta f_eta> = <f_eta(left)> - <f_eta(right)>.
        # The f_eta(current wealth) term cancels between left and right.
        return (
            _mean_isoelastic_utility(l_up, l_down, w, dynamic, eta)
            - _mean_isoelastic_utility(r_up, r_down, w, dynamic, eta)
        )

    if method == "map":
        used_priors = {**DEFAULT_PRIORS, **(priors or {})}
    elif method == "mle":
        used_priors = None
    else:
        raise ValueError("method must be 'map' or 'mle'.")

    rng = np.random.default_rng(random_seed)
    starts = [np.array([0.5, 0.0])]
    for _ in range(max(0, n_starts - 1)):
        starts.append(np.array([rng.uniform(-2.0, 3.0), rng.uniform(-2.0, 2.0)]))

    best = None
    for x0 in starts:
        res = minimize(
            _neg_log_posterior,
            x0,
            args=(delta_f_of_eta, y, used_priors),
            method="Nelder-Mead",
            bounds=_BOUNDS,
        )
        if best is None or res.fun < best.fun:
            best = res

    eta_hat, log_beta_hat = best.x
    # Report the plain log-likelihood (priors excluded) for comparability.
    loglik = -_neg_log_posterior(best.x, delta_f_of_eta, y, None)

    return {
        "eta": float(eta_hat),
        "beta": float(np.exp(log_beta_hat)),
        "n_trials": int(len(y)),
        "loglik": float(loglik),
        "success": bool(best.success),
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

    rows = []
    for pid in pd.unique(participant_ids):
        mask = participant_ids == pid
        result = estimate_eta(
            gamble_data.loc[mask],
            choices[mask],
            wealth[mask],
            dynamic,
            method=method,
            priors=priors,
            n_starts=n_starts,
            random_seed=random_seed,
        )
        result["participant_id"] = pid
        rows.append(result)

    columns = ["participant_id", "eta", "beta", "n_trials", "loglik", "success"]
    return pd.DataFrame(rows)[columns]
