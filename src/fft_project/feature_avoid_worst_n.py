# This module implements the 'avoid worst n ranks' cue as a standalone feature function.
#
# DESIGN NOTE — isolation:
#   This file is intentionally self-contained and does not import from cue_class.py
#   or cue_features.py. Those files are being developed on feature/12_cue_and_decision_classes
#   (PR #22), which is not yet merged to main. Once PR #22 merges, the function
#   avoid_worst_n_ranks() below can be moved into cue_features.py and wrapped in a
#   Cue object with type="boolean" and params={'n': n} with no changes to the logic.
#
# DESIGN NOTE — signed cue framework:
#   Following the theoretical framework in docs/notes/index.md, cues are defined
#   as signed differences:
#       F(g1, g2) = f(g1) - f(g2)
#   where f is a 1-sided feature that is non-decreasing as g1 gets better.
#
#   For this cue, f(g) = 1 if g avoids all worst-n fractal ranks, else 0.
#   This gives three possible outcomes:
#       F =  1  →  left avoids worst-n, right does not  →  prefer left
#       F = -1  →  right avoids worst-n, left does not  →  prefer right
#       F =  0  →  both or neither contain a worst-n fractal  →  undecided
#   The undecided case (F=0) is the natural resolution of the open question in
#   issue #25: when both gambles contain a worst-n fractal, or neither does,
#   the cue does not fire and the decision passes to the next cue in the tree.
#
# DESIGN NOTE — rank convention:
#   Fractal ranks are 0-indexed integers sorted in ascending order of value,
#   so rank 0 is the worst fractal and rank f-1 is the best (where f = total
#   number of fractals, 9 in the current experiment). The worst n ranks are
#   therefore {0, 1, ..., n-1}.

import pandas as pd


def avoid_worst_n_ranks(g1_up, g1_down, g2_up, g2_down, n):
    """
    1-sided feature for the 'avoid worst n ranks' cue.

    Returns True if the first gamble (g1) does NOT contain any fractal
    among the worst n ranks, and False otherwise. The second gamble's values
    (g2_up, g2_down) are accepted to keep the interface consistent with the
    Cue class in cue_class.py, which calls f(g1, g2) and f(g2, g1) in turn.

    Parameters
    ----------
    g1_up : int
        Rank of the up-fractal of gamble 1. 0-indexed, 0 = worst.
    g1_down : int
        Rank of the down-fractal of gamble 1.
    g2_up : int
        Rank of the up-fractal of gamble 2. Not used; kept for interface consistency.
    g2_down : int
        Rank of the down-fractal of gamble 2. Not used; kept for interface consistency.
    n : int
        Number of worst fractal ranks to avoid. Must satisfy 1 <= n < total fractals.

    Returns
    -------
    bool
        True if g1 avoids all worst-n ranks, False otherwise.
    """
    worst_ranks = set(range(n))
    return not ({g1_up, g1_down} & worst_ranks)


def signed_cue(left_up, left_down, right_up, right_down, n):
    """
    Computes the signed cue value F = f(left) - f(right).

    Parameters
    ----------
    left_up, left_down : int
        Fractal ranks for the left gamble.
    right_up, right_down : int
        Fractal ranks for the right gamble.
    n : int
        Number of worst fractal ranks to avoid.

    Returns
    -------
    int
        1  if left avoids worst-n and right does not (prefer left),
       -1  if right avoids worst-n and left does not (prefer right),
        0  if both or neither (undecided).
    """
    f_left = avoid_worst_n_ranks(left_up, left_down, right_up, right_down, n)
    f_right = avoid_worst_n_ranks(right_up, right_down, left_up, left_down, n)
    return int(f_left) - int(f_right)


def evaluate_df(df, n,
                left_up_col='fractal_left_up',
                left_down_col='fractal_left_down',
                right_up_col='fractal_right_up',
                right_down_col='fractal_right_down'):
    """
    Applies the avoid-worst-n-ranks cue to every row of a gamble DataFrame.

    Mirrors the interface of Cue.evaluate_df() in cue_class.py so that this
    function can be replaced by a proper Cue object once PR #22 merges.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame of gamble pairs, as produced by simulate_gamble_data().
        Must contain the four fractal-rank columns named in the *_col arguments.
    n : int
        Number of worst fractal ranks to avoid.
    left_up_col, left_down_col, right_up_col, right_down_col : str
        Column names for the four fractal ranks. Defaults match the column
        names produced by simulation_gamble_data.py.

    Returns
    -------
    pd.DataFrame
        Copy of df with two new columns:
        - 'avoid_worst_{n}_value'      : signed cue value (-1, 0, or 1)
        - 'avoid_worst_{n}_side_if_true': 'left', 'right', or None
    """
    required = [left_up_col, left_down_col, right_up_col, right_down_col]
    missing = set(required) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {missing}")

    df = df.copy()

    values = df.apply(
        lambda row: signed_cue(
            row[left_up_col], row[left_down_col],
            row[right_up_col], row[right_down_col],
            n
        ),
        axis=1
    )

    df[f'avoid_worst_{n}_value'] = values
    df[f'avoid_worst_{n}_side_if_true'] = values.map(
        lambda v: 'left' if v > 0 else ('right' if v < 0 else None)
    )

    return df
