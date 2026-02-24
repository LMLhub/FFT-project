import os
import pandas as pd
import itertools
import random
import numpy as np

def simulate_gamble_data(fractals, exclude_nobrainer=True, mirror_gambles=True, file_path=None):
    """
    Simulates all possible gamble pairs for a set of f fractals.

    Parameters:
    - fractals: list of fractal values
    - exclude_nobrainer: if True, exclude gamble pairs where one gamble is obviously dominating, 
      i.e. (max(A) >= max(B) and min(A) >= min(B)) or (max(B) >= max(A) and min(B) >= min(A)),
      where A and B correspond to the value or rank of the gambles.
    - mirror_gambles: if True, includes both (A, B) and (B, A) for each gamble pair, effectively doubling the number of gambles

    Returns:
    - a dataFrame with the following: 
    gamble_id: index and unique identifier for each gamble
    fractal_side_position: rank of the fractal that shows up at side left or right and position up or down
    gamma_side_position: value of the fractal that shows up at side left or right and position up or down
    """
    
    fractals = np.asarray(fractals)

    # Ensure fractals is 1D
    if fractals.ndim != 1:
        raise ValueError("fractals must be a 1D array.")

    # Ensure numeric dtype
    if not np.issubdtype(fractals.dtype, np.number):
        raise TypeError("fractals must contain numeric values.")
    # Ensure that the fractals are ordered in terms of rank (ascending)
    fractals = np.sort(fractals)

    # Check if the number of fractals is at least 4
    f = len(fractals)
    if f < 4:
        raise ValueError("The number of fractals must be at least 4.")

    # Generate all possible single gambles
    possible_gambles = list(itertools.combinations(range(f), 2))

    # Generate all possible gamble pairs
    possible_gamble_pairs = list(itertools.combinations(possible_gambles, 2))
  
    # Filter out no-brainer pairs if exclude_nobrainer is True
    if exclude_nobrainer:
        valid_pairs = []

        for gA, gB in possible_gamble_pairs:
            maxA, minA = max(gA), min(gA)
            maxB, minB = max(gB), min(gB)

            dominates_A = (maxA >= maxB) and (minA >= minB)
            dominates_B = (maxB >= maxA) and (minB >= minA)
            if dominates_A or dominates_B:
                continue

            valid_pairs.append((gA, gB))
    else:
        valid_pairs = possible_gamble_pairs
    
    # Check if there are any valid pairs left after filtering
    if len(valid_pairs) == 0:
        raise ValueError("No valid gamble pairs available with current constraints.")

    # If mirror_gambles is True, include both (A, B) and (B, A) for each pair
    if mirror_gambles:
        mirrored_pairs = []
        for gA, gB in valid_pairs:
            mirrored_pairs.append((gA, gB))
            mirrored_pairs.append((gB, gA))
        valid_pairs = mirrored_pairs
    
    # Create a DataFrame to store the gamble pairs
    rows = []
    for idx, (left, right) in enumerate(valid_pairs):
        rows.append({
            "gamble_id": idx,
            "fractal_left_up": left[0],
            "fractal_left_down": left[1],
            "fractal_right_up": right[0],
            "fractal_right_down": right[1],
        })

    df = pd.DataFrame(rows)
    df = df.set_index("gamble_id")

    # Add columns for fractal values (gamma) based on the fractal rank
    for col in df.columns:
        col_name = col.replace("fractal_", "gamma_")
        df[col_name] = fractals[df[col].values]
    
    # Save to csv if file_path is provided
    if file_path is not None:
        file_name = "gamble_data.csv"
        full_path = os.path.join(file_path, file_name)
        df.to_csv(full_path, index=False)
        print("Gamble data saved to:", full_path)
    else:
        print("No file path provided - gamble data not saved.")

    return df

def main():
    "Run main function for testing the simulate_gamble_data function."
    df = simulate_gamble_data([10, 20, 30, 40, 50, 60, 70, 80, 90],True,False)
    print(df)
    return 0

if __name__ == "__main__":
  returncode = main()

