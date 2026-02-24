import os
import pandas as pd
import itertools
import random
import numpy as np

def all_valid_gambles(fractals, exclude_nobrainer=True, file_path=None):
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
        file_name = "valid_gambles.csv"
        full_path = os.path.join(file_path, file_name)
        df.to_csv(full_path, index=False)
        print("Gamble data saved to:", full_path)
    else:
        print("No file path provided - all valid gamble data not saved.")

    return df

def simulate_gamble_data(n,fractals, 
                          exclude_nobrainer=True,
                          mirror_gambles=True, 
                          random_draw = True, 
                          file_path=None,
                          random_seed=None):
    """
    Creates a synthetic dataset with n gambles, drawn from all valid gamble pairs.

    Parameters:
    - fractals: list of fractal values
    - exclude_nobrainer: if True, exclude no-brainer gamble pairs
    - mirror_gambles: if True, include mirrored gamble pairs
    - random_draw: if True, draw gambles randomly; if False, draw sequentially
    - file_path: directory where the generated gamble data CSV should be saved (optional)
    - random_reproducible_number: seed for reproducible random draws (optional)

    Returns:
    - DataFrame containing the simulated gamble data
    """
    n = int(n)  # Ensure n is an integer

    valid_gambles = all_valid_gambles(fractals, exclude_nobrainer, file_path)
    
    # Mirroring doubles the number of gambles, so we only need to draw n/2 pairs
    if mirror_gambles:
        if n % 2 != 0:
            raise ValueError("n must be even when mirror_gambles is True.")
        n = int(n/2)

    # Draw gambles from the valid pairs
    if random_draw:
        sampled_df = valid_gambles.sample(n, replace=True, random_state=random_seed)
    else:
        sampled_df = valid_gambles.head(n)
       
    # If mirror_gambles is True, include both (A, B) and (B, A) for each pair
    if mirror_gambles:
        mirrored_df = sampled_df.copy()
        mirrored_df = mirrored_df.rename(columns={
            "fractal_left_up": "fractal_right_up",
            "fractal_left_down": "fractal_right_down",
            "fractal_right_up": "fractal_left_up",
            "fractal_right_down": "fractal_left_down",
            "gamma_left_up": "gamma_right_up",
            "gamma_left_down": "gamma_right_down",
            "gamma_right_up": "gamma_left_up",
            "gamma_right_down": "gamma_left_down",
        })
        sampled_df = pd.concat([sampled_df, mirrored_df], ignore_index=True)

    # Save to csv if file_path is provided
    if file_path is not None:
        file_name = "synthetic_gamble_data.csv"
        full_path = os.path.join( file_path, file_name)
        sampled_df.to_csv(full_path, index=False)
        print("Synthetic gamble data saved to:", full_path)
    else:
        print("No file path provided - synthetic gamble data not saved.")
    
    #Reset index after sampling and mirroring
    sampled_df.reset_index(drop=True, inplace=True)

    return sampled_df

def main():
    "Run main function for testing the simulate_gamble_data function."
    df = simulate_gamble_data(100,[10, 20, 30, 40, 50, 60, 70, 80, 90],
                              True,True,True,None,random_seed=43 )
    print(len(df))
    print(df.head())
    return 0

if __name__ == "__main__":
  returncode = main()

