import os
import pandas as pd
import itertools
import random
 
def simulate_gamble_data(f, exclude_nobrainer=True, mirror_gambles=True, file_path=None):
    """
    Simulates all possible gamble pairs for a set of f fractals.

    Parameters:
    - f: number of fractals to be combined (assumed ordered in terms of rank)
    - exclude_nobrainer: if True, exclude gamble pairs where one gamble is obviously dominating, 
      i.e. (max(A) > max(B) and min(A) > min(B)) or (max(B) > max(A) and min(B) > min(A)),
      where A and B correspond to the value or rank of the gambles.
    - mirror_gambles: if True, includes both (A, B) and (B, A) for each gamble pair, effectively doubling the number of gambles

    Returns:
    - a dataFrame with the following: 
    gamble_id: unique identifier for each gamble
    fractal_left_up
    fractal_left_down
    fractal_right_up
    fractal_right_down
    """
    
    # Check if the number of fractals is at least 4

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

            dominates_A = (maxA > maxB) and (minA > minB)
            dominates_B = (maxB > maxA) and (minB > minA)
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

    # Save to csv if file_path is provided
    if file_path is not None:
        file_name = "gamble_data.csv"
        full_path = os.path.join(file_path, file_name)
        df.to_csv(full_path, index=False)
        print("Gamble data saved to:", full_path)
    else:
        print("No file path provided. Returning DataFrame.")
    return df

def main():
    df = simulate_gamble_data(9,True,False,"C:\\Users\\emili\\OneDrive\\Documents\\Heuristics\\test_folder")
    print(len(df))
    return 0

if __name__ == "__main__":
  returncode = main()

