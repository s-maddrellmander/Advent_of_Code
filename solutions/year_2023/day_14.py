# solutions/year_2023/day_14.py
from typing import Dict, List, Optional, Union

import numpy as np
from tqdm import tqdm

from logger_config import logger
from utils import Timer


def roll_fast_with_obstacles(grid_array: np.ndarray) -> np.ndarray:
    R, C = grid_array.shape  # R is the number of rows, C is the number of columns

    # Process each column one by one
    for c in range(C):
        # Extract the current column
        column = grid_array[:, c]

        # Get the positions of 'O' and '#' in the current column
        # np.where() returns the indices of elements that match the condition
        positions_O = np.where(column == "O")[0]
        positions_hash = np.where(column == "#")[0]

        # Create a new column initially filled with '.'
        # This will be used to build the new state of the current column
        new_column = np.array(["."] * R)

        # Place '#' in the new column at the same positions as in the original column
        # '#' are immovable obstacles, so their positions don't change
        for pos in positions_hash:
            new_column[pos] = "#"

        # Now, process each 'O' in the column
        for pos in positions_O:
            # Find the nearest '#' below this 'O'
            # This is where the 'O' will stop falling
            obstacles_below = positions_hash[positions_hash > pos]
            new_pos = obstacles_below[0] - 1 if obstacles_below.size > 0 else R - 1

            # If there are other 'O's or '#' already placed below in the new column,
            # move the 'O' upwards until a free space is found
            while new_pos >= 0 and new_column[new_pos] in ["O", "#"]:
                new_pos -= 1

            # If there's space available, place the 'O' in the new column
            if new_pos >= 0:
                new_column[new_pos] = "O"

        # Replace the original column with the new one in the grid
        grid_array[:, c] = new_column

    return grid_array


def score_grid(grid_array: np.ndarray) -> int:
    # Count the numbers of 0s per row
    # multiply that by the row number and sum the total
    return np.sum(
        np.sum(grid_array == "O", axis=1) * np.arange(1, grid_array.shape[0] + 1)
    )


def part1(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 1 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 1: Rolling north"):
        # Your solution for part 1 goes here
        # Convert the grid to a NumPy array for easier manipulation
        grid = [list(row) for row in input_data]
        grid_array = np.array(grid)
        # Rotate the grid 180 degrees so N at the bottom
        grid_array = np.rot90(grid_array, 2)

        # Roll the grid
        grid_array = roll_fast_with_obstacles(grid_array)

        score = score_grid(grid_array)

        return score


def part2(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 2 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 2"):
        seen_states: Dict = dict()
        grid = [list(row) for row in input_data]
        grid_array = np.array(grid)
        # Rotate the grid 180 degrees so N at the bottom
        grid_array = np.rot90(grid_array, 2)
        # This is the starting place.
        for cycle in tqdm(range(0, 1000000000)):
            # Cycle the four directions (north, west, south, east) with rotations and rolling
            for _ in range(4):
                grid_array = roll_fast_with_obstacles(grid_array)
                grid_array = np.rot90(grid_array, -1)
            score = score_grid(grid_array)
            # logger.debug(f"Cycle {cycle}: {score} ")
            # Do this at the end
            grid_hash = hash(grid_array.tostring())  # type: ignore
            if grid_hash in seen_states:
                # We've seen this state before, so we can break
                logger.debug(f"Cycle {cycle}: {score} ")
                cycle_length = cycle - seen_states[grid_hash]
                remainder = (1000000000 - cycle) % cycle_length
                break

            seen_states[grid_hash] = cycle

        # import ipdb; ipdb.set_trace()
        # Assume cycle is < max_cycles, therefore we can just do the remainder

        for extra in range(remainder - 1):
            # Cycle the four directions (north, west, south, east) with rotations and rolling
            for _ in range(4):
                grid_array = roll_fast_with_obstacles(grid_array)
                grid_array = np.rot90(grid_array, -1)
            score = score_grid(grid_array)

        score = score_grid(grid_array)
        return score
