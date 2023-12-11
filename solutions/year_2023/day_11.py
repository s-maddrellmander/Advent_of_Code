# solutions/year_2023/day_1.py
from itertools import combinations
from typing import List, Optional, Tuple, Union

import numpy as np

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str], expansion_constant=2) -> np.ndarray:
    """
    Parse the input data into a usable format.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        np.ndarray: The parsed input data.
    """
    # Convert input data to a NumPy array
    grid = np.array([list(line) for line in input_data])

    # Identify rows and columns that are all '.'
    all_dot_rows = np.all(grid == ".", axis=1)
    all_dot_cols = np.all(grid == ".", axis=0)

    # Repeat rows and columns that are all '.' expansion_constant times
    row_repeats = np.where(all_dot_rows, expansion_constant, 1)
    col_repeats = np.where(all_dot_cols, expansion_constant, 1)

    # Repeat rows and columns
    grid = np.repeat(grid, row_repeats, axis=0)
    grid = np.repeat(grid, col_repeats, axis=1)

    return grid


def parse_input_part2(input_data, expansion_constant=2):
    # Convert input data to a NumPy array
    grid = np.array([list(line) for line in input_data])

    # Identify rows and columns that are all '.'
    all_dot_rows = np.all(grid == ".", axis=1)
    all_dot_cols = np.all(grid == ".", axis=0)

    # Expansion factors
    row_exp = np.where(all_dot_rows, expansion_constant - 1, 0)
    col_exp = np.where(all_dot_cols, expansion_constant - 1, 0)

    # Then get all the coordinates of the '#'s using numpy
    coords = np.argwhere(grid == "#")

    # Add expansion_constant to the coordinates for the number of expansions required
    # for values lower than the original coordinates
    for i, (x, y) in enumerate(coords):
        coords[i, 0] += np.sum(row_exp[:x])
        coords[i, 1] += np.sum(col_exp[:y])

    return coords


def all_combinations(
    coords: List[Tuple[int, int]]
) -> List[Tuple[Tuple[int, int], Tuple[int, int]]]:
    # Get all pairs of pairs of coordinates
    all_pairs = list(combinations(coords, 2))
    return all_pairs


def grid_to_coords(grid: List[str]) -> List[tuple]:
    """
    Convert a grid to a list of coordinates.

    Args:
        grid (List[str]): The grid to convert.

    Returns:
        List[tuple]: The coordinates of the grid.
    """
    coords = []
    for y, line in enumerate(grid):
        for x, char in enumerate(line):
            if char == "#":
                coords.append((y, x))
    return coords


def manhattan_distances(coords: List[Tuple[int, int]]) -> List[int]:
    """
    Calculate the Manhattan distance between all pairs of coordinates.

    Args:
        coords (List[Tuple[int, int]]): The list of coordinates.

    Returns:
        List[int]: The Manhattan distances between all pairs of coordinates.
    """
    distances = []
    logger.debug(f"Calculating distances for {len(coords)} pairs of coordinates")
    for (x1, y1), (x2, y2) in coords:
        distance = abs(x1 - x2) + abs(y1 - y2)
        distances.append(distance)
    return distances


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

    with Timer("Part 1"):
        galaxies = parse_input(input_data)
        coords = grid_to_coords(galaxies)
        all_pairs = all_combinations(coords)
        distances = manhattan_distances(all_pairs)
        return sum(distances)


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
        galaxies = parse_input_part2(input_data, expansion_constant=1000000)
        all_pairs = all_combinations(galaxies)
        distances = manhattan_distances(all_pairs)
        return sum(distances)
