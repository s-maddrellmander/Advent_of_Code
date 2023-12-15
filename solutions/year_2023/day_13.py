# solutions/year_2023/day_13.py
from typing import List, Optional, Union

import numpy as np
from tqdm import tqdm

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str]) -> List[np.ndarray]:
    # Splitting the content into blocks
    # Splitting the content into blocks based on empty strings
    blocks = "\n".join(input_data).split("\n\n")

    # Function to convert a block into a numpy array
    def convert_to_numpy_array(block):
        return np.array(
            [[1 if char == "#" else 0 for char in line] for line in block.split("\n")]
        )

    # Converting each block
    numpy_arrays = [convert_to_numpy_array(block) for block in blocks]
    return numpy_arrays


def consequtive(arr: np.ndarray, axis=0) -> np.ndarray:
    if axis == 1:
        arr = arr.T
    # In this function, axis=0 checks for consecutive equal rows, and axis=1 checks for consecutive equal columns.
    consecutive_equal = np.all(arr[:-1] == arr[1:], axis=1)
    # Finding the indexes where consecutive rows are equal
    indexes = np.where(consecutive_equal)[0]
    return indexes


def find_reflection(
    arr: np.ndarray, rows: np.ndarray, axis=0, smudge=False
) -> List[int]:
    if axis == 1:
        arr = arr.T
    # Check if the array is symetric reflected across the rows
    reflections = []
    # Find the min distnace to the edge.
    for row in rows:
        row += 1

        side2 = arr[row : row + row, :]
        side1 = arr[row - side2.shape[0] : row, :]
        assert side1.shape == side2.shape
        if smudge:
            # The difference in total above  below the line will be 1
            if np.sum(side1 != np.flip(side2, axis=0)) == 1:
                reflections.append(row)

        elif np.array_equal(side1, np.flip(side2, axis=0)):
            reflections.append(row)
    return reflections


def score_fn(row, cols):
    if len(row):
        return row[0] * 100
    if len(cols):
        return cols[0]


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

    score = 0
    with Timer("Part 1"):
        mazes = parse_input(input_data)
        for maze in tqdm(mazes):
            rows = consequtive(maze, axis=0)
            mirror_row = find_reflection(maze, rows)
            cols = consequtive(maze, axis=1)
            mirror_col = find_reflection(maze, cols, axis=1)
            score += score_fn(mirror_row, mirror_col)
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

    score = 0
    with Timer("Part 2"):
        mazes = parse_input(input_data)
        for maze in tqdm(mazes):
            rows = np.array([_ for _ in range(maze.shape[0])])
            mirror_row = find_reflection(maze, rows, smudge=True)
            cols = np.array([_ for _ in range(maze.shape[1])])
            mirror_col = find_reflection(maze, cols, axis=1, smudge=True)
            score += score_fn(mirror_row, mirror_col)
        return score
