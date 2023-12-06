# solutions/year_2023/day_06.py
from typing import List, Optional, Tuple, Union

import numpy as np

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str]) -> Tuple:
    times = [int(_) for _ in input_data[0].split(" ")[1:] if _ != ""]
    distances = [int(_) for _ in input_data[1].split(" ")[1:] if _ != ""]
    return (times, distances)


def quadratic_solution(time: int, distance: int) -> Tuple[int, int]:
    # Use the quadratic solution to calculate the times
    x_1 = (-1.0 * time + np.sqrt(pow(time, 2) - 4 * (-1.0 * (-1) * distance))) / (-2.0)
    x_2 = (-1.0 * time - np.sqrt(pow(time, 2) - 4 * (-1.0 * (-1) * distance))) / (-2.0)
    low = np.ceil(min(x_1, x_2))
    if low == x_1:
        low += 1
    high = np.floor(max(x_1, x_2))
    if high == x_2:
        high -= 1
    return low, high


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
        # Parse data
        times, distances = parse_input(input_data)
        product = 1
        for time, distance in zip(times, distances):
            lower, upper = quadratic_solution(time, distance)
            delta = upper - lower + 1
            product *= delta
            logger.debug(
                f"Time: {time}, Distance: {distance}, Lower: {lower}, Upper: {upper}"
            )
    return int(product)


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
        # Your solution for part 2 goes here
        times, distances = parse_input(input_data)
        time = int("".join([str(_) for _ in times]))
        distance = int("".join([str(_) for _ in distances]))
        lower, upper = quadratic_solution(time, distance)
        delta = upper - lower + 1
        return int(delta)
