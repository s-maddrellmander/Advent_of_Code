# solutions/year_2023/day_00.py
from collections import defaultdict
from math import prod
from typing import Dict, List, Optional, Union

import numpy as np

from logger_config import logger
from utils import Timer


def is_symbol(char: str) -> bool:
    return char != "." and not char.isdigit()


def is_number(char):
    return char.isdigit()


def sum_part_numbers(grid):
    gears = defaultdict(list)

    def find_numbers():
        numbers = {}
        for i, row in enumerate(grid):
            j = 0
            while j < len(row):
                char = row[j]
                if is_number(char):
                    # Find the full number
                    num = char
                    k = j + 1
                    while k < len(row) and is_number(row[k]):
                        num += row[k]
                        k += 1

                    # Check if the number is not part of a larger number
                    if k == len(row) or not is_number(row[k]):
                        if num not in numbers:
                            numbers[num] = []
                        numbers[num].append(
                            (i, j, k - 1)
                        )  # Store start and end indices of the number

                    j = k  # Skip to the end of the number
                else:
                    j += 1  # Move to the next character
        return numbers

    def count_adjacent_symbols(num_positions):
        def check_adjacent(i, start_j, end_j):
            for j in range(start_j, end_j + 1):
                for di in range(-1, 2):
                    for dj in range(-1, 2):
                        ni, nj = i + di, j + dj
                        if 0 <= ni < len(grid) and 0 <= nj < len(grid[ni]):
                            if is_symbol(grid[ni][nj]):
                                logger.debug(
                                    f"Number {num} is adjacent to symbol {grid[ni][nj]}"
                                )
                                if grid[ni][nj] == "*":
                                    logger.debug(
                                        f"Adding {num} to gears at position {(ni, nj)}, {grid[ni][nj]}"
                                    )
                                    gears[(ni, nj)].append(num)
                                return True
            return False

        count = 0
        for position in num_positions:
            if check_adjacent(*position):
                count += 1
        return count

    # Find all numbers and check their adjacency to symbols
    numbers = find_numbers()
    logger.debug(f"Numbers: {numbers.keys()}")
    total_sum = 0
    for num, positions in numbers.items():
        # Count the number of symbols adjacent to the number
        count = count_adjacent_symbols(positions)
        assert int(num) > 0
        # Add the number to the total sum for each adjacent symbol
        total_sum += int(num) * count

    gear_sum = sum(
        prod([int(num) for num in gear]) for gear in gears.values() if len(gear) == 2
    )

    return total_sum, gear_sum


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
        return sum_part_numbers(input_data)[0]


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
        return sum_part_numbers(input_data)[1]
