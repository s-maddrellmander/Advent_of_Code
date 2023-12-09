# solutions/year_2023/day_09.py
from typing import List, Optional, Union

from logger_config import logger
from utils import Timer


def next_in_sequence(seq: List[int]) -> int:
    """
    Function to find the next number in a sequence using the method of differences.
    The time complexity of this function is O(n^2) because of the nested loop used to fill the difference table.
    The space complexity is also O(n^2) because a 2D table of size n*n is created to store the differences.
    """
    n = len(seq)
    if n == 0:
        raise ValueError("Sequence must have at least one element")

    # Create a table to store the differences
    # This is a 2D list of size n*n, initialized with the input sequence
    table = [seq.copy()]

    # Fill the difference table
    # This is done in O(n^2) time because of the nested loop
    while len(table[-1]) > 1:
        table.append(
            [table[-1][i + 1] - table[-1][i] for i in range(len(table[-1]) - 1)]
        )

    # Use the last non-zero differences to predict the next number
    # This is done in O(n) time because we iterate over the table once
    next_num = seq[-1]
    for i in range(1, len(table)):
        if table[i][-1] != 0:
            next_num += table[i][-1]

    return next_num


def previous_in_sequence(seq: List[int]) -> int:
    """
    Function to find the prev number in a sequence using the method of differences.
    The time complexity of this function is O(n^2) because of the nested loop used to fill the difference table.
    The space complexity is also O(n^2) because a 2D table of size n*n is created to store the differences.
    """
    n = len(seq)
    if n == 0:
        raise ValueError("Sequence must have at least one element")

    # Create a table to store the differences
    # This is a 2D list of size n*n, initialized with the input sequence
    table = [seq.copy()]

    # Fill the difference table
    # This is done in O(n^2) time because of the nested loop
    while len(table[-1]) > 1:
        table.append(
            [table[-1][i + 1] - table[-1][i] for i in range(len(table[-1]) - 1)]
        )

    # Use the last non-zero differences to predict the prev number
    # This is done in O(n) time because we iterate over the table once
    prev_num = seq[0]
    diff = 0
    for i in range(1, len(table)):
        diff = table[-i][0] - diff
    prev_num -= diff

    return prev_num


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
        data = [[int(x) for x in row.strip("\n").split(" ")] for row in input_data]
        next_values = [next_in_sequence(data[i]) for i in range(len(data))]
        return sum(next_values)


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
        data = [[int(x) for x in row.strip("\n").split(" ")] for row in input_data]
        prev_values = [previous_in_sequence(data[i]) for i in range(len(data))]
        return sum(prev_values)
