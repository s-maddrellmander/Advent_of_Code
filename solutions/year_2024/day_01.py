# solutions/year_2024/day_00.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer

def parse_input(input_data):
    pairs = []
    for line in input_data:
        x = line.split("   ")
        pairs.append((int(x[0]), int(x[1])))
    return pairs



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
        pairs = parse_input(input_data)
        list1 = sorted([x[0] for x in pairs])
        list2 = sorted([x[1] for x in pairs])

        deltas = [abs(x2 - x1) for x1, x2 in zip(list1, list2)]

        return sum(deltas)

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
        pairs = parse_input(input_data)
        list1 = sorted([x[0] for x in pairs])
        list2 = sorted([x[1] for x in pairs])

        # counts in list2
        counts = {}
        for x in list2:
            if x not in counts:
                counts[x] = 1
            else:
                counts[x] += 1

        scores = 0
        for y in list1:
            if y in counts:
                scores += y * counts[y]

        return scores
