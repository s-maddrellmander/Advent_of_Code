# solutions/year_2024/day_00.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parse_lines(input_data):
    lines = []
    for line in input_data:
        line = line.strip().split(" ")
        lines.append([int(x) for x in line])
    return lines


def is_safe(line, damp=0):
    # Is safe when all values are either increasing by 1, decreasingb y 1
    # Different levels must be no more than 1 - 3 difference
    """
    How can this work?
    1. All increasing / decreasing in the right range
    2.
    """
    if damp == 1 and line[0] == line[1]:
        line = line[1:]
        damp -= 1

    if line[0] < line[1]:
        # Do increase
        y = line[0]
        for x in line[1:]:
            if x > y and abs(x - y) < 4:
                y = x
                continue
            else:
                if damp == 1:
                    damp -= 1
                else:
                    return False
        return True

    elif line[1] < line[0]:
        # do decrease
        y = line[0]
        for x in line[1:]:
            if x < y and abs(y - x) < 4:
                y = x
                continue
            else:
                if damp == 1:
                    damp -= 1
                else:
                    return False
        return True
    else:
        # same - then unsafe
        return False


def make_combs(line):
    # Here we can skip any single value
    lines = []
    for idx in range(len(line)):
        new_line = line[:idx] + line[idx + 1 :]
        lines.append(new_line)
    return lines


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
        lines = parse_lines(input_data)
        res = 0
        for line in lines:
            if is_safe(line):
                res += 1
        return res


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
        lines = parse_lines(input_data)
        res = 0
        for line in lines:
            safe = is_safe(line, damp=0)
            if safe:
                res += 1
            else:
                perms = make_combs(line)
                for opt in perms:
                    safe = is_safe(opt)
                    if safe:
                        res += 1
                        break
            # Now loop if not safe
        return res
