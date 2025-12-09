# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer

"""
How do we find these rouge IDs? 
- strings that are duplicates, could do an O(N) search and just scan bc if we have  

"""


def parse_input(data: list[str]) -> list[tuple[int]]:
    ranges = []
    for dat in data:
        A, B = dat.split("-")
        ranges.append((int(A), int(B)))
    return ranges


def scan_range(low: int, high: int) -> int:
    count = 0 
    value = 0
    for x in range(low, high+1):
        str_x = str(x)
        if len(str_x) % 2 == 0:
            idx = len(str_x) // 2
            front, back = str_x[:idx], str_x[idx:]
            if front == back:
                count += 1
                value += x
    return count, value


def scan_range_full(low, high):
    """
    Can scan as before, but this time have to set the str - check if the set len goes into the full len
    Then can check the slices does second one match the previous one
    Would make O(N) main scan and 

    The set doesn't work because we get repeated values
    """
    count  = 0
    value  = 0
    for x in range(low, high+1):
        str_x = str(x)
        set_x = set(str_x)
        delta = len(set_x)
        i = 1
        while len(str_x) / i >= 2: 
            query = str_x[:i+1]
            reps = len(str_x) // i
            target = query * reps
            print(target)

            if target == str_x:
                count += 1
                value += x
                i = 1e6
            else:
                i += 1
    return count, value 





def part1(input_data: list[str] | None) -> str | int:
    """
    Solve part 1 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")
    if len(input_data) == 1:
        input_data = input_data[0].split(',')
    with Timer("Part 1"):
        ranges = parse_input(input_data)
        total = 0
        value = 0
        for rang in ranges:
            c, v = scan_range(rang[0], rang[1])
            total += c
            value += v
        return value


def part2(input_data: list[str] | None) -> str | int:
    """
    Solve part 2 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    input_data = input_data[0].split(',')
    with Timer("Part 2"):
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
