# solutions/year_2024/day_07.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parse_lines(data: list[str]) -> list[tuple[int, list[int]]]:
    parsed = []
    for line in data:
        line = line.split(": ")
        line[1] = [int(x) for x in line[1].split()]
        parsed.append((int(line[0]), line[1]))
    return parsed


def do_maths(current: int, operator: str, new_value: int) -> int:
    if operator == "+":
        return current + new_value
    elif operator == "*":
        return current * new_value
    elif operator == "||":
        return int(str(current) + str(new_value))


def branching_operation(values, target, ops=["+", "*"]):
    """
    Have a queue containing the running sum for each path of the sum
    Then at each level copy + add each new combination.
    """
    queue = [(values[0], str(values[0]))]
    found = False
    count = 0
    for idx in range(1, len(values)):
        for q_idx in range(len(queue)):
            running_sum, path = queue.pop(0)
            for op in ops:
                count += 1
                new_sum = do_maths(running_sum, op, values[idx])
                queue.append((new_sum, path + op + str(values[idx])))
                if new_sum == target and idx == len(values) - 1:
                    return True
    return False


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
        running_total = 0
        solutions = 0
        lines = parse_lines(input_data)
        for i, (target, query) in enumerate(lines):
            result = branching_operation(query, target)
            if result == True:
                solutions += 1
                running_total += target
        return running_total


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
        running_total = 0
        solutions = 0
        lines = parse_lines(input_data)
        for i, (target, query) in enumerate(lines):
            print(f"{i}/{len(lines)}")
            result = branching_operation(query, target, ["+", "*", "||"])
            if result == True:
                solutions += 1
                running_total += target
        return running_total
