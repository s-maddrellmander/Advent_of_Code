# solutions/year_2023/day_18.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parse_data(input_data: List[str]) -> List[Tuple[str, int, str]]:
    instructions = []
    for line in input_data:
        line = line.strip()
        d, v, colour = line.split(" ")
        instructions.append((d, int(v), colour))
    return instructions


directions = {"R": (0, 1), "D": (1, 0), "L": (0, -1), "U": (-1, 0)}
hex_dirs = {
    0: directions["R"],
    1: directions["D"],
    2: directions["L"],
    3: directions["U"],
}


def stokes_integral(x, dy):
    return x * dy


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
        instructions = parse_data(input_data)
        area = 0
        perimiter = 0
        x, y = 0, 0
        for direction, length, _ in instructions:
            dy, dx = directions[direction]
            dx, dy = dx * length, dy * length
            x, y = x + dx, y + dy
            area += stokes_integral(x, dy)
            perimiter += abs(length)
        logger.debug(f"Area: {area}")
        logger.debug(f"Perimiter: {perimiter}")
        logger.debug(f"Area + Perimiter // 2 + 1: {area + perimiter // 2 + 1}")
    return area + perimiter // 2 + 1


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
        instructions = parse_data(input_data)
        area = 0
        perimiter = 0
        x, y = 0, 0
        for _, _, hex_dir in instructions:
            hex_dir, length = int(hex_dir[-2]), int(hex_dir[2:-2], 16)
            dy, dx = hex_dirs[hex_dir]
            dx, dy = dx * length, dy * length
            x, y = x + dx, y + dy
            area += stokes_integral(x, dy)
            perimiter += abs(length)
        logger.debug(f"Area: {area}")
        logger.debug(f"Perimiter: {perimiter}")
        logger.debug(f"Area + Perimiter // 2 + 1: {area + perimiter // 2 + 1}")
    return area + perimiter // 2 + 1
