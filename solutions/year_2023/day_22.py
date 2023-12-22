# solutions/year_2023/day_22.py
from collections import defaultdict, namedtuple
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer

Brick = namedtuple("Brick", ["id", "cells", "minZ"])


def parse_input(input_data):
    """
    Parses the input data and converts it into a list of tuples representing bricks.

    Args:
    input_data (str): Multiline string where each line represents a brick in the format 'x1,y1,z1~x2,y2,z2'.

    Returns:
    List[Tuple[Tuple[int, int, int], Tuple[int, int, int]]]: List of bricks represented as tuples of coordinates.
    """
    bricks = []
    for line in input_data:
        if line.strip():  # Ensure the line is not empty
            parts = line.split("~")
            start = tuple(map(int, parts[0].split(",")))
            end = tuple(map(int, parts[1].split(",")))
            bricks.append((start, end))
    bricks = sorted(bricks, key=lambda brick: brick[0][2])
    return bricks


def tuples_to_bricks(brick_tuples):
    bricks = []
    for i, l in enumerate(brick_tuples):
        (x1, y1, z1), (x2, y2, z2) = l
        cells, minZ = [], 1 << 16
        for x in range(min(x1, x2), max(x1, x2) + 1):
            for y in range(min(y1, y2), max(y1, y2) + 1):
                for z in range(min(z1, z2), max(z1, z2) + 1):
                    cells.append((x, y, z))
                    minZ = min(minZ, z)
        bricks.append(Brick(i, tuple(cells), minZ))
    return sorted(bricks, key=lambda b: b.minZ)


def lower_brick(brick: Brick) -> Brick:
    return Brick(
        brick.id, tuple((x, y, z - 1) for x, y, z in brick.cells), brick.minZ - 1
    )


def can_lower(brick: Brick, settled_bricks: Dict[int, Brick]) -> bool:
    for cell in brick.cells:
        if cell[2] <= 0 or (cell[0], cell[1], cell[2] - 1) in settled_bricks:
            return False
    return True


def simulate_settling(bricks: List[Brick]):
    # Initialize a dictionary to store the final positions of bricks
    settled_bricks: Dict = dict()
    final_bricks: Dict = dict()

    for b in bricks:
        # Lower the brick until it can't be lowered anymore
        while can_lower(b, settled_bricks):
            b = lower_brick(b)
        settled_bricks.update({cell: b.id for cell in b.cells})
        final_bricks[b.id] = b

    return final_bricks


def check_to_remove(bricks):
    can_remove = 0
    id_can_remove = []

    total_falling = 0

    # This is super slow, looping looping looping

    for reduced_set in combinations(bricks.values(), len(bricks) - 1):
        temp_settled = simulate_settling(reduced_set)
        # import ipdb; ipdb.set_trace()
        if all([temp_settled[x.id] == x for x in reduced_set]):
            can_remove += 1
            # The one id not in the reduced set is the one that can be removed
            id_can_remove.append(
                list(set(bricks.keys()) - set([x.id for x in reduced_set]))[0]
            )
        else:
            # import ipdb; ipdb.set_trace()
            total_falling += sum([temp_settled[x.id] != x for x in reduced_set])
    # import ipdb; ipdb.set_trace()
    logger.debug(f"Can remove {can_remove} bricks")
    return can_remove, id_can_remove, total_falling


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
        parsed_bricks = parse_input(input_data)
        bricks = tuples_to_bricks(parsed_bricks)
        settled_bricks = simulate_settling(bricks)
        return check_to_remove(settled_bricks)[0]


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
        parsed_bricks = parse_input(input_data)
        bricks = tuples_to_bricks(parsed_bricks)
        settled_bricks = simulate_settling(bricks)
        return check_to_remove(settled_bricks)[2]
