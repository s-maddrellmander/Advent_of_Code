# solutions/year_2023/day_17.py
import heapq
from typing import Dict, List, Optional, Tuple, Union

import numpy as np

from logger_config import logger
from utils import Timer


def parser(input_data: List[str]) -> dict:
    coords: dict = {}
    for i, line in enumerate(input_data):
        line = line.strip()
        for j, loc in enumerate(line):
            coords[(int(j), int(i))] = int(loc)
    return coords


def print_grid_and_path(coordinates, path):
    if not coordinates:
        print("No coordinates provided.")
        return

    # Determine the bounds of the grid
    max_col = max(coordinates, key=lambda x: x[0])[0]
    max_row = max(coordinates, key=lambda x: x[1])[1]

    # Iterate over each row and column, printing the appropriate character
    print("=" * (max_col + 1))
    for row in range(max_row + 1):
        for col in range(max_col + 1):
            if (col, row) in coordinates:
                if (col, row) in path:
                    print("O", end="")
                else:
                    if path == []:
                        print(coordinates[(col, row)], end="")
                    else:
                        print("#", end="")
            else:
                print(".", end="")
        print()  # New line at the end of each row
    print("=" * (max_col + 1))


def dijkstra(coords, start, end, min_step=1, step_limit=3):
    visited = set()
    pq = [
        (0, start, (1, 0), 1),
        (0, start, (0, 1), 1),
    ]  # (distance, coordinate, direction, steps_in_direction)
    while pq:
        current_dist, current_coord, current_dir, current_steps = heapq.heappop(pq)
        if (current_coord, current_dir, current_steps) in visited:
            continue
        else:
            visited.add((current_coord, current_dir, current_steps))
        new_coord = (
            current_coord[0] + current_dir[0],
            current_coord[1] + current_dir[1],
        )
        if new_coord not in coords:
            continue
        alt_dist = current_dist + coords[new_coord]
        if current_steps >= min_step and current_steps <= step_limit:
            if new_coord == end:
                return alt_dist
        for direction in [(1, 0), (0, 1), (-1, 0), (0, -1)]:
            # No backtracking
            if (
                current_dir[0] + direction[0] == 0
                and current_dir[1] + direction[1] == 0
            ):
                continue
            new_steps = current_steps + 1 if direction == current_dir else 1
            # Continue exploring the neighbor only if the direction change is allowed
            # and the step limit in the same direction has not been exceeded
            if (
                direction != current_dir and current_steps < min_step
            ) or new_steps > step_limit:
                continue
            heapq.heappush(pq, (alt_dist, new_coord, direction, new_steps))


def path_cost(path, coords):
    return sum([coords[coord] for coord in path])


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
        mapping = parser(input_data)
        cost = dijkstra(
            mapping,
            (0, 0),
            (len(input_data) - 1, len(input_data) - 1),
            min_step=1,
            step_limit=3,
        )
    return cost


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
        mapping = parser(input_data)
        cost = dijkstra(
            mapping,
            (0, 0),
            (len(input_data) - 1, len(input_data) - 1),
            min_step=4,
            step_limit=10,
        )
    return cost
