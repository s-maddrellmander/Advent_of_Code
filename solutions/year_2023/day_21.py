# solutions/year_2023/day_21.py
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


@dataclass
class Locationn:
    value: str
    state: bool = False
    visits: int = 0


def parse_input(input_data: List[str]) -> Dict[complex, Locationn]:
    grid = {}
    for y, line in enumerate(input_data):
        for x, char in enumerate(line):
            if char == "S":
                char = "O"
            grid[complex(x, y)] = Locationn(char, state=False if char != "O" else True)
    return grid


def get_neighbours(grid: Dict[complex, Locationn], loc: complex) -> List[complex]:
    neighbours = []
    for direction in [1, -1, 1j, -1j]:
        neighbour = loc + direction
        if neighbour in grid:
            neighbours.append(neighbour)
    return neighbours


def turn_all_states_false(grid: Dict[complex, Locationn]) -> Dict[complex, Locationn]:
    for loc in grid:
        grid[loc].state = False
    return grid


def print_grid(grid: Dict[complex, Locationn]) -> None:
    max_x = max(grid, key=lambda x: x.real).real
    max_y = max(grid, key=lambda x: x.imag).imag
    for y in range(int(max_y) + 1):
        for x in range(int(max_x) + 1):
            print(grid[complex(x, y)].value, end="")
        print()


def soft_reset_grid(grid: Dict[complex, Locationn]) -> Dict[complex, Locationn]:
    new_grid = {}
    for loc in grid:
        new_grid[loc] = Locationn(
            "#" if grid[loc].value == "#" else ".", False, grid[loc].visits
        )
    return new_grid


def count_O(grid: Dict[complex, Locationn]) -> int:
    count = 0
    for loc in grid:
        if grid[loc].value == "O":
            count += 1
    return count


def step_all(grid: Dict[complex, Locationn]) -> Dict[complex, Locationn]:
    new_grid = soft_reset_grid(grid)
    grid = turn_all_states_false(grid)
    for loc in grid:
        # new_grid[loc] = Locationn(grid[loc].value, grid[loc].state)
        if grid[loc].value == "O":
            neighbours = get_neighbours(grid, loc)
            logger.debug(f"Neighbours: {neighbours}")
            for neighbour in neighbours:
                if grid[neighbour].value == "#":
                    continue
                else:
                    # This can recieve multiple visits and state and value won't change
                    logger.debug(f"Visiting {neighbour}")
                    new_grid[neighbour].state = True
                    new_grid[neighbour].value = "O"
                    new_grid[neighbour].visits += 1

    return new_grid


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
        grid = parse_input(input_data)
        for step in range(64):
            grid = step_all(grid)

        return count_O(grid)


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
        logger.debug("Brute force is not going to work here.")
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
