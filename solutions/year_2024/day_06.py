# solutions/year_2024/day_06.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def create_map(input_data: List[str]) -> Dict[str, str]:
    carte = {}
    start = None
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            carte[(i, j)] = {"type": input_data[i][j], "visited": False}
            if input_data[i][j] == "^":
                start = (i, j)
    return carte, start


def move_guard(carte: Dict[str, str], position: Tuple[int, int], direction: str):
    if direction == "^":
        if (position[0] - 1, position[1]) in carte:
            if carte[(position[0] - 1, position[1])]["type"] == "#":
                # Rotate right
                return (position[0], position[1]), ">"
            else:
                return (position[0] - 1, position[1]), "^"
        else:
            # OUt of the map - so end of the game
            return None, None
    elif direction == "v":
        if (position[0] + 1, position[1]) in carte:
            if carte[(position[0] + 1, position[1])]["type"] == "#":
                # Rotate right
                return (position[0], position[1]), "<"
            else:
                return (position[0] + 1, position[1]), "v"
        else:
            # OUt of the map - so end of the game
            return None, None
    elif direction == ">":
        if (position[0], position[1] + 1) in carte:
            if carte[(position[0], position[1] + 1)]["type"] == "#":
                # Rotate right
                return (position[0], position[1]), "v"
            else:
                return (position[0], position[1] + 1), ">"
        else:
            # OUt of the map - so end of the game
            return None, None
    elif direction == "<":
        if (position[0], position[1] - 1) in carte:
            if carte[(position[0], position[1] - 1)]["type"] == "#":
                # Rotate right
                return (position[0], position[1]), "^"
            else:
                return (position[0], position[1] - 1), "<"
        else:
            # OUt of the map - so end of the game
            return None, None


def move_guard_until_end(carte: Dict[str, str], position: Tuple[int, int], direction: str):
    path = set(position)
    carte[position]["visited"] = True
    carte[position]["type"] = "X"
    counter = 1
    while position is not None:
        assert carte[position]["type"] != "#"
        position, direction = move_guard(carte, position, direction)
        if position is not None: # THat means we are still in the map
            if not carte[position]["visited"]: # If we have not visited this position
                counter += 1 # Increase the counter
        path.add(position)
        # print(counter)
        if position is not None:
            carte[position]["visited"] = True
            carte[position]["type"] = "X"
    return len(path), carte



def print_carte(carte: Dict[str, str]):
    # from the dict constract the array again
    max_x = max([x[0] for x in carte.keys()])
    max_y = max([x[1] for x in carte.keys()])
    for i in range(max_x + 1):
        row = ""
        for j in range(max_y + 1):
            row += carte[(i, j)]["type"]
        print(row)    


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
        carte, position = create_map(input_data)
        direction = carte[position]["type"]
        path, carte = move_guard_until_end(carte, position, direction)
        # print_carte(carte)
        # Count X in carte
        counter = sum(carte[pos]["visited"] for pos in carte)
        return counter


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
        return "Part 2 solution not implemented."
