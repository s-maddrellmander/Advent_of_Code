# solutions/year_2024/day_06.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def create_map(input_data: List[str]) -> Dict[str, str]:
    carte = {}
    start = None
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            carte[(i, j)] = {"type": input_data[i][j], "visited": False, "direction": None}
            if input_data[i][j] == "^":
                start = (i, j)
    return carte, start


def move_guard(carte: Dict[str, str], position: Tuple[int, int], direction: str):
    if direction == "^":
        next_pos = (position[0] - 1, position[1])
        if next_pos in carte:
            if carte[next_pos]["type"] == "#":
                # Try moving right immediately instead of staying in place
                next_pos = (position[0], position[1] + 1)
                if next_pos in carte and carte[next_pos]["type"] != "#":
                    return next_pos, ">"
                # If that direction isn't viable either, return None
                return None, None
            return next_pos, "^"
        return None, None
    elif direction == "v":
        next_pos = (position[0] + 1, position[1])
        if next_pos in carte:
            if carte[next_pos]["type"] == "#":
                next_pos = (position[0], position[1] - 1)
                if next_pos in carte and carte[next_pos]["type"] != "#":
                    return next_pos, "<"
                return None, None
            return next_pos, "v"
        return None, None
    elif direction == ">":
        next_pos = (position[0], position[1] + 1)
        if next_pos in carte:
            if carte[next_pos]["type"] == "#":
                next_pos = (position[0] + 1, position[1])
                if next_pos in carte and carte[next_pos]["type"] != "#":
                    return next_pos, "v"
                return None, None
            return next_pos, ">"
        return None, None
    elif direction == "<":
        next_pos = (position[0], position[1] - 1)
        if next_pos in carte:
            if carte[next_pos]["type"] == "#":
                next_pos = (position[0] - 1, position[1])
                if next_pos in carte and carte[next_pos]["type"] != "#":
                    return next_pos, "^"
                return None, None
            return next_pos, "<"
        return None, None


def move_guard_until_end(carte: Dict[str, str], position: Tuple[int, int], direction: str, part2: bool = False) -> Tuple[int, Dict[str, str]]:
    path = set(position)
    carte[position]["visited"] = True
    carte[position]["type"] = "X"
    carte[position]["direction"] = direction
    counter = 1
    while position is not None:
        assert carte[position]["type"] != "#"
        position, direction = move_guard(carte, position, direction)
        if part2:
            if position in path:
                # We have been here before
                # Check if the direction is the same
                if carte[position]["direction"] == direction:
                    # We have a loop
                    # print("Loop at", position)
                    return True
        if position is not None: # THat means we are still in the map
            if not carte[position]["visited"]: # If we have not visited this position
                counter += 1 # Increase the counter
        path.add(position)
        # print(counter)
        if position is not None:
            carte[position]["visited"] = True
            carte[position]["type"] = "X"
            carte[position]["direction"] = direction
    if part2:
        return False
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
        # Get initial path
        original_carte, start_position = create_map(input_data)
        start_direction = original_carte[start_position]["type"]
        path, marked_carte = move_guard_until_end(original_carte.copy(), start_position, start_direction)
        
        # Find all visited positions that could be blocked
        visited_positions = [pos for pos in marked_carte if marked_carte[pos]["visited"]]
        
        loops = 0
        for _, pos in enumerate(visited_positions):
            # Create a fresh map for each test
            test_carte = create_map(input_data)[0]
            # Add the test blocker
            test_carte[pos]["type"] = "#"
            
            # Try to find a loop with this configuration
            has_loop = move_guard_until_end(test_carte, start_position, start_direction, True)
            if has_loop:
                loops += 1
                
        return loops