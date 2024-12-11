# solutions/year_2024/day_06.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def create_map(input_data: List[str]) -> Dict[str, str]:
    carte = {}
    start = None
    for i in range(len(input_data)):
        for j in range(len(input_data[i])):
            carte[(i, j)] = {"type": input_data[i][j], "visited": False, "direction": []}
            if input_data[i][j] == "^":
                start = (i, j)
    return carte, start


def move_guard(position, direction, carte):
    directions = [">", "v", "<", "^"]
    direction_index = directions.index(direction)
    
    for _ in range(4):
        # print(direction)
        if direction == ">":
            next_pos = (position[0], position[1] + 1)
        elif direction == "v":
            next_pos = (position[0] + 1, position[1])
        elif direction == "<":
            next_pos = (position[0], position[1] - 1)
        elif direction == "^":
            next_pos = (position[0] - 1, position[1])
        
        if next_pos in carte and carte[next_pos]["type"] != "#":
            return next_pos, direction

        if next_pos in carte and carte[next_pos]["type"] == "#":
            # Turn right to the next direction
            direction_index = (direction_index + 1) % 4
            direction = directions[direction_index]
            # print("New direction", direction)
        else:
            # Not in the map and we exit
            return None, None
    return None, None


def move_guard_until_end(carte: Dict[str, str], position: Tuple[int, int], direction: str, part2: bool = False) -> Tuple[int, Dict[str, str]]:
    path = [position]
    carte[position]["visited"] = True
    carte[position]["type"] = "X"
    carte[position]["direction"].extend(direction)
    counter = 1
    while position is not None:
        assert carte[position]["type"] != "#"
        position, direction = move_guard(position, direction, carte, )
        if part2:
            if position in path:
                # We have been here before
                # Check if the direction is the same
                if direction in carte[position]["direction"]:
                    # We have a loop
                    # print("Loop at", position)
                    return True
        if position is not None: # THat means we are still in the map
            if not carte[position]["visited"]: # If we have not visited this position
                counter += 1 # Increase the counter
        path.append(position)
        # print(counter)
        if position is not None:
            carte[position]["visited"] = True
            carte[position]["type"] = "X"
            carte[position]["direction"].extend(direction)
    if part2:
        return False
    return path, carte



def print_carte(carte: dict[str, str]):
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
        print(len(visited_positions), "<---- unique positions")
        print(len(path), "<---- path length")
        # import ipdb; ipdb.set_trace()   
        # Try to block at every position on the map
        # visited_positions = [(x, y) for x in range(1, len(input_data[0]) - 1) for y in range(1, len(input_data) - 1) if (x, y) != start_position][1:]

        
        loops = set()
        
        for _, pos in enumerate(visited_positions):
            print(_, "/", len(visited_positions), end="\r")
            # Create a fresh map for each test
            test_carte = create_map(input_data)[0]
            # Add the test blocker
            test_carte[pos] = {"type": "#", "visited": False, "direction": []}
            
            # Try to find a loop with this configuration
            has_loop = move_guard_until_end(test_carte, start_position, start_direction, True)
            if has_loop:
                loops.add(pos)
                
        return len(list(loops))