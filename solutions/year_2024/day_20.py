# solutions/year_2024/day_20.py

from logger_config import logger
from utils import Timer
import heapq
from collections import Counter

def print_map(carte, path=None, limit=15):
    for y in range(limit):
        for x in range(limit):
            if path and complex(x, y) in path:
                print("o", end="")
            elif complex(x, y) in carte:
                print(".", end="")
            else:
                print("#", end="")
        print()


def parse_map(input_data: list[str]) -> dict[complex, str]:
    carte = {}
    for y, line in enumerate(input_data):
        for x, char in enumerate(line):
            if char != "#":
                carte[complex(x, y)] = char
            if char == "S":
                start = complex(x, y)

            if char == "E":
                end = complex(x, y)
    return carte, start, end

def a_star(carte, start, end):
    frontier = []
    counter = 0
    heapq.heappush(frontier, (0, counter, start))
    came_from = dict()
    cost_so_far = dict()
    came_from[start] = None
    cost_so_far[start] = 0
    
    while frontier:
        current = heapq.heappop(frontier)[-1]
        if current == end:
            break
        for dir in [1, 1j, -1, -1j]:
            if current + dir in carte:
                new_cost = cost_so_far[current] + 1
                new_loc = current + dir
                if new_loc not in cost_so_far or new_cost < cost_so_far[new_loc]:
                    cost_so_far[new_loc] = new_cost
                    priority = new_cost + abs(end - new_loc)
                    counter += 1
                    heapq.heappush(frontier, (priority, counter, new_loc))
                    came_from[new_loc] = current
    return came_from, cost_so_far

def reconstruct_path(came_from, start, end):
    if end not in came_from:
        return None
    current = end

    path = []
    while current != start:
        path.append(current)
        current = came_from[current]
    path.append(start)
    path.reverse()
    return path


def shortcut(path: list[complex]) -> list:
    base_cost = len(path) - 1
    # Shortcuts are found by taking two steps through a wall and back onto the path
    # Can be stright lines, or right angles
    valid_shortcuts = []
    for step in path: # At each step on the path
        for short_1 in [1, 1j, -1, -1j]: # Check all 4 directions
            if step + short_1 not in path: # If the first step is not on the path, it's not a shortcut, it's just the path
                for short_2 in [1, 1j, -1, -1j]: # Check all 4 directions
                    new_loc = step + short_1 + short_2 # The new location is the step + the two shortcuts
                    if new_loc in path:            # If the new location is on the path - otherwise we're just going through a wall
                        new_idx = path.index(new_loc) # Get the index of the new location
                        if new_idx < path.index(step): # If the new location is before the current step, we're going backwards
                            continue # Not a short cut
                        step_delta = new_idx - path.index(step) - 2 # The number of steps we're skipping
                        
                        # If the step delta is 2 or more, we have a shortcut
                        if step_delta >= 1:
                            valid_shortcuts.append(step_delta)
    return valid_shortcuts
                        
                    

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

    with Timer("Part 1"):
        carte, start, end = parse_map(input_data)
        # Start by finding the baseline path 
        came_from, cost_so_far = a_star(carte, start, end)
        path = reconstruct_path(came_from, start, end)
        # print_map(carte, path, limit=len(input_data))
        shortcuts = shortcut(path)
        # Find all greater than 100
        counter = Counter(shortcuts)    

        total = 0
        for k, v in counter.items():
            if k >= 100:
                total += v
        
        return total


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

    with Timer("Part 2"):
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
