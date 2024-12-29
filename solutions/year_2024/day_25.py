# solutions/year_2024/day_25.py

from logger_config import logger
from utils import Timer

def parse_data(input_data: list[str]) -> tuple[list[list[int]], list[list[int]]]:
    """
    Parse the input data into locks and keys.
    """
    keys = []
    locks = []
    current_schematic = []
    
    for line in input_data:
        if line == "":
            if current_schematic:
                heights = parse_schematic(current_schematic)
                if current_schematic[0][0] == '#':
                    locks.append(heights)
                else:
                    keys.append(heights)
                current_schematic = []
        else:
            current_schematic.append(line)
    
    # Handle the last schematic if it exists
    if current_schematic:
        heights = parse_schematic(current_schematic)
        if current_schematic[0][0] == '#':
            locks.append(heights)
        else:
            keys.append(heights)
    
    return locks, keys

def parse_schematic(lines: list[str]) -> list[int]:
    """
    Parse a single schematic into heights.
    For locks: count from top down
    For keys: count from bottom up
    """
    heights = []
    is_lock = lines[0][0] == '#'
    
    # For each column
    for col in range(len(lines[0])):
        column = [line[col] for line in lines]
        
        if is_lock:
            # Count continuous '#' from top
            height = 0
            for c in column:
                if c == '#':
                    height += 1
                elif height > 0:  # Stop at first gap
                    break
        else:
            # Count continuous '#' from bottom
            height = 0
            for c in reversed(column):
                if c == '#':
                    height += 1
                elif height > 0:  # Stop at first gap
                    break
        
        heights.append(height)
    
    return heights

def part1(input_data: list[str] | None) -> str | int:
    """
    Solve part 1 of the day's challenge.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")
    
    with Timer("Part 1"):
        locks, keys = parse_data(input_data)
        logger.info(f"Num locks: {len(locks)}, Num keys: {len(keys)}")
        
        counter = 0
        for lock in locks:
            for key in keys:
                overlap = False
                for l, k in zip(lock, key):
                    if l + k > 7:
                        overlap = True
                        break
                if not overlap:
                    counter += 1
        
        return counter

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
