# solutions/year_2023/day_00.py
from typing import List, Optional, Union

from logger_config import logger
from utils import Timer
import numpy as np


def process_array(input_data: List[str]) -> np.array:
    # We need a mapping to convert chars to ints
    wildcard_mapping = {
        '*': -1,
        '@': -2,
        '#': -3,
        '$': -4,
        '+': -5,
        '-': -6,
        '=': -7,
        '%': -8,
        '&': -9,
        '/': -10,
        '.': -99
        }
    # Now we can convert the input data to a numpy array and  keep the original numbers
    processed_array = np.array([[wildcard_mapping[char] if char in wildcard_mapping else int(char) for char in row]
                                for row in input_data])
    return processed_array
    
def get_wildcards(grid: np.array) -> np.array:
    # Find the locations of all wildcards (-ve values)
    wildcards = np.where(grid < 0)
    return wildcards

def get_numbers(grid: np.array) -> np.array:
    # Find the locations of all numbers (positive values)
    numbers = np.where(grid > 0)
    return numbers

def find_adjacent_numbers(wildcards: np.array, numbers: np.array) -> np.array:
    adjacent_coords = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Iterate over each point in wildcards
    for i in range(wildcards.shape[1]):
        x, y = wildcards[:, i]
        
        # Check all 8 directions
        for dx, dy in directions:
            adjacent_x, adjacent_y = x + dx, y + dy
            adjacent_point = np.array([adjacent_x, adjacent_y])

            # Check if the adjacent point is in numbers
            for j in range(numbers.shape[1]):
                if np.array_equal(adjacent_point, numbers[:, j]):
                    adjacent_coords.append(adjacent_point.tolist())

    # Remove duplicates and convert to numpy array
    if not adjacent_coords:
        return np.array([[], []])
    unique_adjacent_coords = np.unique(adjacent_coords, axis=0).T

    return unique_adjacent_coords

def get_number_sequences(grid: np.array, numbers, is_adjacent: np.array) -> np.array:
    nums_to_save = []
    for coord in is_adjacent.T:
        x, y = coord
        logger.info((y, x, grid[y, x]))

        # Skip if the number at the current coordinate is <= 0
        if grid[y, x] < 0:
            continue

        left_bound = max(0, x - 2)
        right_bound = min(grid.shape[1], x + 3)
        row_segment = grid[y, left_bound:right_bound]


        logger.info(row_segment)
        # Process for different cases
        if x > 0 and x < grid.shape[1] - 1 and grid[y, x - 1] > 0 and grid[y, x + 1] > 0:
            # Both sides
            nums_to_save.append(int("".join(map(str, row_segment[x - 1 - left_bound : x + 2 - left_bound]))))
        elif x > 0 and grid[y, x - 1] > 0:
            # Left side
            if x > 1 and grid[y, x - 2] > 0:
                nums_to_save.append(int("".join(map(str, row_segment[x - 2 - left_bound : x + 1 - left_bound]))))
            else:
                nums_to_save.append(int("".join(map(str, row_segment[x - 1 - left_bound : x + 1 - left_bound]))))
        elif x < grid.shape[1] - 1 and grid[y, x + 1] > 0:
            # Right side
            if x < grid.shape[1] - 2 and grid[y, x + 2] > 0:
                nums_to_save.append(int("".join(map(str, row_segment[x - left_bound : x + 3 - left_bound]))))
            else:
                nums_to_save.append(int("".join(map(str, row_segment[x - left_bound : x + 2 - left_bound]))))
        else:
            nums_to_save.append(grid[y, x])

    logger.info(nums_to_save)
    return nums_to_save
        
    

def tuple_array_to_array(array) -> np.array:
    return np.array([array[0], array[1]])
    
    
    
    

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
        # Process the input data into a 2D numpy array
        grid = process_array(input_data)
        # Find the locations of all wildcards (-ve values)
        wildcards = tuple_array_to_array(get_wildcards(grid))
        # Find the locations of all numbers (positive values)
        numbers = tuple_array_to_array(get_numbers(grid))
        # Check if any of the wildcards are adjacent to any of the numbers
        is_adjacent = find_adjacent_numbers(wildcards, numbers)
        # Now we need to find which numbers are part of a sequence of numbers in the array
        number_sequences = get_number_sequences(grid, numbers, np.array([is_adjacent[1], is_adjacent[0]]))
         
        
        return -1

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