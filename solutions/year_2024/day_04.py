# solutions/year_2024/day_04.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer

"""
Simplest approach:
    - Just scan the array for all X values
    - Then look around each x in all directions for the required rest of the word
    - I guess this is a dynamic programing approach? 
    - Or the 8 defined searches for each X with appropriate padding.

"""

def find_x_coord(array, query="X"):
    coords = []
    for i, row in enumerate(array):
        for j, c in enumerate(row):
            if c == query:
                coords.append(complex(i, j))
    return coords

def full_map(input_data):
    full_map = {}
    for row in range(len(input_data)):
        for col in range(len(input_data[0])):
            full_map[complex(row, col)] = input_data[row][col]
    return full_map

def find_xmas(coord, M):
    score = 0

    # Check horizontal right
    if (coord + (1 + 0j) in M and coord + (2 + 0j) in M and coord + (3 + 0j) in M and
        M[coord + (1 + 0j)] == "M" and M[coord + (2 + 0j)] == "A" and M[coord + (3 + 0j)] == "S"):
        score += 1

    # Diagonal down-right
    if (coord + (1 + 1j) in M and coord + (2 + 2j) in M and coord + (3 + 3j) in M and
        M[coord + (1 + 1j)] == "M" and M[coord + (2 + 2j)] == "A" and M[coord + (3 + 3j)] == "S"):
        score += 1

    # Vertical down
    if (coord + (0 + 1j) in M and coord + (0 + 2j) in M and coord + (0 + 3j) in M and
        M[coord + (0 + 1j)] == "M" and M[coord + (0 + 2j)] == "A" and M[coord + (0 + 3j)] == "S"):
        score += 1

    # Diagonal up-left
    if (coord + (-1 + -1j) in M and coord + (-2 + -2j) in M and coord + (-3 + -3j) in M and
        M[coord + (-1 + -1j)] == "M" and M[coord + (-2 + -2j)] == "A" and M[coord + (-3 + -3j)] == "S"):
        score += 1

    # Horizontal left
    if (coord + (-1 + 0j) in M and coord + (-2 + 0j) in M and coord + (-3 + 0j) in M and
        M[coord + (-1 + 0j)] == "M" and M[coord + (-2 + 0j)] == "A" and M[coord + (-3 + 0j)] == "S"):
        score += 1

    # Diagonal up-right
    if (coord + (-1 + 1j) in M and coord + (-2 + 2j) in M and coord + (-3 + 3j) in M and
        M[coord + (-1 + 1j)] == "M" and M[coord + (-2 + 2j)] == "A" and M[coord + (-3 + 3j)] == "S"):
        score += 1

    # Vertical up
    if (coord + (0 + -1j) in M and coord + (0 + -2j) in M and coord + (0 + -3j) in M and
        M[coord + (0 + -1j)] == "M" and M[coord + (0 + -2j)] == "A" and M[coord + (0 + -3j)] == "S"):
        score += 1

    # Diagonal down-left
    if (coord + (1 + -1j) in M and coord + (2 + -2j) in M and coord + (3 + -3j) in M and
        M[coord + (1 + -1j)] == "M" and M[coord + (2 + -2j)] == "A" and M[coord + (3 + -3j)] == "S"):
        score += 1

    return score


def find_crossed_mas(coord, M):
    diagonals = [
        ((1 + 1j), (-1 - 1j)),
        ((-1 + 1j), (1 - 1j)),
        ((1 - 1j), (-1 + 1j)),
        ((-1 - 1j), (1 + 1j))
    ]
    
    sols = 0
    for d1, d2 in diagonals:
        if (coord + d1 in M and coord + d2 in M and
                M[coord + d1] == "M" and M[coord + d2] == "S"):
            sols += 1
    if sols == 2:
        return True
    return False
        

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
        data = full_map(input_data)
        coords = find_x_coord(input_data)

        score = 0
        for coord in coords:
            score += find_xmas(coord, data)
        return score

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
        
        data = full_map(input_data)
        coords = find_x_coord(input_data, "A")

        score = 0
        for coord in coords:
            score += find_crossed_mas(coord, data)
        return score
        
