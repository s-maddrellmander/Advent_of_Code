# solutions/year_2024/day_19.py

from logger_config import logger
from utils import Timer
from functools import lru_cache

def parse_data(input_data: list[str]):
    towels = input_data[0].split(', ')
    
    patterns = input_data[2:]
    return towels, patterns


def match_towel(towel: str, patterns: list[str]) -> int:
    @lru_cache(maxsize=None)
    def count_from_index(start_idx: int) -> int:
        # Base case: if we've matched the entire towel
        if start_idx == len(towel):
            return 1
            
        # Count solutions from the current position
        total = 0
        for pat in patterns:
            end_idx = start_idx + len(pat)
            # Check if pattern matches at current position
            if (end_idx <= len(towel) and 
                towel[start_idx:end_idx] == pat):
                total += count_from_index(end_idx)
                
        return total

    return count_from_index(0)


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
        valid = 0
        towels, patterns = parse_data(input_data)
        for i, towel in enumerate(patterns):
            print(f"{i}/{len(patterns)}", end="\r")
            if match_towel(towel, towels) > 0:
                valid += 1
        return valid
        


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
        valid = 0
        towels, patterns = parse_data(input_data)
        for i, towel in enumerate(patterns):
            print(f"{i}/{len(patterns)}", end="\r")
            results = match_towel(towel, towels)
            valid += results
        return valid
