# solutions/year_2023/day_12.py
import re
from functools import lru_cache
from typing import List, Optional, Tuple, Union

from tqdm import tqdm

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str]) -> Tuple[str, List[int]]:
    lines = [line.strip() for line in input_data]
    output = []
    for line in lines:
        nums = re.findall(r"\d+", line)
        springs = line.split(" ")[0]
        output.append((springs, [int(x) for x in nums]))
    return output


# Adding a pruning strategy
def valid_combination(combination, expected_groups):
    # Function to check if a combination matches the expected group sizes
    groups = []
    count = 0
    for c in combination:
        if c == "#":
            count += 1
        elif count > 0:
            groups.append(count)
            count = 0
    if count > 0:
        groups.append(count)
    return groups == expected_groups


def count_damaged_springs(springs, pattern):
    # Function to count contiguous damaged springs
    def count_groups(combination):
        # Initialize a dynamic programming table with zeros
        # dp[i] will hold the count of contiguous damaged springs ending at index i
        dp = [0] * len(combination)

        # Iterate over the springs
        for i, s in enumerate(combination):
            # If the spring at index i is damaged
            if s == "#":
                # If it's the first spring or the previous spring is not damaged, start a new count
                # Otherwise, increment the count from the previous spring
                dp[i] = dp[i - 1] + 1 if i > 0 else 1

        # Initialize a list to store the sizes of the groups of contiguous damaged springs
        damaged_groups = []
        for i in range(len(dp)):
            # If the spring at index i is damaged and it's the last spring or the next spring is not damaged
            # Then it's the end of a group of damaged springs, so add the count to damaged_groups
            if dp[i] > 0 and (i == len(dp) - 1 or dp[i + 1] == 0):
                damaged_groups.append(dp[i])

        # Return the sizes of the groups of contiguous damaged springs
        return damaged_groups

    # Initialize a list to store all possible combinations of operational and damaged springs
    combinations = []
    # Momoisation
    memo = {}
    # Generate all possible combinations
    generate_combinations(
        springs, 0, "", combinations, expected_groups=pattern, memo=memo
    )

    # Initialize a list to store valid combinations
    valid_combinations = []
    # Iterate over all combinations
    for comb in combinations:
        # If the sizes of the groups of contiguous damaged springs in the combination match the given pattern
        # Then the combination is valid, so add it to valid_combinations
        if count_groups(comb) == pattern:
            valid_combinations.append(comb)

    # Return the valid combinations
    return valid_combinations


# @lru_cache(maxsize=None)  # maxsize=None means the cache can grow without bound
def generate_combinations(springs, index, current, results, expected_groups, memo):
    # Check if the current state is already computed
    if (index, current) in memo:
        for comb in memo[(index, current)]:
            results.append(current + comb)
        return

    if index == len(springs):
        if valid_combination(current, expected_groups):
            results.append(current)
        return

    # If the spring at index is unknown
    if springs[index] == "?":
        # Consider the spring as operational and recurse
        generate_combinations(
            springs, index + 1, current + ".", results, expected_groups, memo=memo
        )
        # Consider the spring as damaged and recurse
        generate_combinations(
            springs, index + 1, current + "#", results, expected_groups, memo=memo
        )
    else:
        # If the spring at index is known, add it to the current combination and recurse
        generate_combinations(
            springs,
            index + 1,
            current + springs[index],
            results,
            expected_groups,
            memo=memo,
        )


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
        input_data = parse_input(input_data)
        total = 0
        for row in tqdm(input_data):
            springs, pattern = row
            valid_combinations = count_damaged_springs(springs, pattern)
            total += len(valid_combinations)
    return total


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
        input_data = parse_input(input_data)

        total = 0
        for row in tqdm(input_data):
            springs, pattern = row
            springs = [springs + "?"] * 5
            pattern = [pattern] * 5
            # # Flatten the springs list
            springs = "".join(x for x in [s for sublist in springs for s in sublist])

            # # # Flatten the pattern list
            pattern = [p for sublist in pattern for p in sublist]
            # import ipdb; ipdb.set_trace()
            valid_combinations = count_damaged_springs(springs, pattern)
            total += len(valid_combinations)
    return total
