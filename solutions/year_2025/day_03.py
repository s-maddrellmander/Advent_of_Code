# solutions/year_2024/day_03.py

from logger_config import logger
from utils import Timer


def two_max_voltage(batteries: str) -> int:
    # We want to select the two values in the string which make the largest number
    # The minimum is the final two elements from the battery
    # Find the largest vlaue not the final one
    # Then second pointer scan 
    idx1 = 0
    max_value = 0
    for i in range(len(batteries[:-1])): # Not the final number 
        if int(batteries[i]) > max_value:
            max_value = int(batteries[i])
            idx1 = i
    # Then find the second digit
    second_val = 0
    for i in range(idx1+1, len(batteries)): # Include the final one 
        if int(batteries[i]) > second_val:
            second_val = int(batteries[i])
    output = "".join([str(max_value), str(second_val)])
    return int(output)

# def remove_three(batteries: str) -> int:
#     # Three passes - remove the three smallest values
#     # at the three smallest indexs
#     pass




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
        score = 0
        for row in input_data:
            score += two_max_voltage(row)
    return score

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
