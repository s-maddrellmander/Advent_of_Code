# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer


def parse_input(input_data: list[str]) -> list[tuple[list[int]]]:
    """Take the input data and return a list of equations in the form
    x_t = x_1 * A + x_2 * B, y_t = y_1 * A + y_2 * B
    Combine the X and Y into a tuple 
    """
    equations = []
    for idx in range(0, len(input_data), 4):
        lines = input_data[idx:idx+4]
        x_1, y_1 = lines[0].split(": ")[1].split(", ")
        x_1 = int(x_1[2:])
        y_1 = int(y_1[2:])
        
        x_2, y_2 = lines[1].split(": ")[1].split(", ")
        x_2 = int(x_2[2:])
        y_2 = int(y_2[2:])
        
        x_t, y_t = lines[2].split(": ")[1].split(", ")
        x_t = int(x_t[2:])
        y_t = int(y_t[2:])
        
        equations.append(([x_t, x_1, x_2], [y_t, y_1, y_2]))
    return equations
        

def solve_pair(eq_1: list[int], eq_2: list[int]) -> tuple[int, int]:
    """Take the equations in the form
    x_t = x_1 * A + x_2 * B
    y_t = y_1 * A + y_2 * B
    
    And return A, B
    """
    x_t, x_1, x_2 = eq_1
    y_t, y_1, y_2 = eq_2
    
    B = (x_1 * y_t - y_1 * x_t)/(y_2 * x_1 - y_1 * x_2)
    # If B is not an integer, then the equations are invalid
    if B % 1 != 0:
        return -1, -1
    
    A = (x_t - x_2 * B) / x_1
    if A % 1 != 0:
        return -1, -1
    
    return int(A), int(B)


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
        equations = parse_input(input_data)
        prize = 0
        for eq in equations:
            A, B = solve_pair(eq[0], eq[1])
            if A != -1 and B != -1:
                prize += A * 3 + B * 1
        return prize


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
        equations = parse_input(input_data)
        prize = 0
        for equation in equations:
            # Add 10000000000000 to x_t and y_t to get the new equations
            equation[0][0] += 10000000000000
            equation[1][0] += 10000000000000
            A, B = solve_pair(equation[0], equation[1])
            if A != -1 and B != -1:
                prize += A * 3 + B
        return prize
