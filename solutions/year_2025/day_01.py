# solutions/year_2025/day_01.py

from logger_config import logger
from utils import Timer


def parse_input(input_data: list[str]) -> list[int]:
    moves = []
    for step in input_data:
        if step[0] == "L":
            # Left -> negative 
            moves.append(-1*int(step[1:]))
        else:
            # Right so positive 
            moves.append(1*int(step[1:]))
    return moves
    
    


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
        zeros = 0
        coord = 50
        moves = parse_input(input_data)
        for move in moves:
            coord += move
            coord = coord % 100
            if coord == 0:
                zeros += 1
    return zeros


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
        rotations = parse_input(input_data)
        turns = 0
        dial = 50
        for rot in rotations:
            clicks, rotation = divmod(abs(rot), 100)
            turns += clicks
            if rot >=0:
                if dial + rotation >= 100:
                    turns += 1
                dial = (dial + rotation) % 100
            else: 
                if dial and dial - rotation <= 0:
                    turns += 1
                dial = (dial - rotation) % 100

        return turns
