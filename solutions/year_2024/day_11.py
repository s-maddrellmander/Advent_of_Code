# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer


def parse_input(input_string: str) -> list[str]:
    line = input_string[0].split(' ')
    return line

def blink(stones: list[str]) -> list[str]:
    new_stones = []
    for stone in stones:
        if int(stone) == 0:
            new_stones.append("1")
        elif len(stone) % 2 == 0:
            split_idx = len(stone) // 2
            new_stones.append(stone[:split_idx])
            new_stones.append(str(int(stone[split_idx:])))
        else:
            new_stones.append(str(int(stone) * 2024))
    return new_stones

def fast_blink(stones: dict[str, int]) -> dict[str, int]:
    # Don't need to track the full list, obvs, just the number of each stone
    # The order didn't actually matter, that was a red herring. 
    def add_key(d, k, v=1):
        if k not in d:
            d[k] = 0
        d[k] += v

    new_stones = {}
    for stone in stones: # Here we get the keys
        current = stones[stone]
        # Apply the rules
        if stone == "0":
            add_key(new_stones, "1", current)
        elif len(stone) % 2 == 0:
            split_idx = len(stone) // 2
            add_key(new_stones, stone[:split_idx], current)
            add_key(new_stones, str(int(stone[split_idx:])), current)
        else:
            add_key(new_stones, str(int(stone) * 2024), current)
    return new_stones


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
        _stones = parse_input(input_data)
        stones = {stone: 1 for stone in _stones}
        for _ in range(25):
            stones = fast_blink(stones)
        
        val = sum(stones.values()) 
        return val


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
        _stones = parse_input(input_data)
        stones = {stone: 1 for stone in _stones}
        for _ in range(75):
            stones = fast_blink(stones)
        
        val = sum(stones.values()) 
        return val
