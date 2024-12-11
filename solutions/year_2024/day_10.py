# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer

def get_map(input_data: list[str]) -> tuple[dict[complex, int], list[complex]]:
    top_map = {}
    trail_heads = []
    for y, line in enumerate(input_data):
        for x, char in enumerate(line):
            if char.isdigit():
                top_map[x + y * 1j] = int(char)
                if char == '0':
                    trail_heads.append(x + y * 1j)
    return top_map, trail_heads


def bfs(top_map: dict[complex, int], trail_head: complex) -> tuple[int, list[complex]]:
    # Take in each trailhead and do a BFS to find all "hiking trails" - end at 9, only increasing in values
    
    queue = [trail_head]
    num_trails = 0
    trails_end = []
    while queue:
        step = queue.pop(0)
        if top_map[step] == 9:
            num_trails += 1 
            trails_end.append(step)
            continue
        # Step in <v>^ directions
        for direction in [1, -1, 1j, -1j]:
            next_step = step + direction
            if next_step in top_map and top_map[next_step] == top_map[step] + 1:
                queue.append(next_step)
    return num_trails, trails_end
                

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
        total_trails = 0
        top_map, trail_heads = get_map(input_data)
        for trail_head in trail_heads:
            num_trails, trails_end = bfs(top_map, trail_head)
            total_trails += len(set(trails_end))
        return total_trails


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
        total_trails = 0
        top_map, trail_heads = get_map(input_data)
        for trail_head in trail_heads:
            num_trails, trails_end = bfs(top_map, trail_head)
            total_trails += num_trails 
        return total_trails
