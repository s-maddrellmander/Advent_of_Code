# solutions/year_2024/day_00.py

from logger_config import logger
from utils import Timer
from itertools import combinations

def parse_data(data: list[str]) -> dict[str, list[complex]]:
    # Take the map, collect nodes together with the same frequency, as complex coordinates
    nodes: dict[str, list[complex]] = {}
    for y, row in enumerate(data):
        for x, cell in enumerate(row):
            if cell != '.':
                if cell in nodes:
                    nodes[cell].append(complex(x, y))
                else:
                    nodes[cell] = [complex(x, y)]

    return nodes

def find_all_pairs(node_class: list[complex]) -> list[tuple[complex, complex]]:
    # Find all the pairs within the same class
    pairs = combinations(node_class, 2)
    return list(pairs)


def return_antinode(a: complex, b: complex) -> complex:
    delta = b - a
    return a - delta if a + delta == b else a + delta
    
def add_antinodes(pairs: list[tuple[complex, complex]], bounds: list[int]) -> set[complex]:
    # Takes the pairs, finds the distance between, then returns the antinodes
    # Filtering for the bounds of the map
    anti_nodes = set()
    for pair in pairs:
        anti_1 = return_antinode(*pair)
        anti_2 = return_antinode(*pair[::-1])
        for anti in [anti_1, anti_2]:
            if 0 <= anti.real < bounds[0] and 0 <= anti.imag < bounds[1]:
                anti_nodes.add(anti)
    return anti_nodes


def return_all_antinodes(pair: tuple[complex, complex], bounds: list[int]) -> set[complex]:
    a, b = pair
    delta = b - a
    set_antinodes = set([a, b])
    if a + delta == b:
        # Path 1
        while 0 <= (a-delta).real < bounds[0] and 0 <= (a-delta).imag < bounds[1]:
            a -= delta
            set_antinodes.add(a)
    else:
        # Same for path 2 but going the other way
        while 0 <= (b+delta).real < bounds[0] and 0 <= (b+delta).imag < bounds[1]:
            b += delta
            set_antinodes.add(b)
    return set_antinodes
        

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
        bounds = [len(input_data[0]), len(input_data)]
        nodes = parse_data(input_data)
        unique_antinodes: complex = set()
        for key in nodes:
            pairs = find_all_pairs(nodes[key])
            anti_nodes = add_antinodes(pairs, bounds)
            unique_antinodes = unique_antinodes | anti_nodes # Set union
        return len(unique_antinodes)


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
        bounds = [len(input_data[0]), len(input_data)]
        nodes = parse_data(input_data)
        unique_antinodes: complex = set()
        for key in nodes:
            pairs = find_all_pairs(nodes[key])
            for pair in pairs:
                antinodes1 = return_all_antinodes(pair, bounds)
                antinodes2 = return_all_antinodes(pair[::-1], bounds)
                unique_antinodes = unique_antinodes | antinodes1 | antinodes2
        return len(unique_antinodes)
