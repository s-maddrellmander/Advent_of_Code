# solutions/year_2023/day_00.py
import sys
from typing import Dict, List, Optional, Set, Tuple, Union

from logger_config import logger
from utils import Timer

sys.setrecursionlimit(1000000)


import matplotlib.pyplot as plt
import numpy as np

ID = 0


def plot_map(
    map_data: Dict[Tuple[int, int], str], visited: Set[Tuple[int, int]]
) -> None:
    global ID
    x_coords = [coord[0] for coord in map_data.keys()]
    y_coords = [coord[1] for coord in map_data.keys()]

    min_x, max_x = min(x_coords), max(x_coords)
    min_y, max_y = min(y_coords), max(y_coords)

    img = np.zeros((max_y - min_y + 1, max_x - min_x + 1, 3))

    for y in range(min_y, max_y + 1):
        for x in range(min_x, max_x + 1):
            if (x, y) in map_data and map_data[(x, y)] != "#":
                if (x, y) in visited:
                    img[y - min_y, x - min_x] = [0, 1, 0]  # Green for visited paths
                else:
                    img[y - min_y, x - min_x] = [0, 0, 1]  # Blue for unvisited paths
            else:
                img[y - min_y, x - min_x] = [1, 0, 0]  # Red for walls

    plt.imshow(img)
    plt.axis("off")
    plt.savefig(f"map_{ID}.png")
    plt.close()
    ID += 1


def parse_map(input_data: List[str]) -> Dict[Tuple[int, int], str]:
    """
    Parse the input data into a dictionary of coordinates and their values.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Dict[Tuple[int, int], str]: The parsed input data.
    """
    parsed_data = {}
    for y, line in enumerate(input_data):
        for x, char in enumerate(line):
            if char != "#":
                parsed_data[(x, y)] = char
    return parsed_data


def dfs(
    node: Tuple[int, int],
    end: Tuple[int, int],
    map_data: Dict[Tuple[int, int], str],
    visited: Set[Tuple[int, int]],
    max_length: List[int],
    path: List[Tuple[int, int]],
    part: int = 1,
) -> None:
    path.append(node)
    # plot_map(map_data, visited=visited)
    logger.debug(f"Path: {len(path)}")
    if node == end:
        max_length[0] = max(max_length[0], len(path) - 1)
    else:
        neighbors = get_neighbors(node, map_data, part=part)
        if len(neighbors) > 2:
            logger.debug(f"Node: {node}, Neighbors: {neighbors}")
        for neighbor in neighbors:
            if neighbor not in visited:
                visited.add(neighbor)
                dfs(neighbor, end, map_data, visited, max_length, path, part=part)
                visited.remove(neighbor)
    path.pop()


def longest_path(
    start: Tuple[int, int],
    end: Tuple[int, int],
    map_data: Dict[Tuple[int, int], str],
    part: int = 1,
) -> int:
    visited: Set = set()
    max_length = [-1]
    dfs(start, end, map_data, visited, max_length, [], part=part)
    return max_length[0]


def get_neighbors(
    node: Tuple[int, int], map_data: Dict[Tuple[int, int], str], part: int = 1
) -> List[Tuple[int, int]]:
    """
    Get the neighboring nodes of a given node.

    Args:
        node (Tuple[int, int]): The coordinates of the node.
        map_data (Dict[Tuple[int, int], str]): The map data.

    Returns:
        List[Tuple[int, int]]: The neighboring nodes.
    """
    x, y = node
    neighbors = []
    # Check if the direction is <, >, ^, or v
    if part == 1 and map_data[node] in "<>^v":
        # Filter only in the gvien direction
        if map_data[node] == "<":
            if (x - 1, y) in map_data:
                neighbors.append((x - 1, y))
        elif map_data[node] == ">":
            if (x + 1, y) in map_data:
                neighbors.append((x + 1, y))
        elif map_data[node] == "^":
            if (x, y - 1) in map_data:
                neighbors.append((x, y - 1))
        elif map_data[node] == "v":
            if (x, y + 1) in map_data:
                neighbors.append((x, y + 1))
        return neighbors
    else:
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            neighbor = (x + dx, y + dy)
            if neighbor in map_data:
                neighbors.append(neighbor)
        return neighbors


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
        parsed_map = parse_map(input_data)
        gap_top = (input_data[0].find("."), 0)
        gap_bottom = (input_data[-1].find("."), len(input_data) - 1)
        length = longest_path(gap_top, gap_bottom, parsed_map)
        return length


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
        parsed_map = parse_map(input_data)
        # plot_map(parsed_map, visited=set())
        gap_top = (input_data[0].find("."), 0)
        gap_bottom = (input_data[-1].find("."), len(input_data) - 1)
        length = longest_path(gap_top, gap_bottom, parsed_map, part=2)
        return length
