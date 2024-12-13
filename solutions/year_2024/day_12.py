# solutions/year_2024/day_12.py

from logger_config import logger
from utils import Timer


def parse_map(input_data: list[str]) -> dict[complex, str]:
    # Take the input data and parse it into a dictionary
    mapping = {}
    for x, row in enumerate(input_data):
        for y, col in enumerate(row):
            mapping[complex(y, x)] = col
    return mapping


def find_island(mapping: dict[complex, str], start: complex, visited: list[list[bool]]) -> tuple[set[complex], list[list[bool]]]:
    # Find the island of the given start point
    island = set()
    stack = [start]
    letter = mapping[start]
    while stack:
        current = stack.pop()
        if visited[int(current.real)][int(current.imag)]: # If visited we don't need to look at it again
            continue
        
        if mapping[current] == letter:
            visited[int(current.real)][int(current.imag)] = True # set this point to visited 
            island.add(current)
            for direction in [1, -1, 1j, -1j]:
                if current + direction in mapping:
                    # Only keep the ones in the map
                    stack.append(current + direction)
    return island, visited


def island_perimeter(island: set[complex]) -> int:
    # Calculate the perimeter of the island
    perimeter = 0
    for point in island:
        for direction in [1, -1, 1j, -1j]:
            if point + direction not in island:
                perimeter += 1
    return perimeter

def island_sides(island: set[complex]) -> int:
    # Calculate the number of straight sides in the island - not the perimeter
    # Get the actual perimeter coordinates first
    perimeter = []
    for point in island:
        exposed_faces = 0
        on_perimeter = False
        for direction in [1, -1, 1j, -1j]:
            
            if point + direction not in island:
                
                exposed_faces += 1
                if on_perimeter == False:
                    perimeter.append(point)
                    on_perimeter = True
    
    # For each point in the perimeter, calculate its vertices
    vertices = set()
    for point in perimeter:
        x, y = point.real, point.imag
        vertices.add(complex(x - 0.0, y - 0.0))
        vertices.add(complex(x + 1.0, y - 0.0))
        vertices.add(complex(x - 0.0, y + 1.0))
        vertices.add(complex(x + 1.0, y + 1.0))
    
    # Now vertices contains all the vertices for each point in the perimeter
    print(vertices)

    # Loop through all verticies and find the number of exposed squared in 9 directions
    corner = 0
    print(len(vertices))
    for vertex in vertices:
        exposed_faces = 0
        for direction in [1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]:
            if vertex + direction not in island:
                exposed_faces += 1
        if exposed_faces == 1:
            # Corner fully enclosed, except for one diagonal direction
            corner += 1
        elif exposed_faces == 6:
            # 90 degree corner 
            corner += 1
    return corner, vertices

  




def find_all_islands(mapping: dict[complex, str], map_dims: tuple[int]) -> list[set[complex]]:
    # Find all the islands in the mapping
    visited = [[False for _ in range(map_dims[0])] for _ in range(map_dims[1])]
    islands = []
    for point in mapping:
        if visited[int(point.real)][int(point.imag)]:
            continue
        island, visited = find_island(mapping, point, visited)
        islands.append(island)
    return islands


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
        mapping = parse_map(input_data)
        islands = find_all_islands(mapping, (len(input_data[0]), len(input_data)))
        total_fence = 0
        for island in islands:
            perimeter = island_perimeter(island)
            total_fence += perimeter * len(island) 
        return total_fence


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
        mapping = parse_map(input_data)
        islands = find_all_islands(mapping, (len(input_data[0]), len(input_data)))
        total_sides = 0
        for island in islands:
            sides = island_sides(island)
            total_sides += sides * len(island)
        return total_sides
