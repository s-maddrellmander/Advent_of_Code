import pytest
from solutions.year_2024.day_12 import *


@pytest.fixture
def simple_map():
    return [
            "AAAA",
            "BBCD",
            "BBCC",
            "EEEC",
            ]


def test_parse_map(simple_map):
    parsed_map = parse_map(simple_map)
    assert parsed_map[complex(0, 0)] == "A"
    assert parsed_map[complex(1, 0)] == "A"
    assert parsed_map[complex(0, 1)] == "B"


def test_find_island(simple_map):
    # Start with just (0, 0) and A
    parsed_map = parse_map(simple_map)
    visited = [[False for _ in range(4)] for _ in range(4)]
    island, visited = find_island(parsed_map, complex(0, 0), visited)
    print(visited)

    assert len(island) == 4
    assert visited[1][0] == True
    assert visited[2][0] == True
    assert visited[3][0] == True

    assert island == {complex(1, 0), complex(0,0), complex(2, 0), complex(3, 0)}

    # Now start with (1, 0) and B - and update visited
    island, visited = find_island(parsed_map, complex(0, 1), visited)
    print(island, visited)
    assert parsed_map[complex(0, 1)] == "B"
    assert len(island) == 4

    # Now C 
    island, visited = find_island(parsed_map, complex(2, 1), visited)
    assert parsed_map[complex(2, 1)] == "C"
    assert len(island) == 4

def test_auto_find_islands(simple_map):
    parsed_map = parse_map(simple_map)
    visited = [[False for _ in range(4)] for _ in range(4)]
    islands = []
    for point in parsed_map:
        if visited[int(point.real)][int(point.imag)]:
            continue
        island, visited = find_island(parsed_map, point, visited)
        islands.append(island)
    assert len(islands) == 5

def test_calculate_island_perimeter():
    island = {complex(0, 0), complex(1, 0), complex(2, 0), complex(3, 0)}
    assert island_perimeter(island) == 4 + 4 + 2

    island = {complex(0, 0), complex(1, 0), complex(2, 0), complex(3, 0), complex(3, 1), complex(3, 2)}
    assert island_perimeter(island) == 14


@pytest.fixture
def full_test_map():
    return [
        "RRRRIICCFF",
        "RRRRIICCCF",
        "VVRRRCCFFF",
        "VVRCCCJFFF",
        "VVVVCJJCFE",
        "VVIVCCJJEE",
        "VVIIICJJEE",
        "MIIIIIJJEE",
        "MIIISIJEEE",
        "MMMISSJEEE",
    ]

def test_part1(full_test_map):
    assert part1(full_test_map) == 1930

def test_island_sides(simple_map):
    island = {complex(0, 0), complex(1, 0), complex(2, 0), complex(3, 0)}
    assert island_sides(island)[0] == 4

    # C from simple_map
    island, visited = find_island(parse_map(simple_map), complex(2, 1), [[False for _ in range(4)] for _ in range(4)])
    assert island_perimeter(island) == 10
    corners, vertices = island_sides(island)
    assert len(vertices) == 10
    assert corners == 8

def test_part2(full_test_map):
    assert part2(full_test_map) == 1206


def test_check_vertex_is_corner():
    def is_corner(vertex, island):
        corner = 0
        exposed_faces = 0
        for direction in [1, -1, 1j, -1j, 1 + 1j, 1 - 1j, -1 + 1j, -1 - 1j]:
            if vertex + direction not in island:
                exposed_faces += 1
        print(exposed_faces)
        if exposed_faces == 1:
            # Corner fully enclosed, except for one diagonal direction
            corner += 1
        elif exposed_faces == 6:
            # 90 degree corner 
            corner += 1
        return corner
    
    island = {complex(0, 0), complex(1, 0), complex(2, 0), complex(3, 0), complex(3, 1), complex(3, 2)}
    vertices = set()
    for point in island:
        x, y = point.real, point.imag
        vertices.add(complex(x - 0.0, y - 0.0))
        vertices.add(complex(x + 1.0, y - 0.0))
        vertices.add(complex(x - 0.0, y + 1.0))
        vertices.add(complex(x + 1.0, y + 1.0))
    
    assert is_corner(complex(0, 0), vertices) == 1
