import pytest

from solutions.year_2024.day_18 import *

@pytest.fixture
def eg_data():
    return   ["5,4",
        "4,2",
        "4,5",
        "3,0",
        "2,1",
        "6,3",
        "2,4",
        "1,5",
        "0,6",
        "3,3",
        "2,6",
        "5,1",
        "1,2",
        "5,5",
        "2,5",
        "6,5",
        "1,4",
        "0,4",
        "6,4",
        "1,1",
        "6,1",
        "1,0",
        "0,5",
        "1,6",
        "2,0",]

def test_parse_map(eg_data):
    carte = parse_map(eg_data)
    assert len(carte) == 25
    assert carte[complex(5, 4)] == '#'
    assert carte[complex(2, 6)] == '#'



def test_part1(eg_data):
    carte = parse_map(eg_data, 12)
    print_map(carte)
    start = complex(0, 0)
    end = complex(6, 6)
    came_from, cost_so_far = a_star(carte, start, end, 7)
    path = reconstruct_path(came_from, start, end)
    print(path)
    
    print_map(carte, path)
    assert len(path) - 1 == 22

def test_part2(eg_data):
    left = 0
    right = len(eg_data)
    while left < right:
        mid = (left + right) // 2
        carte = parse_map(eg_data, mid)
        start = complex(0, 0)
        end = complex(6, 6)
        came_from, cost_so_far = a_star(carte, start, end, 7)
        path = reconstruct_path(came_from, start, end)
        if path:
            left = mid + 1
        else:
            right = mid
    assert eg_data[left-1] == "6,1"