from solutions.year_2024.day_06 import *
import pytest

@pytest.fixture
def data():
    return [
            "....#.....",
            ".........#",
            "..........",
            "..#.......",
            ".......#..",
            "..........",
            ".#..^.....",
            "........#.",
            "#.........",
            "......#...",
                ]

def test_create_map(data):
    carte, start = create_map(data)
    assert carte[(0, 0)] == {"type": ".", "visited": False, "direction": []}
    assert carte[(6, 4)] == {"type": "^", "visited": False,  "direction": []}
    assert carte[(0, 4)] == {"type": "#", "visited": False, "direction": []}
    assert start == (6, 4)
   
 
@pytest.mark.parametrize("direction", ["^", "v", ">", "<"])
def test_move_guard(direction, data):
    carte, _ = create_map(data)
    position = (1, 1)
    new_position, new_direction = move_guard(position, direction, carte)
    if direction == "^":
        assert new_position == (0, 1)
        assert new_direction == "^"
    elif direction == "v":
        assert new_position == (2, 1)
        assert new_direction == "v"
    elif direction == ">":
        assert new_position == (1, 2)
        assert new_direction == ">"
    elif direction == "<":
        assert new_position == (1, 0)
        assert new_direction == "<"
    else:
        assert new_position is None
        assert new_direction is None
        
        
        
def test_part1(data):
    assert part1(data) == 41
    
def test_part2(data):
    assert part2(data) == 6