import pytest

from solutions.year_2023.day_10 import *


def test_part1():
    input_data = [".....", ".S-7.", ".|.|.", ".L-J.", ".....", "....."]
    path = part1(input_data=input_data)
    assert path


def test_step_by_step():
    input_data = [".....", ".S-7.", ".|.|.", ".L-J.", ".....", "....."]
    parsed_grid, start_pos = parse_grid(input_data)
    plot_map(parsed_grid.keys())
    assert len(parsed_grid.keys()) == 8
    path = bfs(parsed_grid, start_pos)
    assert len(path) == 8


def test_busy_step_by_step():
    input_data = ["-L|F7", "7S-7|", "L|7||", "-L-J|", "L|-JF"]
    parsed_grid, start_pos = parse_grid(input_data)
    plot_map(parsed_grid.keys())
    # assert len(parsed_grid.keys()) == 8
    path = bfs(parsed_grid, start_pos)
    assert len(path) == 8
