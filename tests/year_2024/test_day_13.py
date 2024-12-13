import pytest
from solutions.year_2024.day_13 import *

def test_solve_pair():
    eq_1 = [8400, 94, 22]
    eq_2 = [5400, 34, 67]
    
    A, B = solve_pair(eq_1, eq_2)
    assert A == 80
    assert B == 40
    
def test_solve_pair_invalid():
    eq_1 = [12748, 26, 67]
    eq_2 = [12176, 66, 21]
    
    A, B = solve_pair(eq_1, eq_2)
    assert A == -1
    assert B == -1


@pytest.fixture
def test_data():
    return ["Button A: X+94, Y+34",
        "Button B: X+22, Y+67",
        "Prize: X=8400, Y=5400",
        "",
        "Button A: X+26, Y+66",
        "Button B: X+67, Y+21",
        "Prize: X=12748, Y=12176",
        "",
        "Button A: X+17, Y+86",
        "Button B: X+84, Y+37",
        "Prize: X=7870, Y=6450",
        "",
        "Button A: X+69, Y+23",
        "Button B: X+27, Y+71",
        "Prize: X=18641, Y=10279",]

def test_parse_input(test_data):
    equations = parse_input(test_data)
    
    assert equations[0] == ([8400, 94, 22], [5400, 34, 67])


def test_part1(test_data):
    assert part1(test_data) == 480
    
def test_part2(test_data):
    assert part2(test_data) == 875318608908