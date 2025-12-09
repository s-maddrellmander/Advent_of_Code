import pytest
from solutions.year_2025.day_01 import *

test_data = [
            "L68",
            "L30",
            "R48",
            "L5",
            "R60",
            "L55",
            "L1",
            "L99",
            "R14",
            "L82",
            ]

def test_part1():
    assert part1(test_data) == 3

def test_part2():
    assert part2(test_data) == 6 
    
def test_divmod():
    A, B = divmod(50, 100)
    assert A == 0
    assert B == 50

    A, B = divmod(-50, 100)
    assert A == -1
    assert B == 50
    