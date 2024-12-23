import pytest

from solutions.year_2024.day_17 import *


def test_eg_1():
    register_A = 0
    register_B = 0
    register_C = 9
    operand = 2

    A, B, C, val = op_6(register_A, register_B, register_C, operand)
    print(A, B, C, val)
    assert B == 1


@pytest.fixture
def example_data():
    data = [
        "Register A: 729",
        "Register B: 0",
        "Register C: 0",
        "",
        "Program: 0,1,5,4,3,0",
    ]
    return data


def test_part1(example_data):
    assert part1(example_data) == 4635635210
