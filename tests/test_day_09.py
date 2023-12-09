import pytest

from solutions.year_2023.day_09 import *


def test_next_in_seq():
    # Test with sequence of squares
    seq = [1, 4, 9, 16]
    assert next_in_sequence(seq) == 25

    # Test with sequence of multiples of 3
    seq = [0, 3, 6, 9, 12, 15]
    assert next_in_sequence(seq) == 18

    # Test with sequence of triangular numbers
    seq = [1, 3, 6, 10, 15, 21]
    assert next_in_sequence(seq) == 28

    # Test with a custom sequence
    seq = [10, 13, 16, 21, 30, 45]
    assert next_in_sequence(seq) == 68


def test_part1():
    input_data = ["0 3 6 9 12 15", "1 3 6 10 15 21", "10 13 16 21 30 45"]
    assert part1(input_data=input_data) == 114


def test_part2():
    input_data = [
        "1 3 6 10 15 21",
        "10 13 16 21 30 45",
        "0 3 6 9 12 15",
    ]
    assert part2(input_data=input_data) == 2
