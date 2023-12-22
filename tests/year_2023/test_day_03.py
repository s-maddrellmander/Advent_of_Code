import pytest

from solutions.year_2023.day_03 import *

# Example engine schematic
example_schematic = [
    "467..114..",
    "...*......",
    "..35..633.",
    "......#...",
    "617*......",
    ".....+.58.",
    "..592.....",
    "......755.",
    "...$.*....",
    ".664.598..",
]


def test_example():
    assert sum_part_numbers(example_schematic)[0] == 4361


def test_part1():
    assert part1(example_schematic) == 4361


def test_part2():
    assert part2(example_schematic) == 467835
