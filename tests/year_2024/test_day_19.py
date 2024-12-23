import pytest

from solutions.year_2024.day_19 import *


@pytest.fixture
def example_data():
    return [
        "r, wr, b, g, bwu, rb, gb, br",
        "",
        "brwrr",
        "bggr",
        "gbbr",
        "rrbgbr",
        "ubwu",
        "bwurrg",
        "brgr",
        "bbrgwb",
    ]


def test_parse_data(example_data):
    towels, patterns = parse_data(example_data)
    assert towels == ["r", "wr", "b", "g", "bwu", "rb", "gb", "br"]
    assert patterns == [
        "brwrr",
        "bggr",
        "gbbr",
        "rrbgbr",
        "ubwu",
        "bwurrg",
        "brgr",
        "bbrgwb",
    ]


def test_match_towel(example_data):
    towels, patterns = parse_data(example_data)
    assert match_towel("brwrr", towels) == True
    assert match_towel("bggr", towels) == True
    assert match_towel("gbbr", towels) == True
    assert match_towel("rrbgbr", towels) == True
    assert match_towel("ubwu", towels) == False
    assert match_towel("bwurrg", towels) == True
    assert match_towel("brgr", towels) == True
    assert match_towel("bbrgwb", towels) == False


def test_part1(example_data):
    assert part1(example_data) == 6


def test_part2(example_data):
    assert part2(example_data) == 16
