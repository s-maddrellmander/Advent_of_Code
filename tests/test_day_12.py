import pytest

from solutions.year_2023.day_12 import *


def test_count_example_without_unknown():
    # Example without unknowns
    example = "#.#.###"
    assert ["#.#.###"] == count_damaged_springs(example, pattern=[1, 1, 3])


def test_generate_combinations():
    springs = ".??..??...?##. 1,1,3"
    results = []
    generate_combinations(springs, 0, "", results, [1, 1, 3], memo={})
    assert len(results) == 4


def test_count_example_with_unknown():
    # Example with unknowns
    example = "???.###"
    assert len(count_damaged_springs(example, pattern=[1, 1, 3])) == 1
    example = "?#?#?#?#?#?#?#?"
    assert len(count_damaged_springs(example, pattern=[1, 3, 1, 6])) == 1
    example = ".??..??...?##."
    assert len(count_damaged_springs(example, pattern=[1, 1, 3])) == 4


def test_parse_output():
    input_data = ["#.#.### 1,1,3", ".#...#....###. 1,1,3"]
    results = parse_input(input_data)
    assert results == [("#.#.###", [1, 1, 3]), (".#...#....###.", [1, 1, 3])]


def test_part1():
    input_data = [
        "???.### 1,1,3",
        ".??..??...?##. 1,1,3",
        "?#?#?#?#?#?#?#? 1,3,1,6",
        "????.#...#... 4,1,1",
        "????.######..#####. 1,6,5",
        "?###???????? 3,2,1",
    ]
    assert part1(input_data=input_data) == 21


def test_valid_combination():
    combination = ".##..#...###.."
    expected_groups = [2, 1, 3]
    assert valid_combination(combination, expected_groups) == True

    combination = ".##..#...##.."
    expected_groups = [2, 1, 3]
    assert valid_combination(combination, expected_groups) == False
