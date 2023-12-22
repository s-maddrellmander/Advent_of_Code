import pytest

from solutions.year_2023.day_13 import *


def test_parser():
    # The input file content
    file_content = [
        "#.##..##.",
        "..#.##.#.",
        "##......#",
        "##......#",
        "..#.##.#.",
        "..##..##.",
        "#.#.##.#.",
        "",
        "#...##..#",
        "#....#..#",
        "..##..###",
        "#####.##.",
        "#####.##.",
        "..##..###",
        "#....#..#",
    ]
    out = parse_input(file_content)
    assert len(out) == 2
    assert out[0].shape == (7, 9)


def test_consequtive_rows():
    data = np.array(
        [
            [1, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 0],
        ]
    )
    # Find rows fist
    duplicates = consequtive(data, axis=0)
    assert np.array_equal(duplicates, np.array([2, 6]))

    # Find cols second
    duplicates = consequtive(data, axis=1)
    assert np.array_equal(duplicates, np.array([4]))


def test_reflection():
    data = np.array(
        [
            [1, 0, 1, 1, 0, 0, 1, 1, 0],
            [0, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [1, 1, 0, 0, 0, 0, 0, 0, 1],
            [0, 0, 1, 0, 1, 1, 0, 1, 0],
            [0, 0, 1, 1, 0, 0, 1, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 0],
            [1, 0, 1, 0, 1, 1, 0, 1, 0],
        ]
    )
    # Find rows fist
    duplicates = consequtive(data, axis=0)
    assert np.array_equal(duplicates, np.array([2, 6]))
    # Find the reflections:
    reflections = find_reflection(data, duplicates, axis=0)
    assert reflections == [7]

    # Find cols second
    duplicates = consequtive(data, axis=1)
    assert np.array_equal(duplicates, np.array([4]))
    # Find the reflections:
    reflections = find_reflection(data, duplicates, axis=1)
    assert reflections == [5]


def test_part1():
    file_content = [
        "#.##..##.",
        "..#.##.#.",
        "##......#",
        "##......#",
        "..#.##.#.",
        "..##..##.",
        "#.#.##.#.",
        "",
        "#...##..#",
        "#....#..#",
        "..##..###",
        "#####.##.",
        "#####.##.",
        "..##..###",
        "#....#..#",
    ]
    assert part1(input_data=file_content) == 405


def test_part2():
    file_content = [
        "#.##..##.",
        "..#.##.#.",
        "##......#",
        "##......#",
        "..#.##.#.",
        "..##..##.",
        "#.#.##.#.",
        "",
        "#...##..#",
        "#....#..#",
        "..##..###",
        "#####.##.",
        "#####.##.",
        "..##..###",
        "#....#..#",
    ]
    assert part2(input_data=file_content) == 400
