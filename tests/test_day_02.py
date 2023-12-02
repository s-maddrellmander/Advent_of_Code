import pytest

from solutions.year_2023.day_02 import *


def test_process_input():
    input_data = ["Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green"]
    expected_output = np.array([[[4, 0, 3], [1, 2, 6], [0, 2, 0]]])
    generated_result = process_input(input_data)
    logger.info(f"generated_result: {generated_result}")
    logger.info(f"expected_output: {expected_output}")
    assert np.array_equal(generated_result, expected_output)
    input_data = [
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
    ]
    expected_output = np.array(
        [
            [[4, 0, 3], [1, 2, 6], [0, 2, 0]],
            [[4, 0, 3], [1, 2, 6], [0, 2, 0]],
            [[4, 0, 3], [1, 2, 6], [0, 2, 0]],
            [[4, 0, 3], [1, 2, 6], [0, 2, 0]],
        ]
    )
    generated_result = process_input(input_data)
    logger.info(f"generated_result: {generated_result}")
    logger.info(f"expected_output: {expected_output}")
    assert np.array_equal(generated_result, expected_output)


def test_part1():
    # Your test for part 1 goes here
    test_input = [
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
        "Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue",
        "Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red",
        "Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red",
        "Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green",
    ]
    assert part1(test_input) == 8


def test_part2():
    # Your test for part 2 goes here
    test_input = [
        "Game 1: 3 blue, 4 red; 1 red, 2 green, 6 blue; 2 green",
        "Game 2: 1 blue, 2 green; 3 green, 4 blue, 1 red; 1 green, 1 blue",
        "Game 3: 8 green, 6 blue, 20 red; 5 blue, 4 red, 13 green; 5 green, 1 red",
        "Game 4: 1 green, 3 red, 6 blue; 3 green, 6 red; 3 green, 15 blue, 14 red",
        "Game 5: 6 red, 1 blue, 3 green; 2 blue, 1 red, 2 green",
    ]
    assert part2(test_input) == 2286
