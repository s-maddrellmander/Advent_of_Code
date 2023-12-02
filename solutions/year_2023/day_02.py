# solutions/year_2023/day_02.py
from typing import List

import numpy as np

from logger_config import logger
from utils import Timer

# from numpy.typing import ArrayLike, NDArray


def extract_number_after_color(round_data: str) -> List[int]:
    split_data = round_data.split(",")
    colors = ["red", "green", "blue"]
    round_data = [0, 0, 0]
    for i, color in enumerate(colors):
        for data in split_data:
            if color in data:
                round_data[i] = int(data.split(" ")[1])
    return round_data


def process_input(input_data) -> np.array:
    # Each line of the format
    # Game 1: 1 blue, 2 red, 3 green; 3 red, 2 green; 15 blue
    # i.e. game iD, followed by colon, followed by list of colours and counts
    # separated by commas and each round separated by semicolons.
    # Convert this to a tensor of shape (n_games, n_rounds, 3)
    game_ids = []
    total_data = []
    max_n_rounds = 0
    for line in input_data:
        game_data = []
        game_id, game = line.split(":")
        game_ids.append(int(game_id.split(" ")[1]))
        game = game.split(";")
        if len(game) > max_n_rounds:
            max_n_rounds = len(game)
        for round_data in game:
            round_data = extract_number_after_color(round_data)
            assert len(round_data) == 3
            # round_data = [int(x.split(" ")[1]) for x in round_data]
            game_data.append(round_data)
        total_data.append(game_data)
    # Now we need to pad the total_data to make it a tensor
    # of shape (n_games, max(n_rounds), 3)
    for game_data in total_data:
        while len(game_data) < max_n_rounds:
            game_data.append([0, 0, 0])
    # Combine game_ids and game_data into a single tensor
    game_ids = np.array(game_ids)
    assert len(total_data) == len(input_data)
    assert len(total_data[0]) == max_n_rounds
    total_data = np.array(total_data)

    # assert total_data.shape[0][1] == 3

    return total_data


def part1(input_data: List[str]) -> str | int:
    with Timer("Part1"):
        # Your solution for part 1 goes here
        # This is an (n_games, n_rounds, 3) np.array
        processed_data = process_input(input_data)
        # Max values for the red, green and blue
        num_rounds = processed_data.shape[1]
        num_games = processed_data.shape[0]
        max_values = [12, 13, 14]
        logger.info("Take the max values and shape to (num_games, num_rounds, 3)")
        max_values = np.tile(max_values, (num_games, num_rounds, 1))

        # Compare the processed_data with the max_values
        # Make a boolean array of shape (num_games, num_rounds, 3) for whether
        # processed_data is less than max_values
        possible = processed_data <= max_values
        # Then get the row indicies where all values are true for all rounds in the game
        # i.e. where the game is possible - this is a boolean array of shape (num_games)
        possible = np.all(possible, axis=(1, 2))
        # Then we convert to integers and sum their indexes to get the number of possible games
        # get the indexes of the possible games, starting at 1
        valid = np.where(possible)

        num_possible = np.sum(valid) + valid[0].shape[0]
    return num_possible


def part2(input_data: List[str]) -> str | int:
    with Timer("Part2"):
        processed_data = process_input(input_data)
        # We are going to max pool the array along the color axis
        # i.e. we want to get the max number of each color for each game
        max_colors = np.max(processed_data, axis=1)
        assert max_colors.shape == (len(input_data), 3)
        # Then we multiply the max colors together and sum them
        product = np.prod(max_colors, axis=1)
        return np.sum(product)
