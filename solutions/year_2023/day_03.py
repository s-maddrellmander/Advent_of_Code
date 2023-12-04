# solutions/year_2023/day_00.py
from typing import List, Optional, Union

import numpy as np

from logger_config import logger
from utils import Timer


def process_array(input_data: List[str]) -> np.array:
    # We need a mapping to convert chars to ints
    wildcard_mapping = {
        "*": -1,
        "@": -2,
        "#": -3,
        "$": -4,
        "+": -5,
        "-": -6,
        "=": -7,
        "%": -8,
        "&": -9,
        "/": -10,
        ".": -99,
    }
    # Now we can convert the input data to a numpy array and  keep the original numbers
    processed_array = np.array(
        [
            [
                wildcard_mapping[char] if char in wildcard_mapping else int(char)
                for char in row
            ]
            for row in input_data
        ]
    )
    return processed_array


def get_wildcards(grid: np.array) -> np.array:
    # Find the locations of all wildcards (-ve values)
    wildcards = np.where(grid < 0)
    return wildcards


def get_numbers(grid: np.array) -> np.array:
    # Find the locations of all numbers (positive values)
    numbers = np.where(grid > 0)
    return numbers


def find_adjacent_numbers(wildcards: np.array, numbers: np.array) -> np.array:
    adjacent_coords = []
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]

    # Iterate over each point in wildcards
    for i in range(wildcards.shape[1]):
        x, y = wildcards[:, i]

        # Check all 8 directions
        for dx, dy in directions:
            adjacent_x, adjacent_y = x + dx, y + dy
            adjacent_point = np.array([adjacent_x, adjacent_y])

            # Check if the adjacent point is in numbers
            for j in range(numbers.shape[1]):
                if np.array_equal(adjacent_point, numbers[:, j]):
                    adjacent_coords.append(adjacent_point.tolist())

    # Remove duplicates and convert to numpy array
    if not adjacent_coords:
        return np.array([[], []])
    unique_adjacent_coords = np.unique(adjacent_coords, axis=0).T

    return unique_adjacent_coords


def get_number_sequences(grid: np.array, numbers, is_adjacent: np.array) -> np.array:
    nums_to_save = []
    for coord in is_adjacent.T:
        x, y = coord
        logger.info((y, x, grid[y, x]))

        # Skip if the number at the current coordinate is <= 0
        if grid[y, x] < 0:
            continue

        left_bound = max(0, x - 2)
        right_bound = min(grid.shape[1], x + 3)
        row_segment = grid[y, left_bound:right_bound]

        logger.info(row_segment)
        # Process for different cases
        if (
            x > 0
            and x < grid.shape[1] - 1
            and grid[y, x - 1] > 0
            and grid[y, x + 1] > 0
        ):
            # Both sides
            nums_to_save.append(
                int(
                    "".join(
                        map(str, row_segment[x - 1 - left_bound : x + 2 - left_bound])
                    )
                )
            )
        elif x > 0 and grid[y, x - 1] > 0:
            # Left side
            if x > 1 and grid[y, x - 2] > 0:
                nums_to_save.append(
                    int(
                        "".join(
                            map(
                                str,
                                row_segment[x - 2 - left_bound : x + 1 - left_bound],
                            )
                        )
                    )
                )
            else:
                nums_to_save.append(
                    int(
                        "".join(
                            map(
                                str,
                                row_segment[x - 1 - left_bound : x + 1 - left_bound],
                            )
                        )
                    )
                )
        elif x < grid.shape[1] - 1 and grid[y, x + 1] > 0:
            # Right side
            if x < grid.shape[1] - 2 and grid[y, x + 2] > 0:
                nums_to_save.append(
                    int(
                        "".join(
                            map(str, row_segment[x - left_bound : x + 3 - left_bound])
                        )
                    )
                )
            else:
                nums_to_save.append(
                    int(
                        "".join(
                            map(str, row_segment[x - left_bound : x + 2 - left_bound])
                        )
                    )
                )
        else:
            nums_to_save.append(grid[y, x])

    logger.info(nums_to_save)
    return nums_to_save


def tuple_array_to_array(array) -> np.array:
    return np.array([array[0], array[1]])


def parse_schematic(schematic):
    return [list(line) for line in schematic.splitlines()]


def is_digit(c):
    return c.isdigit()


def get_adjacent_cells(x, y, grid):
    offsets = [-1, 0, 1]
    for dx in offsets:
        for dy in offsets:
            if dx == 0 and dy == 0:
                continue
            new_x, new_y = x + dx, y + dy
            if 0 <= new_x < len(grid[0]) and 0 <= new_y < len(grid):
                yield new_x, new_y


def extract_number(x, y, grid):
    num_str = ""
    while x < len(grid[0]) and is_digit(grid[y][x]):
        num_str += grid[y][x]
        x += 1
    return int(num_str) if num_str else None


def sum_part_numbers(schematic):
    grid = parse_schematic(schematic)
    part_numbers = set()

    for y in range(len(grid)):
        for x in range(len(grid[0])):
            if grid[y][x] not in {".", " "} and not grid[y][x].isdigit():
                for adj_x, adj_y in get_adjacent_cells(x, y, grid):
                    if grid[adj_y][adj_x].isdigit():
                        # Trace the number horizontally
                        start_x = adj_x
                        while start_x > 0 and grid[adj_y][start_x - 1].isdigit():
                            start_x -= 1
                        num_str = ""
                        while start_x < len(grid[0]) and grid[adj_y][start_x].isdigit():
                            num_str += grid[adj_y][start_x]
                            start_x += 1
                        part_numbers.add(int(num_str))
    logger.info(part_numbers)
    return sum(part_numbers)


def one_last_go(input_data: List[str]) -> np.array:
    save = []
    # Split grid on new lines
    grid = input_data.split("\n")
    # grid = [line for line in grid]
    # Pad the grid with . to avoid index errors all around the grid
    grid = ["." + line + "." for line in grid]
    # import ipdb; ipdb.set_trace()
    # Now add top and bottom rows of .
    grid = ["." * len(grid[0])] + grid + ["." * len(grid[0])]
    # import ipdb; ipdb.set_trace()

    logger.info(grid)
    for i in range(len(grid)):
        for j in range(1, len(grid[0]) - 1):
            tmp = ""
            logger.info((i, j))
            if grid[i][j].isdigit():
                # Check the block around the 3 digit number for wildcards
                for k in range(-1, 2):
                    for l in range(-1, 2):
                        if grid[i + k][j + l] in [
                            "*",
                            "@",
                            "#",
                            "$",
                            "+",
                            "-",
                            "=",
                            "%",
                            "&",
                            "/",
                        ]:
                            tmp += grid[i][j]
                            if grid[i][j + 1].isdigit():
                                tmp += grid[i][j + 1]
                                if grid[i][j + 2].isdigit():
                                    tmp += grid[i][j + 2]
                            save.append(int(tmp))
            j += 3
    return sum(save)


def final_go(input_data: List[str]) -> np.array:
    grid = input_data.split("\n")
    #  Pad the grid with . to avoid index errors all around the grid
    grid = ["." + line + "." for line in grid]
    # import ipdb; ipdb.set_trace()
    # Now add top and bottom rows of .
    grid = ["." * len(grid[0])] + grid + ["." * len(grid[0])]
    directions = [(-1, 0), (1, 0), (0, -1), (0, 1), (-1, -1), (-1, 1), (1, -1), (1, 1)]
    numbers = []
    to_check = []
    for i in range(len(grid)):
        for j in range(1, len(grid[0]) - 1):
            tmp = ""
            jump = 0
            if grid[i][j].isdigit():
                tmp += grid[i][j]
                if grid[i][j + 1].isdigit():
                    tmp += grid[i][j + 1]
                    jump += 1
                    if grid[i][j + 2].isdigit():
                        tmp += grid[i][j + 2]
                        jump += 1
                # Save the number and it's coordinates for all numbers
                numbers.append({int(tmp): [(i, j), (i, j + 1), (i, j + 2)]})
            j += jump
            if grid[i][j] in ["*", "@", "#", "$", "+", "-", "=", "%", "&", "/"]:
                # Then add the coordinates in a ring around the widlcard
                to_check.append(
                    [(i + k, j + l) for k in range(-1, 2) for l in range(-1, 2)]
                )
    # Then we want to count all the numbers who have a coordinate in the list to_check
    # import ipdb; ipdb.set_trace()
    for number in numbers:
        for num, coords in number.items():
            logger.info((num, coords))
            for coord in coords:
                logger.info(coord)
                if coord in to_check:
                    logger.info(num, coord)

    return -1


def part1(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 1 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 1"):
        # Process the input data into a 2D numpy array
        # grid = process_array(input_data)
        # # Find the locations of all wildcards (-ve values)
        # wildcards = tuple_array_to_array(get_wildcards(grid))
        # # Find the locations of all numbers (positive values)
        # numbers = tuple_array_to_array(get_numbers(grid))
        # # Check if any of the wildcards are adjacent to any of the numbers
        # is_adjacent = find_adjacent_numbers(wildcards, numbers)
        # # Now we need to find which numbers are part of a sequence of numbers in the array
        # number_sequences = get_number_sequences(grid, numbers, np.array([is_adjacent[1], is_adjacent[0]]))

        # Convert list of string to list with /n
        schematic = "\n".join(input_data)

        return sum_part_numbers(schematic)

        # return -1


def part2(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 2 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 2"):
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
