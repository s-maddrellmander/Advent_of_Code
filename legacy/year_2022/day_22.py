import logging
import re
from collections import deque

from utils import load_file


def parse_map_and_path(inputs):
    turns = {"R": 1, "L": -1, "": 0}
    instructions = [
        (int(s), turns[t]) for (s, t) in re.findall("(\d+)([RL]?)", inputs[-1])
    ]

    mapp = dict()
    start = None
    _width = 0
    _depth = 0
    for y, line in enumerate(inputs[:-1], 1):
        for x, val in enumerate(line, 1):
            if val != " ":
                if start is None:
                    start = complex(x, y)
                mapp[complex(x, y)] = val
                if x > _width:
                    _width = x
                if y > _depth:
                    _depth = y
    return mapp, instructions, start, _width, _depth


def rotate_directions(directions, turn):
    directions = deque(directions)
    directions.rotate(-turn)
    directions = list(directions)
    return directions


def part_1(inputs):
    mapp, instructions, start, _width, _depth = parse_map_and_path(inputs)
    directions = [1, 1j, -1, -1j]  # R D L U
    path = dict()
    position = start
    facing = 0
    width = max([x.real for x in mapp.keys()])
    depth = max([x.imag for x in mapp.keys()])
    assert _width == width
    assert _depth == _depth
    for steps, turn in instructions:
        for _ in range(steps):
            new = position
            new += directions[facing]
            # switch = False
            while new not in mapp:
                new += directions[facing]
                x = new.real
                x = width if x < 1 else x
                x = 1 if x > width else x
                y = new.imag
                y = depth if y < 1 else y
                y = 1 if y > depth else y
                new = x + y * 1j
            if mapp[new] == ".":
                position = new
                # if switch:
                #     facing = new_facing
            else:
                break
        # Then after all the steps we turn
        facing += turn
        facing = facing % 4
        # directions = rotate_directions(directions, turn)
    score = 1000 * position.imag + 4 * position.real + facing % 4
    logging.info(f"Part 1: Final coord {position}, score = {score}")
    return position, score


def part_2(inputs):
    # Same as part 1 but the wrapping rules are different
    pass


def control():
    inputs = load_file("year_2022/data/data_22.txt")
    part_1(inputs)
    part_2(inputs)
