import logging
from utils import load_file
import re
import numpy as np
from tqdm import tqdm

def parse_senor_line(line):
    coords = re.findall(r'-?\d+', line)
    coords = [int(x) for x in coords]
    return dict(sensor=complex(coords[0], coords[1]),
                beacon=complex(coords[2], coords[3]))


def scope_manhatten_distance(sensor, beacon, cave):
    # Take the sensor location and the beacon and calcualte the manhatten dist
    # Then for all purmutations within that add those locations to the cave map
    abs_x = int(abs(sensor.real - beacon.real))
    abs_y = int(abs(sensor.imag - beacon.imag))
    sqr = abs_x + abs_y
    # All combinations must have the same (or loess than the) total sum
    # This loop here is slow. 
    for x in tqdm(range(-sqr, sqr + 1, 1)):
        for y in range(-sqr, sqr + 1, 1):
            if abs(x) + abs(y) <= sqr:
                # Add to cave
                if complex(sensor.real + x, sensor.imag + y) not in cave.keys(): 
                    cave[complex(sensor.real + x, sensor.imag + y)] = "#"
    return cave, sqr

def probe_level(cave, level):
    imag = set([key.imag for key in cave.keys()])
    valid = [key for key in cave.keys() if key.imag == level and cave[key] != "B"]
     # print("\n", [cave[key] for key in valid])
    return len(valid)

def plot_cave(cave):
    # Simple visualisation to make sure the shapes are right
    cave_x = [x.real for x in cave.keys()]
    cave_y = [y.imag for y in cave.keys()]
    x_range = [0, int(min(cave_x) + max(cave_x))]
    y_range = [0, int(min(cave_y) + max(cave_y))]
    base = np.zeros((x_range[1] +10, y_range[1]+10))
    for loc in cave.keys():
        base[int(min(cave_x) + loc.real)][int(min(cave_y) + loc.imag)] = 1
    print(base)

def check_level(coords, cave, level):
    sensor = coords["sensor"]
    beacon = coords["beacon"]
    cave[coords["sensor"]] = "S"
    cave[coords["beacon"]] = "B"
    abs_x = int(abs(sensor.real - beacon.real))
    abs_y = int(abs(sensor.imag - beacon.imag))
    sqr = abs_x + abs_y 
    # Check the level in within the sensor range
    if abs(level - sensor.imag) < sqr:
        # test level within range
        
        d_y = abs(level - sensor.imag)
        d_x = int(abs(sqr - d_y))
        for dx in range(-d_x, d_x+1, 1):
            if complex(sensor.real + dx, level) not in cave.keys(): 
                cave[complex(sensor.real + dx, level)] = "#"
    return cave

def check_level_x(coords, cave, level):
    sensor = coords["sensor"]
    beacon = coords["beacon"]
    cave[coords["sensor"]] = "S"
    cave[coords["beacon"]] = "B"
    abs_x = int(abs(sensor.real - beacon.real))
    abs_y = int(abs(sensor.imag - beacon.imag))
    sqr = abs_x + abs_y 
    # Check the level in within the sensor range
    if abs(level - sensor.real) < sqr:
        # test level within range
        
        d_x = abs(level - sensor.real)
        d_y = int(abs(sqr - d_x))
        for dy in range(-d_y, d_y+1, 1):
            if complex(level, sensor.imag + dy) not in cave.keys(): 
                cave[complex(level, sensor.real + dy)] = "#"
    return cave

def count_part_2(cave, level):
    possibles = []
    for y in range(0, level):
        probe = complex(level, y)
        if probe not in cave.keys():
            possibles.append(probe)
    return possibles

def part_1(inputs, level):
    cave = dict()
    for line in tqdm(inputs):
        coords = parse_senor_line(line)
        cave = check_level(coords, cave, level)
    count = probe_level(cave, level)
    logging.info(f"Part 1: {count}")
    return count

def part_2(inputs, level):
    cave = dict()
    for line in tqdm(inputs):
        coords = parse_senor_line(line)
        cave = check_level_x(coords, cave, level)
    possible = count_part_2(cave, level)
    print(possible)
    assert len(possible) == 1


def control():
    inputs = load_file("year_2022/data/data_15.txt")
    part_1(inputs, level=2000000)
    part_2(inputs)