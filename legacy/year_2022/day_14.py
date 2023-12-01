import cmath
import logging

from utils import load_file


def path_to_rock_coord(line):
    coords = []
    line = line.split(" -> ")
    for i in range(len(line) - 1):
        start = tuple(int(a) for a in line[i].split(","))
        end = tuple(int(a) for a in line[i + 1].split(","))
        if start[0] == end[0]:
            for j in range(abs(end[1] - start[1]) + 1):
                if end[1] - start[1] > 0:
                    factor = 1
                else:
                    factor = -1
                coords.append(complex(start[0], factor * j + start[1]))
        elif start[1] == end[1]:
            for j in range(abs(end[0] - start[0]) + 1):
                if end[0] - start[0] > 0:
                    factor = 1
                else:
                    factor = -1
                coords.append(complex(start[0] + factor * j, start[1]))

    return list(set(coords))


class Sand:
    def __init__(self, source) -> None:
        self.loc = source

    def step(self, cave_map):
        # Step the sand, first, try down, then diag left, then diag right
        if self.loc + complex(0, 1) not in cave_map:
            self.loc += +complex(0, 1)
            return True
        elif self.loc + complex(-1, 1) not in cave_map:
            self.loc += +complex(-1, 1)
            return True
        elif self.loc + complex(1, 1) not in cave_map:
            self.loc += +complex(1, 1)
            return True
        else:
            return False


class Map:
    def __init__(self, source=complex(500, 0)):
        self.source = source
        self.map = {}  # Dict with complex keys as coords, "#" for rock,. "o" for sand
        self.rock = []
        self.sand = []
        self.abyss = None  # This is the lowest coordinate we have, below this is abyss

    def set_abyss(self):
        min_val = max([y.imag for y in self.rock])
        self.abyss = min_val

    def set_rock(self, rock):
        for loc in rock:
            self.map[loc] = "#"
            self.rock.append(loc)
        self.set_abyss()

    def set_floor(self, rock):
        for loc in rock:
            self.map[loc] = "#"
            self.rock.append(loc)

    def fill_with_sand(self, part=1):
        grain = Sand(self.source)
        state = True
        while state is True:
            state = grain.step(self.map)
            if grain.loc.imag > self.abyss and part == 1:
                return True
            if grain.loc == (500 + 0j) and part == 2:
                return True
        self.sand.append(grain.loc)
        self.map[grain.loc] = "o"
        return False


def get_coords(inputs):
    coords = []
    for path in inputs:
        segment = path_to_rock_coord(path)
        coords.extend(segment)
    coords = list(set(coords))
    return coords


def part_1(inputs):
    cave = Map()
    rock_coords = get_coords(inputs)
    cave.set_rock(rock_coords)
    test_point = False
    sand_counter = 0
    while test_point is False:
        test_point = cave.fill_with_sand()
        if test_point is False:
            sand_counter += 1
    logging.info(f"Part 1 {sand_counter}")
    return sand_counter


def part_2(inputs):
    cave = Map()
    rock_coords = get_coords(inputs)

    cave.set_rock(rock_coords)
    floor_level = int(cave.abyss + 2)
    # Update the abyss
    cave.abyss = floor_level + 10
    floor = [f"-1000,{floor_level} -> 1000,{floor_level}"]
    floor_coords = get_coords(floor)
    cave.set_floor(floor_coords)
    test_point = False
    sand_counter = 0
    while test_point is False:
        test_point = cave.fill_with_sand(part=2)
        sand_counter += 1

    logging.info(f"Part 2: {sand_counter}")
    return sand_counter


def control():
    inputs = load_file("year_2022/data/data_14.txt")
    part_1(inputs)
    part_2(inputs)
