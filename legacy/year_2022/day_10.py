import logging

from utils import Queue, load_file


class CathodRay:
    def __init__(self) -> None:
        self.register = 1
        self.clock = 0
        self.register_history = []
        self.clock_history = []
        self.buffer = 0
        self.history = {}

    def update_register(self, val, clock):
        if val == 0:
            self.clock += 1
            self.register_history.append(self.register)
            self.clock_history.append(self.clock)
            self.history[self.clock] = self.register
        else:
            self.clock += 1
            self.register_history.append(self.register)
            self.clock_history.append(self.clock)
            self.history[self.clock] = self.register

            self.clock += 1
            self.register_history.append(self.register)
            self.clock_history.append(self.clock)
            self.history[self.clock] = self.register
            self.register += val

    def get_register_from_cycle(self, cycle):
        # Get the cycle history index
        # index = self.clock_history.index(cycle)
        # return self.register_history[index]
        return self.history[cycle]


def loop_instructions(cathode_ray, inputs):
    for line in inputs:
        val, cycles = parse_line(line)
        cathode_ray.update_register(val, cycles)


def get_results_from_index(cathode_ray, index_list):
    totals = []
    for index in index_list:
        val = cathode_ray.get_register_from_cycle(index)
        totals.append(val * index)
    return totals


def parse_line(line):
    if line == "noop":
        return 0, 1
    else:
        return int(line.split(" ")[1]), 2


def update_pixels(X, cycles, pixels):
    pos = (cycles - 1) % 40
    if pos in {X - 1, X, X + 1}:
        pixels[cycles - 1] = "#"


def get_image(cathode_ray):
    pixels = ["." for _ in range(240)]
    for i in range(1, 241, 1):
        val = cathode_ray.get_register_from_cycle(i)
        update_pixels(val, i, pixels)
    pixels = "".join(x for x in pixels)
    return pixels


def part_1(inputs):
    cathode_ray = CathodRay()
    loop_instructions(cathode_ray, inputs)
    index_list = [20, 60, 100, 140, 180, 220]
    vals = get_results_from_index(cathode_ray, index_list)
    results = sum(vals)
    logging.info(f"Part 1: {results}")
    return results


def part_2(inputs):
    cathode_ray = CathodRay()
    loop_instructions(cathode_ray, inputs)
    pixels = get_image(cathode_ray)
    # Display the Cathode Ray Tube
    for i in range(6):
        print(pixels[i * 40 : i * 40 + 40])


def control():
    inputs = load_file("year_2022/data/data_10.txt")
    print(len(inputs))
    part_1(inputs)
    part_2(inputs)
