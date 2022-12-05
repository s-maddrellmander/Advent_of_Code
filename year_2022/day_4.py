from utils import load_file
import logging

import jax
import jax.numpy as jnp


def get_ranges(line):
    parts = line.split(",")
    e1 = [int(x) for x in parts[0].split("-")]
    e2 = [int(x) for x in parts[1].split("-")]
    
    range_1 = jnp.arange(e1[0], e1[1] + 1, 1)
    range_2 = jnp.arange(e2[0], e2[1] + 1, 1)
    return range_1, range_2

def check_intersection(range_1, range_2, part=1):
    # Check both ways, does one list contain the other
    inter, comm_1, comm_2 = jnp.intersect1d(range_1, range_2, return_indices=True)
    if part == 1:
        if len(comm_1) == len(range_1) or len(comm_2) == len(range_2):
            return 1
        else:
            return 0 
    else:
        if len(inter) > 0:
            return 1
        else:
            return 0

def part_1(inputs):
    count = 0
    for line in inputs:
        r1, r2 = get_ranges(line)
        count += check_intersection(r1, r2)
    logging.info(f"Part 1: {count}")
    return count

def part_2(inputs):
    count = 0
    for line in inputs:
        r1, r2 = get_ranges(line)
        count += check_intersection(r1, r2, part=2)
    logging.info(f"Part 2: {count}")
    return count

def control():
    inputs = load_file("year_2022/data/data_4.txt")
    part_1(inputs)
    part_2(inputs)