import logging

import jax
import jax.numpy as jnp

from utils import load_file


def grouping(calories):
    groups = [[]]
    for val in calories:
        if val == "":
            groups.append([])
        else:
            groups[-1].append(int(val))
    calories = [jnp.array(group) for group in groups if group != []]
    return calories


def part_1(calories):
    # Sum the calories per elf
    sums = jnp.array([jnp.sum(x) for x in calories])
    max_val = jnp.max(sums)
    logging.info(f"Part 1: {max_val}")
    return max_val


def part_2(calories):
    sums = jnp.array([jnp.sum(x) for x in calories])
    sorted = jnp.sort(sums)
    assert sorted[0] < sorted[-1]
    top_3 = jnp.sum(sorted[-3:])
    logging.info(f"Part 2: {top_3}")
    return top_3


def control():
    calories = load_file("year_2022/data/data_1.txt")
    calories = grouping(calories)
    part_1(calories)
    part_2(calories)
