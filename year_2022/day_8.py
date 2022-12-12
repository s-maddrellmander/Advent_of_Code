import logging
from utils import load_file
import jax.numpy as jnp
from functools import partial
from tqdm import tqdm

def parse_array_to_jax(inputs):
    arr = []
    for line in inputs:
        row = [int(x) for x in line]
        arr.append(row)
    return jnp.array(arr)

def pad_jnp_array(arr):
    # Pad with zeros around the array
    padded = jnp.pad(arr, 1, constant_values=-1) 
    return padded

def get_views(probe, arr):
    i, j = probe
    # TODO: Should these be sorted wrt the central node
    return dict(node=probe,
                up=jnp.flip(arr[:i, j]),
                down=arr[i + 1:, j],
                left=jnp.flip(arr[i, :j]),
		right=arr[i, j + 1:],
                node_value=arr[i, j])

def check_any_view_clear(view):
    # Needs a special handling for 0
    # Solution: Pad with -1 as tree height 0 is just shorted, not 0
    if jnp.max(view["up"]) < view["node_value"]:
        return True
    elif jnp.max(view["down"]) < view["node_value"]:
        return True
    elif jnp.max(view["left"]) < view["node_value"]:
        return True
    elif jnp.max(view["right"]) < view["node_value"]:
        return True
    else:
        return False


def find_first_greater_than(numbers, threshold):
    # Exception handeling for the edges properly
    try:
        return jnp.where(numbers >= threshold)[0][0] + 1
    except IndexError as e:
        # Can return one as this goes into a multiplication and won't increase anything
        if len(numbers) == 1:  # i.e. just the padding
            return 1
        else:
            return len(numbers[:-1]) 

def get_view_length(view):
    threshold = view["node_value"]
    directions = ["up", "down", "left", "right"]
    distances = [find_first_greater_than(view[key], threshold) for key in directions]
    return distances

def multiple_distances(distances):
    return jnp.prod(jnp.array(distances))

def loop_all_nodes(arr):
    counter = 0
    for i in tqdm(range(1, arr.shape[0] -1)):
         for j in range(1, arr.shape[1] -1):
             view = get_views((i, j), arr)
             if check_any_view_clear(view):
                 counter += 1
    return counter


def get_best_view(arr):
    max_score = 0
    for i in tqdm(range(1, arr.shape[0])):
         for j in range(1, arr.shape[1]): 
            view = get_views((i, j), arr)
            distances = get_view_length(view)
            dist_prod = multiple_distances(distances)
            max_score = max(max_score, dist_prod)
    logging.info(f"Part 2: {max_score}")
    return max_score


def part_1(inputs):
    forest = parse_array_to_jax(inputs)
    padded_forest = pad_jnp_array(forest)
    count = loop_all_nodes(padded_forest)
    logging.info(f"Part 1: {count}")
    return count

def part_2(inputs):
    forest = parse_array_to_jax(inputs)
    padded_forest = pad_jnp_array(forest)
    score = get_best_view(padded_forest)
    return score
    

def control():
    inputs = load_file("year_2022/data/data_8.txt")
    inputs = [[int(x) for x in y] for y in inputs]
    part_1(inputs)
    part_2(inputs)
