import logging
from utils import load_file
import jax.numpy as jnp

def parse_array_to_jax(inputs):
    arr = []
    for line in inputs:
        row = [int(x) for x in line]
        arr.append(row)
    return jnp.array(arr)

def pad_jnp_array(arr):
    # Pad with zeros around the array
    padded = jnp.pad(arr, 1) 
    return padded

def get_views(probe, arr):
    i, j = probe
    return dict(node=probe,
                up=arr[:i, j],
                down=arr[i + 1:, j],
                left=arr[i, :j],
		right=arr[i, j + 1:],
                node_value=arr[i, j])

def check_any_view_clear(view):
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

def loop_all_nodes(arr):
    counter = 0
    for i in range(1, arr.shape[0] -1):
         for j in range(1, arr.shape[1] -1):
             view = get_views((i, j), arr)
             if check_any_view_clear(view):
                 counter += 1
    return counter
   

def part_1(inputs):
    pass

def part_2(inputs):
    pass

def control():
    inputs = load_file("year_2022/data/data_8.txt")
    part_1(inputs)
    part_2(inputs)
