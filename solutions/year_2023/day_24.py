# solutions/year_2023/day_24.py
import itertools
import math
from typing import Dict, List, Optional, Tuple, Union

from sympy import Symbol, solve_poly_system

from logger_config import logger
from utils import Timer


def parse_data(input_data: List[str]) -> List[List[int]]:
    """
    Parse the input data into a list of coordinates and velocities
    Slpit on the @ symbol

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        List[List[int]]: The parsed input data.
    """
    parsed_data = []
    for line in input_data:
        coords, velocity = line.split("@")
        coords = coords.strip().split(",")
        velocity = velocity.strip().split(",")
        parsed_data.append([int(coord) for coord in coords + velocity])
    return parsed_data


def crossing_paths(
    coord_velocities: List[List[int]],
    min_val: int = 200000000000000,
    max_val: int = 400000000000000,
) -> List[Tuple[int, int]]:
    """
    y = mx + b
    x = (y - b) / m
    y = (dy / dx) * x + b

    f.1 = f.2
    Solve for x
    Substitute x in f.1 to get y
    check if f.1(x) == f.2(x)

    (dy1 / dx1) * x + b1 = (dy2 / dx2) * x + b2
    x = (b2 - b1) / ((dy1 / dx1) - (dy2 / dx2))

    y1 = (dy1 / dx1) * x + b1
    y2 = (dy2 / dx2) * x + b2

    """

    min_x = min_y = min_val
    max_x = max_y = max_val

    valid_intersections = []

    for xyz_vxvyvz in itertools.combinations(coord_velocities, 2):
        x1, y1, vx1, vy1 = (
            xyz_vxvyvz[0][0],
            xyz_vxvyvz[0][1],
            xyz_vxvyvz[0][3],
            xyz_vxvyvz[0][4],
        )
        x2, y2, vx2, vy2 = (
            xyz_vxvyvz[1][0],
            xyz_vxvyvz[1][1],
            xyz_vxvyvz[1][3],
            xyz_vxvyvz[1][4],
        )

        # Check if parallel
        if (vy1 / vx1) == (vy2 / vx2):
            continue

        # Get the b1 and b2 values - intercepts
        b1 = y1 - (vy1 / vx1) * x1
        b2 = y2 - (vy2 / vx2) * x2

        # Get the x value of the intersection
        x = (b2 - b1) / ((vy1 / vx1) - (vy2 / vx2))

        # Get the y value of the intersection
        y = (vy1 / vx1) * x + b1

        # Time for the intersection
        t1 = (x - x1) / vx1
        t2 = (x - x2) / vx2
        if t1 < 0 or t2 < 0:
            continue

        # Check if the intersection is within the bounds of the test area
        if min_x <= x <= max_x and min_y <= y <= max_y:
            valid_intersections.append((int(x), int(y)))

    return valid_intersections


def single_solution(coord_velocities: List[List[int]]) -> List[int]:
    # Get the first three shards
    shards = coord_velocities[:3]
    # Get the x, y, z, vx, vy, vz values for each shard
    x0, y0, z0, xv, yv, zv = shards[0]
    x1, y1, z1, xv, yv, zv = shards[1]
    x2, y2, z2, xv, yv, zv = shards[2]

    # Set up the equations
    x = Symbol("x")
    y = Symbol("y")
    z = Symbol("z")
    vx = Symbol("vx")
    vy = Symbol("vy")
    vz = Symbol("vz")

    equations = []
    t_syms = []

    for idx, shard in enumerate(shards[:3]):
        # Get the x, y, z, vx, vy, vz values for each shard
        x0, y0, z0, xv, yv, zv = shard
        # Get the time for each shard
        t = Symbol("t" + str(idx))

        # Set up the equations
        eqx = x + vx * t - x0 - xv * t
        eqy = y + vy * t - y0 - yv * t
        eqz = z + vz * t - z0 - zv * t

        equations.append(eqx)
        equations.append(eqy)
        equations.append(eqz)
        t_syms.append(t)

    # Solve the equations
    result = solve_poly_system(equations, *([x, y, z, vx, vy, vz] + t_syms))
    return result[0][0] + result[0][1] + result[0][2]


def part1(
    input_data: Optional[List[str]],
    min_val: int = 200000000000000,
    max_val: int = 400000000000000,
) -> Union[str, int]:
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
        coord_velocities = parse_data(input_data)
        intersect = crossing_paths(coord_velocities, min_val, max_val)
        return len(intersect)


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
        coord_velocities = parse_data(input_data)
        return single_solution(coord_velocities)
