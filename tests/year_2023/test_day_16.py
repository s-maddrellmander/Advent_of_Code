import pytest

from solutions.year_2023.day_16 import *


def test_ray_tracing_pipe():
    # | test
    start = (0, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "|"
    path = ray_tracing(coords, start)
    assert set(path) == set([(0, 1), (1, 1), (1, 2), (1, 0)])

    start = (1, 0)
    prev = (1, -1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "|"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 0), (1, 1), (1, 2)])

    start = (2, 1)
    prev = (3, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "|"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(2, 1), (1, 1), (1, 0), (1, 2)])

    start = (1, 2)
    prev = (1, 3)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "|"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 2), (1, 1), (1, 0)])


def test_ray_tracing_dash():
    # - test
    start = (0, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "-"
    path = ray_tracing(coords, start)
    assert set(path) == set([(0, 1), (1, 1), (2, 1)])

    start = (1, 0)
    prev = (1, -1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "-"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 0), (1, 1), (0, 1), (2, 1)])

    start = (2, 1)
    prev = (3, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "-"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(0, 1), (1, 1), (2, 1)])

    start = (1, 2)
    prev = (1, 3)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "-"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 2), (1, 1), (0, 1), (2, 1)])


def test_ray_tracing_forward_slash():
    # / test
    start = (0, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "/"
    path = ray_tracing(coords, start)
    assert set(path) == set([(0, 1), (1, 1), (1, 0)])

    start = (1, 0)
    prev = (1, -1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "/"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 0), (1, 1), (0, 1)])

    start = (2, 1)
    prev = (3, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "/"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 2), (1, 1), (2, 1)])

    start = (1, 2)
    prev = (1, 3)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "/"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 2), (1, 1), (2, 1)])


def test_ray_tracing_back_slash():
    # \ test
    start = (0, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "\\"
    path = ray_tracing(coords, start)
    assert set(path) == set([(0, 1), (1, 1), (1, 2)])

    start = (1, 0)
    prev = (1, -1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "\\"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 0), (1, 1), (2, 1)])

    start = (2, 1)
    prev = (3, 1)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "\\"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(2, 1), (1, 1), (1, 0)])

    start = (1, 2)
    prev = (1, 3)
    coords = {(i, j): "." for i in range(3) for j in range(3)}
    coords[(1, 1)] = "\\"
    path = ray_tracing(coords, start, prev)
    assert set(path) == set([(1, 2), (1, 1), (0, 1)])


def test_example():
    data = [
        ".|...\\....",
        "|.-.\\.....",
        ".....|-...",
        "........|.",
        "..........",
        ".........\\",
        "..../.\\\\..",
        ".-.-/..|..",
        ".|....-|.\\",
        "..//.|....",
    ]
    coords = parser(data)
    assert len(coords) == 100
    path = ray_tracing(coords, (0, 0))
    logger.debug(path)
    assert len(set(path)) == 46


def test_part1():
    data = [
        ".|...\\....",
        "|.-.\\.....",
        ".....|-...",
        "........|.",
        "..........",
        ".........\\",
        "..../.\\\\..",
        ".-.-/..|..",
        ".|....-|.\\",
        "..//.|....",
    ]
    assert part1(data) == 46


def test_part2():
    data = [
        ".|...\\....",
        "|.-.\\.....",
        ".....|-...",
        "........|.",
        "..........",
        ".........\\",
        "..../.\\\\..",
        ".-.-/..|..",
        ".|....-|.\\",
        "..//.|....",
    ]
    assert part2(data) == 51
