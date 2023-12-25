import math

import numpy as np
import pytest

from solutions.year_2023.day_25 import *

input_data = [
    "jqt: rhn xhk nvd",
    "rsh: frs pzl lsr",
    "xhk: hfx",
    "cmg: qnr nvd lhk bvb",
    "rhn: xhk bvb hfx",
    "bvb: xhk hfx",
    "pzl: lsr hfx nvd",
    "qnr: nvd",
    "ntq: jqt hfx bvb xhk",
    "nvd: lhk",
    "lsr: lhk",
    "rzs: qnr cmg lsr rsh",
    "frs: qnr lhk lsr",
]


def test_parse_data():
    node_names, adj_matrix = parse_map(input_data)
    assert len(set(node_names)) == 15
    assert set(node_names) == set(
        [
            "jqt",
            "rsh",
            "xhk",
            "cmg",
            "rhn",
            "bvb",
            "pzl",
            "qnr",
            "ntq",
            "nvd",
            "lsr",
            "rzs",
            "frs",
            "hfx",
            "lhk",
        ]
    )
    assert np.sum(adj_matrix) == 66


def test_bfs_disconnected():
    adjacency_matrix = [
        [0, 1, 0, 0, 0],
        [1, 0, 1, 1, 0],
        [0, 1, 0, 0, 0],
        [0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1],
    ]
    node_names = ["A", "B", "C", "D", "E"]
    counts, clusters = bfs_count_nodes(adjacency_matrix, node_names)
    assert counts == 5
    assert len(clusters) == 2


def test_bfs():
    # Define your graph's adjacency matrix and node names
    adjacency_matrix = [[0, 1, 0, 0], [1, 0, 1, 1], [0, 1, 0, 0], [0, 1, 0, 0]]
    node_names = ["A", "B", "C", "D"]

    # Calculate the number of nodes using BFS
    counts, clusters = bfs_count_nodes(adjacency_matrix, node_names)
    assert counts == 4
    assert len(clusters) == 1


@pytest.mark.skip("Slow")
def test_part1_step_by_step():
    node_names, adj_matrix = parse_map(input_data)
    total_count, clusters = bfs_count_nodes(adj_matrix, node_names)
    assert total_count == 15
    assert len(clusters) == 1
    assert np.sum(adj_matrix) == 66

    adj, clus, combin = cut_three_edges(adj_matrix, node_names)
    assert len(clus) == 2
    assert set(clus) == set([6, 9])
    assert np.sum(adj) == 60

    assert math.prod(clus) == 54


@pytest.mark.skip(reason="Visual test")
def test_plot_graph():
    node_names, adj_matrix = parse_map(input_data)
    plot_graph(adj_matrix, node_names)
