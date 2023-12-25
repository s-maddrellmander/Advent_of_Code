# solutions/year_2023/day_00.py
import itertools
import math
from copy import deepcopy
from itertools import combinations
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from tqdm import tqdm

from logger_config import logger
from utils import Timer


def plot_graph(adj_matrix, node_names):
    G = nx.Graph()
    # import ipdb; ipdb.set_trace()
    G.add_nodes_from(node_names)
    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix)):
            if adj_matrix[i][j] == 1:
                # import ipdb; ipdb.set_trace()
                G.add_edge(node_names[i], node_names[j])

    nx.draw(G, with_labels=True)
    # plt.show()
    plt.close()


def parse_map(input_data: List[str], real=False):
    # Parsing the input data
    # If it's real we cut these edges
    # zlv -> bmx
    # lrd -> qpg
    # tpb -> xsl
    start_end_cut_combs = [
        ("zlv", "bmx"),
        ("lrd", "qpg"),
        ("tpb", "xsl"),
        ("bmx", "zlv"),
        ("qpg", "lrd"),
        ("xsl", "tpb"),
    ]
    edges = [line.split(": ") for line in input_data]
    nodes = set()
    for edge in edges:
        start_node, end_nodes = edge

        nodes.add(start_node)
        for end_node in end_nodes.split():
            if real:
                # import ipdb; ipdb.set_trace()
                # if start_node == "zlv" and end_node == "bmx":
                #     import ipdb; ipdb.set_trace()
                if (start_node, end_node) in start_end_cut_combs:
                    continue
            nodes.add(end_node)

    # Create a list of all nodes
    node_list = list(nodes)

    # Initialize the adjacency matrix
    adj_matrix = [[0 for _ in node_list] for _ in node_list]
    node_to_index = {node: i for i, node in enumerate(node_list)}

    # Fill the adjacency matrix based on edges
    for src, dests in edges:
        src_index = node_to_index[src]
        # import ipdb; ipdb.set_trace()
        for dest in dests.split():
            if (src, dest) in start_end_cut_combs:
                continue
            dest_index = node_to_index[dest]
            adj_matrix[src_index][dest_index] = 1
            adj_matrix[dest_index][src_index] = 1

    return node_list, adj_matrix


def bfs_count_nodes(adj_matrix, node_names):
    num_nodes = len(adj_matrix)
    visited = [False] * num_nodes
    node_to_index = {node: index for index, node in enumerate(node_names)}

    def bfs(start_index):
        queue = [start_index]
        visited[start_index] = True
        count = 1

        while queue:
            node_index = queue.pop(0)
            for adj_node_index, is_connected in enumerate(adj_matrix[node_index]):
                if is_connected and not visited[adj_node_index]:
                    queue.append(adj_node_index)
                    visited[adj_node_index] = True
                    count += 1

        return count

    total_count = 0
    clusters = []
    for node, index in node_to_index.items():
        if not visited[index]:
            current_count = bfs(index)
            total_count += current_count
            clusters.append(current_count)

    return total_count, clusters


def cut_three_edges(adj_matrix, node_names):
    # Remove the three random edges (both directions)

    # Get all unique combinations of three pairs of nodes
    node_pairs = combinations(range(len(adj_matrix)), 2)
    all_combinations = combinations(node_pairs, 3)

    num_nodes = len(adj_matrix)
    num_pairs = math.comb(num_nodes, 2)
    num_combinations = math.comb(num_pairs, 3)
    num_edges = np.sum(adj_matrix)

    logger.debug(f"Number of combinations: {num_combinations}")

    # For each combination of three pairs, remove the corresponding edges
    for combination in tqdm(all_combinations, total=num_combinations):
        # Copy the adjacency matrix so we don't modify the original one
        adj_matrix_copy = deepcopy(adj_matrix)
        # logger.debug(f"Size of adj {np.sum(adj_matrix_copy)}")
        if np.sum(adj_matrix_copy) != num_edges:
            import ipdb

            ipdb.set_trace()
        # import ipdb; ipdb.set_trace()
        valid = True

        for pair in combination:
            if adj_matrix_copy[pair[0]][pair[1]] == 0:
                valid = False
        # if all([adj_matrix_copy[pair[0], pair[1]] == 1 for pair in combination]):
        #     logger.debug(f"Edge {pair} already removed")
        #     continue
        if valid:
            assert adj_matrix_copy[combination[0][0]][combination[0][1]] == 1
            assert adj_matrix_copy[combination[1][0]][combination[1][1]] == 1
            assert adj_matrix_copy[combination[2][0]][combination[2][1]] == 1
            for pair in combination:
                adj_matrix_copy[pair[0]][pair[1]] = 0
                adj_matrix_copy[pair[1]][pair[0]] = 0

            assert np.sum(adj_matrix_copy) == num_edges - 6

            # Now adj_matrix_copy is the adjacency matrix with the edges removed
            # Calculate the number of nodes using BFS
            counts, clusters = bfs_count_nodes(adj_matrix_copy, node_names)

            if len(clusters) == 2 and min(clusters) > 1:
                # import ipdb; ipdb.set_trace()
                logger.debug(f"Found two clusters: {clusters}")
                return adj_matrix_copy, clusters, combination

    return None, None, None


def part1(input_data: Optional[List[str]], real=True) -> Union[str, int]:
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
        verbose = False
        node_names, adj_matrix = parse_map(input_data, real=real)
        if verbose:
            plot_graph(adj_matrix, node_names)

        if real:
            count, clus = bfs_count_nodes(adj_matrix, node_names)
            assert len(clus) == 2
        else:
            adj, clus, combin = cut_three_edges(adj_matrix, node_names)
            assert len(clus) == 2

        return math.prod(clus)


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
        # Your solution for part 2 goes here
        return "Part 2 solution not implemented."
