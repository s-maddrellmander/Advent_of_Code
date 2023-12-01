import cmath
import logging

import numpy as np

from utils import load_file


def get_start_end(inputs):
    for i, line in enumerate(inputs):
        if "S" in line:
            start = complex(i, line.index("S"))
        if "E" in line:
            end = complex(i, line.index("E"))
    return start, end


def convert_alpha_to_numeric(inputs):
    # Parse the string per line to grid, then to number
    # Find the start and end locations
    start, end = get_start_end(inputs)
    grid = {}
    for i, line in enumerate(inputs):
        for j, x in enumerate(line):
            grid[complex(i, j)] = ord(x) - 97
    grid[complex(start)] = 0
    grid[complex(end)] = 26
    # Return the grid as a complex dict
    return grid, start, end


class Climber:
    def __init__(self, start, end, map) -> None:
        self.location = start
        self.destination = end
        self.map_store = map
        self.map = self._map

    def _map(self, coord):
        if coord in self.map_store.keys():
            return self.map_store[coord]
        return

    def get_options(self):
        def is_valid(a, b):
            if a == None:
                return False
            elif a - b <= 1:
                return True
            else:
                return False

        cardinals = [complex(0, 1), complex(0, -1), complex(1, 0), complex(-1, 0)]
        options = [
            self.location + direction
            for direction in cardinals
            if is_valid(self.map(self.location + direction), self.map(self.location))
        ]
        return options

    # Breadth First Search will work nicely here

    def bfs_shortest_path(self, start, goal):
        # finds shortest path between 2 nodes of a graph using BFS
        # keep track of explored nodes
        explored = []
        # keep track of all the paths to be checked
        queue = [[start]]

        # return path if start is goal
        if start == goal:
            return "That was easy! Start = goal"

        # keeps looping until all possible paths have been checked
        while queue:
            # pop the first path from the queue
            path = queue.pop(0)
            # get the last node from the path
            node = path[-1]
            self.location = node
            if node not in explored:
                # TODO: This needs changing
                neighbours = self.get_options()
                # neighbours = graph[node]
                # go through all neighbour nodes, construct a new path and
                # push it into the queue
                for neighbour in neighbours:
                    new_path = list(path)
                    new_path.append(neighbour)
                    queue.append(new_path)
                    # return path if neighbour is goal
                    if neighbour == goal:
                        return new_path

                # mark node as explored
                explored.append(node)

        # in case there's no path between the 2 nodes
        return False


# Shortest Path - this is a classic Dijkstra problem
# Prim's alogrithm in O(V^2)
# 1. Build the map into a graph
# 2. Then use a search algorithm
# 3. Complex number for x,y coordinate system
# 4. Build a dict for the map with complex keys
# 4.1. Then build the tree from this dict, each step recursively adds children
#      from the subset of the 4 directons that
#      (a) Haven't been visited
#      (b) Are valid steps


def part_1(inputs):
    parsed, start, end = convert_alpha_to_numeric(inputs)
    bfs = Climber(start, end, parsed)
    path = bfs.bfs_shortest_path(start, end)[:-1]
    logging.info(f"Part 1: {len(path)}")
    return len(path)


def part_2(inputs):
    parsed, start, end = convert_alpha_to_numeric(inputs)
    lowest = [coord for coord in parsed if parsed[coord] == 0]
    senic_paths_lengths = []
    for start in lowest:
        bfs = Climber(start, end, parsed)
        path = bfs.bfs_shortest_path(start, end)
        if path is not False:
            senic_paths_lengths.append(len(path[:-1]))
    shortest = min(senic_paths_lengths)
    logging.info(f"Part 2: {shortest}")
    return shortest


def control():
    inputs = load_file("year_2022/data/data_12.txt")
    part_1(inputs)
    part_2(inputs)
