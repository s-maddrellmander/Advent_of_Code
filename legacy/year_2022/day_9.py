import logging

import numpy as np

from utils import load_file

"""
Day 9 Problem Description:
    - Tracking the head and tail nodes locations as they traverse around a grid
    - Head (H) moves first, then each move it makes Tail (T) must follow as
      well such that after both moves the T is in an adjacent location to H
    - This is probably best tracked using dicts I think, or class objects 
"""


class Node:
    def __init__(self, current) -> None:
        self.current = current
        self.history = {}  # key = (x, y), value = num times visited
        self.next_node = None
        self.prev_node = None
        self.update_history()

    def update_history(self):
        if self.current in self.history.keys():
            self.history[self.current] += 1
        else:
            self.history[self.current] = 1

    def step(self, kernel):
        # TODO: Needs tweaking for direction
        # This does along the key axes
        x = kernel[0]
        y = kernel[1]
        # These are going to be either or
        if x != 0 and y == 0:
            for _ in range(abs(x)):
                if x > 0:
                    self.move((1, 0))
                else:
                    self.move((-1, 0))
                # if self.next_node is not None:
                #     self.next_node.follow()
        if x == 0 and y != 0:
            for _ in range(abs(y)):
                if y > 0:
                    self.move((0, 1))
                else:
                    self.move((0, -1))

                # if self.next_node is not None:
                #     self.next_node.follow()
        if x != 0 and y != 0:
            # import ipdb; ipdb.set_trace()
            # Handle the diagonals
            pass

    def follow(self):
        diff = np.array(self.prev_node.current) - np.array(
            self.current
        )  # Gives the 2-tuple
        diff = list(diff)
        if diff[0] > 1 or diff[1] > 1:
            if diff[0] > 1:
                diff[0] = diff[0] - 1
            if diff[1] > 1:
                diff[1] = diff[1] - 1
            self.step(diff)
        if diff[0] < 1 or diff[1] < 1:
            if diff[0] < 1:
                diff[0] = diff[0] + 1
            if diff[1] < 1:
                diff[1] = diff[1] + 1
            self.step(diff)
        # import ipdb; ipdb.set_trace()
        # pass
        # If the tail can't move on the cardinal axes it always moves diagonally

    def move(self, kernel: tuple):
        # Move the current node using an (x, y) kernel
        self.current = tuple([self.current[0] + kernel[0], self.current[1] + kernel[1]])
        self.update_history()


def part_1(inputs):
    pass


def part_2(inputs):
    pass


def control():
    inputs = load_file("year_2022/data/data_9.txt")
    part_1(inputs)
    part_2(inputs)
