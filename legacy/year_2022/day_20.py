import logging
from collections import deque

from utils import load_file

"""
How does this work?
- Iterate through the numbers in the original order
    - Move the number in the list the same position +/- as the value
- Repeat. The tricks here are that the list is cirular, the mixing wraps round
- The order of the moves change each loop
"""


# Use a circulary linked list
class circular_linked_list:
    def __init__(self, base_inputs) -> None:
        self.queue = deque(base_inputs)
        self.original_order = base_inputs
        self.track_inds = [i for i in range(len(base_inputs))]
        # The elements in the queue are not unique, so we need to track two lists

    def insert(self, index, item):
        self.queue.insert(index, item)

    def mix_once(
        self,
    ):
        for index in self.track_inds:
            self.mixing(index)

    def mixing(self, index):
        # Track index is the location in the queue, given in order original
        # index = self.queue.index(value)
        # TODO Check this
        # new_index = index + value
        # Rotate the list so the element can be popped
        # import ipdb; ipdb.set_trace()
        # index is the index of the original, we keep track of where it is in the queue with track_index
        rot = self.track_inds[index]
        self.queue.rotate(-rot)
        item = self.queue.popleft()
        assert item == self.original_order[index], (
            item,
            self.original_order[index],
            index,
            self.track_inds,
        )
        # Then insert the item at the the index given by the value
        new = (index + item) % (len(self.queue))
        self.queue.rotate(rot)

        self.queue.insert(new, item)
        # This should update the track inds + queue
        self.track_inds.insert(new, self.track_inds.pop(index))
        import ipdb

        ipdb.set_trace()
        assert self.queue[new] == self.original_order[self.track_inds[new]]

        # TODO: This whole list needs updating though...

        # Roate the queue back into the correct order - this is not needed for
        # The problem but will make testing easier.
        # if new == 0 and self.original_order[i] != 0:
        #     # Add an extra one to account for coming back from the end to the start
        #     self.queue.rotate(-1)

    def get_index(self, index):
        # Return the value "index" steps after 0
        # Get the zero index and zero out the queue
        zero_index = self.queue.index(0)
        # self.queue.rotate(-zero_index)
        # This should just make the location a bit easier
        index = (index + zero_index) % len(self.queue)
        return self.queue[index]


def part_1(inputs):
    queue = circular_linked_list(inputs)
    queue.mix_once()
    sum_coords = sum([queue.get_index(index) for index in [1000, 2000, 3000]])
    logging.info(f"Part 1: {sum_coords}")
    return sum_coords


def part_2(inputs):
    pass


def control():
    inputs = load_file("year_2022/data/data_20.txt")
    inputs = [int(x) for x in inputs]
    part_1(inputs)
    part_2(inputs)
