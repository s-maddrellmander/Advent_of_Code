import logging

import numpy as np

from utils import Cache, Queue, TreeNode, load_file


class SumTreeNode(TreeNode):
    def tree_size(self):
        total = 0
        for sub in self.sub_tree:
            total += sub.tree_size()
        for leaf in self.leaves:
            total += leaf.size
        return total


def build_tree(queue) -> SumTreeNode:
    root_node = queue.dequeue()[-1]
    # Start with the ROOT node
    tree = SumTreeNode(root_node, "ROOT")
    current_node = tree
    while len(queue) > 0:
        line = queue.dequeue()
        if line[0] == "$":
            # Command line instruction
            if "cd" in line:
                name = line.split(" ")[2]
                if name == "..":
                    # Change the current node, where we are adding sub nodes
                    # Moves up ones level in the tree
                    current_node = current_node.parent
                else:
                    # The cd command goes inside a new dir (size = 0 default)
                    # Make the new node here
                    new_node = SumTreeNode(name, parent=current_node)
                    # Add it to the current node sub_tree
                    current_node.add_sub_tree(new_node)
                    # Then `move` into the new sub_tree
                    current_node = new_node
        else:
            # This is wherewe add the actual files with the data
            size, obj_name = line.split(" ")
            # This might return a directory which we don't want
            if size.isnumeric():
                # Then we add as a leaf - if we need the directories we can
                # Add them with size = 0, but they should be covered above,
                # This would only be for empty ones we don't look in
                current_node.add_leaf(SumTreeNode(obj_name, current_node, int(size)))
    return tree


def build_inst_queue(inst):
    queue = Queue()
    queue.build_queue(inst)
    queue.cache = []
    return queue


def parse_command(line):
    assert line[0] == "$"
    line = line.split(" ")
    if line[1] == "cd":  # Change directory
        return line[2]
    if line[1] == "ls":  # This is listing the directories
        return


def check_for_small_dirs(current_node, small_directories, target_size=100000):
    for sub in current_node.sub_tree:
        if sub.tree_size() < target_size:
            small_directories.append(sub)

        small_directories = check_for_small_dirs(sub, small_directories)
    return small_directories


def check_to_del(current_node, to_del, target_size=100000):
    for sub in current_node.sub_tree:
        if sub.tree_size() >= target_size:
            to_del.append(sub)
        to_del = check_to_del(sub, to_del)
    return to_del


def part_1(inputs):
    queue = Queue()
    queue.build_queue(inputs)

    tree = build_tree(queue)
    small_directories = []
    small_directories = check_for_small_dirs(
        tree, small_directories, target_size=100000
    )
    sizes = [small.tree_size() for small in small_directories]
    total = sum(sizes)

    logging.info(f"Part 1 {total}")
    return total


def part_2(inputs):
    """Find the smallest directory that, if deleted, would free up enough space
    on the filesystem to run the update. What is the total size of that directory?"""
    queue = Queue()
    queue.build_queue(inputs)
    tree = build_tree(queue)
    current_total_space = tree.tree_size()
    TOTALSPACE = 70000000
    SPACEREQUIRED = 30000000
    USEDSPACE = current_total_space
    SPACENEEDEDTOBECLREAED = SPACEREQUIRED - (TOTALSPACE - USEDSPACE)
    all_dir_sizes = []
    all_dir_sizes = check_to_del(
        tree, all_dir_sizes, target_size=SPACENEEDEDTOBECLREAED
    )
    all_dir_sizes = [x.tree_size() for x in all_dir_sizes]
    valid_dirs = [x for x in all_dir_sizes if x > SPACENEEDEDTOBECLREAED]
    # import ipdb; ipdb.set_trace()
    minner = min(valid_dirs)
    logging.info(f"Part 2: {minner}")
    return minner


def control():
    inputs = load_file("year_2022/data/data_7.txt")
    part_1(inputs)
    part_2(inputs)
