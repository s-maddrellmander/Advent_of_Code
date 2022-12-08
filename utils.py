import time
import logging
from collections import deque


def select_day(args, day):
    if args.day == 0:
        return True
    elif args.day == day:
        return True
    else:
        return False

def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip('\n') for line in f]
    return lines


class Timer:
    def __init__(self, name):
        self.name = name

    def __enter__(self):
        logging.info(f"\U00002B50 {self.name}:")
        self.t = time.time()

    def __exit__(self, *args, **kwargs):
        elapsed_time = time.time() - self.t
        elapsed_time = time.strftime("%H:%M:%S", time.gmtime(elapsed_time))
        # Timedelta makes for human readable format - not safe for maths operations
        logging.info(f"\U0001F551 Elapsed time for {self.name}: {elapsed_time} (HH:MM:SS)")


class Queue():
    def __init__(self) -> None:
        self.queue = deque()
    
    def build_queue(self, inputs: str) -> deque:
        tmp = [self.enqueue(char) for char in inputs]

    def dequeue(self):
        return self.queue.popleft()
    
    def enqueue(self, x):
        self.queue.append(x)
    
    def __len__(self):
        return len(self.queue)

class Cache(Queue):
    """Cache with max length 
    This class will return the elements automatically if the length exceeds max
    """
    def __init__(self, max_length) -> None:
        super().__init__()
        self.max_length = max_length

    def enforce_max_length(self):
        if len(self.queue) > self.max_length:
            return self.dequeue()
        else:
            return None
    
    def enqueue(self, x):
        super().enqueue(x)
        return self.enforce_max_length()
    
    def dequeue(self):
        return super().dequeue()


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


class TreeNode:
    def __init__(self, name, parent, size=0) -> None:
        self.name = name
        self.parent = parent
        self.sub_tree = []
        self.leaves = []
        self.size = size
    
    def add_sub_tree(self, sub_tree):
        self.sub_tree.append(sub_tree)
    
    def add_leaf(self, leaf):
        self.leaves.append(leaf)