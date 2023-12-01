import logging
import time
from collections import deque
from typing import List, Optional, Union

from logger_config import logger as LOGGER


def select_day(args, day):
    if args.day == 0:
        return True
    elif args.day == day:
        return True
    else:
        return False


def load_file(filename):
    with open(filename) as f:
        lines = [line.rstrip("\n") for line in f]
    return lines


class Timer:
    def __init__(self, name, logger=None, log_level=0):
        self.name = name
        self.logger = LOGGER
        self.log_level = True

    def _log(self, message):
        if self.logger:
            self.logger.info(message)
        else:
            print(message)

    def __enter__(self):
        self.start_time = time.perf_counter()
        self._log(f"⭐ {self.name}: Timer started.")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        elapsed_time = time.perf_counter() - self.start_time
        hours, rem = divmod(elapsed_time, 3600)
        minutes, seconds = divmod(rem, 60)
        # Formatting time to HH:MM:SS.mmm (milliseconds)
        elapsed_time_str = f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}.{int((seconds - int(seconds)) * 1000):03}"

        if exc_type is not None:
            self._log(f"⚠️ {self.name}: Timer stopped with an exception: {exc_val}")
        else:
            self._log(
                f"⏱️ Elapsed time for {self.name}: {elapsed_time_str} (HH:MM:SS.mmm)"
            )
        return False  # Do not suppress exceptions

    def get_elapsed_time(self):
        return time.perf_counter() - self.start_time


class Queue:
    def __init__(self) -> None:
        self.queue = deque()  # type: deque

    def build_queue(self, inputs: str):
        tmp = [self.enqueue(char) for char in inputs]

    def dequeue(self):
        return self.queue.popleft()

    def enqueue(self, x):
        self.queue.append(x)

    def __len__(self):
        return len(self.queue)

    def clear(self):
        self.queue.clear()


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
    __setattr__ = dict.__setitem__  # type: ignore
    __delattr__ = dict.__delitem__  # type: ignore


class TreeNode:
    def __init__(self, name, parent, size=0) -> None:
        self.name = name
        self.parent = parent
        self.sub_tree: List[TreeNode] = []
        self.leaves = []  # type: ignore
        self.size = size

    def add_sub_tree(self, sub_tree):
        self.sub_tree.append(sub_tree)

    def add_leaf(self, leaf):
        self.leaves.append(leaf)
