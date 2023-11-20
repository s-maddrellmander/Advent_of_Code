import logging
from utils import load_file, Queue, Cache

def part_1(queue):
    cache = Cache(max_length=4)
    i = 1
    while len(queue) > 0:
        cache.enqueue(queue.dequeue())
        if len(cache.queue) == 4:
            if len(set(cache.queue)) == 4:
                logging.info(f"Part 1 {i}")
                return i
        i += 1
        
def part_2(queue):
    cache = Cache(max_length=14)
    i = 1
    while len(queue) > 0:
        cache.enqueue(queue.dequeue())
        if len(cache.queue) == 14:
            if len(set(cache.queue)) == 14:
                logging.info(f"Part 1 {i}")
                return i
        i += 1


def control():
    inputs = load_file("year_2022/data/data_6.txt")[0]
    queue = Queue()
    queue.build_queue(inputs)
    part_1(queue)
    queue = Queue()
    queue.build_queue(inputs)
    part_2(queue)