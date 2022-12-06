import logging
from utils import select_day, Timer

from . import (day_7, day_8, day_8, day_9)

def day_selector(args):
    if select_day(args, 7):
        with Timer("Day 7"):
            day_7.control()
    if select_day(args, 8):
        with Timer("Day 8"):
            day_8.control()
    if select_day(args, 9):
        with Timer("Day 9"):
            day_9.control()