import logging
from utils import select_day, Timer

from . import (day_1, day_2, day_3, day_4, day_5, day_6, day_7)

def day_selector(args):
    if select_day(args, 1):
        with Timer("Day 1"):
            day_1.control()
    if select_day(args, 2):
        with Timer("Day 2"):
            day_2.control()
    if select_day(args, 3):
        with Timer("Day 3"):
            day_3.control()
    if select_day(args, 4):
        with Timer("Day 4"):
            day_4.control()
    if select_day(args, 5):
        with Timer("Day 5"):
            day_5.control()
    if select_day(args, 6):
        with Timer("Day 6"):
            day_6.control()
    if select_day(args, 7):
        with Timer("Day 7"):
            day_7.control()
    # if select_day(args, 8):
    #     with Timer("Day 8"):
    #         day_8.control()