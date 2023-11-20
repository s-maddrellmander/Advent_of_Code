import logging
from utils import select_day, Timer

from . import (day_1, day_2, day_3, day_4, day_5, day_6, day_7, day_8, day_9,
               day_10, day_11, day_12, day_13, day_14, day_15, day_16, day_17,
               day_18, day_20, day_22)

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
    if select_day(args, 8):
        with Timer("Day 8"):
            day_8.control()
    if select_day(args, 9):
        with Timer("Day 9"):
            day_9.control()
    if select_day(args, 10):
        with Timer("Day 10"):
            day_10.control()
    if select_day(args, 11):
        with Timer("Day 11"):
            day_11.control()
    if select_day(args, 12):
        with Timer("Day 12"):
            day_12.control()
    if select_day(args, 13):
        with Timer("Day 13"):
            day_13.control()
    if select_day(args, 14):
        with Timer("Day 14"):
            day_14.control()
    if select_day(args, 15):
        with Timer("Day 15"):
            day_15.control()
    if select_day(args, 16):
        with Timer("Day 16"):
            day_16.control()
    if select_day(args, 17):
        with Timer("Day 17"):
            day_17.control()
    if select_day(args, 18):
        with Timer("Day 18"):
            day_18.control()
    if select_day(args, 20):
        with Timer("Day 20"):
            day_20.control()
    if select_day(args, 22):
        with Timer("Day 22"):
            day_22.control()