import logging
# Some issues around JAX imports need this here
logging.basicConfig(level="INFO")

from jsonargparse import ArgumentParser
from year_2017 import (day_7, day_8, day_9)
from year_2022 import(day_1, day_2, day_3)

from utils import select_day, Timer

def main(args):
    # TODO: Need to split these into seperate files
    if args.year == 2017:
        if select_day(args, 7):
            day_7.control()
        if select_day(args, 8):
            with Timer("Day 8"):
                day_8.control()
        if select_day(args, 9):
            with Timer("Day 9"):
                day_9.control()
    elif args.year == 2022:
        if select_day(args, 1):
            with Timer("Day 1"):
                day_1.control()
        if select_day(args, 2):
            with Timer("Day 2"):
                day_2.control()
        if select_day(args, 3):
            with Timer("Day 3"):
                day_3.control()

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--day", type=int, default=0, help="Select the day to run")
    parser.add_argument("--year", type=int, default=2022, help="Select the year to run")

    args = parser.parse_args()
    main(args)