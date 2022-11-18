import logging
from jsonargparse import ArgumentParser

from year_2017 import (day_7)
from utils import select_day

def main(args):
    if select_day(args, 7):
        day_7.control()

if __name__ == "__main__":
    logging.basicConfig(level="INFO")
    parser = ArgumentParser()
    parser.add_argument("--day", type=int, default=0, help="Select the day to run")

    args = parser.parse_args()

    main(args)