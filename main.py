import logging
# Some issues around JAX imports need this here
logging.basicConfig(level="INFO")

from jsonargparse import ArgumentParser
import year_2022.day_selector
import year_2017.day_selector

from utils import select_day, Timer

def main(args):
    # TODO: Need to split these into seperate files
    if args.year == 2017:
        year_2017.day_selector.day_selector(args)
    elif args.year == 2022:
        year_2022.day_selector.day_selector(args)

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--day", type=int, default=0, help="Select the day to run")
    parser.add_argument("--year", type=int, default=2022, help="Select the year to run")

    args = parser.parse_args()
    main(args)