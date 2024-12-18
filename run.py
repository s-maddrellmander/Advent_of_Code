import importlib
import sys

from data_parser import load_file
from logger_config import logger

YEAR = "2024"


def run_solution(day, part, input_data):
    try:
        solution = getattr(day, f"part{part}")(input_data)
        logger.info(f"Solution for {day.__name__} part {part}: {solution}")
    except AttributeError as e:
        logger.warning(f"Solution function for {day.__name__} part {part} not found.")
        return


def main():
    if len(sys.argv) < 2 or len(sys.argv) > 3:
        logger.info("Usage: python run.py <day_number> [part_number]")
        return

    day = sys.argv[1]
    part = sys.argv[2] if len(sys.argv) == 3 else None

    # Ensure day is in the correct format
    day_module_name = f"day_{int(day):02d}"

    # Read the corresponding input file
    try:
        data = load_file(f"inputs/year_{YEAR}/{day_module_name}.txt")
    except FileNotFoundError:
        logger.warning(f"Input file for {day_module_name} not found.")
        return

    # Dynamically import the day module from the correct year package
    try:
        module = importlib.import_module(f"solutions.year_{YEAR}.{day_module_name}")
    except ModuleNotFoundError as e:
        logger.warning(f"Module for {day_module_name} not found.")
        return e

    if part:
        run_solution(module, part, data)
    else:
        # Run both parts
        for part in [1, 2]:
            run_solution(module, part, data)


if __name__ == "__main__":
    main()
