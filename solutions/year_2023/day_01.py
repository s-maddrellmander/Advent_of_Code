# solutions/year_2023/day_01.py
import re
from typing import List, Tuple

from data_parser import load_file
from logger_config import logger
from utils import Timer

import numpy as np

"""
This problem takes the first and last didgets in the string and
combines them into a two digit number.

"""


def get_numbers_from_string(number: str) -> str:
    # String contains letters and numbers
    # Time complexity: O(n) because we iterate through the string
    number = "".join([i for i in number if i.isdigit()])
    return number


def get_first_and_last_digits(number: str) -> Tuple[str, str]:
    # String contains letters and numbers
    # Time complexity: O(n) because we iterate through the string
    number = get_numbers_from_string(number)
    # Get the first and last digits
    if len(number) == 0:
        return ("", "")
    first_digit = number[0]
    last_digit = number[-1]
    return first_digit, last_digit


def replace_spelled_with_numbers(number: str) -> str:
    spelled_numbers = {
        "zero": "0",
        "one": "1",
        "two": "2",
        "three": "3",
        "four": "4",
        "five": "5",
        "six": "6",
        "seven": "7",
        "eight": "8",
        "nine": "9",
    }

    result = ""
    i = 0
    while i < len(number):
        for spelled, digit in spelled_numbers.items():
            if number[i:].startswith(spelled):
                result += digit
                i += 1
                break
        else:
            if number[i].isdigit():
                result += number[i]
            i += 1

    return result


def part1(input_data: List[str]) -> int:
    # Your solution for part 1 goes here
    with Timer("Part 1"):
        result = 0
        for line in input_data:
            first_digit, last_digit = get_first_and_last_digits(line)
            if first_digit == "" and last_digit == "":
                continue
            result += int(first_digit + last_digit)
    return result


def part2(input_data: List[str]) -> int:
    # This time we are looking for spelled words as well as numbers
    with Timer("Part 2"):
        result = 0
        for line in input_data:
            line = replace_spelled_with_numbers(line)
            first_digit, last_digit = get_first_and_last_digits(line)
            if first_digit == "" or last_digit == "":
                continue
            # logger.debug(f"first_digit: {first_digit}, last_digit: {last_digit}, {first_digit + last_digit} {line}")
            result += int(first_digit + last_digit)
    return result
