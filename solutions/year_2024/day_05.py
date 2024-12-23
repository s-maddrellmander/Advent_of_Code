# solutions/year_2024/day_05.py
from typing import Dict, List, Optional, Tuple, Union

from logger_config import logger
from utils import Timer


def parse_rule(rules: List[str]) -> Dict[int, list[int]]:
    rule_dict = {}
    reverse_rules_dict = {}
    for rule in rules:
        rule = rule.split("|")
        if int(rule[0]) not in rule_dict:
            rule_dict[int(rule[0])] = [int(rule[1])]
        else:
            rule_dict[int(rule[0])].append(int(rule[1]))
        # Add the reverse rule
        if int(rule[1]) not in reverse_rules_dict:
            reverse_rules_dict[int(rule[1])] = [int(rule[0])]
        else:
            reverse_rules_dict[int(rule[1])].append(int(rule[0]))
    return rule_dict, reverse_rules_dict


def parse_pages(pages: List[str]) -> List[List[int]]:
    collection = []
    for page in pages:
        page = [int(p) for p in page.split(",")]
        collection.append(page)
    return collection


def check_page(page, rules):
    for i in range(len(page) - 1):
        for j in range(i + 1, len(page)):
            if page[i] not in rules:
                # print(f"Page {page} is not valid - {page[i]} is not in {rules}")
                return False
            if page[j] not in rules[page[i]]:
                # print(f"Page {page} is not valid - {page[j]} is not in {rules[page[i]]}")
                return False
    return True


def fix_page(page, rules, reverse_rules_dict):
    # Need to take the elements and adjust the ordering based on the rules
    # This is basically building a sort with a set of non sequential numeric rules I guess?

    while check_page(page, rules) == False:
        for i in range(len(page) - 1):
            for j in range(i + 1, len(page)):
                # print(i, j, page, page[j], rules[page[i]])
                if page[i] not in rules:
                    page[i], page[j] = page[j], page[i]
                elif page[j] not in rules[page[i]]:
                    # print(f"Page {i}: {page[i]}, {rules[page[i]]}")
                    # Swap the elements
                    page[i], page[j] = page[j], page[i]
    return page


def part1(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 1 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 1"):
        split_idx = input_data.index("")
        rules = input_data[:split_idx]
        pages = input_data[split_idx + 1 :]
        rules = parse_rule(rules)[0]
        pages = parse_pages(pages)
        valid_pages = 0
        score = 0
        for page in pages:
            if check_page(page, rules):
                valid_pages += 1
                mid_page = len(page) // 2
                score += page[mid_page]
        return score


def part2(input_data: Optional[List[str]]) -> Union[str, int]:
    """
    Solve part 2 of the day's challenge.

    Args:
        input_data (List[str]): The puzzle input as a list of strings.

    Returns:
        Union[str, int]: The solution to the puzzle.
    """
    if not input_data:
        raise ValueError("Input data is None or empty")

    with Timer("Part 2"):
        split_idx = input_data.index("")
        rules = input_data[:split_idx]
        pages = input_data[split_idx + 1 :]
        rules, reverse_rules_dict = parse_rule(rules)
        pages = parse_pages(pages)
        valid_pages = 0
        score = 0
        for page in pages:
            if not check_page(page, rules):
                page = fix_page(page, rules, reverse_rules_dict)
                valid_pages += 1
                mid_page = len(page) // 2
                score += page[mid_page]
        return score
