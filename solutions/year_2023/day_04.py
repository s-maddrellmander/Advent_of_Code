# solutions/year_2023/day_00.py
from typing import List, Optional, Union

from logger_config import logger
from utils import Timer


def parse_cards_to_dictionaries(data: List[str]) -> dict:
    cards = {}
    for line in data:
        if line.strip():
            parts = line.split(':')
            card_num = parts[0].strip()
            # reaplce card num with just the number
            card_num = int(card_num.replace('Card ', ''))
            numbers = parts[1].split('|')
            # logger.info(numbers)

            left_numbers = set(map(int, numbers[0].strip().split()))
            right_numbers = set(map(int, numbers[1].strip().split()))
            cards[card_num] = [left_numbers, right_numbers]
    return cards

def calculate_score(cards: set) -> int:
    if len(cards) == 0:
        return 0
    return 1 * 2**(len(cards)-1)

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

    # logger.info(input_data) 
    cards = parse_cards_to_dictionaries(input_data)
    total_score =  0
    with Timer("Part 1"):
        # Get the match
        for game in range(1, len(cards)+1):
            # Match which cards from the right are in the left
            intersection = cards[game][0].intersection(cards[game][1])
            game_score = calculate_score(intersection)
            total_score += game_score
            
    return total_score   
        
        
def parse_cards(card_lines):
    cards = {}
    for line in card_lines:
        parts = line.split(':')
        card_num = int(parts[0].split()[1])  # Convert card number to integer
        numbers = parts[1].split('|')
        left_numbers = set(map(int, numbers[0].strip().split()))
        right_numbers = set(map(int, numbers[1].strip().split()))
        cards[card_num] = [left_numbers, right_numbers]
    return cards

def count_matches(left_numbers, right_numbers):
    return len(left_numbers.intersection(right_numbers))

def process_cards(cards):
    total_cards = {card_num: 0 for card_num in cards}  # Initialize all counts to 0
    to_process = []  # A list to keep track of cards and their copies to process

    # Initial population of the list with original cards
    for card_num in sorted(cards.keys()):
        left_numbers, right_numbers = cards[card_num]
        matches = count_matches(left_numbers, right_numbers)
        logger.info(f"Card {card_num} has {matches} matches")
        
        # if matches > 0:
        #     total_cards[card_num] = 1  # Keep the original card
        #     to_process.append((card_num, matches))

    # Process each card and its copies
    # while to_process:
    #     card_num, matches = to_process.pop(0)
    #     for i in range(1, matches + 1):
    #         next_card = card_num + i
    #         if next_card in cards:
    #             if total_cards[next_card] == 0:
    #                 # This is to ensure that each card's winnings are only processed once
    #                 next_matches = count_matches(*cards[next_card])
    #                 if next_matches > 0:
    #                     to_process.append((next_card, next_matches))
    #             total_cards[next_card] += 1

    return total_cards




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
        parsed_cards = parse_cards(input_data)
        logger.info(parsed_cards)
        results = process_cards(parsed_cards)
        logger.info(results)
        logger.info(f"Total cards: {sum(results.values())   }")
        for card, count in sorted(results.items()):
            logger.info(f"Card {card} wins {count} more cards.")
            


        return -1