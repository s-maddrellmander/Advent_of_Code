# solutions/year_2023/day_07.py
from collections import Counter
from typing import List, Optional, Union

from logger_config import logger
from utils import Timer


def parse_input(input_data: List[str]):
    # Parse the input data
    hands = []
    bids = []
    for line in input_data:
        hand, bid = line.split(" ")
        hands.append(hand)
        bids.append(int(bid))

    return hands, bids


def convert_card_to_base14(card, part: int = 1):
    # Takes each card in the hand and return the base 14 score
    if part == 1:
        mapping = {
            "1": "1",
            "2": "2",
            "3": "3",
            "4": "4",
            "5": "5",
            "6": "6",
            "7": "7",
            "8": "8",
            "9": "9",
            "T": "A",
            "J": "B",
            "Q": "C",
            "K": "D",
            "A": "E",
        }
    else:
        mapping = {
            "J": "1",
            "1": "2",
            "2": "3",
            "3": "4",
            "4": "5",
            "5": "6",
            "6": "7",
            "7": "8",
            "8": "9",
            "9": "A",
            "T": "B",
            "Q": "C",
            "K": "D",
            "A": "E",
        }

    return mapping[card]


def compare_base14_numbers(num1, num2):
    # Zero-pad the numbers to ensure correct comparison
    num1 = num1.zfill(7)
    num2 = num2.zfill(7)

    # Compare the numbers as strings
    if num1 > num2:
        return 1
    elif num1 < num2:
        return -1
    else:
        return 0


from itertools import chain, combinations_with_replacement


# Function to evaluate the score of a hand
def evaluate_score(hand, categories):
    # Determine the category
    counts = tuple(sorted(hand.values(), reverse=True))
    # remove any 0s from the tuple
    counts = tuple(filter(lambda x: x != 0, counts))
    category = categories[counts]
    return category


# Function to find the best score with jokers
def best_score_with_jokers(hand, num_jokers, categories):
    original_score = evaluate_score(hand, categories=categories)
    best_score = original_score
    best_hand = hand

    # Generate all possible hands with jokers
    # import ipdb; ipdb.set_trace()

    for j in range(1, num_jokers + 1):
        for jokers_combination in combinations_with_replacement("123456789ABCDE", j):
            # logger.info(f"jokers_combination: {jokers_combination}")

            new_hand = dict(hand)
            if "1" in new_hand:
                new_hand["1"] -= len(jokers_combination)
            for x in jokers_combination:
                if x in new_hand:
                    new_hand[x] += 1
                else:
                    new_hand[x] = 1

            new_score = evaluate_score(new_hand, categories)
            # logger.info(f"new_hand: {new_hand}, new_score: {new_score}")

            if new_score < best_score:
                best_score = new_score
                best_hand = new_hand
                # import ipdb; ipdb.set_trace()
    # best_hand = tuple(filter(lambda x: x != 0, best_hand.values()))
    if "1" in best_hand:
        if best_hand["1"] == 0:
            best_hand.pop("1")
    return best_hand, best_score


def convert_jokers(hand_counter: Counter, categories: dict) -> Counter:
    # Convert the jokers
    num_jokers = hand_counter["1"]
    # Find the best score
    best_hand, best_score = best_score_with_jokers(hand_counter, num_jokers, categories)
    # logger.info(f"best_hand: {best_hand}, best_score: {best_score}")
    # import ipdb; ipdb.set_trace()
    return best_hand


def score_hand(hand, part: int = 1):
    # Define the categories - return the index to update
    categories = {
        (5,): 0,  # Five of a kind
        (4, 1): 1,  # Four of a kind
        (3, 2): 2,  # Full house
        (3, 1, 1): 3,  # Three of a kind
        (2, 2, 1): 4,  # Two pairs
        (2, 1, 1, 1): 5,  # One pair
        (1, 1, 1, 1, 1): 6,  # High card
    }

    # Convert hand to base 14 and count
    hand = "".join([convert_card_to_base14(card, part=part) for card in hand])
    hand_counter = Counter(hand)
    if part == 2:
        # Convert the jokers
        hand_counter = convert_jokers(hand_counter, categories)

    # import ipdb; ipdb.set_trace()
    # Determine the category
    counts = tuple(sorted(hand_counter.values(), reverse=True))
    category = categories[counts]

    # Create the score
    hex_size = 11
    score = ["0"] * hex_size
    index_to_reaplce = categories[counts]
    score[index_to_reaplce : min(index_to_reaplce + 5, hex_size)] = hand[
        0 : hex_size - index_to_reaplce
    ]
    score = "".join(score)
    return score


def sort_hands(hands: List[str], bids: List[int]):
    # Simple zipped sorting based on the score hand function
    with Timer("Sorting costs"):
        hands, bids = zip(*sorted(zip(hands, bids), key=lambda x: x[0], reverse=True))
    return hands, bids


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
        hands, bids = parse_input(input_data)
        hands = [
            score_hand(
                hand,
            )
            for hand in hands
        ]
        # assert len(set(hands)) == len(hands), f"{len(set(hands))}, {len(hands)}"
        hands_sorted, bids_sorted = sort_hands(hands, bids)
        # logger.info(f"sorted_hands: {hands_sorted}")
        total_score = sum(
            [
                bids_sorted[i] * (len(bids_sorted) - i)
                for i in range(0, len(bids_sorted))
            ]
        )
        # logger.info(f"sorted_bids: {bids_sorted}")
    return total_score


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
        hands, bids = parse_input(input_data)
        hands = [score_hand(hand, 2) for hand in hands]
        # assert len(set(hands)) == len(hands), f"{len(set(hands))}, {len(hands)}"
        hands_sorted, bids_sorted = sort_hands(hands, bids)
        # logger.info(f"sorted_hands: {hands_sorted}")
        total_score = sum(
            [
                bids_sorted[i] * (len(bids_sorted) - i)
                for i in range(0, len(bids_sorted))
            ]
        )
        # logger.info(f"sorted_bids: {bids_sorted}")
    return total_score
