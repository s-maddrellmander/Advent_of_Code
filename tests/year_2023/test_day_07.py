import pytest

from solutions.year_2023.day_07 import *


def test_compare_base14_numbers():
    assert compare_base14_numbers("0000000", "0000000") == 0
    assert compare_base14_numbers("0000000", "0000001") == -1
    assert compare_base14_numbers("0000001", "0000000") == 1
    assert compare_base14_numbers("00000C1", "0000002") == 1
    # some more complex tests
    assert compare_base14_numbers("034BC10", "04009B0") == -1


@pytest.mark.parametrize(
    "card, expected",
    [
        ("1", "1"),
        ("2", "2"),
        ("3", "3"),
        ("4", "4"),
        ("5", "5"),
        ("6", "6"),
        ("7", "7"),
        ("8", "8"),
        ("9", "9"),
        ("T", "A"),
        ("J", "B"),
        ("Q", "C"),
        ("K", "D"),
        ("A", "E"),
    ],
)
def test_convert_card_to_base14(card, expected):
    assert convert_card_to_base14(card) == expected


@pytest.mark.parametrize(
    "hand, expected",
    [
        ("AA123", "00000EE1230"),
        ("32T3K", "0000032A3D0"),
        ("KK677", "0000DD67700"),
        ("KTJJT", "0000DABBA00"),
        ("AAAAA", "EEEEE000000"),
    ],
)
def test_score_hand(hand, expected):
    assert score_hand(hand) == expected


def test_parse_input():
    input_data = ["32T3K 765", "T55J5 684", "KK677 28", "KTJJT 220", "QQQJA 483"]
    expected_hands = ["32T3K", "T55J5", "KK677", "KTJJT", "QQQJA"]
    expected_bids = [765, 684, 28, 220, 483]

    hands, bids = parse_input(input_data)

    assert hands == expected_hands
    assert bids == expected_bids


def test_sort_hands():
    input_data = ["32T3K 765", "T55J5 684", "KK677 28", "KTJJT 220", "QQQJA 483"]
    hands, bids = parse_input(input_data)
    logger.info(hands)
    hands = [score_hand(hand) for hand in hands]

    hands_sorted, bids_sorted = sort_hands(hands, bids)
    assert bids_sorted == (483, 684, 28, 220, 765)

    total_score = [
        bids_sorted[i] * (len(bids_sorted) - i) for i in range(0, len(bids_sorted))
    ]
    assert total_score[0] == 483 * 5
    assert total_score[1] == 684 * 4
    assert total_score[2] == 28 * 3
    assert total_score[3] == 220 * 2
    assert total_score[4] == 765 * 1


def test_part1():
    input_data = ["32T3K 765", "T55J5 684", "KK677 28", "KTJJT 220", "QQQJA 483"]
    assert part1(input_data) == 6440


def test_part1_alt_input():
    input_data = [
        "2345A 1",
        "Q2KJJ 13",
        "Q2Q2Q 19",
        "T3T3J 17",
        "T3Q33 11",
        "2345J 3",
        "J345A 2",
        "32T3K 5",
        "T55J5 29",
        "KK677 7",
        "KTJJT 34",
        "QQQJA 31",
        "JJJJJ 37",
        "AAAJA 43",
        "AAAAJ 59",
        "AAAAA 61",
        "2AAAA 23",
        "2JJJJ 53",
        "JJJJ2 41",
    ]
    assert part1(input_data) == 6592
    input_data = ["AK132 1", "AK123 2"]
    assert part1(input_data) == 4
    input_data = ["AK132 1", "AK123 2", "AQ123 3"]
    assert part1(input_data) == 3 + 4 + 3


def test_part2():
    input_data = ["32T3K 765", "T55J5 684", "KK677 28", "KTJJT 220", "QQQJA 483"]
    assert part2(input_data) == 5905


def test_part2_alt():
    input_data = [
        "2345A 1",
        "Q2KJJ 13",
        "Q2Q2Q 19",
        "T3T3J 17",
        "T3Q33 11",
        "2345J 3",
        "J345A 2",
        "32T3K 5",
        "T55J5 29",
        "KK677 7",
        "KTJJT 34",
        "QQQJA 31",
        "JJJJJ 37",
        "JAAAA 43",
        "AAAAJ 59",
        "AAAAA 61",
        "2AAAA 23",
        "2JJJJ 53",
        "JJJJ2 41",
    ]
    assert part2(input_data) == 6839


def test_specific_part2():
    # input_data = ["AAAAA 61", "AAAJA 43", "2JJJJ 53", "JJJJ2 41", "AAAAJ 59"]
    # input_data = ["AAAJA 43", "2JJJJ 53"]
    input_data = [
        "2345A 1",
        "Q2KJJ 13",
        "Q2Q2Q 19",
        "T3T3J 17",
        "T3Q33 11",
        "2345J 3",
        "J345A 2",
        "32T3K 5",
        "T55J5 29",
        "KK677 7",
        "KTJJT 34",
        "QQQJA 31",
        "JJJJJ 37",
        "JAAAA 43",
        "AAAAJ 59",
        "AAAAA 61",
        "2AAAA 23",
        "2JJJJ 53",
        "JJJJ2 41",
    ]
    hands, bids = parse_input(input_data)
    logger.info(f"hands: {hands}, bids: {bids}")
    hands = [score_hand(hand, 2) for hand in hands]
    logger.info(f"hands: {hands}, bids: {bids}")
    # assert len(set(hands)) == len(hands), f"{len(set(hands))}, {len(hands)}"
    hands_sorted, bids_sorted = sort_hands(hands, bids)
    logger.info(f"sorted_hands: {hands_sorted}, sorted_bids: {bids_sorted}")
    assert hands_sorted == (
        "EEEEE000000",
        "EEEE1000000",
        "31111000000",
        "1EEEE000000",
        "11113000000",
        "11111000000",
        "0DB11B00000",
        "0CCC1E00000",
        "0B661600000",
        "03EEEE00000",
        "00C3C3C0000",
        "00B4B410000",
        "000C3D11000",
        "000B4C44000",
        "0000DD78800",
        "0000043B4D0",
        "00000345610",
        "000001456E0",
        "0000003456E",
    )
    # logger.info(f"sorted_hands: {hands_sorted}")
    total_score = sum(
        [bids_sorted[i] * (len(bids_sorted) - i) for i in range(0, len(bids_sorted))]
    )
    logger.info(f"sorted_bids: {bids_sorted}")


def test_convert_jokers():
    categories = {
        (5,): 0,  # Five of a kind
        (4, 1): 1,  # Four of a kind
        (3, 2): 2,  # Full house
        (3, 1, 1): 3,  # Three of a kind
        (2, 2, 1): 4,  # Two pairs
        (2, 1, 1, 1): 5,  # One pair
        (1, 1, 1, 1, 1): 6,  # High card
    }
    hand = "T55JJ"
    hand = "".join([convert_card_to_base14(card, part=2) for card in hand])
    hand = Counter(hand)
    logger.info(hand)
    best_hand = convert_jokers(hand, categories)
    assert best_hand == ({"B": 1, "6": 4})
    hand = "T55JJ"
    score = score_hand(hand, part=2)
    assert score == "0B661100000"
