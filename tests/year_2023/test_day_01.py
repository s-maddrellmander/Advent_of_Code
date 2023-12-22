from solutions.year_2023.day_01 import *


def test_get_first_and_last_digits():
    assert get_first_and_last_digits("123abc456") == ("1", "6")
    assert get_first_and_last_digits("a1b2c3") == ("1", "3")
    assert get_first_and_last_digits("abc") == ("", "")
    assert get_first_and_last_digits("123") == ("1", "3")
    assert get_first_and_last_digits("") == ("", "")


def test_part1():
    assert part1(["1abc2", ""]) == 12
    assert part1(["1abc2", "1abc2"]) == 24
    assert part1(["1abc2", "pqr3stu8vwx", "a1b2c3d4e5f", "treb7uchet"]) == 142


def test_basic_func():
    number = "abc123def456"
    number = get_numbers_from_string(number)
    assert number == "123456"


def test_replace_spelled_with_numbers():
    assert replace_spelled_with_numbers("one") == "1"
    assert replace_spelled_with_numbers("two") == "2"
    assert replace_spelled_with_numbers("three") == "3"
    assert replace_spelled_with_numbers("four") == "4"
    assert replace_spelled_with_numbers("five") == "5"
    assert replace_spelled_with_numbers("six") == "6"
    assert replace_spelled_with_numbers("seven") == "7"
    assert replace_spelled_with_numbers("eight") == "8"
    assert replace_spelled_with_numbers("nine") == "9"
    assert replace_spelled_with_numbers("one1two2three3") == "112233"
    assert replace_spelled_with_numbers("abc") == ""


def test_part2():
    test_data = [
        "two1nine",
        "eightwothree",
        "abcone2threexyz",
        "xtwone3four",
        "4nineeightseven2",
        "zoneight234",
        "7pqrstsixteen",
    ]
    result = part2(test_data)
    assert result == 281


def test_replace_spelled_with_numbers_complex_single():
    assert replace_spelled_with_numbers("two1nine") == "219"
    assert replace_spelled_with_numbers("fiveight") == "58"
    assert replace_spelled_with_numbers("eightwothree") == "823"


def test_replace_spelled_with_numbers_complex():
    test_data = ["two1nine", "eightwothree", "fiveight"]

    expected_results = ["219", "823", "58"]

    for i, test_string in enumerate(test_data):
        assert replace_spelled_with_numbers(test_string) == expected_results[i]


def test_part2_debug():
    data = load_file(f"tests/day_01_debug_data.txt")
    expected_results = [
        (5, 6),
        (9, 6),
        (4, 4),
        (2, 2),
        (8, 8),
        (9, 3),
        (2, 3),
        (7, 9),
        (8, 3),
        (5, 6),
    ]
    result = 0
    for line, exp in zip(data[: len(expected_results)], expected_results):
        line = replace_spelled_with_numbers(line)
        first_digit, last_digit = get_first_and_last_digits(line)
        assert first_digit == str(exp[0])
        assert last_digit == str(exp[1])
        old_result = result
        result += int(first_digit + last_digit)
        assert result == old_result + int(str(exp[0]) + str(exp[1]))
        # logger.debug(f"{result}")
