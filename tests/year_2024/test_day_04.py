from solutions.year_2024.day_04 import *
import pytest


@pytest.fixture
def small_data():
    dat = [
            "..X...",
            ".SAMX.",
            ".A..A.",
            "XMAS.S",
            ".X....",
           ]
    return dat

def test_find_x(small_data):
    coords = find_x_coord(small_data)
    assert len(coords) == 4


def test_full_map(small_data):
    data = full_map(small_data)

    assert data[complex(0, 0)] == "."
    assert data[complex(0, 2)] == "X"


def test_find_xmas(small_data):
    data = full_map(small_data)
    coords = find_x_coord(small_data)

    score = 0
    for coord in coords:
        score += find_xmas(coord, data)
        print(score)

    assert score == 4


def test_crossed_mas():
    data = [
        ".M.S......",
        "..A..MSMS.",
        ".M.S.MAA..",
        "..A.ASMSM.",
        ".M.S.M....",
        "..........",
        "S.S.S.S.S.",
        ".A.A.A.A..",
        "M.M.M.M.M.",
        "..........",
    ]
    _data = full_map(data)
    coords = find_x_coord(data, "A")
    
    assert len(coords) == 9
    assert complex(1, 2) in coords
    
    score = 0
    for coord in coords:
        score += find_crossed_mas(coord,_data)
    assert score == 9
