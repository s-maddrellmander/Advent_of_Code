import pytest
from solutions.year_2024.day_23 import *

@pytest.fixture
def data():
    
    return ["kh-tc",
            "qp-kh",
            "de-cg",
            "ka-co",
            "yn-aq",
            "qp-ub",
            "cg-tb",
            "vc-aq",
            "tb-ka",
            "wh-tc",
            "yn-cg",
            "kh-ub",
            "ta-co",
            "de-co",
            "tc-td",
            "tb-wq",
            "wh-td",
            "ta-ka",
            "td-qp",
            "aq-cg",
            "wq-ub",
            "ub-vc",
            "de-ta",
            "wq-aq",
            "wq-vc",
            "wh-yn",
            "ka-de",
            "kh-ta",
            "co-tc",
            "wh-qp",
            "tb-vc",
            "td-yn",]


def test_parse_lan(data):
    lan_party = parse_lan(data)
    assert lan_party["kh"] == {'qp', 'ta', 'tc', 'ub'}

def test_find_interconnected_triplets(data):
    lan_party = parse_lan(data)
    triplets = find_interconnected_triplets(lan_party)
    assert len(triplets) == 7
    # assert {"kh", "ta", "tc"} in triplets
    # assert {"kh", "qp", "ub"} in triplets
    
def test_print_adjacency_matrix(data):
    lan_party = parse_lan(data)
    adjacency_matrix = print_adjacency_matrix(lan_party)
    # assert adjacency_matrix == [[0, 0, 1, 1]]
    