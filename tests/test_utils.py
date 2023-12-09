from unittest.mock import Mock

import pytest

from utils import load_file, select_day


def test_select_day():
    # Test when args.day is 0
    args = Mock(day=0)
    assert select_day(args, 1) == True

    # Test when args.day is equal to day
    args = Mock(day=1)
    assert select_day(args, 1) == True

    # Test when args.day is not equal to day and not 0
    args = Mock(day=2)
    assert select_day(args, 1) == False


def test_load_file(tmp_path):
    # Create a temporary file with some content
    file_path = tmp_path / "test.txt"
    file_path.write_text("line1\nline2\nline3\n")

    # Test load_file function
    lines = load_file(file_path)
    assert lines == ["line1", "line2", "line3"]
