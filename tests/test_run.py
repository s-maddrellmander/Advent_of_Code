import importlib
import sys
from unittest.mock import patch

import pytest

from data_parser import load_file
from run import main, run_solution


@pytest.mark.skip(reason="Not implemented yet.")
def test_main_with_specific_part(capsys, monkeypatch):
    monkeypatch.setattr(sys, "argv", ["run.py", "1", "1"])
    main()
    captured = capsys.readouterr()
    assert "⭐ Part 1: Timer started." in captured.out
    assert "⭐ Part 2: Timer started." not in captured.out


@pytest.mark.skip(reason="Not implemented yet.")
def test_run_solution_with_part_1(capsys):
    module = importlib.import_module("solutions.year_2023.day_01")
    data = load_file("inputs/year_2023/day_01.txt")
    run_solution(module, 1, data)
    captured = capsys.readouterr()
    assert "⭐ Part 1: Timer started." in captured.out


@pytest.mark.skip(reason="Not implemented yet.")
def test_run_solution_with_part_2(capsys):
    module = importlib.import_module("solutions.year_2023.day_01")
    data = load_file("inputs/year_2023/day_01.txt")
    run_solution(module, 2, data)
    captured = capsys.readouterr()
    assert "⭐ Part 2: Timer started." in captured.out
