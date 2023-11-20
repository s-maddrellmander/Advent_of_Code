from utils import load_file
import logging

import jax
import jax.numpy as jnp


def score(player, outcome):
    select = dict(rock=1, paper=2, sissors=3)
    game = {-1: 0, 0: 3, 1: 6}
    return select[player] + game[outcome]

def game_outcome(opponent, player):
    rules = dict(rock=dict(rock=0, paper=1, sissors=-1),
                 paper=dict(rock=-1, paper=0, sissors=1),
                 sissors=dict(rock=1, paper=-1, sissors=0)
                 )
    return rules[opponent][player]

def game_inputs(game, code):
    game = game.split(" ")
    opponent = code["opponent"][game[0]]
    player = code["player"][game[1]]
    return opponent, player

def choice_from_outcome(opponent, outcome):
    rules = dict(rock=dict(rock=0, paper=1, sissors=-1),
                 paper=dict(rock=-1, paper=0, sissors=1),
                 sissors=dict(rock=1, paper=-1, sissors=0)
                 )
    for move in rules[opponent].keys():
        if rules[opponent][move] == outcome:
            return move

def part_1(strategy):
    code = dict(opponent=dict(A="rock", B="paper", C="sissors"),
            player=dict(Y="paper", X="rock", Z="sissors"))
    total_score = 0
    for game in strategy:
        opponent, player = game_inputs(game, code)
        outcome = game_outcome(opponent, player)
        game_score = score(player, outcome)
        total_score += game_score
    logging.info(f"Part 1: {total_score}")
    return total_score

def part_2(strategy):
    code = dict(opponent=dict(A="rock", B="paper", C="sissors"),
            player=dict(Y=0, X=-1, Z=1))
    total_score = 0
    for game in strategy:
        opponent, player = game_inputs(game, code)
        player = choice_from_outcome(opponent, player) 
        outcome = game_outcome(opponent, player)
        game_score = score(player, outcome)
        total_score += game_score
    logging.info(f"Part 2: {total_score}")
    return total_score

def control():
    inputs = load_file("year_2022/data/data_2.txt")
    part_1(inputs)
    part_2(inputs)