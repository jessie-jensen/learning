import time
import random

import make_env
import player_random

pX = player_random.PlayerRand('X')
pO = player_random.PlayerRand('O')

def reset_state():
    return [[None, None, None],[None, None, None],[None,None,None]]





env = make_env.MakeEnv()
player_turn = random.choice(['X','O'])

while env.game_state == 'IN PROGRESS':
    time.sleep(.25)

    if player_turn == 'X':
        new_state = pX.make_move(env.available_moves, env.state)
        player_turn = 'O'
    elif player_turn == 'O':
        new_state = pO.make_move(env.available_moves, env.state)
        player_turn = 'X'

    env = make_env.MakeEnv(new_state)