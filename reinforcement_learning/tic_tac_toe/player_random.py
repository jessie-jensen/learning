import random

class PlayerRand():

    def __init__(self, x_or_o):
        self.x_or_o = x_or_o

    def make_move(self, available_moves, state):
        move = random.choice(available_moves)
        state[move[1]][move[0]] = self.x_or_o
        return state



if __name__=='__main__':
    state = [['X', 'O', None],['O', 'O', None],['X','O','X']]
    moves = [[2, 0], [2, 1]]
    
    p = PlayerRand('O')
    print(p.make_move(moves, state))