import tkinter as tk

class MakeEnv:

    def __init__(self, state=[[None, None, None],[None, None, None],[None,None,None]]):
        self.state = state
        self.winner = None
        self.root = tk.Tk()

        self.c1 = 'orange'
        self.c2 = 'grey'
        self.c3 = 'red'
        self.fnt = 'Comic Sans MS'

        self.canvas = tk.Canvas(self.root, width=300, height=300, bg=self.c2)
        self.canvas.pack()

        self.draw_borders()
        self.draw_positions()
        self.check_game_state()


    def draw_borders(self):
        self.canvas.create_line(0,100,300,100, fill=self.c1)
        self.canvas.create_line(0,200,300,200, fill=self.c1)
        self.canvas.create_line(100,0,100,300, fill=self.c1)
        self.canvas.create_line(200,0,200,300, fill=self.c1)

    def draw_xo(self, x, y):
        self.canvas.create_text((100*x)+50, (100*y)+50, text=self.state[y][x], font=(self.fnt, 60, 'bold'), fill=self.c1)


    def draw_positions(self):
        for x in range(3):
            for y in range(3):
                if self.state[y][x] != None:
                    self.draw_xo(x, y)

    
    def draw_win_line(self, direction, x_or_y=None):
        if direction=='horizontal':
            self.canvas.create_line(0, (100*x_or_y)+50, 300, (100*x_or_y)+50, fill=self.c3, width=10)
        elif direction=='vertical':
            self.canvas.create_line((100*x_or_y)+50, 0, (100*x_or_y)+50, 300, fill=self.c3, width=10)
        elif direction=='diagonal_1':
            self.canvas.create_line(0, 0, 300, 300, fill=self.c3, width=10)
        elif direction=='diagonal_2':
            self.canvas.create_line(300,0,0,300, fill=self.c3, width=10)


    def check_for_winner(self):
        for i in range(3):
            #check horizontal
            if (self.state[i][0] == self.state[i][1] == self.state[i][2]) and (self.state[i][0] != None):
                self.winner = self.state[i][0]
                self.draw_win_line('horizontal', i)

            #check vertical
            if (self.state[0][i] == self.state[1][i] == self.state[2][i]) and (self.state[0][i] != None):
                self.winner = self.state[0][i]
                self.draw_win_line('vertical', i)

        #check diagonals
        if (self.state[0][0] == self.state[1][1] == self.state[2][2]) and (self.state[1][1] != None):
            self.winner = self.state[1][1]
            self.draw_win_line('diagonal_1')
        
        if (self.state[0][2] == self.state[1][1] == self.state[2][0]) and (self.state[1][1] != None):
            self.winner = self.state[1][1]
            self.draw_win_line('diagonal_2')


    def check_game_state(self):
        self.check_for_winner()
        self.get_aviailable_moves()
        self.draw_and_set_game_state()




    def get_aviailable_moves(self):
        self.available_moves = []
        for x in range(3):
            for y in range(3):
                if self.state[y][x] == None:
                    self.available_moves.append([x,y])
        
        return self.available_moves


    def draw_and_set_game_state(self):
        if self.available_moves == [] and self.winner == None:
            self.game_state = 'DRAW'
            print('cats game')
            self.canvas.create_text(150,150, text='DRAW', font=(self.fnt, 90, 'bold'), fill=self.c3)
        elif self.winner in ['X','O']:
            self.game_state = self.winner
            print(self.winner, 'wins!')
        else:
            self.game_state='IN PROGRESS'

        self.root.mainloop()



if __name__=='__main__':
    state = [['X', 'O', 'O'],['O', 'O', 'X'],['X','O','O']]
    env = MakeEnv(state)