import vizdoom

import random
import time


# params
EPISODES = 25
# '/Users/jj/anaconda3/lib/python3.6/site-packages/vizdoom/scenarios/basic.cfg'

doom_game = vizdoom.DoomGame()
doom_game.load_config('/Users/jj/anaconda3/lib/python3.6/site-packages/vizdoom/scenarios/basic.cfg')
doom_game.init()



# encode actions
shoot = [0,0,1]
left = [1,0,0]
right = [0,1,0]
actions = [shoot, left, right]

# render & random actions
for i in range(EPISODES):
    doom_game.new_episode()

    while not doom_game.is_episode_finished():
        state = doom_game.get_state()

        img = state.screen_buffer
        misc = state.game_variables

        # take action
        action = random.choice(actions)
        reward = doom_game.make_action(action)

        print('reward:', reward)
        time.sleep(.02)
    
    print('*** FINISHED EPISODE {} ***'.format(i+1))
    time.sleep(3)