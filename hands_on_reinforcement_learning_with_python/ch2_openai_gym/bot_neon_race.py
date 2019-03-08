import gym
import universe

import random

# init neon race env
env = gym.make('flashgames.NeonRace-v0')
env.configure(remotes=1) # creates docker



# define moves
left = [
    ('KeyEvent', 'ArrowUp', True),
    ('KeyEvent', 'ArrowLeft', True),
    ('KeyEvent', 'ArrowRight', False)
]
right = [
    ('KeyEvent', 'ArrowUp', True),
    ('KeyEvent', 'ArrowLeft', False),
    ('KeyEvent', 'ArrowRight', True)
]
forward = [
    ('KeyEvent', 'ArrowUp', True),
    ('KeyEvent', 'ArrowLeft', False),
    ('KeyEvent', 'ArrowRight', False),
    ('KeyEvent', 'n', True)
]


def main(buffer=100):
    # inits
    observation_n = env.reset()
    turn = 0
    rewards = []
    action = forward

    while True:
        turn -= 1
        if turn <= 0:
            action = forward
            turn = 0

        action_n = [action for ob in observation_n] #assign action
        
        observation_n, reward_n, done_n, info = env.step(action_n) # take action & get next state + rewards
        rewards.append(reward_n[0])

        print('OBSERVATION:\t{}\nREWARDS:\t{}'.format(observation_n, rewards))

        if len(rewards) >= buffer:
            if (sum(rewards)*1.0 / len(rewards) == 0):
                turn = 20
                if random.random() < .5:
                    action = right
                else:
                    action = left
            
            rewards = []
        
        if done_n == True:
            break

        env.render()



if __name__ == "__main__":
    main()