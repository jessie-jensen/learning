import gym
import gym_bandits

import numpy as np
import random

env = gym.make('BanditTenArmedGaussian-v0')
env.reset()



# params
ARMS = 10
EPISODES = 20*1000
EPSILON = .1

# inits
counts = np.zeros(ARMS)
rewards_sum = np.zeros(ARMS)
Q = np.zeros(ARMS)

##############################################

#
### EPSILON GREEDY ALGO
# 

def policy_epsilon_greedy(epsilon):
    if random.random() < epsilon:
        action = env.action_space.sample()
    else:
        action = np.argmax(Q)
    
    return action

##############################################


# run simulation

for i in range(EPISODES):
    # take step
    action = policy_epsilon_greedy(EPSILON)
    observation, reward, done, info = env.step(action)

    # updates
    counts[action] += 1
    rewards_sum[action] += reward
    Q[action] = rewards_sum[action] / counts[action]


print('''mean rewards: \n{}\n
counts: \n{}\n
optimal arm: {}
'''.format(Q, counts, np.argmax(Q)))