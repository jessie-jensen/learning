import gym
import gym_bandits

import numpy as np
import random
import math

env = gym.make('BanditTenArmedGaussian-v0')
env.reset()



# params
ARMS = 10
EPISODES = 20*1000

# inits
counts = np.zeros(ARMS)
rewards_sum = np.zeros(ARMS)
Q = np.zeros(ARMS)

##############################################

#
### UCB1 Algo (Upper Confidence Bound)
# 

# UCB1 policy = argmax [ Q(a) + sqrt( 2log(t) / N(a) ) ]

def policy_ucb1(episode):
    ucb = np.zeros(10)

    # init all arms with 1 obs
    if episode < ARMS:
        return episode
    # return max upper confidence bound value
    else:
        for arm in range(ARMS):
            upper_bound = math.sqrt( (2 * math.log(episode)) / counts[arm] )
            ucb[arm] = Q[arm] + upper_bound
        
        return np.argmax(ucb)
        

##############################################


# run simulation

for i in range(EPISODES):
    # take step
    action = policy_ucb1(i)
    observation, reward, done, info = env.step(action)

    # updates
    counts[action] += 1
    rewards_sum[action] += reward
    Q[action] = rewards_sum[action] / counts[action]


print('''mean rewards: \n{}\n
counts: \n{}\n
optimal arm: {}
'''.format(Q, [int(i) for i in counts], np.argmax(Q)))