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

alpha = np.ones(ARMS)
beta = np.ones(ARMS)



##############################################

#
### thompson sampling
# 

# randomly sample probs from a distribution

def policy_thompson_sampling():
    thompson_samples = [np.random.beta(alpha[i] + 1, beta[i] + 1) for i in range(ARMS)]
    action = np.argmax(thompson_samples)

    return action
        

##############################################


# run simulation

for i in range(EPISODES):
    # take step
    action = policy_thompson_sampling()
    observation, reward, done, info = env.step(action)

    # updates
    counts[action] += 1
    rewards_sum[action] += reward
    Q[action] = rewards_sum[action] / counts[action]

    #
    ### THOMPSON BETA DISTRIBUTION UPDATES (note forcing to binary in this case)
    #
    if reward > 0:
        alpha[action] += 1
    else:
        beta[action] += 1


print('''mean rewards: \n{}\n
counts: \n{}\n
optimal arm: {}
'''.format(Q, [int(i) for i in counts], np.argmax(Q)))