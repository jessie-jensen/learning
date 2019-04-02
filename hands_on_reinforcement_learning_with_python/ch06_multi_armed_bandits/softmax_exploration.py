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
TAU = .5

# inits
counts = np.zeros(ARMS)
rewards_sum = np.zeros(ARMS)
Q = np.zeros(ARMS)

##############################################

#
### SOFTMAX EXPLORATION
# 

#  Pt(a) = (exp(Qt(a)/T) / SUMn,i( exp(Qt(i)/T) )
# T (tau) == temperature factor param

def policy_softmax(tau):
    softmax_denominator = sum([math.exp(i / tau) for i in Q])
    softmax_probs = [(math.exp(i / tau) / softmax_denominator) for i in Q]

    thres = random.random()
    cumulative_prob = 0.0

    for i in range(len(softmax_probs)):
        cumulative_prob += softmax_probs[i]
        if (cumulative_prob > thres):
            return i

    return np.argmax(softmax_probs)

##############################################



# run simulation

for i in range(EPISODES):
    # take step
    action = policy_softmax(TAU)
    observation, reward, done, info = env.step(action)

    # updates
    counts[action] += 1
    rewards_sum[action] += reward
    Q[action] = rewards_sum[action] / counts[action]



print('''mean rewards: \n{}\n
counts: \n{}\n
optimal arm: {}
'''.format(Q, counts, np.argmax(Q)))