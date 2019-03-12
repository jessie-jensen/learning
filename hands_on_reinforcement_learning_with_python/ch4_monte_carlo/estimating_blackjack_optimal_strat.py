import gym

import numpy as np
import functools
from collections import defaultdict

import matplotlib.pyplot as plt 
from mpl_toolkits.mplot3d import Axes3D
plt.style.use('ggplot')

# make env
env = gym.make('Blackjack-v0')



def example_policy(observation):
    score, dealer_score, usable_ace = observation
    # simple policy to hit on < 20
    action = 0 if score >= 20 else 1
    return action


def generate_episode(policy, env, verbose=False):
    states = []
    actions = [] 
    rewards = []

    done = False
    observation = env.reset()
    if verbose:
        print('\n', observation) 

    while done==False:
        states.append(observation)
        # get action from policy
        action = policy(observation)
        actions.append(action)
        # take step
        observation, reward, done, info = env.step(action)
        rewards.append(reward)

        if verbose:
            print(action, ' - ', observation, reward, done, info)

    return states, actions, rewards


def first_visit_mc_prediction(policy, env, n_episodes, verbose=False):
    value_table = defaultdict(float)
    N = defaultdict(int)

    for _ in range(n_episodes):
        states, _actions, rewards = generate_episode(policy, env, verbose=verbose)
        total_reward = 0
        for t in range(len(states)-1, -1, -1):
            # save reward & state for each step
            R = rewards[t]
            S = states[t]
            total_reward += R

            if S not in states[:t]:
                N[S] += 1
                value_table[S] += ((total_reward - value_table[S]) / N[S])

    return value_table



print(first_visit_mc_prediction(example_policy, env, 10, verbose=True))

######


def plot_blackjack(V, ax1, ax2):
    player_sum = np.arange(12, 21+1)
    dealer_show = np.arange(1, 10+1)
    usable_ace = np.array([False, True])

    state_values = np.zeros((len(player_sum),
                                len(dealer_show),
                                len(usable_ace)))

    for i, player in enumerate(player_sum):
        for j, dealer in enumerate(dealer_show):
            for k, ace in enumerate(usable_ace):
                state_values[i, j, k] = V[player, dealer, ace]

    X, Y = np.meshgrid(player_sum, dealer_show)

    ax1.plot_wireframe(X, Y, state_values[:, :, 0])
    ax2.plot_wireframe(X, Y, state_values[:, :, 1])

    for ax in ax1, ax2:
        ax.set_zlim(-1, 1)
        ax.set_ylabel('player sum')
        ax.set_xlabel('dealer showing')
        ax.set_zlabel('state-value')



value_table = first_visit_mc_prediction(example_policy, env, n_episodes=500*1000)

fig, axes = plt.subplots(nrows=2, figsize=(5, 8), subplot_kw ={'projection': '3d'})
axes[0].set_title('value function without usable ace')
axes[1].set_title('value function with usable ace')

plot_blackjack(value_table, axes[0], axes[1])
plt.show()