import gym

import random
import time
import os



env = gym.make('Taxi-v2')
# Taxi-v2
# This task was introduced in [Dietterich2000] to illustrate some issues in hierarchical reinforcement learning. 
# There are 4 locations (labeled by different letters) and your job is to pick up the passenger at one location and drop him off in another. 
# You receive +20 points for a successful dropoff, and lose 1 point for every timestep it takes. 
# There is also a 10 point penalty for illegal pick-up and drop-off actions.
# Rendering:
# - blue: passenger
# - magenta: destination
# - yellow: empty taxi
# - green: full taxi
# - other letters (R, G, B and Y): locations for passengers and destinations
# actions:
# - 0: south
# - 1: north
# - 2: east
# - 3: west
# - 4: pickup
# - 5: dropoff


# params
LEARNING_RATE = .4
DISCOUNT_RATE = .999
EXPLORE_RATE  = .017



def init_q_table(env):
    q = {}
    for state in range(env.observation_space.n):
        for action in range(env.action_space.n):
            q[(state, action)] = 0
    
    return q



def policy_epsilon_greedy(q, env, state, explore_rate):
    if (random.uniform(0,1) < explore_rate):
        # explore
        print('***EXPLORE***')
        action = env.action_space.sample()
    else:
        # exploit greedy action
        print('---EXPLOIT---')
        action = max(list(range(env.action_space.n)), key= lambda a: q[(state, a)])
    
    return action



def sarsa_learner(env, learning_rate, discount_rate, explore_rate, episodes, episode_render_l=[]):
    q = init_q_table(env)

    episode_rewards_l = []
    episode_actions_l = []
    for i in range(episodes+1):
        # single episode

        # inits
        total_reward = 0
        done = False
        turns = 0

        state = env.reset()
        action = policy_epsilon_greedy(q, env, state, explore_rate)

        while done==False:
            if (i+1 in episode_render_l) and (turns < 30):
                os.system('cls' if os.name == 'nt' else 'clear')
                print('EPISODE: {}\nLAST EPISODE REWARDS: {}\nLAST EPISODE ACTIONS: {}'.format(i+1, episode_rewards_l[-1], episode_actions_l[-1]))
                env.render()
                time.sleep(.5)

            # take action & get obs + reward
            next_state, reward, done, _info = env.step(action)

            # get next state action from policy
            next_action = policy_epsilon_greedy(q, env, next_state, explore_rate)

            # update current state-action q value:
            # q(s,a) + a(r + g*q(s',a') - q(s,a))
            q[(state, action)] += learning_rate * (reward + (discount_rate * q[(next_state, next_action)]) - q[(state, action)])

            # prep for next state
            state = next_state
            action = next_action
            total_reward += reward
            turns += 1

        episode_rewards_l.append(total_reward)
        episode_actions_l.append(turns)
        print('EPISODE #{}, ACTIONS = {}, TOTAL REWARD = {}'.format(i+1, turns, total_reward))

    print(episode_rewards_l)


if __name__ == "__main__":
    sarsa_learner(env, LEARNING_RATE, DISCOUNT_RATE, EXPLORE_RATE, 10000, [10, 100, 1000, 10000])