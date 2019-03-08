import gym

import numpy as np

import time
import os


env = gym.make('FrozenLake-v0')



def compute_value_function(env, policy, gamma=1.0, threshold=1e-10):
    value_table = np.zeros(env.observation_space.n)

    while True:
        updated_value_table = np.copy(value_table)
        for state in range(env.observation_space.n):
            action = policy[state]

            next_state_rewards = []
            for next_state_properties in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _done = next_state_properties
                next_state_reward = transition_prob * (reward_prob + gamma * value_table[next_state])
                next_state_rewards.append(next_state_reward)
            value_table[state] = np.sum(next_state_rewards)

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
            
    return value_table



def extract_policy(value_table, gamma=1.0):
    # init policy & q to zeros
    policy = np.zeros(env.observation_space.n)
    q_table = np.zeros([env.observation_space.n, env.action_space.n])

    for state in range(env.observation_space.n):

        for action in range(env.action_space.n):
            for next_state_properties in env.env.P[state][action]:
                transition_prob, next_state, reward_prob, _done = next_state_properties

                # Q(s,a) = transition_prob * (reward_prob + (gamma * next_state_value))
                next_state_reward = transition_prob * (reward_prob + gamma * value_table[next_state])
                q_table[state][action] += next_state_reward

        policy[state] = np.argmax(q_table[state])

    print('CURRENT policy is:\n', policy)
    return policy, q_table



def policy_iteration(env, gamma=1.0, iterations=100000):
    policy = np.zeros(env.observation_space.n)

    for i in range(iterations):
        new_value_func = compute_value_function(env, policy, gamma=gamma)
        new_policy, q_table = extract_policy(new_value_func, gamma=gamma)

        if (np.all(policy == new_policy)):
            print('Policy iteration converged at step #{}'.format(i+1))
            break
        
        policy = new_policy
    
    print('OPTIMAL policy is:\n', policy)
    return policy, q_table



def main(env, gamma=1.0, speed=2, clear=False):
    optimal_policy, q_table = policy_iteration(env, gamma=gamma)
    print(q_table)

    # # render
    # observation = env.reset()
    # done = False
    # total_rewards = 0
    # print(observation)

    # for _ in range(1000):
    #     state = observation
    #     # print('TOTAL REWARD: ', total_rewards)
    #     env.render()
    #     if done:
    #         break
        
    #     # make optimal step acording to policy
    #     optimal_action = int(optimal_policy[state])
    #     observation, reward, done, info = env.step(optimal_action)
    #     total_rewards += reward
    #     print('\n\n', observation, reward, done, info)


    #     time.sleep(1.0/speed)
        
    #     if clear:
    #         os.system('cls' if os.name == 'nt' else 'clear')



if __name__ == "__main__":
    main(env)