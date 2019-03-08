import gym

import numpy as np

import time
import os


env = gym.make('FrozenLake-v0')



def value_iteration(env, iterations=100000, threshold=1e-20, gamma=1.0, verbose=False):
    '''returns optimal value table'''

    # init value to 0's
    value_table = np.zeros(env.observation_space.n)

    for i in range(iterations):
        updated_value_table = np.copy(value_table)

        for state in range(env.observation_space.n):
            # calc list of q values, for given state
            q_values = []
            
            for action in range(env.action_space.n):
                next_state_rewards = []

                for next_state_properties in env.env.P[state][action]:
                    transition_prob, next_state, reward_prob, _done = next_state_properties

                    # Q(s,a) = transition_prob * (reward_prob + (gamma * next_state_value))
                    next_state_reward = transition_prob * (reward_prob + gamma * updated_value_table[next_state])
                    next_state_rewards.append(next_state_reward)

                q_values.append(np.sum(next_state_rewards))
            
            value_table[state] = max(q_values)

        if verbose:
            print(value_table)
        
        if (np.sum(np.fabs(updated_value_table - value_table)) <= threshold):
            print('Value iteration converged at iteration #{}'.format(i+1))
            break

    print('Optimal value function is:\n', value_table)
    return value_table



def extract_policy(env, value_table, gamma=1.0):
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

    print('Optimal policy is:\n', policy)
    return policy, q_table



def main(env, gamma=1.0, speed=2, clear=False):
    optimal_value_table = value_iteration(env, gamma=gamma)
    optimal_policy, q_table = extract_policy(env, optimal_value_table, gamma=gamma)
    print(q_table)

    # render
    observation = env.reset()
    done = False
    total_rewards = 0
    print(observation)

    for _ in range(1000):
        state = observation
        # print('TOTAL REWARD: ', total_rewards)
        env.render()
        if done:
            break
        
        # make optimal step acording to policy
        optimal_action = int(optimal_policy[state])
        observation, reward, done, info = env.step(optimal_action)
        total_rewards += reward
        print('\n\n', observation, reward, done, info)


        time.sleep(1.0/speed)
        
        if clear:
            os.system('cls' if os.name == 'nt' else 'clear')



if __name__ == "__main__":
    main(env)