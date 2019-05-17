import gym

env = gym.make('Breakout-v0')
env.reset()

for _ in range(1000):
    env.render()
    
    action = env.action_space.sample()
    next_state, reward, done, _info = env.step(action)