import gym

env = gym.make('BipedalWalker-v2')

for episode in range(10):
    observation = env.reset()

    for i in range(1000):
        env.render()
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)

        if done:
            print('\t\t',observation, reward, done, info)
            print("EPISODE {}\tTIMESTEPS = {}".format(episode+1, i+1))
            break