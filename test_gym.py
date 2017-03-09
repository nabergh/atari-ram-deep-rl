import gym
from itertools import count

env = gym.make('Breakout-ram-v0')
observation = env.reset()
# print(observation)
# print(env.action_space)
# action = env.action_space.sample()
# observation, reward, done, info = env.step(action)
# print(observation)
# print(reward)
# print(done)
# print(info)

for i_episode in range(5):
    observation = env.reset()
    total_reward = 0
    for t in count():
        env.render()
        # print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        total_reward += reward
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            print("Reward " + str(total_reward))
            break