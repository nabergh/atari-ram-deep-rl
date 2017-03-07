import gym

env = gym.make('Breakout-ram-v0')
observation = env.reset()
print(observation)
print(env.action_space)
action = env.action_space.sample()
observation, reward, done, info = env.step(action)
print(observation)
print(reward)
print(done)
print(info)

for i_episode in range(5):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break