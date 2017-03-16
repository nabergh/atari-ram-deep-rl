# Inspired by https://github.com/ikostrikov/pytorch-a3c/blob/master/envs.py

import numpy as np
import gym
from universe import vectorized
from universe.wrappers import Unvectorize, Vectorize

def create_atari_env(env_name):
	env = gym.make(env_name)
	env = Vectorize(env)
	env = Bitwise(env)
	env = Unvectorize(env)
	return env

# a not so pretty way to turn bytes into bits
def to_bits(byte_list):
    l = []
    for b in byte_list:
        for i in reversed(range(8)):
            l.append(b >> i & 1)
    return np.array(l)

# Wrapper to convert our bytes into bits
class Bitwise(vectorized.ObservationWrapper):
	def __init__(self, env = None):
		super(Bitwise, self).__init__(env)
	
	def _observation(self, observations):
		return [to_bits(observation) for observation in observations]
