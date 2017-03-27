import gym
import numpy as np
from collections import namedtuple
from itertools import count
import random
import math

import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torch.nn.functional as F
from torch.autograd import Variable

import cma

from a3c_envs import create_atari_env

# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.FloatTensor

env = create_atari_env('Qbert-ram-v0')
state = env.reset()


class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(state_space, 32)
        self.l2 = nn.Linear(32, 16)
        self.l3 = nn.Linear(16, 8)
        self.l4 = nn.Linear(8, action_space.n)
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        x = F.relu(self.l3(x))
        return self.l4(x)



def np_to_state_dict(params):
    params = [solution[1:1+numpy.prod(shape)].reshape(shape) for shape in shapes]
    model_params = model.state_dict()
    for model_param, param in zip(model.state_dict(), params):
        model_params[model_param] = torch.from_numpy(param)
    return model_params

def state_dict_to_np():
    return np.concatenate([tensor.numpy().flatten() for _,tensor in model.state_dict().items()])



def min_function(params):
    model.load_state_dict(np_to_state_dict(params))
    
    state = torch.from_numpy(env.reset()).type(dtype)
    total_reward = 0
    
    for t in count():
        action = np.argmax(model(Variable(state.unsqueeze(0), volatile = True)))
        next_state, reward, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).type(dtype)
        
        state = next_state
        
        total_reward += reward
        if done:
            print('Total reward for episode of length ' + str(t) + ' is ' + str(total_reward))
            
            return -total_reward



model = DQN(state.shape[0], env.action_space)
model.type(dtype)
shapes = [params.shape for params in state_dict_to_np()]
print(len(state_dict_to_np()))
cma.fmin(min_function, state_dict_to_np(), 1.0, {"maxfevals": 1e4, "tolx": 0, "tolfun": 0, "tolfunhist": 0})


