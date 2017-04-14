import cma
import gym
import numpy as np
from itertools import count
from datetime import date
import time
import pickle
import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
import torch.nn.init as init
from torch.autograd import Variable

from a3c_envs import create_atari_env

from tensorboard_logger import configure, log_value

# dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor
dtype = torch.FloatTensor

env = create_atari_env('Asteroids-ram-v0')
state = env.reset()

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.xavier_uniform(m.weight.data)
#         m.bias.data.fill_(0)

class DQN(nn.Module):
    def __init__(self, state_space, action_space):
        super(DQN, self).__init__()
        self.l1 = nn.Linear(state_space, 16)
        self.l2 = nn.Linear(16, 8)
        self.l3 = nn.Linear(8, action_space.n)
        # self.apply(weights_init)
        self.train()
    
    def forward(self, x):
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        # x = F.relu(self.l3(x))
        return self.l3(x)

class ActorCritic(torch.nn.Module):
    def __init__(self, num_inputs, action_space):
        super(ActorCritic, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 32)
        self.lstm = nn.LSTMCell(32, 32)
        self.actor_linear = nn.Linear(32, action_space.n)

        self.train()

    def forward(self, inputs):
        inputs, (hx, cx) = inputs
        x = F.elu(self.linear1(inputs))
        hx, cx = self.lstm(x, (hx, cx))
        x = hx
        return self.actor_linear(x), (hx, cx)



def np_to_state_dict(params):
    param_list = []
    for shape in shapes:
        param_list.append(params[:np.prod(shape)].reshape(shape))
        params = params[np.prod(shape):]
    model_params = model.state_dict()
    for model_param, param in zip(model.state_dict(), param_list):
        model_params[model_param] = torch.from_numpy(param)
    return model_params

def state_dict_to_np():
    return np.concatenate([tensor.numpy().flatten() for _,tensor in model.state_dict().items()])


def min_function(params):
    model.load_state_dict(np_to_state_dict(params))
    cx = Variable(torch.zeros(1, 32).type(dtype), volatile = True)
    hx = Variable(torch.zeros(1, 32).type(dtype), volatile = True)
    state = torch.from_numpy(env.reset()).type(dtype)
    total_reward = 0
    
    for t in count():
        action_probs = model(Variable(state.unsqueeze(0), volatile = True))
        action = np.argmax(action_probs.data.cpu().numpy())
        next_state, reward, done, _ = env.step(action)
        next_state = torch.from_numpy(next_state).type(dtype)
        
        state = next_state
        
        total_reward += reward
        if done:
            print('Total reward for episode of length ' + str(t) + ' is ' + str(total_reward))
            min_function.ctr += 1
            log_value('Reward', total_reward, min_function.ctr)
            log_value('Episode length', t, min_function.ctr)
            if min_function.ctr % 50 == 0:
                pickle.dump(model.state_dict(), open('asteroids_cma' + '.p', 'wb'))
            return -total_reward
min_function.ctr = 0

model = DQN(state.shape[0], env.action_space)
model.type(dtype)
shapes = [tensor.numpy().shape for _,tensor in model.state_dict().items()]

timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
run_name = 'asteroids_cma' + '_' + timestring
configure("logs/" + run_name, flush_secs=5)
cma.fmin(min_function, state_dict_to_np(), 1.0, {"maxfevals": 1e4, "tolx": 0, "tolfun": 0, "tolfunhist": 0, 'CMA_diagonal': True})
