import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
from torch.autograd import Variable


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        m.bias.data.fill_(0)
        m.weight.data.fill_(0)

class EvolutionNet(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(EvolutionNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 64)
        self.linear2 = nn.Linear(64, 32)
        self.linear3 = nn.Linear(32, 32)
        self.linear4 = nn.Linear(32, 16)
        self.linear5 = nn.Linear(16, 8)
        self.linear6 = nn.Linear(8, action_space.n)
        self.apply(weights_init)
        
        self.shapes = [tensor.numpy().shape for _,tensor in self.state_dict().items()]

    def forward(self, inputs):
        x = F.tanh(self.linear1(inputs))
        x = F.tanh(self.linear2(x))
        x = F.tanh(self.linear3(x))
        x = F.tanh(self.linear4(x))
        x = F.tanh(self.linear5(x))
        return self.linear6(x)

    def set_weights_np(self, params):
        param_list = []
        for shape in self.shapes:
            param_list.append(params[:np.prod(shape)].reshape(shape))
            params = params[np.prod(shape):]
        model_params = self.state_dict()
        for model_param, param in zip(self.state_dict(), param_list):
            model_params[model_param] = torch.from_numpy(param)
        self.load_state_dict(model_params)
        

    def get_weights_np(self):
        return np.concatenate([tensor.cpu().numpy().flatten() for _,tensor in self.state_dict().items()])