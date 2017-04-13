import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

class EvolutionNet(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(EvolutionNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 16)
        self.linear2 = nn.Linear(16, 8)
        self.linear3 = nn.Linear(8, action_space.n)

        self.shapes = [tensor.numpy().shape for _,tensor in self.state_dict().items()]

    def forward(self, inputs):
        x = F.tanh(self.linear1(inputs))
        x = F.tanh(self.linear2(x))

        return self.linear3(x)

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