import numpy as np

import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

# def weights_init(m):
#     classname = m.__class__.__name__
#     if classname.find('Linear') != -1:
#         init.xavier_uniform(m.weight.data)
#         m.bias.data.fill_(0)


class EvolutionNet(torch.nn.Module):

    def __init__(self, num_inputs, action_space):
        super(EvolutionNet, self).__init__()
        self.linear1 = nn.Linear(num_inputs, 256)
        self.linear2 = nn.Linear(256, 256)
        self.linear3 = nn.Linear(256, 256)
        self.linear4 = nn.Linear(256, action_space.n)
        # self.apply(weights_init)

        self.shapes = [tensor.numpy().shape for _,tensor in self.state_dict().items()]

    def forward(self, inputs):
        x = F.leaky_relu(self.linear1(inputs))
        x = F.leaky_relu(self.linear2(x))
        x = F.leaky_relu(self.linear3(x))

        return self.linear4(x)

    def set_weights_np(self, params):
        params = [params[1:1+np.prod(shape)].reshape(shape) for shape in self.shapes]
        model_params = self.state_dict()
        for model_param, param in zip(self.state_dict(), params):
            model_params[model_param] = torch.from_numpy(param)
        self.load_state_dict(model_params)
        

    def get_weights_np(self):
        return np.concatenate([tensor.cpu().numpy().flatten() for _,tensor in self.state_dict().items()])