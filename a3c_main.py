import argparse

import torch
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

import torch.nn as nn
import torch.nn.functional as F
import gym
from a3c_model import ActorCritic
from a3c_train import train
from a3c_test import test
from a3c_envs import create_atari_env

# Based on
# https://github.com/pytorch/examples/tree/master/mnist_hogwild
# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--gamma', type=float, default=0.99, metavar='G',
                    help='discount factor for rewards (default: 0.99)')
parser.add_argument('--tau', type=float, default=1.00, metavar='T',
                    help='parameter for GAE (default: 1.00)')
parser.add_argument('--seed', type=int, default=1, metavar='S',
                    help='random seed (default: 1)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--num-steps', type=int, default=30, metavar='NS',
                    help='number of forward steps in A3C (default: 20)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='Breakout-ram-v0', metavar='ENV',
                    help='environment to train on (default: PongDeterministic-v3)')
parser.add_argument('--fname', default='a3c_model', metavar='FN',
                    help='path/prefix for the filename of the shared model\'s parameters')


if __name__ == '__main__':
    args = parser.parse_args()

    torch.manual_seed(args.seed)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    env = create_atari_env(args.env_name)
    state = env.reset()
    shared_model = ActorCritic(state.shape[0], env.action_space).type(dtype)
    shared_model.share_memory()

    # train(1,args,shared_model,dtype)
    processes = []

    p = mp.Process(target=test, args=(args.num_processes, args, shared_model, dtype))
    p.start()
    processes.append(p)

    for rank in range(0, args.num_processes):
        p = mp.Process(target=train, args=(rank, args, shared_model, dtype))
        p.start()
        processes.append(p)
    for p in processes:
        p.join()
