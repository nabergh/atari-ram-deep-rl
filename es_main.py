import argparse
import pickle
from random import randint

import torch
import torch.multiprocessing as _mp
mp = _mp.get_context('spawn')

import torch.nn as nn
import torch.nn.functional as F
import gym
from es_train import train
from es_test import test

# Training settings
parser = argparse.ArgumentParser(description='A3C')
parser.add_argument('--lr', type=float, default=0.0001, metavar='LR',
                    help='learning rate (default: 0.0001)')
parser.add_argument('--sigma', type=float, default=1.00, metavar='S',
                    help='standard deviation for gaussian noise (default: 1.00)')
parser.add_argument('--seed', type=int, default=-1, metavar='S',
                    help='random seed (default: random int)')
parser.add_argument('--num-processes', type=int, default=4, metavar='N',
                    help='how many training processes to use (default: 4)')
parser.add_argument('--max-episode-length', type=int, default=10000, metavar='M',
                    help='maximum length of an episode (default: 10000)')
parser.add_argument('--env-name', default='Breakout-ram-v0', metavar='ENV',
                    help='environment to train on (default: Breakout-ram-v0)')
parser.add_argument('--save-name', default='es_model', metavar='FN',
                    help='path/prefix for the filename to save shared model\'s parameters')
parser.add_argument('--load-name', default=None, metavar='SN',
                    help='path/prefix for the filename to load shared model\'s parameters')
parser.add_argument('--monitor', action="store_true",
                    help='whether to monitor testing (note: not for evalutation)')
parser.add_argument('--evaluate', action="store_true",
                    help='whether to evaluate trained model')
parser.add_argument('--evaluate-episodes', type=int, default=100, metavar='EE',
                    help='number of episodes to evaluate')
parser.add_argument('--upload', action="store_true",
                    help='whether to upload results for trained model (--evaluate must be set)')

if __name__ == '__main__':
    args = parser.parse_args()

    if args.seed < 0:
        args.seed = randint(0, 9999999999)

    dtype = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.FloatTensor

    processes = []

    reward_queues = [mp.Queue(args.num_processes - 1) for i in range(args.num_processes)]

    if not args.evaluate:
        for rank in range(0, args.num_processes):
            p = mp.Process(target=train, args=(rank, args, reward_queues, dtype))
            p.start()
            processes.append(p)
    else:
        p = mp.Process(target=test, args=(0, args, dtype))
        p.start()
        processes.append(p)

    for p in processes:
        p.join()
