import time
import numpy as np
from datetime import date
from itertools import count
import pickle

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from es_model import EvolutionNet
from a3c_envs import create_atari_env

from tensorboard_logger import configure, log_value

def train(rank, args, reward_queues, dtype):

    if rank is 0:
        timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
        run_name = args.save_name + '_' + timestring
        configure("logs/es_" + run_name, flush_secs=5)

    curr_seed = args.seed
    
    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    state = env.reset()

    model = EvolutionNet(state.shape[0], env.action_space).type(dtype)
    pert_model = EvolutionNet(state.shape[0], env.action_space).type(dtype)
    
    for step in count():
        torch.manual_seed(curr_seed + rank)

        # create model with weights perturbed by N(0, 1) * args.sigma
        params = model.get_weights_np()
        perturbs = torch.normal(torch.zeros(len(params)), torch.FloatTensor([args.sigma] * len(params))).numpy()
        pert_model.set_weights_np(params + perturbs)
    
        # evaluate on environment
        done = False
        total_reward = 0
        steps = 0
        state = env.reset()
        while not done:
            state = torch.from_numpy(state).type(dtype)
            action_probs = pert_model((Variable(state.unsqueeze(0), volatile = True)))
            action = np.argmax(action_probs.data.cpu().numpy())
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            
            total_reward += reward
            steps += 1

        print('Reward from process ' + str(rank) + ': ' + str(total_reward) + ' after ' + str(steps) + ' steps')

        # put reward in queue for other processes to receive
        for n in range(args.num_processes - 1):
            reward_queues[rank].put(total_reward)
        
        # reconstruct perturbations from other processes and add them to total based on their reward
        perturbs = perturbs * total_reward
        max_reward = 0
        for n in range(args.num_processes):
            if n != rank:
                other_reward = reward_queues[n].get()
                total_reward += other_reward
                max_reward = max(max_reward, other_reward)
                torch.manual_seed(curr_seed + n)
                other_perturbs = torch.normal(torch.zeros(len(params)), torch.FloatTensor([args.sigma] * len(params))).numpy()
                perturbs += other_perturbs * other_reward

        # adjust for learning rate, number of processes and standard deviation
        perturbs = perturbs * args.lr / (args.sigma * args.num_processes)
        model.set_weights_np(params + perturbs)

        curr_seed = curr_seed + args.num_processes

        # logs average reward and maximum reward for the training step
        if rank == 0:
            log_value('Reward', total_reward / args.num_processes, step)
            log_value('Maximum Reward', max_reward, step)

        # save weights of model
        if rank == 1 and step % 80 == 0:
            pickle.dump(model.state_dict(), open(args.save_name + '.p', 'wb'))