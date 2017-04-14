import time
import numpy as np
from datetime import date
from itertools import count
import pickle
import gym

import torch
import torch.nn.functional as F
from torch.autograd import Variable

from es_model import EvolutionNet
from a3c_envs import create_atari_env

from tensorboard_logger import configure, log_value

def test(rank, args, dtype):

    if args.upload:
        api_key = ''
        with open('api_key.json', 'r+') as api_file:
            api_key = json.load(api_file)['api_key']

    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    run_name = args.load_name + '_' + timestring
    configure("logs/es_evaluate" + run_name, flush_secs=5)

    torch.manual_seed(args.seed)
    curr_seed = args.seed

    env = create_atari_env(args.env_name, True, run_name)
    env.seed(args.seed + rank)
    state = env.reset()

    model = EvolutionNet(state.shape[0], env.action_space).type(dtype)
    if args.load_name is not None:
        model.load_state_dict(pickle.load(open('models/' + args.load_name + '.p', 'rb')))
    else:
        print('A model is needed to train. Use the --load-name argument to point to a saved model\'s pickled state dictionary')

    for step in range(100):
        # evaluate on environment
        done = False
        total_reward = 0
        steps = 0
        state = env.reset()
        while not done:
            state = torch.from_numpy(state).type(dtype)
            action_probs = model((Variable(state.unsqueeze(0), volatile = True)))
            action = np.argmax(action_probs.data.cpu().numpy())
            next_state, reward, done, _ = env.step(action)
            
            state = next_state
            
            total_reward += reward
            steps += 1

        print('Reward from process ' + str(rank) + ': ' + str(total_reward) + ' after ' + str(steps) + ' steps')

        # logs average reward
        log_value('Reward', total_reward, step)

    env.close()

    if args.upload:
        gym.upload('monitor/' + run_name, api_key=api_key)

