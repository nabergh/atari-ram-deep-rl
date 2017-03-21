import time
import pickle
from datetime import date

import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.autograd import Variable
from collections import deque
from a3c_model import ActorCritic
from a3c_envs import create_atari_env

from tensorboard_logger import configure, log_value

def test(rank, args, shared_model, dtype):
    test_ctr = 0
    torch.manual_seed(args.seed + rank)

    env = create_atari_env(args.env_name)
    env.seed(args.seed + rank)
    state = env.reset()

    model = ActorCritic(state.shape[0], env.action_space).type(dtype)

    model.eval()

    state = torch.from_numpy(state).type(dtype)
    reward_sum = 0
    done = True

    start_time = time.time()

    #set up logger
    timestring = str(date.today()) + '_' + time.strftime("%Hh-%Mm-%Ss", time.localtime(time.time()))
    configure("logs/run_" + args.save_name + '_' + timestring, flush_secs=5)

    # a quick hack to prevent the agent from stucking
    actions = deque(maxlen=200)
    episode_length = 0
    while True:
        episode_length += 1
        # Sync with the shared model
        if done:
            model.load_state_dict(shared_model.state_dict())
            cx = Variable(torch.zeros(1, 256).type(dtype), volatile=True)
            hx = Variable(torch.zeros(1, 256).type(dtype), volatile=True)
        else:
            cx = Variable(cx.data.type(dtype), volatile=True)
            hx = Variable(hx.data.type(dtype), volatile=True)

        value, logit, (hx, cx) = model(
            (Variable(state.unsqueeze(0), volatile=True), (hx, cx)))
        prob = F.softmax(logit)
        action = prob.max(1)[1].data.cpu().numpy()

        state, reward, done, _ = env.step(action[0, 0])
        done = done or episode_length >= args.max_episode_length
        reward_sum += reward

        # a quick hack to prevent the agent from stucking
        actions.append(action[0, 0])
        if actions.count(actions[0]) == actions.maxlen:
            print("Agent stuck doing action " + str(actions[0]))
            stuck = True
            done = True

        if done:
            print("Time {}, episode reward {}, episode length {}".format(
                time.strftime("%Hh %Mm %Ss",
                              time.gmtime(time.time() - start_time)),
                reward_sum, episode_length))

            if not stuck:
                log_value('Reward', reward_sum, test_ctr)
                log_value('Episode length', episode_length, test_ctr)

            reward_sum = 0
            episode_length = 0
            stuck = False
            actions.clear()
            state = env.reset()
            test_ctr += 1

            if test_ctr % 10 == 0:
                pickle.dump(shared_model.state_dict(), open(args.save_name + '.p', 'wb'))
            time.sleep(60)

        state = torch.from_numpy(state).type(dtype)
