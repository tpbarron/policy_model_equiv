import argparse
from collections import deque
from itertools import count
import csv, copy, os, sys, time

import gym
import numpy as np

import torch
import torch.multiprocessing as mp
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data.sampler import BatchSampler, SubsetRandomSampler
torch.set_default_tensor_type('torch.FloatTensor')

# ppo imports
from baselines.common.vec_env.subproc_vec_env import SubprocVecEnv
from envs import make_env
from eval import run_eval_episodes
from model import *
from storage import RolloutStorage
from logger import Logger

import pybullet_envs

parser = argparse.ArgumentParser()
parser.add_argument('--env-name', type=str, default='AcrobotContinuousVisionX-v0', help='Env to train.')
parser.add_argument('--n-iter', type=int, default=250, help='Num iters')
parser.add_argument('--max-episode-steps', type=int, default=1000, help='Max num ep steps')
parser.add_argument('--render', action='store_true', help='Render env observations')
parser.add_argument('--shared-actor-critic', action='store_true', help='Whether to share params between pol and val in network')

# PPO args
parser.add_argument('--lr', type=float, default=3e-4, help='learning rate (default: 7e-4)')
parser.add_argument('--eps', type=float, default=1e-5, help='RMSprop optimizer epsilon (default: 1e-5)')
parser.add_argument('--alpha', type=float, default=0.99, help='RMSprop optimizer apha (default: 0.99)')
parser.add_argument('--gamma', type=float, default=0.99, help='discount factor for rewards (default: 0.99)')
parser.add_argument('--use-gae', action='store_true', default=False, help='use generalized advantage estimation')
parser.add_argument('--tau', type=float, default=0.95, help='gae parameter (default: 0.95)')
parser.add_argument('--ppo-epoch', type=int, default=4, help='number of ppo epochs (default: 4)')
parser.add_argument('--num-processes', type=int, default=16, help='how many training CPU processes to use (default: 16)')
parser.add_argument('--num-steps', type=int, default=5, help='number of forward steps in A2C (default: 5)')
parser.add_argument('--batch-size', type=int, default=64, help='ppo batch size (default: 64)')
parser.add_argument('--clip-param', type=float, default=0.2, help='ppo clip parameter (default: 0.2)')
parser.add_argument('--entropy-coef', type=float, default=0.01, help='entropy term coefficient (default: 0.01)')
parser.add_argument('--value-loss-coef', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--model-loss-coef', type=float, default=0.5, help='model loss coefficient (default: 0.5)')

parser.add_argument('--max-grad-norm', type=float, default=0.5, help='value loss coefficient (default: 0.5)')
parser.add_argument('--seed', type=int, default=1, help='random seed (default: 1)')

parser.add_argument('--log-dir', default='/tmp/gym/', help='directory to save agent logs (default: /tmp/gym)')
parser.add_argument('--log-interval', type=int, default=1, help='log interval, one log per n updates (default: 10)')
parser.add_argument('--vis-interval', type=int, default=100, help='vis interval, one log per n updates (default: 100)')
parser.add_argument('--num-stack', type=int, default=1, help='number of frames to stack (default: 4)')
parser.add_argument('--num-frames', type=int, default=10e6, help='number of frames to train (default: 10e6)')
parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
parser.add_argument('--no-vis', action='store_true', default=False, help='disables visdom visualization')
parser.add_argument('--no-mean-encode', action='store_true', default=False, help='use tanh instead of mean encoding')
parser.add_argument('--model', action='store_true', default=False, help='Update value fn with model')

args = parser.parse_args()
args.cuda = not args.no_cuda and torch.cuda.is_available()
args.vis = not args.no_vis
print (args.log_dir)
os.makedirs(args.log_dir, exist_ok=True)

# some bookkeeping
log = Logger(args)
log.save_args()
log.create_csv_log()

# NOTE: in case someone is searching as I was, this wrapper will also reset the
# envs as each one finishes done
envs = SubprocVecEnv([
    make_env(args.env_name, args.seed, i, args.log_dir, args.max_episode_steps) #, args.cam_type)
    for i in range(args.num_processes)
])
# NOTE: seed breaks multiprocessing
# torch.manual_seed(args.seed)
np.random.seed(args.seed)

# if act discrete and obssize > 1 then discrete pixels
# if act cont and obs > 1 then cont pixels
# if act

obs_shape = envs.observation_space.shape
action_shape = 1
enc_shape = (64,)

# determine action shape
is_continuous = None
if envs.action_space.__class__.__name__ == 'Discrete':
    is_continuous = False
    action_shape = 1
else:
    is_continuous = True
    action_shape = envs.action_space.shape[0]
assert(is_continuous is not None)
# determine observation shape
if len(obs_shape) > 1: # then assume images and add frame stack
    obs_shape = (obs_shape[0] * args.num_stack, *obs_shape[1:])

actor_critic = make_actor_critic(obs_shape, envs.action_space, args.shared_actor_critic, is_continuous, not args.no_mean_encode)
num_inputs = envs.observation_space.shape[0]
num_model_inputs = num_inputs
num_actions = envs.action_space.shape[0]

if args.cuda:
    actor_critic.cuda()
    # dynamics = dynamics.cuda()

old_model = copy.deepcopy(actor_critic)
optimizer = optim.Adam(actor_critic.parameters(), args.lr, eps=args.eps)

def set_optimizer_lr(lr):
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

def ppo_update(num_updates, rollouts, final_rewards):
    # ppo update
    advantages = rollouts.returns[:-1] - rollouts.value_preds[:-1]
    advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-5)
    print ("advantages: ", advantages.max(), advantages.min(), advantages.mean(), advantages.std())

    old_model.load_state_dict(actor_critic.state_dict())
    if hasattr(actor_critic, 'obs_filter'):
        old_model.obs_filter = actor_critic.obs_filter

    decayed_clip = args.clip_param * max(1.0 - float(num_updates * args.num_processes * args.num_steps) / args.num_frames, 0)

    for _ in range(args.ppo_epoch):
        sampler = BatchSampler(SubsetRandomSampler(range(args.num_processes * args.num_steps-1)), args.batch_size * args.num_processes, drop_last=False)
        for indices in sampler:
            indices = torch.LongTensor(indices)
            if args.cuda:
                indices = indices.cuda()
            states_batch = rollouts.states[:-1].view(-1, *obs_shape)[indices]
            actions_batch = rollouts.actions.view(-1, action_shape)[indices]
            return_batch = rollouts.returns[:-1].view(-1, 1)[indices]

            # Reshape to do in a single forward pass for all steps
            values, action_log_probs, dist_entropy, model_preds, rew_preds = actor_critic.evaluate_actions(Variable(states_batch), Variable(actions_batch))

            _, old_action_log_probs, _, _, _ = old_model.evaluate_actions(Variable(states_batch, volatile=True), Variable(actions_batch, volatile=True))

            ratio = torch.exp(action_log_probs - Variable(old_action_log_probs.data))
            adv_targ = Variable(advantages.view(-1, 1)[indices])
            surr1 = ratio * adv_targ
            surr2 = torch.clamp(ratio, 1.0 - decayed_clip, 1.0 + decayed_clip) * adv_targ
            action_loss = -torch.min(surr1, surr2).mean() # PPO's pessimistic surrogate (L^CLIP)

            value_loss = (Variable(return_batch) - values).pow(2).mean()
            if args.model:
                enc_indices = indices + 1
                future_enc_batch = rollouts.model_encs.view(-1, *enc_shape)[enc_indices]
                future_rew_batch = rollouts.rewards.view(-1, 1)[indices]
                model_loss = (Variable(future_enc_batch) - model_preds).pow(2).mean() + (Variable(future_rew_batch) - rew_preds).pow(2).mean()

            optimizer.zero_grad()

            value_loss_coef = args.value_loss_coef if args.shared_actor_critic else 1.0
            model_loss_coef = args.model_loss_coef if args.shared_actor_critic else 1.0
            if args.model:
                (model_loss_coef*model_loss + value_loss_coef*value_loss + action_loss - dist_entropy * args.entropy_coef).backward()
            else:
                (value_loss_coef*value_loss + action_loss - dist_entropy * args.entropy_coef).backward()

            optimizer.step()

    rollouts.states[0].copy_(rollouts.states[-1])

    if num_updates % args.log_interval == 0:
        print("Updates {}, num frames {}, mean/median reward {:.1f}/{:.1f}, min/max reward {:.1f}/{:.1f}, entropy {:.5f}, value loss {:.5f}, policy loss {:.5f}".
            format(num_updates, num_updates * args.num_processes * args.num_steps,
                   final_rewards.mean(),
                   final_rewards.median(),
                   final_rewards.min(),
                   final_rewards.max(), -dist_entropy.data[0],
                   value_loss.data[0], action_loss.data[0]))
        # if final_rewards.mean() == 1.0: #Then have solved env
        #     return True, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])

    if num_updates * args.num_processes * args.num_steps > args.num_frames:
        return True, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])
    return False, (-dist_entropy.data[0], value_loss.data[0], action_loss.data[0])

    # if num_updates % args.vis_interval == 0:
    #     win = visdom_plot(viz, win, args.log_dir, args.env_name, args.algo)

def train():
    pre_bnn_error, post_bnn_error = -1, -1
    rollouts = RolloutStorage(args.num_steps, args.num_processes, obs_shape, envs.action_space, enc_shape)
    current_state = torch.zeros(args.num_processes, *obs_shape)

    def update_current_state(state):
        shape_dim0 = envs.observation_space.shape[0]
        state = torch.from_numpy(state).float()
        if args.num_stack > 1:
            # print (current_state.size())
            current_state[:, :-shape_dim0] = current_state[:, shape_dim0:]
        current_state[:, -shape_dim0:] = state

    state = envs.reset()
    update_current_state(state)

    rollouts.states[0].copy_(current_state)

    # These variables are used to compute average rewards for all processes.
    episode_rewards = torch.zeros([args.num_processes, 1])
    final_rewards = torch.zeros([args.num_processes, 1])

    if args.cuda:
        current_state = current_state.cuda()
        rollouts.cuda()

    for num_update in count(1):
        step = 0

        while step < args.num_steps:
            # episode_step += 1
            # Sample actions
            # print ("data: ", rollouts.states[step].size())
            value, action, enc, mpred, rpred = actor_critic.act(Variable(rollouts.states[step], volatile=True), encode_mean=True)
            # print ("val, act: ", value.size(), action.size())
            cpu_actions = action.data.cpu().numpy()
            if isinstance(envs.action_space, gym.spaces.Box):
                cpu_actions = np.clip(cpu_actions, envs.action_space.low, envs.action_space.high)

            # Obser reward and next state
            state, reward, done, info = envs.step(cpu_actions)
            reward = torch.from_numpy(np.expand_dims(np.stack(reward), 1)).float()
            episode_rewards += reward

            last_state = current_state.cpu().numpy()

            # If done then clean the history of observations.
            masks = torch.FloatTensor([[0.0] if done_ else [1.0] for done_ in done])
            final_rewards *= masks
            final_rewards += (1 - masks) * episode_rewards
            episode_rewards *= masks

            if args.cuda:
                masks = masks.cuda()

            if current_state.dim() == 4:
                current_state *= masks.unsqueeze(2).unsqueeze(2)
            else:
                current_state *= masks

            update_current_state(state)

            if args.render:
                env.render()

            rollouts.insert(step, current_state, action.data, value.data, reward, masks, enc.data)
            step += 1

        next_value = actor_critic(Variable(rollouts.states[-1], volatile=True))[0].data

        rollouts.compute_returns(next_value, args.use_gae, args.gamma, args.tau)
        do_exit, (pol_entropy, value_loss, policy_loss) = ppo_update(num_update, rollouts, final_rewards)

        test_rewards = np.array([-1])
        if num_update % 1 == 0:
            test_rewards = run_eval_episodes(actor_critic, 1, args, obs_shape)

        log.write_row({'updates': num_update,
                    'frames': num_update * args.num_processes * args.num_steps,
                    'mean_reward': test_rewards.mean(),
                    'median_reward': np.median(test_rewards),
                    'min_reward': test_rewards.min(),
                    'max_reward': test_rewards.max(),
                    'pol_entropy': pol_entropy,
                    'value_loss': value_loss,
                    'policy_loss': policy_loss})

        # save model
        torch.save(actor_critic, os.path.join(args.log_dir, 'model'+str(num_update)+'.pth'))

        if do_exit:
            envs.close()
            return

        frames = num_update * args.num_processes * args.num_steps
        set_optimizer_lr(args.lr * max(1.0 - frames / args.num_frames, 0))

if __name__ == '__main__':
    train()
