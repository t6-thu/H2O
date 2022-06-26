import os
import h5py
import torch
import random
import numpy as np
from gym.spaces import Box, Discrete, Tuple

from envs import get_dim
from replay_buffer import ReplayBuffer


class MixedReplayBuffer(ReplayBuffer):
    def __init__(self, reward_scale, reward_bias, clip_action, state_dim, action_dim, task="halfcheetah", data_source="medium_replay",  device="cuda", scale_rewards=True, scale_state=False, buffer_ratio=1, residual_ratio=0.1):
        super().__init__(state_dim, action_dim, device=device)

        self.scale_rewards = scale_rewards
        self.scale_state = scale_state
        self.buffer_ratio = buffer_ratio
        self.residual_ratio = residual_ratio

        # load expert dataset into the replay buffer
        path = os.path.join("../../d4rl_mujoco_dataset", "{}_{}-v2.hdf5".format(task, data_source))
        with h5py.File(path, "r") as dataset:
            total_num = dataset['observations'].shape[0]
            # idx = random.sample(range(total_num), int(total_num * self.residual_ratio))
            idx = np.random.choice(range(total_num), int(total_num * self.residual_ratio), replace=False)
            s = np.vstack(np.array(dataset['observations'])).astype(np.float32)[idx, :] # An (N, dim_observation)-dimensional numpy array of observations
            a = np.vstack(np.array(dataset['actions'])).astype(np.float32)[idx, :] # An (N, dim_action)-dimensional numpy array of actions
            r = np.vstack(np.array(dataset['rewards'])).astype(np.float32)[idx, :] # An (N,)-dimensional numpy array of rewards
            s_ = np.vstack(np.array(dataset['next_observations'])).astype(np.float32)[idx, :] # An (N, dim_observation)-dimensional numpy array of next observations
            done = np.vstack(np.array(dataset['terminals']))[idx, :] # An (N,)-dimensional numpy array of terminal flags

        # whether to bias the reward
        r = r * reward_scale + reward_bias
        # whether to clip actions
        a = np.clip(a, -clip_action, clip_action)
        
        fixed_dataset_size = r.shape[0]
        self.fixed_dataset_size = fixed_dataset_size
        self.ptr = fixed_dataset_size
        self.size = fixed_dataset_size
        self.max_size = (self.buffer_ratio + 1) * fixed_dataset_size

        self.state = np.vstack((s, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.action = np.vstack((a, np.zeros((self.max_size - self.fixed_dataset_size, action_dim))))
        self.next_state = np.vstack((s_, np.zeros((self.max_size - self.fixed_dataset_size, state_dim))))
        self.reward = np.vstack((r, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.done = np.vstack((done, np.zeros((self.max_size - self.fixed_dataset_size, 1))))
        self.device = torch.device(device)
        
        # # State normalization
        self.normalize_states()



    def normalize_states(self, eps=1e-3):
        # STATE: standard normalization
        self.state_mean = self.state.mean(0, keepdims=True)
        self.state_std = self.state.std(0, keepdims=True) + eps
        if self.scale_state:
            self.state = (self.state - self.state_mean) / self.state_std
            self.next_state = (self.next_state - self.state_mean) / self.state_std

    def append(self, s, a, r, s_, done):

        self.state[self.ptr] = s
        self.action[self.ptr] = a
        self.next_state[self.ptr] = s_
        self.reward[self.ptr] = r
        self.done[self.ptr] = done

        # fix the offline dataset and shuffle the simulated part
        self.ptr = (self.ptr + 1 - self.fixed_dataset_size) % (self.max_size - self.fixed_dataset_size) + self.fixed_dataset_size
        self.size = min(self.size + 1, self.max_size)

    def append_traj(self, observations, actions, rewards, next_observations, dones):
        for o, a, r, no, d in zip(observations, actions, rewards, next_observations, dones):
            self.append(o, a, r, no, d)

    def sample(self, batch_size, scope=None, type=None):
        if scope == None:
            ind = np.random.randint(0, self.size, size=batch_size)
        elif scope == "real":
            ind = np.random.randint(0, self.fixed_dataset_size, size=batch_size)
        elif scope == "sim":
            ind = np.random.randint(self.fixed_dataset_size, self.size, size=batch_size)
        else: 
            raise RuntimeError("Misspecified range for replay buffer sampling")

        if type == None:
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
                'rewards': torch.FloatTensor(self.reward[ind]).to(self.device), 
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device), 
                'dones': torch.FloatTensor(self.done[ind]).to(self.device)
                }
        elif type == "sas":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device), 
                'next_observations': torch.FloatTensor(self.next_state[ind]).to(self.device)
                }
        elif type == "sa":
            return {
                'observations': torch.FloatTensor(self.state[ind]).to(self.device), 
                'actions': torch.FloatTensor(self.action[ind]).to(self.device)
                }
        else: 
            raise RuntimeError("Misspecified return data types for replay buffer sampling")

    def get_mean_std(self):
        return torch.FloatTensor(self.state_mean).to(self.device), torch.FloatTensor(self.state_std).to(self.device)