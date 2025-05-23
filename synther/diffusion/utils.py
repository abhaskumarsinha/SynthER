
# ——— Inline dummy modules ———
import sys
import types
import builtins
import numpy as np
# ——— INLINE DUMMY “gym” PACKAGE ———

# ——— 1. Define your Box, Dict & DummyEnv ———
class Box:
    def __init__(self, low, high, shape, dtype):
        self.low, self.high, self.shape, self.dtype = low, high, shape, dtype

class Dict(dict):
    pass

class DummyEnv:
    def __init__(self, name):
        self.name = name
        self.observation_space = Box(low=-np.inf, high=np.inf, shape=(3,), dtype=np.float32)
        self.action_space      = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)
    def reset(self):
        return np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
    def step(self, action):
        obs    = np.zeros(self.observation_space.shape, dtype=self.observation_space.dtype)
        reward = 0.0
        done   = True
        info   = {}
        return obs, reward, done, info
    def render(self): pass
    def close(self):  pass

# ——— 2. Build the gym.wrappers sub‐modules ———
wrappers = types.ModuleType("gym.wrappers")

# 2a. flatten_observation
flatten_mod = types.ModuleType("gym.wrappers.flatten_observation")
class FlattenObservation:
    def __init__(self, env):
        self.env = env
    def reset(self):
        obs = self.env.reset()
        if isinstance(obs, dict):
            return np.concatenate([v.ravel() for v in obs.values()])
        return obs
    def step(self, action):
        _, r, d, i = self.env.step(action)
        return self.reset(), r, d, i
flatten_mod.FlattenObservation = FlattenObservation
wrappers.flatten_observation = flatten_mod

# 2b. RescaleAction & ClipAction
class RescaleAction:
    def __init__(self, env, low, high): self.env = env
    def __getattr__(self, name): return getattr(self.env, name)
class ClipAction:
    def __init__(self, env): self.env = env
    def __getattr__(self, name): return getattr(self.env, name)
wrappers.RescaleAction = RescaleAction
wrappers.ClipAction   = ClipAction

# ——— 3. Build the top‐level gym module ———
gym = types.ModuleType("gym")
gym.Env    = DummyEnv
gym.make   = lambda name: DummyEnv(name)
gym.spaces = types.ModuleType("gym.spaces")
gym.spaces.Box  = Box
gym.spaces.Dict = Dict
gym.wrappers    = wrappers

# ——— 4. Inject into sys.modules & builtins ———
sys.modules["gym"]                               = gym
sys.modules["gym.spaces"]                        = gym.spaces
sys.modules["gym.wrappers"]                      = wrappers
sys.modules["gym.wrappers.flatten_observation"]  = flatten_mod
builtins.gym                                     = gym

# ——— 5. Fake out mujoco & d4rl ———
mujoco = types.ModuleType("mujoco")
sys.modules["mujoco"] = mujoco

d4rl = types.ModuleType("d4rl")
d4rl.load_dataset = lambda env, *a, **k: np.zeros((100,) + env.observation_space.shape, dtype=env.observation_space.dtype)
sys.modules["d4rl"] = d4rl

d4rl = types.ModuleType("d4rl")

# stub for qlearning_dataset (used by make_inputs)
def qlearning_dataset(env, *args, **kwargs):
    print(f"[dummy d4rl] qlearning_dataset({env.name})")
    # return an object with the fields that make_inputs expects,
    # or just a simple dict/namespace if that's enough:
    data = {
        'observations':  np.zeros((100,) + env.observation_space.shape, dtype=env.observation_space.dtype),
        'actions':       np.zeros((100,) + env.action_space.shape, dtype=env.action_space.dtype),
        'rewards':       np.zeros((100,), dtype=np.float32),
        'terminals':     np.zeros((100,), dtype=np.bool_),
        'next_observations': np.zeros((100,) + env.observation_space.shape, dtype=env.observation_space.dtype)
    }
    return data

# alias load_dataset if you also need it elsewhere
d4rl.load_dataset        = qlearning_dataset
d4rl.qlearning_dataset   = qlearning_dataset

sys.modules["d4rl"] = d4rl


# ——— End dummy modules ———

# now your real imports
import gym
import d4rl

# Utilities for diffusion.
from typing import Optional, List, Union

#import d4rl
import gin
#import gym
import numpy as np
import torch
from torch import nn

# GIN-required Imports.
from synther.diffusion.denoiser_network import ResidualMLPDenoiser
from synther.diffusion.elucidated_diffusion import ElucidatedDiffusion
from synther.diffusion.norm import normalizer_factory


# Make transition dataset from data.
@gin.configurable
def make_inputs(
        env: gym.Env,
        modelled_terminals: bool = False,
) -> np.ndarray:
    dataset = d4rl.qlearning_dataset(env)
    obs = dataset['observations']
    actions = dataset['actions']
    next_obs = dataset['next_observations']
    rewards = dataset['rewards']
    inputs = np.concatenate([obs, actions, rewards[:, None], next_obs], axis=1)
    if modelled_terminals:
        terminals = dataset['terminals'].astype(np.float32)
        inputs = np.concatenate([inputs, terminals[:, None]], axis=1)
    return inputs


# Convert diffusion samples back to (s, a, r, s') format.
@gin.configurable
def split_diffusion_samples(
        samples: Union[np.ndarray, torch.Tensor],
        env: gym.Env,
        modelled_terminals: bool = False,
        terminal_threshold: Optional[float] = None,
):
    # Compute dimensions from env
    obs_dim = env.observation_space.shape[0]
    action_dim = env.action_space.shape[0]
    # Split samples into (s, a, r, s') format
    obs = samples[:, :obs_dim]
    actions = samples[:, obs_dim:obs_dim + action_dim]
    rewards = samples[:, obs_dim + action_dim]
    next_obs = samples[:, obs_dim + action_dim + 1: obs_dim + action_dim + 1 + obs_dim]
    if modelled_terminals:
        terminals = samples[:, -1]
        if terminal_threshold is not None:
            if isinstance(terminals, torch.Tensor):
                terminals = (terminals > terminal_threshold).float()
            else:
                terminals = (terminals > terminal_threshold).astype(np.float32)
        return obs, actions, rewards, next_obs, terminals
    else:
        return obs, actions, rewards, next_obs


@gin.configurable
def construct_diffusion_model(
        inputs: torch.Tensor,
        normalizer_type: str,
        denoising_network: nn.Module,
        disable_terminal_norm: bool = False,
        skip_dims: List[int] = [],
        cond_dim: Optional[int] = None,
) -> ElucidatedDiffusion:
    event_dim = inputs.shape[1]
    model = denoising_network(d_in=event_dim, cond_dim=cond_dim)

    if disable_terminal_norm:
        terminal_dim = event_dim - 1
        if terminal_dim not in skip_dims:
            skip_dims.append(terminal_dim)

    if skip_dims:
        print(f"Skipping normalization for dimensions {skip_dims}.")

    normalizer = normalizer_factory(normalizer_type, inputs, skip_dims=skip_dims)

    return ElucidatedDiffusion(
        net=model,
        normalizer=normalizer,
        event_shape=[event_dim],
    )
