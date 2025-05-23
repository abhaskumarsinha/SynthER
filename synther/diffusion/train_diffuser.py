import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')))

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

#----------------------------------------------
# Train diffusion model on D4RL transitions.
import argparse
import pathlib

#import d4rl
import gin
#import gym
import numpy as np
import torch
import wandb

from synther.diffusion.elucidated_diffusion import Trainer
from synther.diffusion.norm import MinMaxNormalizer
from synther.diffusion.utils import make_inputs, split_diffusion_samples, construct_diffusion_model


@gin.configurable
class SimpleDiffusionGenerator:
    def __init__(
            self,
            env: gym.Env,
            ema_model,
            num_sample_steps: int = 128,
            sample_batch_size: int = 100000,
    ):
        self.env = env
        self.diffusion = ema_model
        self.diffusion.eval()
        # Clamp samples if normalizer is MinMaxNormalizer
        self.clamp_samples = isinstance(self.diffusion.normalizer, MinMaxNormalizer)
        self.num_sample_steps = num_sample_steps
        self.sample_batch_size = sample_batch_size
        print(f'Sampling using: {self.num_sample_steps} steps, {self.sample_batch_size} batch size.')

    def sample(
            self,
            num_samples: int,
    ) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        assert num_samples % self.sample_batch_size == 0, 'num_samples must be a multiple of sample_batch_size'
        num_batches = num_samples // self.sample_batch_size
        observations = []
        actions = []
        rewards = []
        next_observations = []
        terminals = []
        for i in range(num_batches):
            print(f'Generating split {i + 1} of {num_batches}')
            sampled_outputs = self.diffusion.sample(
                batch_size=self.sample_batch_size,
                num_sample_steps=self.num_sample_steps,
                clamp=self.clamp_samples,
            )
            sampled_outputs = sampled_outputs.cpu().numpy()

            # Split samples into (s, a, r, s') format
            transitions = split_diffusion_samples(sampled_outputs, self.env)
            if len(transitions) == 4:
                obs, act, rew, next_obs = transitions
                terminal = np.zeros_like(next_obs[:, 0])
            else:
                obs, act, rew, next_obs, terminal = transitions
            observations.append(obs)
            actions.append(act)
            rewards.append(rew)
            next_observations.append(next_obs)
            terminals.append(terminal)
        observations = np.concatenate(observations, axis=0)
        actions = np.concatenate(actions, axis=0)
        rewards = np.concatenate(rewards, axis=0)
        next_observations = np.concatenate(next_observations, axis=0)
        terminals = np.concatenate(terminals, axis=0)

        return observations, actions, rewards, next_observations, terminals


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, default='halfcheetah-medium-replay-v2')
    parser.add_argument('--gin_config_files', nargs='*', type=str, default=['config/resmlp_denoiser.gin'])
    parser.add_argument('--gin_params', nargs='*', type=str, default=[])
    # wandb config
    parser.add_argument('--wandb-project', type=str, default="offline-rl-diffusion")
    parser.add_argument('--wandb-entity', type=str, default="")
    parser.add_argument('--wandb-group', type=str, default="diffusion_training")
    #
    parser.add_argument('--results_folder', type=str, default='./results')
    parser.add_argument('--use_gpu', action='store_true', default=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--save_samples', action='store_true', default=True)
    parser.add_argument('--save_num_samples', type=int, default=int(5e6))
    parser.add_argument('--save_file_name', type=str, default='5m_samples.npz')
    parser.add_argument('--load_checkpoint', action='store_true')
    args = parser.parse_args()

    gin.parse_config_files_and_bindings(args.gin_config_files, args.gin_params)

    # Set seed.
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if args.use_gpu:
        torch.cuda.manual_seed(args.seed)

    # Create the environment and dataset.
    env = gym.make(args.dataset)
    inputs = make_inputs(env)
    inputs = torch.from_numpy(inputs).float()
    dataset = torch.utils.data.TensorDataset(inputs)

    results_folder = pathlib.Path(args.results_folder)
    results_folder.mkdir(parents=True, exist_ok=True)
    with open(results_folder / 'config.gin', 'w') as f:
        f.write(gin.config_str())

    # Create the diffusion model and trainer.
    diffusion = construct_diffusion_model(inputs=inputs)
    trainer = Trainer(
        diffusion,
        dataset,
        results_folder=args.results_folder,
    )

    if not args.load_checkpoint:
        # Initialize logging.
        wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            config=args,
            group=args.wandb_group,
            name=args.results_folder.split('/')[-1],
        )
        # Train model.
        trainer.train()
    else:
        trainer.ema.to(trainer.accelerator.device)
        # Load the last checkpoint.
        trainer.load(milestone=trainer.train_num_steps)

    # Generate samples and save them.
    if args.save_samples:
        generator = SimpleDiffusionGenerator(
            env=env,
            ema_model=trainer.ema.ema_model,
        )
        observations, actions, rewards, next_observations, terminals = generator.sample(
            num_samples=args.save_num_samples,
        )
        np.savez_compressed(
            results_folder / args.save_file_name,
            observations=observations,
            actions=actions,
            rewards=rewards,
            next_observations=next_observations,
            terminals=terminals,
        )
