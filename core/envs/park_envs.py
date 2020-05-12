"""
Make environments of Park Platform
"""
import torch
import numpy as np 
import os
import park

from park.spaces.box import Box
from baselines import bench
from baselines.common.atari_wrappers import make_atari, wrap_deepmind
from baselines.common.vec_env import VecEnvWrapper
from baselines.common.vec_env.dummy_vec_env import DummyVecEnv
from baselines.common.vec_env.shmem_vec_env import ShmemVecEnv
from baselines.common.wrappers import TimeLimit

from .standard_envs import VecNormalize, VecPyTorch, VecPyTorchFrameStack, TimeLimitMask

PARK_ENV_LIST = ['abr', 'abr_sim',
                 'spark', 'spark_sim',
                 'load_balance']

def make_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps=None):
    def _thunk():
        if env_id not in PARK_ENV_LIST:
            raise ValueError("Unsupported environment, expect the environment to be one of "+str(PARK_ENV_LIST)+" but got: "+str(env_id))
        else:
            env = park.make(env_id)

        if max_episode_steps:
            env = TimeLimit(env, max_episode_steps)
            # adding information to env for computing return
            env = TimeLimitMask(env)

        env.seed(seed+rank)

        if log_dir is not None:
            env = bench.Monitor(
                    env,
                    os.path.join(log_dir, str(rank)),
                    allow_early_resets=allow_early_resets)

        obs_shape = env.observation_space.shape
        
        if len(obs_shape) == 3:
            raise NotImplementedError(
                "CNN models work only for atari,\n"
                "please use a custom wrapper for a custom pixel input env.\n"
                "See wrap_deepmind for an example.")
        
        if len(obs_shape) == 3 and obs_shape[2] in [1,3]:
            env = TransposeImage(env, op=[2, 0, 1])

        return env 

    return _thunk

def make_vec_envs(env_name,
                  seed,
                  num_processes,
                  gamma,
                  log_dir,
                  device,
                  allow_early_resets,
                  num_frame_stack=None,
                  max_episode_steps=None):
    envs = [
            make_env(env_name, seed, i, log_dir, allow_early_resets, max_episode_steps)
            for i in range(num_processes)
    ]

    if len(envs) > 1:
        envs = ShmemVecEnv(envs, context='fork')
    else:
        envs = DummyVecEnv(envs)

    # if len(envs.observation_space.shape) == 1:
    #     if gamma is None:
    #         envs = VecNormalize(envs, ret=False)
    #     else:
    #         envs = VecNormalize(envs, gamma=gamma)
    envs = VecPyTorch(envs, device)

    return envs
