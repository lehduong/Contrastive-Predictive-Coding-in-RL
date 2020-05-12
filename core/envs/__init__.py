from .standard_envs import make_env as make_standard_env
from .standard_envs import make_vec_envs as make_standard_vec_envs
from .park_envs import make_env as make_park_env
from .park_envs import make_vec_envs as make_park_vec_envs
from .park_envs import PARK_ENV_LIST

def make_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps=None):
    if env_id in PARK_ENV_LIST:
        return make_park_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps)
    else:
        return make_standard_env(env_id, seed, rank, log_dir, allow_early_resets, max_episode_steps)

def make_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, allow_early_resets, num_frame_stack=None, max_episode_steps=None):
    if env_name in PARK_ENV_LIST:
        return make_park_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, allow_early_resets, num_frame_stack, max_episode_steps)
    else:
        return make_standard_vec_envs(env_name, seed, num_processes, gamma, log_dir, device, allow_early_resets, num_frame_stack, max_episode_steps)
