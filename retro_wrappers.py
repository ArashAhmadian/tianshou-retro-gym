import retro 
from tianshou.env import ShmemVectorEnv
import gym
import numpy as np
from retro_wrapper_helpers import *

def wrap_dm(
    game = "",
    episode_life=True,
    clip_rewards=True,
    frame_stack=4,
    scale=False,
    warp_frame=True,
    **kwargs
):
    """Configure environment for DeepMind-style Atari. The observation is
    channel-first: (c, h, w) instead of (h, w, c).
    :param str game: the atari environment id.
    :param bool episode_life: wrap the episode life wrapper.
    :param bool clip_rewards: wrap the reward clipping wrapper.
    :param int frame_stack: wrap the frame stacking wrapper.
    :param bool scale: wrap the scaling observation wrapper.
    :param bool warp_frame: wrap the grayscale + resize observation wrapper.
    :return: the wrapped atari environment.
    """
    assert 'NoFrameskip' in game
    env = gym.make(game,**kwargs)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    if warp_frame:
        env = WarpFrame(env)
    if scale:
        env = ScaledFloatFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack:
        env = FrameStack(env, frame_stack)
    return env


def make_atari_env(is_retro, task, seed, training_num, test_num, **kwargs):
    """Wrapper function for Atari env.
    If EnvPool is installed, it will automatically switch to EnvPool's Atari env.
    :return: a tuple of (single env, training envs, test envs).
    """


    wrap_deepmind = wrap_dm_retro if is_retro else wrap_dm
    env = wrap_deepmind(task, **kwargs)
    act_shape = env.action_space.shape or env.action_space.n
    obs_shape = env.observation_space.shape or env.observation_space.n 
    if env.spec:
        reward_threshold = env.spec.reward_threshold
    else: 
        reward_threshold = None 
    # We have to close this one-off enviroment because gym retro
    # doesn't allow more than more instance of the emulator to get run 
    # in the one process => have to use shmemvec as below only. 
    env.close() 
    train_envs = ShmemVectorEnv(
        [
            lambda:
            wrap_deepmind(game=task, episode_life=True, clip_rewards=True, **kwargs)
            for _ in range(training_num)
        ]
    )
    test_envs = ShmemVectorEnv(
        [
            lambda:
            wrap_deepmind(game=task, episode_life=False, clip_rewards=False, **kwargs)
            for _ in range(test_num)
        ]
    )
    train_envs.seed(seed)
    test_envs.seed(seed)

    return obs_shape,act_shape,reward_threshold, train_envs, test_envs

def wrap_dm_retro(
    game="", 
    scale=True, 
    frame_stack=4,     
    episode_life=True,
    clip_rewards=True,
    **kwargs):
    """
    Configure environment for retro games, using config similar to DeepMind-style Atari in wrap_deepmind
    """
    env = retro.make(game,**kwargs)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    if frame_stack > 1:
        env = FrameStack(env, frame_stack)
    if scale:
        env = ScaledFloatFrame(env)
    if episode_life:
        # NOTE: leads to error in accessing .env.unwrapped.ale.lives()
        # This is because of the under-the-hood impelmentation of retro
        # which doesn't call super() in its __init__ function eventhough 
        # it inherits from gym. So not all methods (only the ones selected)
        # are visible to the outer API. 
        # TODO: Find workaround 
        # env = EpisodicLifeEnv(env)
        pass 
    return env