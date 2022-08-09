from tf_agents.environments import suite_atari

from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
import gym
import numpy as np

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env):
        """
        Performs N no-op actions every time the evironment is reset.
        N is sampled uniformly in range (0, 30)
        """
        super(NoopResetEnv, self).__init__(env)
        self._env = env
        self.no_ops_range = (0, 30)

    def __getattr__(self, name):
        return getattr(self._env, name)

    def reset(self):
        last_obs = self._env.reset()
        no_ops = np.random.randint(low=self.no_ops_range[0], high=self.no_ops_range[1])
        for _ in range(no_ops):
            last_obs = self._env.step(0)[0]
        return last_obs

    def step(self, action):
        return self._env.step(action)

class ClipRewardsEnv(gym.RewardWrapper):
    def __init__(self, env):
        """
        Clip rewards in range [-1, 0, 1]
        """
        super(ClipRewardsEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)

def get_env(game_name, use_rnn, deterministic=False, clip_reward=True, ):
    env_name = f'{game_name}NoFrameskip-v4'
    wrappers = [AtariPreprocessing]

    if clip_reward:
        wrappers.insert(0, ClipRewardsEnv)

    if not deterministic:
        wrappers.insert(0, NoopResetEnv)
    
    if not use_rnn:
        wrappers.append(FrameStack4)
    env = suite_atari.load(env_name, gym_env_wrappers=wrappers)
    return env