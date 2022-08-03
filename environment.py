from tf_agents.environments import suite_atari

from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
import gym
import numpy as np

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env):
        super(NoopResetEnv, self).__init__(env)
        self._env = env
        self.no_ops_range = (0, 30)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
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
    # Clip rewards in range [-1, 0, 1]
    def __init__(self, env):
        super(ClipRewardsEnv, self).__init__(env)

    def reward(self, reward):
        return np.sign(reward)

def get_env(game_name, deterministic=False, clip_reward=True):
    env_name = f'{game_name}NoFrameskip-v4'
    wrappers = [AtariPreprocessing, FrameStack4]

    if clip_reward:
        wrappers.insert(0, ClipRewardsEnv)

    if not deterministic:
        wrappers.insert(0, NoopResetEnv)
    env = suite_atari.load(env_name, gym_env_wrappers=wrappers)
    return env