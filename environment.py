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

def get_env(game_name):
    env_name = f'{game_name}NoFrameskip-v4'
    env = suite_atari.load(env_name, gym_env_wrappers=[NoopResetEnv, AtariPreprocessing, FrameStack4])
    return env