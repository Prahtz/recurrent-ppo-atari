from tf_agents.environments import suite_gym
import cv2 as cv
import numpy as np
import gym

import collections

class AtariRescaling(gym.Wrapper):
    def __init__(self, env: gym.Env):
        super(AtariRescaling, self).__init__(env)
        self._env = env
        shape = (105, 80, 3)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)
    def reset(self):
        observation = self._env.reset()
        return cv.resize(observation, (80, 105), interpolation=cv.INTER_AREA)
    def step(self, action):
        observation, reward, done, info = self._env.step(action)
        observation = cv.resize(observation, (80, 105), interpolation=cv.INTER_AREA)
        return observation, reward, done, info

class FrameStack4(gym.Wrapper):
    """Stack previous four frames (must be applied to Gym env, not our envs)."""

    STACK_SIZE = 4

    def __init__(self, env: gym.Env):
        super(FrameStack4, self).__init__(env)
        self._env = env
        self._frames = collections.deque(maxlen=FrameStack4.STACK_SIZE)
        space = self._env.observation_space
        shape = space.shape[0:2] + (FrameStack4.STACK_SIZE*3,)
        self.observation_space = gym.spaces.Box(
            low=0, high=255, shape=shape, dtype=np.uint8)

    def __getattr__(self, name):
        """Forward all other calls to the base environment."""
        return getattr(self._env, name)

    def _generate_observation(self):
        return np.concatenate(self._frames, axis=2)

    def reset(self) -> np.ndarray:
        observation = self._env.reset()
        for _ in range(FrameStack4.STACK_SIZE):
            self._frames.append(observation)
        return self._generate_observation()

    def step(self, action: np.ndarray) -> np.ndarray:
        observation, reward, done, info = self._env.step(action)
        self._frames.append(observation)
        return self._generate_observation(), reward, done, info
    
def get_env(game_name):
    env_name = f'{game_name}-v4'
    env = suite_gym.load(env_name, gym_env_wrappers=[AtariRescaling, FrameStack4])
    return env