from tf_agents.environments import suite_atari
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.atari_preprocessing import AtariPreprocessing
import tensorflow as tf
import cv2 as cv
import numpy as np
import gym

import collections

def get_env(game_name):
    env_name = f'{game_name}-v4'
    #env = suite_gym.load(env_name, gym_env_wrappers=[AtariRescaling, FrameStack4])
    env = suite_atari.load(env_name, gym_env_wrappers=[AtariPreprocessing, FrameStack4])
    return env