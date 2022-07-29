from tf_agents.environments import suite_atari
from tf_agents.environments.atari_wrappers import FrameStack4
from tf_agents.environments.atari_preprocessing import AtariPreprocessing

def get_env(game_name):
    env_name = f'{game_name}NoFrameskip-v4'
    env = suite_atari.load(env_name, gym_env_wrappers=[AtariPreprocessing, FrameStack4])
    return env