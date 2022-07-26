import argparse
from config import config
import random
from environment import get_env
from agents import PPOAgent

import functools
from tf_agents.trajectories import StepType
from tf_agents.environments import ParallelPyEnvironment, TFPyEnvironment, BatchedPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
import tf_agents
import tensorflow as tf
import numpy as np
import tensorflow_probability as tfp

from actor_critic import AtariActorCriticNetwork, AtariNetwork
from atari_policy import AtariPolicy

import wandb

def main(args):
    args = args[0]
    cfg_path = args.cfg_path
    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    game_name = args.game_name

    #wandb.init(project="recurrent-pg-atari", entity="prahtz", name=game_name)

    num_workers = cfg.ppo.num_actors
    max_time_steps = cfg.env.max_time_steps
    time_steps = cfg.ppo.horizon
    num_epochs = cfg.ppo.num_epochs
    n_minibatch = cfg.ppo.num_minibatch

    lr = cfg.ppo.lr
    discount = cfg.ppo.discount
    gae_parameter = cfg.ppo.gae_parameter
    clipping_parameter = cfg.ppo.clipping_parameter
    vf_coeff = cfg.ppo.vf_coeff
    entropy_coeff = cfg.ppo.entropy_coeff

    n_collections = max_time_steps // (num_workers*time_steps)

    units = 800

    env_constructor = functools.partial(get_env, game_name)

    envs = [env_constructor for _ in range(num_workers)]
    env = TFPyEnvironment(ParallelPyEnvironment(env_constructors=envs))
    env.seed([args.seed]*num_workers)

    policy_state_spec = tf.TensorSpec(shape=(units), dtype=tf.float32)
    info_spec = (tf.TensorSpec(shape=(env.action_spec().maximum), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32))
    atari_network = AtariNetwork()
    actor_critic_network = AtariActorCriticNetwork(atari_network, AtariNetwork(), env.action_spec().maximum)
    policy = AtariPolicy(time_step_spec=env.time_step_spec(), 
                         action_spec=env.action_spec(), 
                         policy_state_spec=policy_state_spec,
                         info_spec=info_spec,
                         actor_critic_network=actor_critic_network)

    learning_rate_fn = tf.keras.optimizers.schedules.PolynomialDecay(
        initial_learning_rate=lr,
        decay_steps=n_collections,
        end_learning_rate=0,
        power=1.0,
        cycle=False,
        name=None
    )
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    driver = DynamicStepDriver(env, policy=policy, num_steps=time_steps*num_workers)
    
    agent = PPOAgent(num_workers, driver, policy)
    agent.train(optimizer=optimizer,
                num_epochs=num_epochs,
                num_minibatch=n_minibatch,
                n_collections=n_collections,
                discount=discount,
                gae_parameter=gae_parameter,
                clipping_parameter=clipping_parameter,
                vf_coeff=vf_coeff,
                entropy_coeff=entropy_coeff)
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', help='Name of the Atari game')
    parser.add_argument('cfg_path', help='YAML file containing arguments and hyperparameters')
    parser.add_argument('--seed', help='Random seed, for reproducibility', default=42)
    
    args = parser.parse_args()


    
    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf_agents.system.multiprocessing.handle_main(main, [args])





