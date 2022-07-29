import argparse

from config import config
import random
from environment import get_env
from agents import PPOAgent
import os

import functools
from tf_agents.environments import ParallelPyEnvironment, TFPyEnvironment
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
import tf_agents
import tensorflow as tf
import numpy as np

from actor_critic import AtariActorCriticNetwork, AtariNetwork
from atari_policy import AtariPolicy

import wandb
import datetime

import utils

def main(args):
    args = args[0]
    cfg_path = args.cfg_path
    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    game_name = args.game_name
    render = args.render

    now = datetime.datetime.now().timetuple()
    run_name = game_name + '_' + ''.join([str(t) for t in now])
    wandb.init(project="recurrent-pg-atari", entity="prahtz", name=run_name)

    

    num_workers = cfg.ppo.num_actors
    max_time_steps = cfg.env.max_time_steps
    horizon = cfg.ppo.horizon
    num_epochs = cfg.ppo.num_epochs
    n_minibatch = cfg.ppo.num_minibatch

    lr = cfg.ppo.lr
    discount = cfg.ppo.discount
    gae_parameter = cfg.ppo.gae_parameter
    clipping_parameter = cfg.ppo.clipping_parameter
    vf_coeff = cfg.ppo.vf_coeff
    entropy_coeff = cfg.ppo.entropy_coeff
    annealing = cfg.ppo.annealing

    share_params = cfg.model.share_params
    units = cfg.model.rnn_units
    use_rnn = cfg.model.use_rnn
    memory_size = cfg.model.memory_size

    n_collections = max_time_steps // (num_workers*horizon)
    
    wandb.config = {
        "lr": lr,
        "max_time_steps": max_time_steps,
        "epochs": num_epochs,
        "n_minibatch": n_minibatch,
        "horizon": horizon,
        "vf_coeff": vf_coeff,
        "use_rnn": use_rnn,
        "rnn_units": units,
        "share_params": share_params,
        "memory_size": memory_size
    }

    env_constructor = functools.partial(get_env, game_name)

    envs = [env_constructor for _ in range(num_workers)]
    env = TFPyEnvironment(ParallelPyEnvironment(env_constructors=envs))
    env.seed([args.seed]*num_workers)

    if share_params:
        policy_state_spec = (tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16))
    else:
        policy_state_spec = ((tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16)), 
                            (tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16)))
    info_spec = (tf.TensorSpec(shape=(env.action_spec().maximum), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32))

    policy_encoder = AtariNetwork(use_rnn=use_rnn, rnn_units=units, memory_size=memory_size)
    if share_params:
        value_encoder = None
    else:
        value_encoder = AtariNetwork(use_rnn=use_rnn, rnn_units=units, memory_size=memory_size)

    actor_critic_network = AtariActorCriticNetwork(n_actions=env.action_spec().maximum, 
                                                   policy_encoder=policy_encoder, 
                                                   value_encoder=value_encoder)
    policy = AtariPolicy(time_step_spec=env.time_step_spec(), 
                         action_spec=env.action_spec(), 
                         policy_state_spec=policy_state_spec,
                         info_spec=info_spec,
                         actor_critic_network=actor_critic_network)
    if annealing:
        lr_decay = utils.AtariDecay(initial_value=1.0, 
                                    end_value=0.0, 
                                    decay_steps=n_collections, 
                                    decay_after_steps=num_epochs*n_minibatch)
        clipping_decay = utils.AtariDecay(initial_value=1.0, 
                                    end_value=0.0, 
                                    decay_steps=n_collections, 
                                    decay_after_steps=num_epochs*n_minibatch)
        learning_rate_fn = lambda : lr*lr_decay()
        clipping_fn = lambda : clipping_parameter*clipping_decay()
    else:
        learning_rate_fn = lr
        clipping_fn = lambda : clipping_parameter
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate_fn)
    driver = DynamicStepDriver(env, policy=policy, num_steps=horizon*num_workers)
    
    agent = PPOAgent(num_workers, horizon, driver, policy)
    agent.train(optimizer=optimizer,
                num_epochs=num_epochs,
                num_minibatch=n_minibatch,
                n_collections=n_collections,
                discount=discount,
                gae_parameter=gae_parameter,
                clipping_fn=clipping_fn,
                vf_coeff=vf_coeff,
                entropy_coeff=entropy_coeff,
                use_rnn=use_rnn,
                render=render)
    
    policy.actor_critic_network.save_weights(f'models/{game_name}.h5')
if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', help='Name of the Atari game')
    parser.add_argument('cfg_path', help='YAML file containing arguments and hyperparameters')
    parser.add_argument('--seed', help='Random seed, for reproducibility', default=42)
    parser.add_argument('--debug', help='If set, run in debug mode (no logs)', action='store_true')
    parser.add_argument('--render', help='If set, render experiences every 100 collections', action='store_true')

    args = parser.parse_args()
    
    if args.debug:
        os.environ["WANDB_MODE"] = "offline"

    random.seed(args.seed)
    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf_agents.system.multiprocessing.handle_main(main, [args])