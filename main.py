import argparse
from config import config

from environment import get_env

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

def collect_sequences(driver, memory, policy, policy_state, time_steps):
    last_time_step, last_policy_state = driver.run(maximum_iterations=time_steps, policy_state=policy_state)
    last_observation, last_done = last_time_step.observation, tf.equal(last_time_step.step_type, StepType.LAST)
    last_observation, last_done = tf.expand_dims(last_observation, axis=1), tf.expand_dims(last_done, axis=1)
    _, _, last_value, _, _ = policy.actor_critic_network(last_observation, last_done, cell_states=last_policy_state, training=False)
    last_value = tf.squeeze(last_value, axis=1)


    observations = tf.stack([batched_step.observation for batched_step in memory], axis=1)
    rewards = tf.stack([batched_step.reward for batched_step in memory], axis=1)
    dones = tf.cast(tf.stack([batched_step.step_type == StepType.LAST for batched_step in memory] + [tf.squeeze(last_done, 1)], axis=1), dtype=tf.float32)
    actions = tf.stack([batched_step.action for batched_step in memory], axis=1)
    probs = tf.stack([batched_step.policy_info[0] for batched_step in memory], axis=1)
    values = tf.stack([batched_step.policy_info[1] for batched_step in memory] + [last_value], axis=1)

    memory.clear()
    return observations, actions, rewards, dones[:, 1:], probs, values, last_policy_state
    
def compute_truncated_advantages(discount, return_factor, rewards, values, dones):
    non_terminal = 1.0 - dones
    deltas = rewards + discount*non_terminal*values[:, 1:] - values[:, :-1]

    advantages = [deltas[:, -1]]
    for i in reversed(range(deltas.shape[1] - 1)):
        advantages.append(deltas[:, i] + discount*return_factor*non_terminal[:, i]*advantages[-1])
    advantages = tf.stack(advantages[::-1], axis=1)
    return advantages

def ppo_clip_loss(advantages, old_logprobs, logprobs, clip_range):
    prob_ratio = tf.exp(logprobs - old_logprobs)

    loss = tf.maximum(-advantages * prob_ratio, -advantages * tf.clip_by_value(prob_ratio, 1 - clip_range, 1 + clip_range))
    loss = tf.reduce_mean(loss, axis=None)
    return loss

def value_function_loss(values, returns):
    return tf.reduce_mean((values - returns)**2, axis=None)

def entropy_loss(entropy):
    return tf.reduce_mean(entropy, axis=None)

import cv2 as cv
def render_obs(obs):
    cv.namedWindow('env0', cv.WINDOW_NORMAL)
    obs = obs[0]
    for i in range(obs.shape[0]):
        frame = obs[i][:,:,3].numpy()
        #frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow('env0', frame)
        cv.waitKey(delay=100)


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
    epochs = cfg.ppo.num_epochs
    n_minibatch = cfg.ppo.num_minibatch

    lr = cfg.ppo.lr
    discount = cfg.ppo.discount
    gae_parameter = cfg.ppo.gae_parameter
    clipping_parameter = cfg.ppo.clipping_parameter
    vf_coeff = cfg.ppo.vf_coeff
    entropy_coeff = cfg.ppo.entropy_coeff

    n_collections = max_time_steps // (num_workers*time_steps)

    units = 800

    assert num_workers % n_minibatch == 0
    worker_per_minibatch = num_workers // n_minibatch
    minibatch_ids = [(i, i + worker_per_minibatch) for i in range(0, num_workers, worker_per_minibatch)]



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
    

    memory = []
    policy_state, new_policy_state = None, None
    driver = DynamicStepDriver(env, policy=policy, observers=[memory.append], num_steps=time_steps*num_workers)
    from metrics import AverageReward
    episodic_return_fn = AverageReward(num_envs=num_workers)
    k=0
    for collect_n in range(1, n_collections + 1):
        k+=1
        policy_state = new_policy_state
        policy_state = None
        observations, actions, rewards, dones, probs, values, new_policy_state = collect_sequences(driver, memory, policy, None, None)
        
        render = True if k % 10 == 0 else False
        if render:
            render_obs(observations)
        
        episodic_return = episodic_return_fn(rewards, dones)
        #for t in episodic_return.keys():
            #wandb.log({'step': t + (k-1)*time_steps, 'episodic_return':  episodic_return[t]})
        

        advantages = compute_truncated_advantages(discount, gae_parameter, rewards, values, dones)
        advantages = tf.stop_gradient(advantages)
        returns = advantages + values[:, :-1]
        returns = tf.stop_gradient(returns)

        for epoch in range(epochs):
            for i, j in minibatch_ids:
                with tf.GradientTape() as tape:
                    cell_state = None if policy_state is None else policy_state[i:j]
                    _, new_probs, new_values, entropy, _ = actor_critic_network(observations[i:j], dones[i:j], cell_state, actions[i:j], training=True)
                    
                    mb_advantages = (advantages[i:j] - tf.reduce_mean(advantages[i:j], axis=None)) / (tf.math.reduce_std(advantages[i:j], axis=None) + 1e-8)
                    
                    mb_advantages = tf.stop_gradient(mb_advantages)

                    ppo_l = ppo_clip_loss(mb_advantages, probs[i:j], new_probs, clipping_parameter)
                    value_function_l = value_function_loss(new_values, returns[i:j])
                    entropy_l = entropy_loss(entropy)
                    loss = ppo_l + vf_coeff*value_function_l - entropy_coeff*entropy_l
                    print(f'total_loss: {loss} - ppo_loss: {ppo_l} - value_loss: {vf_coeff*value_function_l} - entropy: {entropy_coeff*entropy_l}')
                    grads = tape.gradient(loss, actor_critic_network.trainable_weights)
                    optimizer.apply_gradients(zip(grads, actor_critic_network.trainable_weights))
                    
    tf.keras.models.save_model(
        actor_critic_network,
        f'models/{game_name}',
        overwrite=True,
        include_optimizer=False,
        save_format='tf',
        signatures=None,
        options=None,
        save_traces=True
    )

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', help='Name of the Atari game')
    parser.add_argument('cfg_path', help='YAML file containing arguments and hyperparameters')
    parser.add_argument('--seed', help='Random seed, for reproducibility', default=42)
    
    args = parser.parse_args()



    tf.random.set_seed(args.seed)
    np.random.seed(args.seed)
    tf.keras.utils.set_random_seed(args.seed)
    tf.config.experimental.enable_op_determinism()
    tf_agents.system.multiprocessing.handle_main(main, [args])





