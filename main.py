from environment import get_env

import functools
from tf_agents.trajectories import StepType
from tf_agents.environments import ParallelPyEnvironment, TFPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.drivers.dynamic_step_driver import DynamicStepDriver
import tf_agents
import tensorflow as tf
from keras import optimizers
import numpy as np
import tensorflow_probability as tfp

from actor_critic import AtariActorCriticNetwork, AtariNetwork
from atari_policy import AtariPolicy

class Transition:
    def __init__(self):
        self.time_step = None
        self.policy_step = None
        self.next_time_step = None
    def replace(self, transition):
        self.time_step, self.policy_step, self.next_time_step = transition

def collect_sequences(driver, memory, transition, policy, policy_state, time_steps, render=False):
    
    driver.run(maximum_iterations=time_steps, policy_state=policy_state)

    last_observation, last_done = transition.next_time_step.observation, tf.equal(transition.next_time_step.step_type, StepType.LAST)
    last_observation, last_done = tf.expand_dims(last_observation, axis=1), tf.expand_dims(last_done, axis=1)
    _, _, last_value, _, _ = policy.actor_critic_network(last_observation, last_done, cell_states=transition.policy_step.state, training=True)
    last_value = tf.squeeze(last_value, axis=1)
    
    observations = tf.stack([batched_step.observation for batched_step in memory], axis=1)
    rewards = tf.stack([batched_step.reward for batched_step in memory], axis=1)
    dones = tf.cast(tf.stack([batched_step.step_type == StepType.LAST for batched_step in memory], axis=1), dtype=tf.float32)
    actions = tf.stack([batched_step.action for batched_step in memory], axis=1)
    probs = tf.stack([batched_step.policy_info[0] for batched_step in memory], axis=1)
    values = tf.stack([batched_step.policy_info[1] for batched_step in memory] + [last_value], axis=1)
    if render:
        render_obs(observations)
    return observations, actions, rewards, dones, probs, values, transition.policy_step.state
    
def compute_truncated_advantages(discount, return_factor, rewards, values, dones):
    #Need hangling of terminal states
    #Need last GAE(\lambda)
    non_terminal = 1.0 - dones
    deltas = rewards + discount*tf.multiply(non_terminal, values[:, 1:]) - values[:, :-1]
    advantages = [deltas[:, -1]]
    for i in range(deltas.shape[1] - 2, -1, -1):
        advantages.append(deltas[:, i] + discount*return_factor*tf.multiply(non_terminal[:, i], advantages[-1]))
    advantages = tf.stack(advantages[::-1], axis=1)
    """mean = tf.math.reduce_mean(advantages, axis=1)
    mean = tf.expand_dims(mean, axis=1)
    mean = tf.repeat(mean, advantages.shape[1], axis=1)

    std = tf.math.reduce_std(advantages, axis=1) + 1e-7
    std = tf.expand_dims(std, axis=1)
    std = tf.repeat(std, advantages.shape[1], axis=1)

    advantages = (advantages - mean) / std"""
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
        frame = obs[i][:,:,:3].numpy()
        frame = cv.cvtColor(frame, cv.COLOR_RGB2BGR)
        cv.imshow('env0', frame)
        cv.waitKey(delay=2)

def main(args):
    num_workers = 16
    time_steps = 100
    game_name = 'Pong'
    units = 800
    discount = 0.99
    return_factor = 0.95
    clip_range = 0.1
    value_function_coeff = 0.5
    entropy_coeff = 0.01

    epochs = 4
    n_minibatch = 4

    policy_state_spec = tf.TensorSpec(shape=(units), dtype=tf.float32)
    info_spec = (tf.TensorSpec(shape=(18), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32))

    env_constructor = functools.partial(get_env, game_name)

    envs = [env_constructor for _ in range(num_workers)]
    env = TFPyEnvironment(ParallelPyEnvironment(env_constructors=envs))
    env.seed([42]*num_workers)
    optimizer = optimizers.Adam(learning_rate=2e-5)

    atari_network = AtariNetwork()
    actor_critic_network = AtariActorCriticNetwork(atari_network, 18)
    policy = AtariPolicy(time_step_spec=env.time_step_spec(), 
                         action_spec=env.action_spec(), 
                         policy_state_spec=policy_state_spec,
                         info_spec=info_spec,
                         actor_critic_network=actor_critic_network)
    memory = []
    transition = Transition()
    policy_state = None
    driver = DynamicStepDriver(env, policy=policy, observers=[memory.append], transition_observers=[transition.replace], num_steps=100000)
    while True:
        observations, actions, rewards, dones, probs, values, policy_state = collect_sequences(driver, memory, transition, policy, policy_state, time_steps, render=True)
        #print(observations.shape, actions.shape, rewards.shape, dones.shape, probs.shape, values.shape, policy_state.shape)
        memory.clear()
        advantages = compute_truncated_advantages(discount, return_factor, rewards, values, dones)
        returns = advantages + values[:, :-1]

        assert num_workers % n_minibatch == 0
        worker_per_minibatch = num_workers // n_minibatch
        minibatch_ids = [(i, i + worker_per_minibatch) for i in range(0, num_workers, worker_per_minibatch)]
        for epoch in range(epochs):
            for i, j in minibatch_ids:
                with tf.GradientTape() as tape:
                    _, new_probs, new_values, entropy, _ = actor_critic_network(observations[i:j], dones[i:j], policy_state[i:j], actions[i:j], training=True)

                    mb_advantages = (advantages[i:j] - tf.reduce_mean(advantages[i:j], axis=None)) / (tf.math.reduce_std(advantages[i:j], axis=None) + 1e-8)
                    
                    ppo_l = ppo_clip_loss(mb_advantages, probs[i:j], new_probs, clip_range)
                    value_function_l = value_function_loss(new_values, returns[i:j])
                    entropy_l = entropy_loss(entropy)
                    loss = ppo_l + value_function_coeff*value_function_l - entropy_coeff*entropy_l
                print(loss)
                grads = tape.gradient(loss, actor_critic_network.trainable_weights)
                optimizer.apply_gradients(zip(grads, actor_critic_network.trainable_weights))

if __name__ == '__main__':
    tf.random.set_seed(42)
    np.random.seed(42)
    tf_agents.system.multiprocessing.handle_main(main)





