from environment import get_env

import functools
from tf_agents.trajectories import StepType
from tf_agents.environments import ParallelPyEnvironment, TFPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
import tf_agents
import tensorflow as tf

from actor_critic import AtariActorCriticNetwork, AtariNetwork
from atari_policy import AtariPolicy

def collect_sequences(env, policy, num_workers, time_steps):
    memory = []
    driver = DynamicEpisodeDriver(env, policy=policy, observers=[memory.append], num_episodes=num_workers)

    driver.run(maximum_iterations=time_steps, policy_state=None)
    observations = tf.stack([batched_step.observation for batched_step in memory], axis=1)
    rewards = tf.stack([batched_step.reward for batched_step in memory], axis=1)
    dones = tf.stack([batched_step.step_type == StepType.LAST for batched_step in memory], axis=1)
    actions = tf.stack([batched_step.action for batched_step in memory], axis=1)

    return observations, actions, rewards, dones
    
def compute_truncated_advantages(discount, return_factor, rewards, values):
    advantages = []
    advantage = 0.0
    for i in range(len(rewards)-1, -1, -1):
        advantage = rewards[i] +  discount*values[i+1] - values[i] + discount*return_factor*advantage
        advantages.append(advantage)
    return tf.constant(advantages)


def main(args):
    num_workers = 8
    time_steps = 100
    game_name = 'MontezumaRevenge'
    units = 800
    policy_state_spec = tf.TensorSpec(shape=(units), dtype=tf.float32)


    env_constructor = functools.partial(get_env, game_name)

    envs = [env_constructor for _ in range(num_workers)]
    env = TFPyEnvironment(ParallelPyEnvironment(env_constructors=envs))

    atari_network = AtariNetwork()
    actor_critic_network = AtariActorCriticNetwork(atari_network, 18)
    policy = AtariPolicy(time_step_spec=env.time_step_spec(), action_spec=env.action_spec(), policy_state_spec=policy_state_spec, actor_critic_network=actor_critic_network)
    
    observations, actions, rewards, dones = collect_sequences(env, policy, num_workers, time_steps)

if __name__ == '__main__':
    tf_agents.system.multiprocessing.handle_main(main)

