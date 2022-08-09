import argparse
from actor_critic import AtariActorCriticNetwork, AtariNetwork
from atari_policy import AtariPolicy
from environment import get_env
from config import config
from tf_agents.environments import TFPyEnvironment, ParallelPyEnvironment
from tf_agents.drivers.dynamic_episode_driver import DynamicEpisodeDriver
from tf_agents.trajectories import StepType
import tensorflow as tf
import functools
import tf_agents
import wandb
from tqdm import tqdm
import os

def evaluate(args):
    args = args[0]
    cfg_path = args.cfg_path
    cfg = config.get_cfg_defaults()
    cfg.merge_from_file(cfg_path)
    cfg.freeze()

    game_name = args.game_name
    checkpoint_path = args.checkpoint_path
    run_name = game_name + '_' + cfg_path.split('/')[-1].split('.')[0]
    api = wandb.Api()
    runs = api.runs('prahtz/recurrent-ppo-atari')

    average_episodic_return = None
    for c_run in runs:
        if c_run.name == run_name:
            samples = c_run.scan_history(keys=['episodic_return'])
            episodic_returns = [sample['episodic_return'] for sample in samples][-100:]
            average_episodic_return = sum(episodic_returns) / len(episodic_returns)
            break
    assert average_episodic_return is not None

    share_params = cfg.model.share_params
    units = cfg.model.rnn_units
    conv_net = cfg.model.conv_net
    use_rnn = cfg.model.use_rnn
    memory_size = cfg.model.memory_size
    
    num_episodes = 100
    num_envs = 10
    env_constructor = functools.partial(get_env, game_name, use_rnn, True, False)
    envs = [env_constructor for _ in range(num_envs)]
    env = TFPyEnvironment(ParallelPyEnvironment(env_constructors=envs))

    action_spec = env.action_spec()

    if share_params:
        policy_state_spec = (tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16))
    else:
        policy_state_spec = ((tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16)), 
                            (tf.TensorSpec(shape=(units), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.int16)))
    info_spec = (tf.TensorSpec(shape=(action_spec.maximum + 1), dtype=tf.float32), tf.TensorSpec(shape=(1), dtype=tf.float32))

    policy_encoder = AtariNetwork(conv_net=conv_net, use_rnn=use_rnn, rnn_units=units, memory_size=memory_size)
    if share_params:
        value_encoder = None
    else:
        value_encoder = AtariNetwork(conv_net=conv_net, use_rnn=use_rnn, rnn_units=units, memory_size=memory_size)

    actor_critic_network = AtariActorCriticNetwork(n_actions=action_spec.maximum + 1, 
                                                   policy_encoder=policy_encoder, 
                                                   value_encoder=value_encoder)

    policy = AtariPolicy(time_step_spec=env.time_step_spec(), 
                         action_spec=action_spec, 
                         policy_state_spec=policy_state_spec,
                         info_spec=info_spec,
                         actor_critic_network=actor_critic_network)

    obs = tf.zeros(shape=[num_envs, 1] + env.observation_spec().shape)
    dones = tf.zeros(shape=(num_envs, 1))
    actor_critic_network(obs, dones, cell_states=policy.get_initial_state(num_envs), training=False)
    actor_critic_network.load_weights(checkpoint_path)

    assert num_episodes % num_envs == 0
    results = []
    for _ in tqdm(range(0, num_episodes, num_envs)):
        memory = []
        driver = DynamicEpisodeDriver(env, policy, observers=[memory.append], num_episodes=num_envs)
        last_time_step, _ = driver.run()
        last_done = tf.equal(last_time_step.step_type, StepType.LAST)
        last_done = tf.expand_dims(last_done, axis=1)

        rewards = tf.cast(tf.stack([batched_step.reward for batched_step in memory], axis=1), dtype=tf.float32)
        dones = tf.cast(tf.stack([batched_step.step_type == StepType.LAST for batched_step in memory[1:]] + [tf.squeeze(last_done, 1)], axis=1), dtype=tf.float32)
        
        for i in range(num_envs):
            j = 0
            while j < dones.shape[1] - 1 and not dones[i, j]:
                j+=1
            j+=1
            
            episodic_return = tf.reduce_sum(rewards[i, :j])
            results.append(episodic_return)
        memory.clear()
            

    real_average_episodic_return = sum(results) / len(results)
    real_average_episodic_return = real_average_episodic_return.numpy().item()
    os.makedirs('results/', exist_ok=True)
    with open('results/out.txt', 'a') as f_out:
        f_out.write(f'{run_name} avg_ep_rtn: {average_episodic_return} - real_avg_ep_rtn: {real_average_episodic_return}\n')

    
            

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('game_name', help='Name of the Atari game')
    parser.add_argument('cfg_path', help='YAML file containing arguments and hyperparameters')
    parser.add_argument('checkpoint_path', help='Path of the file containing the weights of the model')
    parser.add_argument('--render', help='If set, render experiences every 100 collections', action='store_true')

    args = parser.parse_args()
    tf_agents.system.multiprocessing.handle_main(evaluate, [args])