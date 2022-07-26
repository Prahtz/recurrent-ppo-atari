from random import shuffle
import tensorflow as tf
from tf_agents.trajectories import StepType
import utils

class PPOAgent:
    def __init__(self, num_workers, driver, policy):
        self.num_workers = num_workers
        self.driver = driver
        self.env = driver._env
        self.policy = policy
        self.memory = []
        self.driver._observers = [self.memory.append]


    def compute_advantages(self, discount, return_factor, rewards, values, dones):
        non_terminal = 1.0 - dones
        deltas = rewards + discount*non_terminal*values[:, 1:] - values[:, :-1]

        advantages = [deltas[:, -1]]
        for i in reversed(range(deltas.shape[1] - 1)):
            advantages.append(deltas[:, i] + discount*return_factor*non_terminal[:, i]*advantages[-1])
        advantages = tf.stack(advantages[::-1], axis=1)
        return advantages
    
    def compute_minibatch_ids(self, num_minibatch, use_rnn=False):
        horizon = self.driver._num_steps // self.num_workers
        step = horizon // num_minibatch

        assert self.driver._num_steps % step == 0
        ids = [[i for i in range(horizon)] for _ in range(self.num_workers)]
        for i in range(len(ids)):
            if not use_rnn:
                shuffle(ids[i])
            ids[i] = [ids[i][j:j+step] for j in range(0, horizon, step)]
        return [[ids[j][i] for j in range(self.num_workers)] for i in range(num_minibatch)]

    def ppo_clip_loss(self, advantages, old_logprobs, logprobs, clip_range):
        prob_ratio = tf.exp(logprobs - old_logprobs)

        loss = tf.maximum(-advantages * prob_ratio, -advantages * tf.clip_by_value(prob_ratio, 1 - clip_range, 1 + clip_range))
        loss = tf.reduce_mean(loss, axis=None)
        return loss

    def value_function_loss(self, values, returns):
        return tf.reduce_mean((values - returns)**2, axis=None)

    def entropy_loss(self, entropy):
        return tf.reduce_mean(entropy, axis=None)

    def collect_experiences(self, policy_state):
        last_time_step, last_policy_state = self.driver.run(policy_state=policy_state)
        last_observation, last_done = last_time_step.observation, tf.equal(last_time_step.step_type, StepType.LAST)
        last_observation, last_done = tf.expand_dims(last_observation, axis=1), tf.expand_dims(last_done, axis=1)
        _, _, last_value, _, _ = self.policy.actor_critic_network(last_observation, last_done, cell_states=last_policy_state, training=False)
        last_value = tf.squeeze(last_value, axis=1)

        experiences = {}
        experiences['observations']= tf.stack([batched_step.observation for batched_step in self.memory], axis=1)
        experiences['rewards'] = tf.stack([batched_step.reward for batched_step in self.memory], axis=1)
        experiences['dones'] = tf.cast(tf.stack([batched_step.step_type == StepType.LAST for batched_step in self.memory[1:]] + [tf.squeeze(last_done, 1)], axis=1), dtype=tf.float32)
        experiences['actions'] = tf.stack([batched_step.action for batched_step in self.memory], axis=1)
        experiences['probs'] = tf.stack([batched_step.policy_info[0] for batched_step in self.memory], axis=1)
        experiences['values'] = tf.stack([batched_step.policy_info[1] for batched_step in self.memory] + [last_value], axis=1)
        experiences['last_policy_state'] = last_policy_state

        self.memory.clear()
        return experiences

    def train(self, optimizer, num_epochs, num_minibatch, n_collections, discount, gae_parameter, clipping_parameter, vf_coeff, entropy_coeff):
        from metrics import AverageReward
        policy_state, new_policy_state = None, None
        episodic_return_fn = AverageReward(num_envs=self.num_workers)
        for collect_n in range(n_collections):
            policy_state = new_policy_state
            policy_state = None
            experiences = self.collect_experiences(policy_state)
            
            episodic_return = episodic_return_fn(experiences['rewards'], experiences['dones'])
            #for t in episodic_return.keys():
                #wandb.log({'step': t + (k-1)*time_steps, 'episodic_return':  episodic_return[t]})
            if collect_n % 100 == 0:
                utils.render_obs(experiences['observations'])
            

            advantages = self.compute_advantages(discount, gae_parameter, experiences['rewards'], experiences['values'], experiences['dones'])
            advantages = tf.stop_gradient(advantages)
            returns = advantages + experiences['values'][:, :-1]
            returns = tf.stop_gradient(returns)
            
            minibatch_ids = self.compute_minibatch_ids(num_minibatch)

            for _ in range(num_epochs):
                for idx in minibatch_ids:
                    with tf.GradientTape() as tape:
                        sample_policy_state = None if policy_state is None else tf.gather(policy_state, idx)
                        sample_observations = tf.gather(experiences['observations'], idx, batch_dims=1)
                        sample_dones = tf.gather(experiences['dones'], idx, batch_dims=1)
                        sample_actions = tf.gather(experiences['actions'], idx, batch_dims=1)
                        sample_probs = tf.gather(experiences['probs'], idx, batch_dims=1)
                        sample_advantages = tf.gather(advantages, idx, batch_dims=1)
                        sample_returns = tf.gather(returns, idx, batch_dims=1)

                        _, new_probs, new_values, entropy, _ = self.policy.actor_critic_network(sample_observations, 
                                                                                                sample_dones, 
                                                                                                sample_policy_state, 
                                                                                                sample_actions, 
                                                                                                training=True)
                        
                        sample_advantages = (sample_advantages - tf.reduce_mean(sample_advantages, axis=None)) / (tf.math.reduce_std(sample_advantages, axis=None) + 1e-8)
                        
                        sample_advantages = tf.stop_gradient(sample_advantages)

                        ppo_l = self.ppo_clip_loss(sample_advantages, sample_probs, new_probs, clipping_parameter)
                        value_function_l = self.value_function_loss(new_values, sample_returns)
                        entropy_l = self.entropy_loss(entropy)
                        loss = ppo_l + vf_coeff*value_function_l - entropy_coeff*entropy_l
                        
                        print(f'total_loss: {loss} - ppo_loss: {ppo_l} - value_loss: {vf_coeff*value_function_l} - entropy: {entropy_coeff*entropy_l}')
                        grads = tape.gradient(loss, self.policy.actor_critic_network.trainable_weights)
                        optimizer.apply_gradients(zip(grads, self.policy.actor_critic_network.trainable_weights))
                        
        tf.keras.models.save_model(
            self.policy.actor_critic_network,
            f'models/{self.env.game_name}',
            overwrite=True,
            include_optimizer=False,
            save_format='tf',
            signatures=None,
            options=None,
            save_traces=True
        )
        