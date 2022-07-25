import tensorflow as tf
from tf_agents.trajectories import StepType

class PPOAgent:
    def __init__(self, num_workers, env, driver, policy):
        self.num_workers = num_workers
        self.env = env
        self.driver = driver
        self.policy = policy
        self.memory = []


    def compute_advantages(self, discount, return_factor, rewards, values, dones):
        non_terminal = 1.0 - dones
        deltas = rewards + discount*non_terminal*values[:, 1:] - values[:, :-1]

        advantages = [deltas[:, -1]]
        for i in reversed(range(deltas.shape[1] - 1)):
            advantages.append(deltas[:, i] + discount*return_factor*non_terminal[:, i]*advantages[-1])
        advantages = tf.stack(advantages[::-1], axis=1)
        return advantages

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

        observations = tf.stack([batched_step.observation for batched_step in self.memory], axis=1)
        rewards = tf.stack([batched_step.reward for batched_step in self.memory], axis=1)
        dones = tf.cast(tf.stack([batched_step.step_type == StepType.LAST for batched_step in self.memory] + [tf.squeeze(last_done, 1)], axis=1), dtype=tf.float32)
        actions = tf.stack([batched_step.action for batched_step in self.memory], axis=1)
        probs = tf.stack([batched_step.policy_info[0] for batched_step in self.memory], axis=1)
        values = tf.stack([batched_step.policy_info[1] for batched_step in self.memory] + [last_value], axis=1)

        self.memory.clear()
        return observations, actions, rewards, dones[:, 1:], probs, values, last_policy_state

    def train(self, optimizer, num_epochs, n_collections, discount, gae_parameter, clipping_parameter, vf_coeff, entropy_coeff):
        from metrics import AverageReward
        episodic_return_fn = AverageReward(num_envs=self.num_workers)
        k=0
        for collect_n in range(1, n_collections + 1):
            k+=1
            policy_state = new_policy_state
            policy_state = None
            observations, actions, rewards, dones, probs, values, new_policy_state = self.collect_experiences(policy_state)
            
            episodic_return = episodic_return_fn(rewards, dones)
            #for t in episodic_return.keys():
                #wandb.log({'step': t + (k-1)*time_steps, 'episodic_return':  episodic_return[t]})
            

            advantages = self.compute_advantages(discount, gae_parameter, rewards, values, dones)
            advantages = tf.stop_gradient(advantages)
            returns = advantages + values[:, :-1]
            returns = tf.stop_gradient(returns)
            
            minibatch_ids = self.compute_minibatch_ids()

            for epoch in range(num_epochs):
                for idx in minibatch_ids:
                    with tf.GradientTape() as tape:
                        sample_policy_state = tf.gather(policy_state, idx)
                        sample_observations = tf.gather(observations, idx)
                        sample_dones = tf.gather(dones, idx)
                        sample_actions = tf.gather(actions, idx)
                        sample_advantages = tf.gather(advantages, idx)
                        sample_probs = tf.gather(probs, idx)
                        sample_returns = tf.gather(returns, idx)


                        cell_state = None if policy_state is None else sample_policy_state
                        _, new_probs, new_values, entropy, _ = self.policy.actor_critic_network(sample_observations, sample_dones, cell_state, sample_actions, training=True)
                        
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
        