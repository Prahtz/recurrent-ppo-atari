from tf_agents.policies import TFPolicy
from tf_agents.trajectories import policy_step

import tensorflow as tf
import tensorflow_probability as tfp

class AtariPolicy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, policy_state_spec, actor_critic_network):
        super().__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec)
        self.action_spec = action_spec
        self.actor_critic_network = actor_critic_network

    def _distribution(self, time_step):
        pass

    def _variables(self):
        return ()

    def _action(self, time_step, policy_state, seed):
        if type(policy_state) is tuple:
            policy_state = None
        observations = time_step.observation
        observations = tf.expand_dims(observations, axis=1)
        probs, values, new_policy_state = self.actor_critic_network(observations, cell_states=policy_state, training=False)
        actions = tfp.distributions.Categorical(probs=probs, dtype=tf.int64).sample()
        actions = tf.squeeze(actions, axis=1)
        return policy_step.PolicyStep(actions, new_policy_state)


