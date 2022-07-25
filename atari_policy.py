from tf_agents.policies import TFPolicy
from tf_agents.trajectories import policy_step, StepType

import tensorflow as tf
import tensorflow_probability as tfp

class AtariPolicy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, policy_state_spec, info_spec, actor_critic_network):
        super().__init__(time_step_spec, action_spec, policy_state_spec=policy_state_spec, info_spec=info_spec)
        self.action_spec = action_spec
        self.actor_critic_network = actor_critic_network

    def _distribution(self, time_step):
        pass

    def _variables(self):
        return ()

    def _action(self, time_step, policy_state, seed):
        observations = time_step.observation
        observations = tf.expand_dims(observations, axis=1)

        if type(policy_state) is tuple:
            policy_state = None

        dones = tf.expand_dims(tf.math.equal(time_step.step_type, StepType.LAST), axis=1)
        dones = tf.cast(dones, tf.float32)

        #probs, values, new_policy_state = self.actor_critic_network(observations, dones, cell_states=policy_state, training=False)
        actions, log_probs, values, _, new_policy_state = self.actor_critic_network(observations, dones, cell_states=policy_state, training=False)
        log_probs = tf.squeeze(log_probs, axis=1)
        values = tf.squeeze(values, axis=1)
        actions = tf.squeeze(actions, axis=1)

        #actions = tfp.distributions.Categorical(probs=probs, dtype=tf.int64).sample()

        #action_probs = tf.gather_nd(params=probs, indices=[a for a in zip(range(observations.shape[0]), list(actions))])
        return policy_step.PolicyStep(actions, new_policy_state, info=(log_probs, values))



