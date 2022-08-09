from tf_agents.policies import TFPolicy
from tf_agents.trajectories import policy_step, StepType

import tensorflow as tf

class AtariPolicy(TFPolicy):
    def __init__(self, time_step_spec, action_spec, policy_state_spec, info_spec, actor_critic_network):
        """
        TFPolicy for collecting experience for Atari games using PPO.
        
        Parameters
        ----------
        time_step_spec: tf_agents.spec.BoundedTensorSpec
            Specification of the time step
        action_spec: tf_agents.spec.BoundedTensorSpec
            Specification of the action
        policy_state_spec: tf_agents.spec.BoundedTensorSpec
            Specification of the policy state
        info_spec: tf_agents.spec.BoundedTensorSpec
            Specification of the log probability and value function tensors
        actor_critic_network: actor_critic.AtariActorCriticNetwork
            Actor-critic network for Atari games
        """
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

        dones = tf.expand_dims(tf.math.equal(time_step.step_type, StepType.LAST), axis=1)
        dones = tf.cast(dones, tf.float32)

        actions, log_probs, values, _, new_policy_state = self.actor_critic_network(observations, dones, cell_states=policy_state, training=False)
        log_probs = tf.squeeze(log_probs, axis=1)
        values = tf.squeeze(values, axis=1)
        actions = tf.squeeze(actions, axis=1)

        return policy_step.PolicyStep(actions, new_policy_state, info=(log_probs, values))



