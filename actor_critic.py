import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

class AtariGRU(tf.keras.layers.Layer):
    def __init__(self, units, memory_size=16):
        # RNN special layer, if episode ends or memory limit reached reset cell states
        super().__init__()
        self.units = units
        self.memory_size = memory_size
        weights_init = tf.keras.initializers.Orthogonal(gain=1.0)
        self.rnn = tf.keras.layers.GRU(units=units, stateful=False, return_state=True, return_sequences=True, kernel_initializer=weights_init)
    def call(self, inputs, dones, cell_states, training=True):
        # inputs.shape = [batch_size, time_steps, units]
        state, step = cell_states
        time_steps = inputs.shape[1]
        x_steps = tf.split(inputs, time_steps, axis=1)
        sequences = []
        for i in range(time_steps):
            hidden_state, state = self.rnn(inputs=x_steps[i], initial_state=state, training=training)
            step += 1

            reset_dones = tf.equal(dones[:, i], 1)
            reset_memory = tf.squeeze(tf.equal(tf.math.mod(step, self.memory_size), 0), axis=1)
            reset = tf.logical_or(reset_dones, reset_memory)
            indices = tf.where(reset)
            if len(indices):
                initial_cells = tf.zeros(shape=(len(indices), self.units))
                state = tf.tensor_scatter_nd_update(state, indices, initial_cells)
                step = tf.tensor_scatter_nd_update(tf.squeeze(step, axis=1), indices, tf.zeros(shape=(len(indices)), dtype=tf.int16))
                step = tf.expand_dims(step, 1)
            sequences.append(hidden_state)
        sequences = tf.concat(sequences, axis=1)
        return sequences, (state, step)
            
class AtariNetwork(tf.keras.Model):
    def __init__(self, conv_net=True, use_rnn=True, rnn_units=64, memory_size=16):
        super().__init__()
        self.use_rnn = use_rnn
        self.rnn_units = rnn_units
        self.memory_size = memory_size
        
        kernel_init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))

        if conv_net:
            self.act_fn = tf.keras.layers.ReLU()
            self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, kernel_initializer=kernel_init)
            self.c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, kernel_initializer=kernel_init)
            self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, kernel_initializer=kernel_init)
            self.flat = tf.keras.layers.Flatten()
            self.dense = tf.keras.layers.Dense(units=512, activation='relu')
            self.network = tf.keras.layers.TimeDistributed(tf.keras.Sequential([self.c1, self.act_fn, self.c2, self.act_fn, self.c3, self.act_fn, self.flat, self.dense]))
        else:
            self.flat = tf.keras.layers.Flatten()
            self.fc1 = tf.keras.layers.Dense(units=64, kernel_initializer=kernel_init, activation='tanh')
            self.fc2 = tf.keras.layers.Dense(units=64, kernel_initializer=kernel_init, activation='tanh')
            self.network = tf.keras.layers.TimeDistributed(tf.keras.Sequential([self.flat, self.fc1, self.fc2]))
            
        if use_rnn:
            self.atari_rnn = AtariGRU(units=rnn_units, memory_size=memory_size)
            self.concat = tf.keras.layers.Concatenate(axis=2)

    def call(self, inputs, dones, cell_states=None, training=True):
        x = self.network(inputs, training=training)
        if self.use_rnn:
            y, cell_states = self.atari_rnn(inputs=x, dones=dones, cell_states=cell_states, training=training)
            x = self.concat([x, y], training=training)
        return x, cell_states

class AtariActorCriticNetwork(tf.keras.Model):
    def __init__(self, n_actions, policy_encoder, value_encoder=None):
        super().__init__()
        #self.flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.policy_encoder = policy_encoder

        if value_encoder is None:
            self.shared_params = True
            self.value_encoder = self.policy_encoder
        else:
            self.shared_params = False
            self.value_encoder = value_encoder

        critic_init = tf.keras.initializers.Orthogonal(gain=1.0)
        actor_init = tf.keras.initializers.Orthogonal(gain=0.01)
        self.actor = tf.keras.layers.Dense(units=n_actions, kernel_initializer=actor_init)
        self.critic = tf.keras.layers.Dense(units=1, kernel_initializer=critic_init)

        self.memory_size = policy_encoder.memory_size

    def call(self, inputs, dones, cell_states=None, action=None, training=True):
        inputs = tf.cast(inputs, tf.float32) / 255.0
        #inputs = self.flat(inputs, training=training)


        policy_states = cell_states
        if self.shared_params:
            value_states = policy_states
        else:
            policy_states, value_states = policy_states

        x, new_policy_states = self.policy_encoder(inputs, dones=dones, cell_states=policy_states, training=training)
        logits = self.actor(x, training=training)

        if not self.shared_params:
            x, new_value_states = self.value_encoder(inputs, dones=dones, cell_states=value_states, training=training)
            new_cell_states = (new_policy_states, new_value_states)
        else:
            new_cell_states = new_policy_states

        values = self.critic(x, training=training)
        values = tf.squeeze(values, axis=2)

        distribution = tfp.distributions.Categorical(logits=logits, dtype=tf.int64)
        if action is None:
            action = distribution.sample()
        return action, distribution.log_prob(action), values, distribution.entropy(), new_cell_states
