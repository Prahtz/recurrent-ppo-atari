import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np

"""class AtariGRU(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.steps = 0
        self.initial_cell = tf.zeros(shape=(units))
        self.rnn = tf.keras.layers.GRU(units=units, return_sequences=True, return_state=True)
        
    
    def call(self, inputs, dones, cell_states, training=True):
        #shape: (num_envs, batch_size, units)
        #dones: (num_envs, batch_size)

        if cell_states is None:
            cell_states = self.initial_cell
            num_envs = inputs.shape[0]
            cell_states = tf.expand_dims(cell_states, axis=0)
            cell_states = tf.repeat(cell_states, num_envs, axis=0)
        
        inputs = tf.split(inputs, inputs.shape[1], axis=1)
        sequences = []
        for x, done in zip(inputs, tf.transpose(dones)):
            indices = tf.where(done)
            #print(cell_states[0][:4])
            if len(indices):
                initial_cells = tf.expand_dims(self.initial_cell, axis=0)
                initial_cells = tf.repeat(initial_cells, len(indices), axis=0)
                cell_states = tf.tensor_scatter_nd_update(cell_states, indices, initial_cells)
            hidden_state, cell_states = self.rnn(inputs=x, initial_state=cell_states, training=training)
            
            sequences.append(hidden_state)
        sequences = tf.concat(sequences, axis=1)
        return sequences, cell_states

class AtariNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=4, activation='tanh')
        self.c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='tanh')
        self.c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='tanh')
        self.fc1 = tf.keras.layers.Flatten()
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.cnn = tf.keras.models.Sequential([self.c1, self.c2, self.c3, self.fc1, self.fc2])
        self.cnn = tf.keras.layers.TimeDistributed(self.cnn)
        self.gru = AtariGRU(units=512)
        self.concatenate = tf.keras.layers.Concatenate(axis=2)

    def call(self, inputs, dones, cell_states=None, training=True):
        x = self.cnn(inputs, training=training)

        sequences, cell_states = self.gru(x, dones, cell_states=cell_states, training=training)
        x = self.concatenate([x, sequences], training=training)
        return x, cell_states

class AtariNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='tanh', kernel_initializer='orthogonal')
        self.c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='tanh', kernel_initializer='orthogonal')
        self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='tanh', kernel_initializer='orthogonal')
        self.fc1 = tf.keras.layers.Flatten()
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.cnn = tf.keras.models.Sequential([self.c1, self.c2, self.c3, self.fc1, self.fc2])
        self.cnn = tf.keras.layers.TimeDistributed(self.cnn)
        #self.gru = AtariGRU(units=800)
        #self.concatenate = layers.Concatenate(axis=2)
    def call(self, inputs, dones, cell_states=None, training=True):
        #inputs must be of shape [num_envs, batch_size, observation_spec]  
        num_envs, batch_size = inputs.shape[:2]
        #inputs = tf.reshape(inputs, shape=[num_envs*batch_size] + inputs.shape[2:])    
        x = self.cnn(inputs, training=training)

        #x = tf.reshape(x, shape=(num_envs, batch_size, x.shape[-1]))
        #sequences, cell_states = self.gru(x, dones, cell_states=cell_states, training=training)
        #x = self.concatenate([x, sequences], training=training)
        return x, cell_states"""



"""class AtariGRU(tf.keras.layers.Layer):
    def __init__(self, units, memory_size=16):
        # RNN special layer, if episode ends or memory limit reached reset cell states
        super().__init__()
        self.units = units
        self.memory_size = memory_size
        self.rnn = tf.keras.layers.GRU(units=units, stateful=False, return_state=True, return_sequences=True)
    def call(self, inputs, dones, cell_states, training=True):
        # inputs.shape = [batch_size, time_steps, units]
 
        num_envs, time_steps = inputs.shape[0], inputs.shape[1]
        if cell_states is None:
            cell_states = tf.zeros(shape=(num_envs, self.units))
            self.cell_steps = tf.zeros(shape=(num_envs,))
            print('reset:', self.cell_steps.shape)
        x_steps = tf.split(inputs, time_steps, axis=1)
        sequences = []
        for i in range(time_steps):
            hidden_state, cell_states = self.rnn(inputs=x_steps[i], initial_state=cell_states, training=training)
            self.cell_steps += 1
            reset_dones = tf.equal(dones[:, i], 1)
            reset_memory = tf.equal(tf.math.mod(self.cell_steps, self.memory_size), 0)
            reset = tf.logical_or(reset_dones, reset_memory)
            indices = tf.where(reset)
            if len(indices):
                initial_cells = tf.zeros(shape=(len(indices), self.units))
                cell_states = tf.tensor_scatter_nd_update(cell_states, indices, initial_cells)

                indices_dones = tf.where(reset_dones)
                if len(indices_dones):
                    self.cell_steps = tf.tensor_scatter_nd_update(self.cell_steps, indices_dones, tf.zeros(shape=(len(indices_dones),)))
            sequences.append(hidden_state)
        sequences = tf.concat(sequences, axis=1)
        return sequences, cell_states"""

class AtariGRU(tf.keras.layers.Layer):
    def __init__(self, units, memory_size=16):
        # RNN special layer, if episode ends or memory limit reached reset cell states
        super().__init__()
        self.units = units
        self.memory_size = memory_size
        self.rnn = tf.keras.layers.GRU(units=units, stateful=False, return_state=True, return_sequences=True)
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
    def __init__(self, use_rnn=True, rnn_units=64, memory_size=16):
        super().__init__()
        self.use_rnn = use_rnn
        self.rnn_units = rnn_units
        self.memory_size = memory_size
        
        hidden_init = tf.keras.initializers.Orthogonal(gain=np.sqrt(2))
        self.fc1 = tf.keras.layers.Dense(units=64, kernel_initializer=hidden_init, activation='tanh')
        self.fc2 = tf.keras.layers.Dense(units=64, kernel_initializer=hidden_init, activation='tanh')
        self.shallow_network = tf.keras.layers.TimeDistributed(tf.keras.Sequential([self.fc1, self.fc2]))
        if use_rnn:
            self.atari_rnn = AtariGRU(units=rnn_units, memory_size=memory_size)
            self.concat = tf.keras.layers.Concatenate(axis=2)

    def call(self, inputs, dones, cell_states=None, training=True):
        x = self.shallow_network(inputs, training=training)
        if self.use_rnn:
            y, cell_states = self.atari_rnn(inputs=x, dones=dones, cell_states=cell_states, training=training)
            x = self.concat([x, y])
        return x, cell_states

class AtariActorCriticNetwork(tf.keras.Model):
    def __init__(self, n_actions, policy_encoder, value_encoder=None):
        super().__init__()
        self.flat = tf.keras.layers.TimeDistributed(tf.keras.layers.Flatten())
        self.policy_encoder = policy_encoder

        if value_encoder is None:
            self.shared_params = True
            self.value_encoder = self.policy_encoder
        else:
            self.shared_params = False
            self.value_encoder = value_encoder

        critic_init = tf.keras.initializers.Orthogonal(gain=1.0)
        actor_init = tf.keras.initializers.Orthogonal(gain=0.01)
        self.actor = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=n_actions, kernel_initializer=actor_init))
        self.critic = tf.keras.layers.TimeDistributed(tf.keras.layers.Dense(units=1, kernel_initializer=critic_init))

        self.memory_size = policy_encoder.memory_size

    def call(self, inputs, dones, cell_states=None, action=None, training=True):
        inputs = tf.cast(inputs, tf.float32) / 255.0
        inputs = self.flat(inputs, training=training)


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

"""encoder = AtariNetwork()
net = AtariActorCriticNetwork(encoder, 18, 2, 10)
memory = []

a = tf.random.uniform(shape=(2, 10, 105, 80, 12), minval=0, maxval=256)
#a = tf.cast(a, dtype=tf.uint8)
dones = tf.zeros(shape=(2, 10))
net(a, dones, memory, training=False)

to_inspect = memory[0]
print(to_inspect[:, 0])
memory.clear()

b = tf.split(a, 10, axis=1)

dones = tf.zeros(shape=(2, 10))
cell_states = None
for c in b:
    _,_,_,_, cell_states = net(c, dones, memory, cell_states=cell_states, training=False)
    print(memory[0][:, 0])
    print(tf.reduce_all(tf.math.equal(memory[0][:, 0], to_inspect[:, 0])))
    break"""

"""inputs = tf.random.uniform(shape=(8, 23, 64))
dones = np.zeros(shape=(8, 23))
dones[3, 6] = 1

cell_states = tf.zeros(shape=(8, 64))

rnn = AtariGRU(64, 16)

h, c = rnn(inputs, dones, cell_states, reset_states=True)
"""
"""reset_dones = tf.random.uniform(shape=(8,1), minval=0, maxval=2, dtype=tf.int32)
reset_dones = tf.equal(reset_dones[:, 0], 1)

indices_dones = tf.where(reset_dones)
print(indices_dones)
step = tf.ones(shape=(8, 1), dtype=tf.int16)
print(len(indices_dones), step.shape)
x = tf.tensor_scatter_nd_update(tf.squeeze(step), indices_dones, tf.zeros(shape=(len(indices_dones)), dtype=tf.int16))
x = tf.expand_dims(x, axis=1)
print(x.shape)"""