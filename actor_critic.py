

import tensorflow as tf
import tensorflow_probability as tfp
import numpy as np
tf.random.set_seed(42)
np.random.seed(42)

class AtariGRU(tf.keras.layers.Layer):
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
        self.c1 = tf.keras.layers.Conv2D(filters=64, kernel_size=8, strides=4, activation='relu')
        self.c2 = tf.keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, activation='relu')
        self.c3 = tf.keras.layers.Conv2D(filters=128, kernel_size=3, strides=1, activation='relu')
        self.fc1 = tf.keras.layers.Flatten()
        self.fc2 = tf.keras.layers.Dense(800, activation='relu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
        self.cnn = tf.keras.models.Sequential([self.c1, self.c2, self.c3, self.fc1, self.fc2])
        self.cnn = tf.keras.layers.TimeDistributed(self.cnn)
        self.gru = AtariGRU(units=800)
        self.concatenate = tf.keras.layers.Concatenate(axis=2)
    def call(self, inputs, dones, cell_states=None, training=True):
        #inputs must be of shape [num_envs, batch_size, observation_spec]  
        num_envs, batch_size = inputs.shape[:2]
        #inputs = tf.reshape(inputs, shape=[num_envs*batch_size] + inputs.shape[2:])    
        x = self.cnn(inputs, training=training)

        #x = tf.reshape(x, shape=(num_envs, batch_size, x.shape[-1]))
        sequences, cell_states = self.gru(x, dones, cell_states=cell_states, training=training)
        x = self.concatenate([x, sequences], training=training)
        return x, cell_states

class AtariNetwork(tf.keras.Model):
    def __init__(self):
        super().__init__()
        self.c1 = tf.keras.layers.Conv2D(filters=32, kernel_size=8, strides=4, activation='tanh')
        self.c2 = tf.keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, activation='tanh')
        self.c3 = tf.keras.layers.Conv2D(filters=64, kernel_size=3, strides=1, activation='tanh')
        self.fc1 = tf.keras.layers.Flatten()
        self.fc2 = tf.keras.layers.Dense(512, activation='relu')
        self.layer_norm = tf.keras.layers.LayerNormalization()
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
        return x, cell_states

class AtariActorCriticNetwork(tf.keras.Model):
    def __init__(self, encoder, encoder_value, n_actions):
        super().__init__()
        self.encoder = encoder
        self.encoder_value = encoder_value
        self.actor = tf.keras.layers.Dense(units=n_actions)
        self.critic = tf.keras.layers.Dense(units=1)

    def call(self, inputs, dones, cell_states=None, action=None, training=True):
        inputs = tf.cast(inputs, tf.float32) / 255.0
        x, new_cell_states = self.encoder(inputs, dones=dones, cell_states=cell_states, training=training)
        logits = self.actor(x, training=training)

        x, new_cell_states = self.encoder_value(inputs, dones=dones, cell_states=cell_states, training=training)
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

