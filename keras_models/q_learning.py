import tensorflow as tf
import numpy as np
from networks.keras_nets import DeepQNetwork, ConvDeepQNetwork
from utils.abstracts import AbsModel
from utils.utils import Buffer, keras_save_weights, keras_load_weights
from settings import *


class DoubleQLearningHasselt(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(BUFFER_SIZE)
        if use_conv:
            self.model = ConvDeepQNetwork(self.state_dim, self.action_dim)
            self.model_target = ConvDeepQNetwork(self.state_dim, self.action_dim)
        else:
            self.model = DeepQNetwork(self.state_dim, self.action_dim)
            self.model_target = DeepQNetwork(self.state_dim, self.action_dim)
        self.opt = tf.keras.optimizers.Adam(LEARNING_RATE)

    def get_val(self, state, eps=0.20):
        state = tf.convert_to_tensor(state, dtype=tf.float32)
        q_value = self.model(state)
        action = np.argmax(np.squeeze(q_value))
        if np.random.randn() < eps:
            return self.random_generator.choice(self.action_dim)
        return action

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        self._update(states, actions, rewards, next_states, done)
        self._update_target(
            self.model_target.trainable_variables, self.model.trainable_variables
        )

    def _update(self, states, actions, rewards, next_states, done):
        gradient = self._compute_loss(states, actions, rewards, next_states, done)
        self.opt.apply_gradients(zip(gradient, self.model.trainable_variables))

    def _compute_loss(self, states, actions, rewards, next_states, done):
        with tf.GradientTape() as tape:
                current_q = self.model(states)
                current_q = tf.gather_nd(
                    current_q, tf.expand_dims(tf.cast(actions, dtype=tf.int32), 1), 1
                )

                next_q = self.model_target(next_states)
                max_next_q = tf.math.reduce_max(next_q, 1)

                expected_q = rewards + (1 - done) * GAMMA * max_next_q
                loss = tf.math.reduce_mean(tf.math.square(current_q - expected_q))
        gradient = tape.gradient(loss, self.model.trainable_variables)
        return gradient

    def _update_target(self, target_weights, weights):
        for (a, b) in zip(target_weights, weights):
            a.assign(TAU * b + (1 - TAU) * a)

    def save_model(self, path, episode):
        keras_save_weights(self.model, path, episode, "_model.h5")
        keras_save_weights(self.model_target, path, episode, "_target.h5")

    def load_model(self, path):
        keras_load_weights(self.model, path, "_model.h5", self.state_dim[0])
        keras_load_weights(self.model_target, path, "_target.h5", self.state_dim[0])

    def __name__(self):
        return "keras_DoubleQLearningHasselt"


class DoubleQLearningFujimoto(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(BUFFER_SIZE)
        if use_conv:
            self.model = ConvDeepQNetwork(self.state_dim, self.action_dim)
            self.model_target = ConvDeepQNetwork(self.state_dim, self.action_dim)
        else:
            self.model = DeepQNetwork(self.state_dim, self.action_dim)
            self.model_target = DeepQNetwork(self.state_dim, self.action_dim)
        self.opt1 = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.opt2 = tf.keras.optimizers.Adam(LEARNING_RATE)

    def get_val(self, state, eps=0.20):
        state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        q_value = self.model(state)
        action = np.argmax(np.squeeze(q_value))
        if np.random.randn() < eps:
            return self.random_generator.choice(self.action_dim)
        return action

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = tf.convert_to_tensor(states, dtype=tf.float32)
        actions = tf.convert_to_tensor(actions, dtype=tf.float32)
        rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
        next_states = tf.convert_to_tensor(next_states, dtype=tf.float32)
        done = tf.convert_to_tensor(done, dtype=tf.float32)
        self._update(states, actions, rewards, next_states, done)

    def _update(self, states, actions, rewards, next_states, done):
        gradient1, gradient2 = self._compute_loss(states, actions, rewards, next_states, done)

        self.opt1.apply_gradients(zip(gradient1, self.model.trainable_variables))
        self.opt2.apply_gradients(zip(gradient2, self.model_target.trainable_variables))

    def _compute_loss(self, states, actions, rewards, next_states, done):
        with tf.GradientTape() as tape:
            curr_q1 = self.model(states)
            curr_q1 = tf.gather_nd(
                curr_q1, tf.expand_dims(tf.cast(actions, dtype=tf.int32), 1), 1
            )

            next_q1 = self.model(next_states)
            next_q2 = self.model_target(next_states)

            next_q = tf.minimum(
                tf.reduce_max(next_q1, axis=1), tf.reduce_max(next_q2, axis=1)
            )

            expected_q = rewards + (1 - done) * GAMMA * next_q
            loss1 = tf.math.reduce_mean(tf.math.square(curr_q1 - expected_q))
        gradient1 = tape.gradient(loss1, self.model.trainable_variables)
        with tf.GradientTape() as tape:
            curr_q2 = self.model_target(states)
            curr_q2 = tf.gather_nd(
                curr_q2, tf.expand_dims(tf.cast(actions, dtype=tf.int32), 1), 1
            )
            loss2 = tf.math.reduce_mean(tf.math.square(curr_q2 - expected_q))

        gradient2 = tape.gradient(loss2, self.model_target.trainable_variables)
        return gradient1, gradient2

    def save_model(self, path, iter):
        keras_save_weights(self.model, path, iter, "_model.h5")
        keras_save_weights(self.model_target, path, iter, "_target.h5")

    def load_model(self, path):
        keras_load_weights(self.model, path, "_model.h5", self.state_dim[0])
        keras_load_weights(self.model_target, path, "_target.h5", self.state_dim[0])

    def __name__(self):
        return "keras_DoubleQLearningFujimoto"
