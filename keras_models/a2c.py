import numpy as np
import tensorflow as tf
from networks.keras_nets import Actor_Critic, Actor, Critic
from utils.abstracts import AbsModel
from utils.utils import Buffer, keras_save_weights, keras_load_weights
from settings import *


class A2C_one_net(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model = Actor_Critic(self.action_dim)
        self.random_generator = np.random.RandomState(SEED)
        self.optimizer = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.buffer = Buffer(max_size=BUFFER_SIZE)

    def get_val(self, state, eps=0.20):
        state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        value, policy = self.model(state)
        normalized_prob = abs(np.squeeze(policy.numpy())) / sum(abs(np.squeeze(policy.numpy())))
        action = np.random.choice(self.action_dim, p=normalized_prob)
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

    def _update(self, state, action, reward, next_state, done):
        gradient = self._compute_loss(state, action, reward, next_state, done)
        self.optimizer.apply_gradients((zip(gradient, self.model.trainable_variables)))

    def _compute_loss(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            q_val, policy = self.model(state)
            q_val_, policy_ = self.model(next_state)
            advantage = tf.squeeze(reward) + (1 + done) * GAMMA * tf.squeeze(q_val_) - tf.squeeze(q_val)
            loss_critic = tf.reduce_mean(tf.pow(advantage, 2))

            dist = tf.gather_nd(policy, tf.expand_dims(tf.cast(action, dtype=tf.int32), 1), 1)
            loss_actor = - dist * advantage
            loss = loss_actor + loss_critic + TAU

        gradient = tape.gradient(loss, self.model.trainable_variables)
        return gradient

    def save_model(self, path, iter):
        keras_save_weights(self.model, path, iter, "_model.h5")

    def load_model(self, path):
        keras_load_weights(self.model, path, "_model.h5", self.state_dim[0])

    def __name__(self):
        return "keras_A2C_one_net"


class A2C_multi_net(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model_actor = Actor(self.action_dim)
        self.model_critic = Critic()
        self.opt_actor = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.opt_critic = tf.keras.optimizers.Adam(LEARNING_RATE)
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(max_size=BUFFER_SIZE)

    def get_val(self, state, eps=0.20):
        state = tf.expand_dims(tf.convert_to_tensor(state), axis=0)
        policy = self.model_actor(state)
        normalized_prob = abs(np.squeeze(policy.numpy())) / sum(abs(np.squeeze(policy.numpy())))
        action = np.random.choice(self.action_dim, p=normalized_prob)
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

    def _update(self, state, action, reward, next_state, done):
        gradient_critic, gradient_actor = self._compute_loss(state, action, reward, next_state, done)
        self.opt_critic.apply_gradients((zip(gradient_critic, self.model_critic.trainable_variables)))
        self.opt_actor.apply_gradients((zip(gradient_actor, self.model_actor.trainable_variables)))

    def _compute_loss(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            q_val = self.model_critic(state)
            q_val_ = self.model_critic(next_state)

            advantage = tf.squeeze(reward) + (1 + done) * GAMMA * tf.squeeze(q_val_) - tf.squeeze(q_val)
            loss_critic = tf.reduce_mean(tf.pow(advantage, 2))
        gradient_critic = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        with tf.GradientTape() as tape:
            policy = self.model_actor(state)
            dist = tf.gather_nd(policy, tf.expand_dims(tf.cast(action, dtype=tf.int32), 1), 1)
            loss_actor = - dist * advantage

        gradient_actor = tape.gradient(loss_actor, self.model_actor.trainable_variables)
        return gradient_critic, gradient_actor

    def save_model(self, path, iter):
        keras_save_weights(self.model_actor, path, iter, "_model_actor.h5")
        keras_save_weights(self.model_critic, path, iter, "_model_critic.h5")

    def load_model(self, path):
        keras_load_weights(self.model_actor, path, "_model_actor.h5", self.state_dim[0])
        keras_load_weights(self.model_critic, path, "_model_critic.h5", self.state_dim[0])

    def __name__(self):
        return "keras_A2C_multi_net"
















