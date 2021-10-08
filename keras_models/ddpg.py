import numpy as np
import tensorflow as tf
from networks.keras_nets import ActorCritic, Actor, Critic
from utils.abstracts import AbsModel
from utils.utils import Buffer, keras_save_weights, keras_load_weights
from settings import *


class DeepDeterministicPolicyGradient(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model_actor = Actor(self.action_dim)
        self.model_critic = Critic()

        self.model_t_actor = Actor(self.action_dim)
        self.model_t_critic = Critic()

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
        self._update_target(self.model_t_actor.trainable_variables, self.model_actor.trainable_variables)
        self._update_target(self.model_t_critic.trainable_variables, self.model_critic.trainable_variables)

    def _update(self, state, action, reward, next_state, done):
        loss_critic, gradient_critic, loss_actor, gradient_actor = \
            self._compute_loss(state, action, reward, next_state, done)
        self.opt_critic.apply_gradients((zip(gradient_critic, self.model_critic.trainable_variables)))
        self.opt_actor.apply_gradients((zip(gradient_actor, self.model_actor.trainable_variables)))

    def _compute_loss(self, state, action, reward, next_state, done):
        with tf.GradientTape() as tape:
            target_policy = self.model_t_actor(next_state, training=True)
            target_val = self.model_t_critic(state, target_policy, training=True)
            y = tf.squeeze(reward) + (1 + done) * GAMMA * tf.squeeze(target_val)
            val = self.model_critic(state, tf.one_hot(tf.cast(action, dtype=tf.int32), 2), training=True)
            loss_critic = tf.math.reduce_mean(tf.math.square(y - val))
        gradient_critic = tape.gradient(loss_critic, self.model_critic.trainable_variables)
        with tf.GradientTape() as tape:
            policy = self.model_actor(state, training=True)
            val = self.model_critic(state, policy, training=True)
            loss_actor = -tf.math.reduce_mean(val)
        gradient_actor = tape.gradient(loss_actor, self.model_actor.trainable_variables)
        return loss_critic, gradient_critic, loss_actor, gradient_actor

    def _update_target(self, target, weight):
        for (a, b) in zip(target, weight):
            a.assign(TAU * b + (1 - TAU) * a)

    def save_model(self, path, iter):
        keras_save_weights(self.model_t_actor, path, iter, "_model_t_actor.h5")
        keras_save_weights(self.model_t_critic, path, iter, "_model_t_critic.h5")
        keras_save_weights(self.model_actor, path, iter, "_model_actor.h5")
        keras_save_weights(self.model_critic, path, iter, "_model_critic.h5")

    def load_model(self, path):
        keras_load_weights(self.model_t_actor, path, "_model_t_actor.h5", self.state_dim[0])
        keras_load_weights(self.model_actor, path, "_model_actor.h5", self.state_dim[0])
        self.model_t_critic(tf.random.uniform((1, self.state_dim[0])), tf.random.uniform((1, self.action_dim)))
        keras_load_weights(self.model_t_critic, path, "_model_t_critic.h5", self.state_dim[0], build=False)
        self.model_critic(tf.random.uniform((1, self.state_dim[0])), tf.random.uniform((1, self.action_dim)))
        keras_load_weights(self.model_critic, path, "_model_critic.h5", self.state_dim[0], build=False)

    def __name__(self):
        return "keras_DeepDeterministicPolicyGradient"
