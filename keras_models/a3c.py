import os
import gym
import numpy as np
import tensorflow as tf
import threading
from queue import Queue
import multiprocessing
from networks.keras_nets import ActorCritic
from utils.utils import Buffer, keras_save_weights, keras_load_weights, Training
from settings import *


class MasterAgent:
    def __init__(self, env_name):
        self.env_name = env_name
        env = gym.make(self.env_name)
        self.state_size = env.observation_space.shape[0]
        self.action_size = env.action_space.n
        self.opt = tf.optimizers.Adam(LEARNING_RATE)
        self.global_model = ActorCritic(self.action_size)
        self.global_model(tf.convert_to_tensor(np.random.random((1, self.state_size)), dtype=tf.float32))

    def train(self):
        res_queue = Queue()
        workers = [Worker(self.state_size, self.action_size, self.global_model, self.opt, res_queue, i, self.env_name)
                   for i in range(multiprocessing.cpu_count())]
        for i, worker in enumerate(workers):
            print(f'Starting worker {i}')
            worker.start()
        moving_average_reward = []
        while True:
            reward = res_queue.get()
            if reward is not None:
                moving_average_reward.append(reward)
            else:
                break

    def play(self):
        env = gym.make(self.env_name)
        state = env.reset()
        model = self.global_model
        path = os.path.join(SAVE_DIR, f"{self.env_name}_a3c_worker")
        keras_load_weights(model, path, ".h5", self.state_size)
        print('Loading model weights ...')
        done = False
        step_counter = 0
        reward_sum = 0
        try:
            while not done:
                env.render(mode='rgb_array')
                policy, value = model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                policy = tf.nn.softmax(policy)
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                reward_sum += reward
                print("{}. Reward: {}, action: {}".format(step_counter, reward_sum, action))
                step_counter += 1
        except KeyboardInterrupt:
            print("Received Keyboard Interrupt. Shutting down.")
        finally:
            env.close()


class Worker(threading.Thread):
    # Set up global variables across different threads
    global_episode = 0
    # Moving average reward
    global_moving_average_reward = 0
    best_score = 0
    save_lock = threading.Lock()

    def __init__(self, state_size, action_size, global_model, opt, result_queue, idx, env_name, update_freq=50):
        super(Worker, self).__init__()
        self.state_size = state_size
        self.action_size = action_size
        self.global_model = global_model
        self.opt = opt
        self.result_queue = result_queue
        self.idx = idx
        self.update_freq = update_freq
        self.env_name = env_name

        self.local_model = ActorCritic(self.action_size)
        self.env = gym.make(self.env_name)
        self.ep_loss = .0

    def run(self):
        total_steps = 1
        mem = Memory()

        while Worker.global_episode < 200:
            done = False
            state = self.env.reset()
            mem.clear()
            ep_reward = 0.0
            ep_steps = 0
            self.ep_loss = 0
            time_count = 0
            while not done:
                _, logits = self.local_model(tf.convert_to_tensor(state[None, :], dtype=tf.float32))
                prob = tf.nn.softmax(logits)
                action = np.random.choice(self.action_size, p=prob.numpy()[0])

                next_state, reward, done, _ = self.env.step(action)

                if done:
                    reward = -1
                ep_reward += reward
                mem.store(state, action, reward, int(done))
                if done or time_count == self.update_freq:
                    with tf.GradientTape() as tap:
                        loss = self.compute_loss(done, next_state, mem)
                    self.ep_loss += loss
                    gradient = tap.gradient(loss, self.local_model.trainable_weights)
                    self.opt.apply_gradients(zip(gradient, self.global_model.trainable_weights))
                    self.local_model.set_weights(self.global_model.get_weights())
                    mem.clear()
                    time_count = 0

                    if ep_reward > Worker.best_score:
                        print(f'best score till now {ep_reward}. model weights saved.')

                        path = os.path.join(SAVE_DIR, f"{self.env_name}_a3c_worker")
                        keras_save_weights(self.global_model, path, Worker.best_score, ".h5")

                        # self.global_model.save_weights(os.path.join('/tmp/', 'model_CartPole_v0.h5'))
                        Worker.best_score = ep_reward
                    Worker.global_episode += 1
                time_count += 1
                state = next_state
                ep_steps += 1
            self.result_queue.put(None)

    def compute_loss(self,
                     done,
                     new_state,
                     memory):
        if done:
            reward_sum = 0.  # terminal
        else:
            reward_sum, _ = self.local_model(
                tf.convert_to_tensor(new_state[None, :],
                                     dtype=tf.float32))[-1].numpy()[0]

        # Get discounted rewards
        discounted_rewards = []
        for reward in memory.rewards[::-1]:  # reverse buffer r
            reward_sum = reward + GAMMA * reward_sum
            discounted_rewards.append(reward_sum)
        discounted_rewards.reverse()

        values, logits = self.local_model(
            tf.convert_to_tensor(np.vstack(memory.states),
                                 dtype=tf.float32))
        # Get our advantages
        advantage = tf.convert_to_tensor(np.array(discounted_rewards)[:, None],
                                         dtype=tf.float32) - tf.squeeze(values)
        # Value loss
        value_loss = advantage ** 2

        # Calculate our policy loss

        policy = tf.nn.softmax(logits)
        entropy = tf.nn.softmax_cross_entropy_with_logits(labels=policy, logits=logits)

        policy_loss = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=memory.actions,
                                                                     logits=logits)
        policy_loss *= tf.stop_gradient(advantage)
        policy_loss -= 0.01 * entropy
        total_loss = tf.reduce_mean((0.5 * value_loss + policy_loss))
        return total_loss


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.done = []

  def store(self, state, action, reward, done):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)
    self.done.append(done)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []
    self.done = []