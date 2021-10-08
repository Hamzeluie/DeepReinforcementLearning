import os
import gym
import torch
import random
import numpy as np
from queue import Queue
import multiprocessing
import threading
from collections import deque
from settings import MAX_EPISODES, MAX_STEPS, BATCH_SIZE, SAVE_DIR, EPISODE_SAVE_POINT
from utils.abstracts import AbsTrain, AbsEval


class Training(AbsTrain):
    def __init__(self, env_name, agent, use_conv=False):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.agent = agent(self.env.observation_space.shape, self.env.action_space.n, use_conv)
        self.max_episodes = MAX_EPISODES
        self.max_steps = MAX_STEPS
        self.batch_size = BATCH_SIZE

    def _step_agent(self, state, eps=0.20):
        return self.agent.get_val(state, eps=eps)

    def _step_env(self, action):
        return self.env.step(np.array(action))

    def train(self):
        episode_rewards = []
        for episode in range(self.max_episodes):
            state = self.env.reset()
            episode_reward = 0
            for step in range(self.max_steps):
                self.env.render()
                action = self._step_agent(state)
                next_state, reward, done, _ = self._step_env(action)
                self.agent.buffer.store(state, action, reward, next_state, done)
                episode_reward += 1
                if self.agent.buffer.__len__() > self.batch_size:
                    self.agent.learn(self.batch_size)
                if done or step == self.max_steps - 1:
                    episode_rewards.append(episode_reward)
                    print(f"Episode {episode} : {episode_reward}")
                    break
                state = next_state
            if (episode + 1) % EPISODE_SAVE_POINT == 0:
                self.save_model(episode + 1)

    def save_model(self, iter):
        path = os.path.join(SAVE_DIR, f"{self.env_name}_{self.agent.__name__().lower()}")
        self.agent.save_model(path, iter)


class AsyncExperiment:
    def __init__(self, env_name, master_agent):
        self.env_name = env_name
        self.master_agent = master_agent

    def start(self):
        agent = self.master_agent(self.env_name)
        # agent.train()
        agent.play()


class Evaluation(AbsEval):
    def __init__(self, env_name, agent, use_conv=False):
        self.env_name = env_name
        self.env = gym.make(self.env_name)
        self.agent = agent(self.env.observation_space.shape, self.env.action_space.n, use_conv)

    def _step_env(self, action):
        return self.env.step(action)

    def _step_agent(self, state, eps=0.20):
        return self.agent.get_val(state, eps=eps)

    def eval(self):
        self.load_model()
        for _ in range(MAX_STEPS):
            state = self.env.reset()
            episode_reward = 0
            while True:
                self.env.render()
                action = self._step_agent(state)
                next_state, reward, done, _ = self.env.step(action)
                episode_reward += 1
                state = next_state
                if done:
                    break
        print(f"maximum reward is {episode_reward}")

    def load_model(self):
        path = os.path.join(SAVE_DIR, f"{self.env_name}_{self.agent.__name__().lower()}")
        self.agent.load_model(path)


class Buffer:
    def __init__(self, max_size):
        self.max_size = max_size
        self.buffer = deque(maxlen=max_size)

    def store(self, state, action, reward, next_state, done):
        experience = (state, action, np.array([reward]), next_state, done)
        self.buffer.append(experience)

    def clear(self):
        self.buffer.clear()

    def sample(self, batch_size):
        state_batch = []
        action_batch = []
        reward_batch = []
        next_state_batch = []
        done_batch = []

        if batch_size > len(self.buffer):
            batch = random.sample(self.buffer, len(self.buffer))
        else:
            batch = random.sample(self.buffer, batch_size)

        for experience in batch:
            state, action, reward, next_state, done = experience
            state_batch.append(state)
            action_batch.append(action)
            reward_batch.append(reward)
            next_state_batch.append(next_state)
            done_batch.append(done)
        return (state_batch, action_batch, reward_batch, next_state_batch, done_batch)

    def __len__(self):
        return len(self.buffer)


def keras_save_weights(model, path, episode, file_extension):
    try:
        len_extension = file_extension.__len__()
        for file in os.listdir(SAVE_DIR):
            if file.split('iter')[0][:-1].lower() == os.path.basename(path).lower()\
                    and file.split('iter')[1][-len_extension:].lower() == file_extension:
                os.remove(os.path.join(SAVE_DIR, file))
                break

        path_model = path + f"_iter_{episode}" + file_extension
        model.save_weights(path_model)
        print("model weights saved.")
    except OSError:
        print("Please Make 'save_dir' directory or folder to save model weights.")


def keras_load_weights(model, path, file_extension, state_dim, build=True):
    try:
        len_extension = file_extension.__len__()
        path_model = ""
        for file in os.listdir(SAVE_DIR):
            if file.split('iter')[0][:-1].lower() == os.path.basename(path).lower():
                if file.split('iter')[1][-len_extension:].lower() == file_extension:
                    path_model = os.path.join(SAVE_DIR, file)
                    break
        if build:
            model.build(input_shape=[1, state_dim])
        model.load_weights(path_model)
        print("model weights loaded.")
    except OSError:
        print("there is no trained model to load.")


def torch_save_weights(model, path, episode, file_extension):
    try:
        len_extension = file_extension.__len__()
        for file in os.listdir(SAVE_DIR):
            if file.split('iter')[0][:-1].lower() == os.path.basename(path).lower()\
                    and file.split('iter')[1][-len_extension:].lower() == file_extension:
                os.remove(os.path.join(SAVE_DIR, file))
                break

        path_model = path + f"_iter_{episode}" + file_extension
        torch.save(model.state_dict(), path_model)
        print(f"{file_extension} weights saved.")
    except OSError:
        print("Please Make 'save_dir' directory or folder to save model weights.")


def torch_load_weights(model, path, file_extension):
    try:
        len_extension = file_extension.__len__()
        path_model = ""
        for file in os.listdir(SAVE_DIR):
            if file.split('iter')[0][:-1].lower() == os.path.basename(path).lower():
                if file.split('iter')[1][-len_extension:].lower() == file_extension:
                    path_model = os.path.join(SAVE_DIR, file)
                    break

        model.load_state_dict(torch.load(path_model))
        print(f"{file_extension} weights loaded.")
    except OSError:
        print("there is no trained model to load.")