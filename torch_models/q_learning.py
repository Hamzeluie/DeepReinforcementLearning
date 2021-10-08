import torch
import torch.nn.functional as F
import numpy as np
from networks.torch_nets import DeepQNetwork, ConvDeepQNetwork
from utils.abstracts import AbsModel
from utils.utils import Buffer, torch_save_weights, torch_load_weights
from settings import *


class DoubleQLearningHasselt(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(BUFFER_SIZE)
        if use_conv:
            self.model = ConvDeepQNetwork(self.state_dim[0], self.action_dim)
            self.model_target = ConvDeepQNetwork(self.state_dim[0], self.action_dim)
        else:
            self.model = DeepQNetwork(self.state_dim[0], self.action_dim)
            self.model_target = DeepQNetwork(self.state_dim[0], self.action_dim)
        for target_param, param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(param)
        self.opt = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)

    def get_val(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0)
        q_value = self.model.forward(state)
        action = np.argmax(q_value.cpu().detach().numpy())
        if np.random.randn() < eps:
            return self.random_generator.choice(self.action_dim)

        return action

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done)
        self._update(states, actions, rewards, next_states, done)
        self._update_target()

    def _update(self, states, actions, rewards, next_states, done):
        loss = self._compute_loss(states, actions, rewards, next_states, done)
        self.opt.zero_grad()
        loss.backward()
        self.opt.step()
        self._update_target()

    def _compute_loss(self, states, actions, rewards, next_states, done):
        actions = actions.view(actions.size(0), 1)
        done = done.view(done.size(0), 1)

        # compute loss
        curr_Q = self.model(states).gather(1, actions)
        next_Q = self.model_target.forward(next_states)
        max_next_Q = torch.max(next_Q, 1)[0]
        max_next_Q = max_next_Q.view(max_next_Q.size(0), 1)

        expected_Q = rewards + (1 - done) * GAMMA * max_next_Q

        loss = F.mse_loss(curr_Q, expected_Q.detach())
        return loss

    def _update_target(self):
        for target_param, param in zip(self.model_target.parameters(), self.model.parameters()):
            target_param.data.copy_(TAU * param + (1 - TAU) * target_param)

    def save_model(self, path, episode):
        torch_save_weights(self.model, path, episode, "_model.pth")
        torch_save_weights(self.model_target, path, episode, "_target.pth")

    def load_model(self, path):
        torch_load_weights(self.model, path, "_model.pth")
        torch_load_weights(self.model_target, path, "_target.pth")

    def __name__(self):
        return "torch_DoubleQLearningHasselt"


class DoubleQLearningFujimoto(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(BUFFER_SIZE)
        if use_conv:
            self.model = ConvDeepQNetwork(self.state_dim[0], self.action_dim)
            self.model_target = ConvDeepQNetwork(self.state_dim[0], self.action_dim)
        else:
            self.model = DeepQNetwork(self.state_dim[0], self.action_dim)
            self.model_target = DeepQNetwork(self.state_dim[0], self.action_dim)
        for target_param, param in zip(self.model.parameters(), self.model_target.parameters()):
            target_param.data.copy_(param)
        self.opt = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.opt_target = torch.optim.Adam(self.model_target.parameters(), LEARNING_RATE)

    def get_val(self, state, eps=0.20):
        state = torch.FloatTensor(state).float().unsqueeze(0)
        q_value = self.model.forward(state)
        action = np.argmax(q_value.cpu().detach().numpy())
        if np.random.randn() < eps:
            return self.random_generator.choice(self.action_dim)

        return action

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = torch.FloatTensor(states)
        actions = torch.LongTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        done = torch.FloatTensor(done)
        self._update(states, actions, rewards, next_states, done)

    def _update(self, states, actions, rewards, next_states, done):
        loss1, loss2 = self._compute_loss(states, actions, rewards, next_states, done)
        self.opt.zero_grad()
        loss1.backward()
        self.opt.step()

        self.opt_target.zero_grad()
        loss2.backward()
        self.opt_target.step()

    def _compute_loss(self, states, actions, rewards, next_states, done):
        actions = actions.view(actions.size(0), 1)
        done = done.view(done.size(0), 1)

        # compute loss
        curr_Q1 = self.model.forward(states).gather(1, actions)
        curr_Q2 = self.model_target.forward(states).gather(1, actions)

        next_Q1 = self.model.forward(next_states)
        next_Q2 = self.model_target.forward(next_states)
        next_Q = torch.min(
            torch.max(next_Q1, 1)[0],
            torch.max(next_Q2, 1)[0]
        )
        next_Q = next_Q.view(next_Q.size(0), 1)
        expected_Q = rewards + (1 - done) * GAMMA * next_Q

        loss1 = F.mse_loss(curr_Q1, expected_Q.detach())
        loss2 = F.mse_loss(curr_Q2, expected_Q.detach())
        return loss1, loss2

    def save_model(self, path, episode):
        torch_save_weights(self.model, path, episode, "_model.pth")
        torch_save_weights(self.model_target, path, episode, "_target.pth")

    def load_model(self, path):
        torch_load_weights(self.model, path, "_model.pth")
        torch_load_weights(self.model_target, path, "_target.pth")

    def __name__(self):
        return "torch_DoubleQLearningFujimoto"
