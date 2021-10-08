import torch
import torch.nn.functional as F
import numpy as np
from networks.torch_nets import Actor, Critic, ActorCritic
from utils.abstracts import AbsModel
from utils.utils import Buffer, torch_save_weights, torch_load_weights
from settings import *


class A2C_one_net(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model = ActorCritic(self.state_dim[0], self.action_dim)
        self.random_generator = np.random.RandomState(SEED)
        self.optimizer = torch.optim.Adam(self.model.parameters(), LEARNING_RATE)
        self.buffer = Buffer(max_size=BUFFER_SIZE)

    def get_val(self, state, eps=0.20):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0)
        _, policy = self.model.forward(state)
        policy = F.softmax(policy, dim=1)
        dist = torch.distributions.Categorical(probs=policy)
        action = dist.sample()
        return int(action)

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        self._update(states, actions, rewards, next_states, done)

    def _update(self, state, action, reward, next_state, done):
        loss = self._compute_loss(state, action, reward, next_state, done)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def _compute_loss(self, state, action, reward, next_state, done):
        q_value, policy = self.model.forward(state)
        policy = F.softmax(policy, dim=1)
        q_value_, _ = self.model.forward(next_state)

        dist = torch.distributions.Categorical(probs=policy)

        advantage = reward.squeeze() + (1 - done) * GAMMA * q_value_.squeeze() - q_value.squeeze()
        loss_critic = advantage.pow(2).mean()
        loss_actor = (-dist.log_prob(action) * advantage.detach()).mean()
        loss = loss_actor + loss_critic + TAU
        return loss

    def save_model(self, path, iter):
        torch_save_weights(self.model, path, iter, "_model.pth")

    def load_model(self, path):
        torch_load_weights(self.model, path, "_model.pth")

    def __name__(self):
        return "torch_A2C_one_net"


class A2C_multi_net(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model_actor = Actor(self.state_dim[0], self.action_dim)
        self.model_critic = Critic(self.state_dim[0])
        self.opt_actor = torch.optim.Adam(self.model_actor.parameters(), LEARNING_RATE)
        self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), LEARNING_RATE)
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(max_size=BUFFER_SIZE)

    def get_val(self, state, eps=0.20):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0)
        policy = self.model_actor.forward(state)
        policy = F.softmax(policy, dim=1)
        dist = torch.distributions.Categorical(probs=policy)
        action = dist.sample()
        return int(action)

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        self._update(states, actions, rewards, next_states, done)

    def _update(self, state, action, reward, next_state, done):
        loss_critic, loss_actor = self._compute_loss(state, action, reward, next_state, done)

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

    def _compute_loss(self, state, action, reward, next_state, done):
        policy = self.model_actor.forward(state)
        policy = F.softmax(policy, dim=1)
        dist = torch.distributions.Categorical(probs=policy)

        advantage = reward.squeeze() + (1 - done) * GAMMA * self.model_critic.forward(
            next_state).squeeze() - self.model_critic.forward(state).squeeze()
        loss_critic = advantage.pow(2).mean()
        loss_actor = (-dist.log_prob(action) * advantage.detach()).mean()
        return loss_critic, loss_actor

    def save_model(self, path, iter):
        torch_save_weights(self.model_actor, path, iter, "_model_actor.pth")
        torch_save_weights(self.model_critic, path, iter, "_model_critic.pth")

    def load_model(self, path):
        torch_load_weights(self.model_actor, path, "_model_actor.pth")
        torch_load_weights(self.model_critic, path, "_model_critic.pth")

    def __name__(self):
        return "torch_A2C_multi_net"












