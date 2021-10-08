import numpy as np
import torch
import torch.nn.functional as F
from networks.torch_nets import Actor, Critic
from utils.abstracts import AbsModel
from utils.utils import Buffer, torch_save_weights, torch_load_weights
from settings import *


class DeepDeterministicPolicyGradient(AbsModel):
    def __init__(self, state_dim, action_dim, use_conv=False):
        self.state_dim = state_dim[0]
        self.action_dim = action_dim
        self.use_conv = use_conv
        self.model_actor = Actor(self.state_dim, self.action_dim)
        self.model_critic = Critic(self.state_dim, self.action_dim)

        self.model_t_actor = Actor(self.state_dim, self.action_dim)
        self.model_t_critic = Critic(self.state_dim, self.action_dim)

        self.opt_actor = torch.optim.Adam(self.model_actor.parameters(), LEARNING_RATE)
        self.opt_critic = torch.optim.Adam(self.model_critic.parameters(), LEARNING_RATE)
        self.random_generator = np.random.RandomState(SEED)
        self.buffer = Buffer(max_size=BUFFER_SIZE)

    def get_val(self, state, eps=0.20):
        state = torch.unsqueeze(torch.tensor(state, dtype=torch.float32), dim=0)
        policy = self.model_actor(state)
        policy = F.softmax(policy, dim=1)
        dist = torch.distributions.Categorical(probs=policy)
        action = dist.sample()
        return int(action)

    def _one_hot(self, x, num_class):
        out = np.zeros((x.__len__(), num_class))
        for idx, val in enumerate(x):
            out[idx][val] = 1
        return out

    def learn(self, batch_size):
        batch = self.buffer.sample(batch_size)
        states, actions, rewards, next_states, done = batch
        states = torch.tensor(states, dtype=torch.float32)
        actions = self._one_hot(actions, num_class=self.action_dim)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        done = torch.tensor(done, dtype=torch.float32)
        self._update(states, actions, rewards, next_states, done)

        self._update_target(self.model_t_actor, self.model_actor)
        self._update_target(self.model_t_critic, self.model_critic)

    def _update(self, state, action, reward, next_state, done):
        loss_critic, loss_actor = \
            self._compute_loss(state, action, reward, next_state, done)
        self.opt_actor.zero_grad()
        loss_actor.backward()
        self.opt_actor.step()

        self.opt_critic.zero_grad()
        loss_critic.backward()
        self.opt_critic.step()

    def _compute_loss(self, state, action, reward, next_state, done):
        target_policy = self.model_t_actor(next_state)
        target_val = self.model_t_critic(state, target_policy)
        y = reward.squeeze() + (1 + done) * GAMMA * target_val.squeeze()
        q_value = self.model_critic(state, action)
        loss_critic = torch.mean(torch.square(y - q_value))

        policy = self.model_actor(state)
        val = self.model_critic(state, policy)
        loss_actor = -torch.mean(val)

        return loss_critic, loss_actor

    def _update_target(self, target, weight):
        for (a, b) in zip(target.parameters(), weight.parameters()):
            a.data.copy_(TAU * b + (1 - TAU) * a)

    def save_model(self, path, iter):
        torch_save_weights(self.model_t_actor, path, iter, "_model_t_actor.pth")
        torch_save_weights(self.model_t_critic, path, iter, "_model_t_critic.pth")
        torch_save_weights(self.model_actor, path, iter, "_model_actor.pth")
        torch_save_weights(self.model_critic, path, iter, "_model_critic.pth")

    def load_model(self, path):
        torch_load_weights(self.model_t_actor, path, "_model_t_actor.pth")
        torch_load_weights(self.model_actor, path, "_model_actor.pth")
        torch_load_weights(self.model_t_critic, path, "_model_t_critic.pth")
        torch_load_weights(self.model_critic, path, "_model_critic.pth")

    def __name__(self):
        return "torch_DeepDeterministicPolicyGradient"
