import torch
import torch.nn as nn
from torch.nn import Linear, ReLU, Tanh, Conv2d


class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            Linear(self.state_dim, 64),
            Tanh(),
            Linear(64, self.hidden_dim),
            Tanh(),
            Linear(self.hidden_dim, self.action_dim)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class Critic(nn.Module):
    def __init__(self, state_dim, hidden_dim=32):
        super(Critic, self).__init__()
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.model = nn.Sequential(
            Linear(self.state_dim, 64),
            ReLU(),
            Linear(64, self.hidden_dim),
            ReLU(),
            Linear(self.hidden_dim, 1)
        )

    def forward(self, x):
        out = self.model(x)
        return out


class CriticConc(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=32):
        super(CriticConc, self).__init__()
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.dense1 = nn.Sequential(
            Linear(self.state_dim, 64),
            ReLU())

        self.dense2 = nn.Sequential(
            Linear(self.action_dim, self.hidden_dim),
            ReLU())

        self.dense3 = Linear(self.hidden_dim + 64, 1)

    def forward(self, x, action):
        d1 = self.dense1(x)
        d2 = self.dense2(action)
        concat = torch.cat((d1, d2), 1)
        d3 = self.dense3(concat)
        return d3


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super(ActorCritic, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.common = nn.Sequential(
        Linear(self.state_dim, 16),
        Tanh(),
        Linear(16, 32),
        Tanh())

        self.critic = nn.Sequential(
        Linear(32, self.hidden_dim),
        ReLU(),
        Linear(self.hidden_dim, 1))

        self.actor = nn.Sequential(
        Linear(32, self.hidden_dim),
        ReLU(),
        Linear(self.hidden_dim, self.action_dim))

    def forward(self, x):
        common = self.common(x)
        actor = self.actor(common)
        critic = self.critic(common)
        return critic, actor


class ConvActorCritic(nn.Module):
    def __init__(self, action_dim, state_dim, hidden_dim=128):
        super(ConvActorCritic, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim

        self.common = nn.Sequential()
        self.common.add(ReLU(Conv2d(self.state_dim, 32, 8, strides=4)))
        self.common.add(ReLU(Conv2d(32, 64, 4, strides=2)))
        self.common.add(ReLU(Conv2d(64, 64, 3, strides=1)))

        self.critic = nn.Sequential()
        self.critic.add(ReLU(Linear(64, self.hidden_dim)))
        self.critic.add(Linear(self.hidden_dim, 1))

        self.actor = nn.Sequential()
        self.actor.add(ReLU(Linear(64, self.hidden_dim)))
        self.actor.add(Linear(64, self.action_dim))

    def forward(self, x):
        common = self.common(x)
        flatten = nn.Flatten(common)
        actor = self.actor(flatten)
        critic = self.critic(flatten)
        return actor, critic


class ConvDeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(ConvDeepQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.conv = nn.Sequential()
        self.conv.add_module("conv_1", Conv2d(self.input_dim, 32, kernel_size=8, strides=4))
        self.conv.add_module("relu", ReLU())
        self.conv.add_module("conv_2", Conv2d(32, 64, kernel_size=4, strides=2))
        self.conv.add_module("relu", ReLU())
        self.conv.add_module("conv_3", Conv2d(64, 64, kernel_size=3, strides=1))
        self.conv.add_module("relu", ReLU())

        self.fc = nn.Sequential()
        self.fc.add_module("dense_1", Linear(64, 128))
        self.fc.add_module("relu", ReLU())
        self.fc.add_module("dense_2", Linear(128, 256))
        self.fc.add_module("relu", ReLU())
        self.fc.add_module("dense_3", Linear(self.output_dim))
        self.fc.add_module("relu", ReLU())

    def forward(self, state):
        features = self.conv(state)
        q_values = self.fc(features)
        return q_values


class DeepQNetwork(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.fc = nn.Sequential()
        self.fc.add_module("dense_1", Linear(self.input_dim, 128))
        self.fc.add_module("relu", ReLU())
        self.fc.add_module("dense_2", Linear(128, 256))
        self.fc.add_module("relu", ReLU())
        self.fc.add_module("dense_3", Linear(256, self.output_dim))

    def forward(self, state):
        q_values = self.fc(state)
        return q_values
