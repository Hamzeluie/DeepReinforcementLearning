import keras
from tensorflow.keras import layers


class Actor(keras.Model):
    def __init__(self, action_dim, hidden_dim=128):
        super(Actor, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim
        self.dense1 = layers.Dense(64, activation="relu")
        self.dense2 = layers.Dense(self.hidden_dim, activation="relu")
        self.dense3 = layers.Dense(self.action_dim)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x


class Critic_conc(keras.Model):
    def __init__(self, hidden_dim=32):
        super(Critic_conc, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(self.hidden_dim, activation="relu")
        self.dense3 = layers.Dense(1)

    def call(self, state, action):
        x1 = self.dense1(state)
        x2 = self.dense2(action)
        concat = layers.Concatenate()([x1, x2])
        x = self.dense3(concat)
        return x


class Critic(keras.Model):
    def __init__(self, hidden_dim=32):
        super(Critic, self).__init__()
        self.hidden_dim = hidden_dim
        self.dense1 = layers.Dense(32, activation="relu")
        self.dense2 = layers.Dense(self.hidden_dim, activation="relu")
        self.dense3 = layers.Dense(1)

    def call(self, x):
        x = self.dense1(x)
        x = self.dense2(x)
        x = self.dense3(x)
        return x

class Actor_Critic(keras.Model):
    def __init__(self, action_dim, hidden_dim=128):
        super(Actor_Critic, self).__init__()
        self.action_dim = action_dim
        self.hidden_dim = hidden_dim

        self.common = keras.Sequential()
        self.common.add(layers.Dense(16, activation="tanh"))
        self.common.add(layers.Dense(32, activation="tanh"))

        self.critic = keras.Sequential()
        self.critic.add(layers.Dense(self.hidden_dim, activation="relu"))
        self.critic.add(layers.Dense(1))

        self.actor = keras.Sequential()
        self.actor.add(layers.Dense(self.hidden_dim, activation="relu"))
        self.actor.add(layers.Dense(self.action_dim, activation="linear"))

    def call(self, x):
        common = self.common(x)
        actor = self.actor(common)
        critic = self.critic(common)
        return critic, actor


class Conv_Actor_Critic(keras.Model):
    def __init__(self, action_dim, state_dim, hidden_dim=128):
        super(Conv_Actor_Critic, self).__init__()
        self.action_dim = action_dim
        self.state_dim = state_dim
        self.hidden_dim = hidden_dim
        self.common = keras.Sequential()
        self.common.add(layers.Conv2D(32, 8, strides=4, activation="relu"))
        self.common.add(layers.Conv2D(64, 4, strides=2, activation="relu"))
        self.common.add(layers.Conv2D(64, 3, strides=1, activation="relu"))

        self.critic = keras.Sequential()
        self.critic.add(layers.Dense(self.hidden_dim, activation="relu"))
        self.critic.add(layers.Dense(1))

        self.actor = keras.Sequential()
        self.actor.add(layers.Dense(self.hidden_dim, activation="relu"))
        self.actor.add(layers.Dense(self.action_dim))

    def call(self, x):
        common = self.common(x)
        flatten = layers.Flatten()(common)
        actor = self.actor(flatten)
        critic = self.critic(flatten)
        return actor, critic


class ConvDeepQNetwork(keras.Model):
    def __init__(self, state_dim, action_dim):
        super(ConvDeepQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim
        self.conv = keras.Sequential()
        self.conv.add(layers.Conv2D(32, kernel_size=8, strides=4))
        self.conv.add(layers.ReLU())
        self.conv.add(layers.Conv2D(64, kernel_size=4, strides=2))
        self.conv.add(layers.ReLU())
        self.conv.add(layers.Conv2D(64, kernel_size=3, strides=1))
        self.conv.add(layers.ReLU())

        self.fc = keras.Sequential()
        self.fc.add(layers.Dense(128))
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(256))
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(self.output_dim))
        self.fc.add(layers.ReLU())

    def __call__(self, state):

        features = self.conv(state)
        q_values = self.fc(features)
        return q_values


class DeepQNetwork(keras.Model):
    def __init__(self, state_dim, action_dim):
        super(DeepQNetwork, self).__init__()
        self.input_dim = state_dim
        self.output_dim = action_dim

        self.fc = keras.Sequential()
        self.fc.add(layers.Dense(128))
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(256))
        self.fc.add(layers.ReLU())
        self.fc.add(layers.Dense(self.output_dim))

    def call(self, state):
        q_values = self.fc(state)
        return q_values

