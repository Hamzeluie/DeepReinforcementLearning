import os
import threading
import gym
import multiprocessing
import numpy as np
from queue import Queue
import torch
from torch import nn
from networks.torch_nets import ActorCritic
from settings import LEARNING_RATE, SAVE_DIR, GAMMA
from utils.utils import torch_save_weights, torch_load_weights


class Memory:
  def __init__(self):
    self.states = []
    self.actions = []
    self.rewards = []

  def store(self, state, action, reward):
    self.states.append(state)
    self.actions.append(action)
    self.rewards.append(reward)

  def clear(self):
    self.states = []
    self.actions = []
    self.rewards = []


class MasterAgent():

  def __init__(self, env_name):
    self.env_name = env_name
    env = gym.make(self.env_name)
    self.state_size = env.observation_space.shape[0]
    self.action_size = env.action_space.n
    self.global_model = ActorCritic(self.state_size, self.action_size)  # global network
    self.opt = torch.optim.Adam(self.global_model.parameters(), lr=LEARNING_RATE)

  def train(self):
    res_queue = Queue()

    workers = [Worker(self.state_size,
                      self.action_size,
                      self.global_model,
                      self.opt, res_queue,
                      i, env_name=self.env_name) for i in range(multiprocessing.cpu_count())]

    for i, worker in enumerate(workers):
      print("Starting worker {}".format(i))
      worker.start()

    moving_average_rewards = []  # record episode reward to plot
    while True:
      reward = res_queue.get()

      if reward is not None:
        moving_average_rewards.append(reward)

      else:

        break

  def play(self):
    env = gym.make(self.env_name).unwrapped
    state = env.reset()
    model = self.global_model
    path = os.path.join(SAVE_DIR, f"{self.env_name}_a3c_worker")
    torch_load_weights(model, path, ".pth")
    print('Loading model from: {}'.format(path))
    done = False
    step_counter = 0
    reward_sum = 0

    try:
      while not done:
        env.render(mode='rgb_array')
        value, policy = model(torch.tensor(state[None, :], dtype=torch.float32))
        policy = nn.Softmax(dim=1)(policy)
        dist = torch.distributions.Categorical(probs=policy)
        action = int(dist.sample())
        state, reward, done, _ = env.step(np.array(action))
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

  def __init__(self,
               state_size,
               action_size,
               global_model,
               opt,
               result_queue,
               idx,
               env_name='CartPole-v0',
               save_dir='/tmp'):
    super(Worker, self).__init__()
    self.state_size = state_size
    self.action_size = action_size
    self.result_queue = result_queue
    self.global_model = global_model
    self.opt = opt
    self.local_model = ActorCritic(self.state_size, self.action_size)
    self.worker_idx = idx
    self.env_name = env_name
    self.env = gym.make(self.env_name).unwrapped
    self.save_dir = save_dir
    self.ep_loss = 0.0
    self.loss = nn.CrossEntropyLoss()

  def run(self):
    total_step = 1
    mem = Memory()
    while Worker.global_episode < 50:
      done = False
      current_state = self.env.reset()
      mem.clear()
      ep_reward = 0.
      ep_steps = 0
      self.ep_loss = 0
      time_count = 0
      while not done:
        _, logits = self.local_model(
            torch.tensor(current_state[None, :],
                                 dtype=torch.float32))
        probs = nn.Softmax(dim=1)(logits)

        action = np.random.choice(self.action_size, p=probs.detach().numpy()[0])
        new_state, reward, done, _ = self.env.step(action)
        if done:
          reward = -1
        ep_reward += reward
        mem.store(current_state, action, reward)

        if time_count == 50 or done:
          # Calculate gradient wrt to local model. We do so by tracking the
          # variables involved in computing the loss by using tf.GradientTape
          total_loss = self.compute_loss(done,
                                       new_state,
                                       mem
                                       )
          self.opt.zero_grad()
          total_loss.backward()
          self.opt.step()

          self.local_model.load_state_dict(self.global_model.state_dict())
          mem.clear()
          time_count = 0
          if ep_reward > Worker.best_score:
            with Worker.save_lock:
              print("Saving best model to {}, "
                    "episode score: {}".format(self.save_dir, ep_reward))
              path = os.path.join(SAVE_DIR, f"{self.env_name}_a3c_worker")
              torch_save_weights(self.global_model, path, Worker.best_score, ".pth")
              Worker.best_score = ep_reward
          Worker.global_episode += 1
        time_count += 1
        current_state = new_state
        ep_steps += 1
        total_step += 1
    self.result_queue.put(None)

  def compute_loss(self,
                   done,
                   new_state,
                   memory):
    if done:
      reward_sum = 0.  # terminal
    else:
      reward_sum, _ = self.local_model(torch.tensor(new_state[None, :],
                                                    dtype=torch.float32))[-1].detach().numpy()[0]

    # Get discounted rewards
    discounted_rewards = []
    for reward in memory.rewards[::-1]:  # reverse buffer r
      reward_sum = reward + GAMMA * reward_sum
      discounted_rewards.append(reward_sum)
    discounted_rewards.reverse()

    values, logits = self.local_model(
        torch.tensor(np.vstack(memory.states),
                             dtype=torch.float32))
    # Get our advantages
    advantage = torch.tensor(np.array(discounted_rewards)[:, None],
                            dtype=torch.float32) - values
    # loss
    value_loss = advantage.pow(2).mean()
    policy = nn.Softmax(dim=1)(logits)
    dist = torch.distributions.Categorical(probs=policy)
    policy_loss = (-dist.log_prob(torch.tensor(memory.actions, dtype=torch.float32))
                   * advantage.detach()).mean()
    total_loss = torch.mean((0.5 * value_loss + policy_loss))
    return total_loss