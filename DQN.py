import gym
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import deque
import numpy as np
import random
import datetime


class CNN(torch.nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.convolution1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.convolution2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=5)
        self.convolution3 = nn.Conv2d(in_channels=32, out_channels=64, kernel_size=7)
        self.fc1 = nn.Linear(in_features=1792, out_features=40)
        self.fc2 = nn.Linear(in_features=40, out_features=9)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.convolution1(x), 3))
        x = F.relu(F.max_pool2d(self.convolution2(x), 3))
        x = F.relu(F.max_pool2d(self.convolution3(x), 3, 2))
        x = x.view(x.size(0), -1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x


model = CNN()
criterion = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)


def train(input, target, epochs=1):
    input = torch.from_numpy(input).float()
    target = torch.from_numpy(target)
    y_pred = 0
    for t in range(1):
        y_pred = model(input)
        loss = criterion(y_pred, target)
        # print(t, loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()


class DQNAgent:
    def __init__(self, action_size=9):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95  # discount rate
        self.epsilon = 1.0  # exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        state_tensor = torch.from_numpy(state).float()
        act_values = self.model(state_tensor).detach().numpy()
        return np.argmax(act_values[0])  # returns action

    def replay(self, batch_size):
        minibatch = random.sample(self.memory, batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (-reward + self.gamma *
                          np.amax(self.model(torch.from_numpy(next_state).float())[0].detach().numpy()))
            target_f = self.model(torch.from_numpy(state).float()).detach().numpy()
            target_f[0][action] = target
            train(state, target_f, epochs=1)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_state_dict(torch.load(name))

    def save(self, name):
        torch.save(self.model.state_dict(), name)


env = gym.make('MsPacman-v0')
state_size = env.observation_space.shape
action_size = env.action_space.n
agent = DQNAgent()
done = False
batch_size = 32
EPISODES = 30
for e in range(EPISODES):
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    vw = cv2.VideoWriter(str(e) + str(datetime.datetime.now()) + '.avi', fourcc, 4, (160, 210))
    state = env.reset()
    state = np.reshape(state, (1, 210, 160, 3)).transpose(0, 3, 1, 2)
    for time in range(1000000000):
        print(time)
        action = agent.act(state)
        next_state, reward, done, _ = env.step(action)
        vw.write(next_state)
        reward = reward if not done else -10
        next_state = np.reshape(next_state, (1, 210, 160, 3)).transpose(0, 3, 1, 2)
        agent.remember(state, action, reward, next_state, done)
        state = next_state
        if done:
            vw.release()
            agent.save(str(e) + str(datetime.datetime.now()) + '.pt')
            print("episode: {}/{}, score: {}, e: {:.2}".format(e, EPISODES, time, agent.epsilon))
            break
        if len(agent.memory) > batch_size:
            agent.replay(batch_size)
