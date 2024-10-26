import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# 波動関数の時間発展行列の定義
def time_evolution_matrix(dim, theta=0.1):
    """
    シュレディンガー方程式の時間発展を模倣するための行列を定義
    dim: 行列の次元
    theta: 回転角、探索の度合いを制御
    """
    U = np.eye(dim) * np.cos(theta) + np.sin(theta) * np.roll(np.eye(dim), 1, axis=0)
    return U

# 環境クラス
class QuantumInspiredEnvironment:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.state_size = (grid_size, grid_size, 1)
        self.action_size = 4
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        return self.get_state()

    def get_state(self):
        state = np.zeros(self.state_size)
        state[self.agent_position[0], self.agent_position[1], 0] = 1
        state[self.goal_position[0], self.goal_position[1], 0] = 2
        return state

    def step(self, action):
        if action == 0 and self.agent_position[0] > 0:
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:
            self.agent_position[1] += 1

        reward = -1
        done = False

        if self.agent_position == self.goal_position:
            reward = 100
            done = True

        return self.get_state(), reward, done

# エージェントクラス
class QuantumInspiredAgent:
    def __init__(self, state_size, action_size, time_evolution_matrix):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.time_evolution_matrix = time_evolution_matrix  # 時間発展行列
        self.model = self._build_model()

    def _build_model(self):
        model = tf.keras.Sequential([
            tf.keras.layers.Flatten(input_shape=(self.state_size[0], self.state_size[1])),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        
        # 擬似量子効果を反映
        act_values = np.dot(self.time_evolution_matrix, act_values[0])
        return np.argmax(act_values)

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                next_target = self.model.predict(next_state)
                next_target = np.dot(self.time_evolution_matrix, next_target[0])
                target = reward + self.gamma * np.amax(next_target)
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# メインのトレーニングループ
if __name__ == "__main__":
    grid_size = 5
    env = QuantumInspiredEnvironment(grid_size=grid_size)
    state_size = env.state_size
    action_size = env.action_size
    time_evolution = time_evolution_matrix(action_size, theta=0.1)
    agent = QuantumInspiredAgent(state_size, action_size, time_evolution)

    episodes = 500
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1, state_size[0], state_size[1]])
        total_reward = 0
        for time in range(500):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1, state_size[0], state_size[1]])
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode: {e+1}/{episodes}, Score: {time}, Total Reward: {total_reward}")
                break
        agent.replay()
