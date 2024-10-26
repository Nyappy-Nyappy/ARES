import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt
from scipy.stats import norm
import hashlib

# ブロックチェーンと因果推論を含むブロッククラス
class Block:
    def __init__(self, index, previous_hash, data):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data
        self.hash = self.calculate_hash()

    def calculate_hash(self):
        return hashlib.sha256((str(self.index) + str(self.previous_hash) + str(self.data)).encode()).hexdigest()

class Blockchain:
    def __init__(self):
        self.chain = [self.create_genesis_block()]

    def create_genesis_block(self):
        return Block(0, "0", "Genesis Block")

    def get_latest_block(self):
        return self.chain[-1]

    def add_block(self, data):
        new_block = Block(len(self.chain), self.get_latest_block().hash, data)
        self.chain.append(new_block)

# 量子インスパイアードベイズ最適化による行動選択
class QuantumInspiredBayesianOptimizer:
    def __init__(self):
        self.samples = []

    def suggest_action(self, q_values):
        mean, std_dev = np.mean(q_values), np.std(q_values)
        probabilities = norm.cdf(q_values, loc=mean, scale=std_dev)
        return np.argmax(probabilities)

# 部分観測での迷路環境
class PartialObservationMaze:
    def __init__(self, grid_size):
        self.grid_size = grid_size
        self.agent_position = [0, 0]
        self.goal_position = [grid_size - 1, grid_size - 1]
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        return self.get_partial_observation()

    def get_partial_observation(self):
        state = np.zeros((self.grid_size, self.grid_size, 1))
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

        return self.get_partial_observation(), reward, done

# エージェントクラス（ベイズ最適化と量子インスパイアードアプローチ）
class QuantumInspiredAgent:
    def __init__(self, state_size, action_size, blockchain):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95
        self.epsilon = 1.0
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.model = self._build_model()
        self.blockchain = blockchain
        self.optimizer = QuantumInspiredBayesianOptimizer()

    def _build_model(self):
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, (3, 3), activation='relu', input_shape=self.state_size),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation='relu'),
            tf.keras.layers.Dense(self.action_size, activation='linear')
        ])
        model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), loss='mse')
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        q_values = self.model.predict(state)
        return self.optimizer.suggest_action(q_values[0])

    def replay(self):
        if len(self.memory) < 32:
            return
        minibatch = random.sample(self.memory, 32)
        for state, action, reward, next_state, done in minibatch:
            target = reward if done else reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# トレーニングループ
if __name__ == "__main__":
    grid_size = 5
    env = PartialObservationMaze(grid_size)
    state_size = (grid_size, grid_size, 1)
    action_size = 4
    blockchain = Blockchain()
    agent = QuantumInspiredAgent(state_size, action_size, blockchain)
    
    episodes = 100
    scores = []
    
    for e in range(episodes):
        state = env.reset()
        state = np.reshape(state, [1] + list(state.shape))
        total_reward = 0

        for time in range(200):
            action = agent.act(state)
            next_state, reward, done = env.step(action)
            next_state = np.reshape(next_state, [1] + list(next_state.shape))
            agent.remember(state, action, reward, next_state, done)
            state = next_state
            total_reward += reward
            if done:
                print(f"Episode {e+1}/{episodes}, Score: {total_reward}, Epsilon: {agent.epsilon:.2}")
                scores.append(total_reward)
                blockchain.add_block({"episode": e, "score": total_reward})
                break

        agent.replay()
    
    # スコアのプロット
    plt.plot(scores)
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.title('Score per Episode')
    plt.show()
