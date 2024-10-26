import tensorflow as tf
import numpy as np
import random
import hashlib
from collections import deque
import matplotlib.pyplot as plt

#tf.keras.mixed_precision.set_global_policy('mixed_float16')

# Simple blockchain class for tamper prevention
class Block:
    def __init__(self, index, previous_hash, data):
        self.index = index
        self.previous_hash = previous_hash
        self.data = data  # Results and rewards of each episode
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

    def tamper_block(self, index, new_data):
        if 0 < index < len(self.chain):
            self.chain[index].data = new_data
            self.chain[index].hash = self.chain[index].calculate_hash()

# Tampering blockchain through the attack agent
class AttackAgent:
    def __init__(self, blockchain):
        self.blockchain = blockchain

    def tamper_blockchain(self, target_index, fake_data):
        self.blockchain.tamper_block(target_index, fake_data)
        print(f"Block {target_index} has been tampered with!")

# Definition of the POMDP environment (Partial Observation + Resolution Change)
class MultiResolutionPOMDPMazeEnvironment:
    def __init__(self, grid_size, observation_range, resolution_levels):
        self.grid_size = grid_size
        self.observation_range = observation_range
        self.resolution_levels = resolution_levels
        self.current_resolution = 0
        self.state_size = (grid_size // (2**self.current_resolution), grid_size // (2**self.current_resolution), 1)
        self.action_size = 4  # Up, Down, Left, Right
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.current_resolution = 0
        self.state_size = (self.grid_size // (2**self.current_resolution), 
                           self.grid_size // (2**self.current_resolution), 1)
        return self.get_partial_observation()

    def get_partial_observation(self):
        obs_size = self.grid_size // (2**self.current_resolution)
        obs = np.zeros((obs_size, obs_size, 1))
        scale = 2 ** self.current_resolution

        for i in range(-self.observation_range, self.observation_range + 1):
            for j in range(-self.observation_range, self.observation_range + 1):
                xi, yj = self.agent_position[0] // scale + i, self.agent_position[1] // scale + j
                if 0 <= xi < obs_size and 0 <= yj < obs_size:
                    obs[xi, yj, 0] = 1
        goal_pos = [self.goal_position[0] // scale, self.goal_position[1] // scale]
        obs[goal_pos[0], goal_pos[1], 0] = 2
        return obs

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

    def increase_resolution(self):
        if self.current_resolution < len(self.resolution_levels) - 1:
            self.current_resolution += 1
            self.state_size = (self.grid_size // (2**self.current_resolution), 
                               self.grid_size // (2**self.current_resolution), 1)
            return True
        return False

# Definition of the Recurrent Dueling DQN model
class RecurrentDuelingDQN(tf.keras.Model):
    def __init__(self, num_actions):
        super(RecurrentDuelingDQN, self).__init__()
        self.conv1 = tf.keras.layers.Conv2D(32, (3, 3), activation='relu')
        self.conv2 = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')
        self.flatten = tf.keras.layers.Flatten()
        self.lstm = tf.keras.layers.LSTM(128, return_sequences=False)
        self.fc1 = tf.keras.layers.Dense(128, activation='relu')
        
        self.value_fc = tf.keras.layers.Dense(1, activation=None)
        self.advantage_fc = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.lstm(tf.expand_dims(x, axis=0))
        x = self.fc1(x)
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values

# Definition of the agent (Supporting ARES-DQN)
class DQNAgent:
    def __init__(self, state_size, action_size, blockchain):
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=2000)
        self.gamma = 0.95    # Discount rate
        self.epsilon = 1.0   # Exploration rate
        self.epsilon_min = 0.01
        self.epsilon_decay = 0.995
        self.learning_rate = 0.001
        self.batch_size = 32
        self.model = self._build_model()
        self.blockchain = blockchain  # Shared blockchain

    def _build_model(self):
        model = RecurrentDuelingDQN(self.action_size)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate), jit_compile=False)
        return model

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def act(self, state):
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self):
        if len(self.memory) < self.batch_size:
            return
        minibatch = random.sample(self.memory, self.batch_size)
        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = reward + self.gamma * np.amax(self.model.predict(next_state)[0])
            target_f = self.model.predict(state)
            target_f[0][action] = target
            self.model.fit(state, target_f, epochs=1, verbose=0)
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

# Main training loop
if __name__ == "__main__":
    grid_size = 10  # Maze size
    num_agents = 3  # Number of agents (one is an attack agent)
    observation_range = 2
    resolution_levels = [0, 1, 2]  # Resolution levels

    blockchain = Blockchain()  # Create blockchain

    # Initialize environment and agents, including attack agent
    envs = [MultiResolutionPOMDPMazeEnvironment(grid_size=grid_size, observation_range=observation_range, 
                                                resolution_levels=resolution_levels) for _ in range(num_agents)]
    state_size = envs[0].state_size
    action_size = envs[0].action_size
    agents = [DQNAgent(state_size, action_size, blockchain) for _ in range(num_agents - 1)]
    attack_agent = AttackAgent(blockchain)  # Create attack agent
    
    episodes = 500
    scores = [[] for _ in range(num_agents)]
    cumulative_rewards = [[] for _ in range(num_agents)]
    success_rates = [[] for _ in range(num_agents - 1)]
    resolution_changes = [[] for _ in range(num_agents - 1)]
    tamper_counts = []

    for e in range(episodes):
        # Reset the state for each agent
        states = [env.reset() for env in envs]
        states = [np.reshape(state, [1] + list(state.shape)) for state in states]
        total_rewards = [0] * num_agents

        for time in range(500):
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones = [], [], []

            for i in range(num_agents - 1):
                next_state, reward, done = envs[i].step(actions[i])
                next_states.append(np.reshape(next_state, [1] + list(next_state.shape)))
                rewards.append(reward)
                dones.append(done)
                total_rewards[i] += reward
                agents[i].remember(states[i], actions[i], reward, next_states[i], done)
                states[i] = next_states[i]
                success_rates[i].append(1 if done else 0)  # Add 1 on success

            if all(dones):
                break  # All agents are done

        # Store episode results in the blockchain
        blockchain.add_block({"episode": e, "rewards": total_rewards[:num_agents-1]})

        # Attack agent tampers with the blockchain
        if np.random.rand() < 0.3:  # 30% chance of tampering with a block
            attack_agent.tamper_blockchain(target_index=random.randint(1, e), fake_data="Tampered Data!")
            tamper_counts.append(e)  # Record the episode of tampering

        # Change resolution for each agent
        for i in range(num_agents - 1):
            if e % (episodes // len(resolution_levels)) == 0 and e != 0:
                envs[i].increase_resolution()
                resolution_changes[i].append(e)  # Record resolution change
            scores[i].append(time)
            cumulative_rewards[i].append(total_rewards[i])
            agents[i].replay()

    # Plot success rate for each agent
    for i in range(num_agents - 1):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(success_rates[i])), success_rates[i])
        plt.xlabel('Episode')
        plt.ylabel('Success Rate')
        plt.title(f'Agent {i+1} Success Rate per Episode')
        plt.show()

    # Plot tamper occurrences per episode
    plt.figure(figsize=(12, 6))
    plt.hist(tamper_counts, bins=episodes // 10)
    plt.xlabel('Episode')
    plt.ylabel('Tamper Count')
    plt.title('Tamper Occurrences per Episode')
    plt.show()

    # Plot resolution change points for each agent
    for i in range(num_agents - 1):
        plt.figure(figsize=(12, 6))
        plt.plot(resolution_changes[i], [1] * len(resolution_changes[i]), 'o')
        plt.xlabel('Episode')
        plt.ylabel('Resolution Change')
        plt.title(f'Agent {i+1} Resolution Change Points')
        plt.show()
