import tensorflow as tf
import numpy as np
import random
from collections import deque
import matplotlib.pyplot as plt

# Definition of the POMDP environment (Partial Observation + Resolution Change)
class MultiResolutionPOMDPMazeEnvironment:
    def __init__(self, grid_size, observation_range, resolution_levels):
        self.grid_size = grid_size
        self.observation_range = observation_range
        self.resolution_levels = resolution_levels  # Resolution levels
        self.current_resolution = 0  # The initial resolution is the lowest (0)
        self.state_size = (grid_size // (2**self.current_resolution), grid_size // (2**self.current_resolution), 1)
        self.action_size = 4  # Up, Down, Left, Right
        self.reset()

    def reset(self):
        self.agent_position = [0, 0]
        self.goal_position = [self.grid_size - 1, self.grid_size - 1]
        self.current_resolution = 0  # Resolution resets to the initial level on reset
        self.state_size = (self.grid_size // (2**self.current_resolution), 
                           self.grid_size // (2**self.current_resolution), 1)
        return self.get_partial_observation()

    def get_partial_observation(self):
        # Returns the partial observation around the agent (adjusted based on resolution)
        obs_size = self.grid_size // (2**self.current_resolution)
        obs = np.zeros((obs_size, obs_size, 1))
        scale = 2 ** self.current_resolution

        for i in range(-self.observation_range, self.observation_range + 1):
            for j in range(-self.observation_range, self.observation_range + 1):
                xi, yj = self.agent_position[0] // scale + i, self.agent_position[1] // scale + j
                if 0 <= xi < obs_size and 0 <= yj < obs_size:
                    obs[xi, yj, 0] = 1  # Agent's position
        goal_pos = [self.goal_position[0] // scale, self.goal_position[1] // scale]
        obs[goal_pos[0], goal_pos[1], 0] = 2  # Goal position
        return obs

    def step(self, action):
        if action == 0 and self.agent_position[0] > 0:  # Up
            self.agent_position[0] -= 1
        elif action == 1 and self.agent_position[0] < self.grid_size - 1:  # Down
            self.agent_position[0] += 1
        elif action == 2 and self.agent_position[1] > 0:  # Left
            self.agent_position[1] -= 1
        elif action == 3 and self.agent_position[1] < self.grid_size - 1:  # Right
            self.agent_position[1] += 1

        reward = -1
        done = False

        if self.agent_position == self.goal_position:
            reward = 100
            done = True

        return self.get_partial_observation(), reward, done

    def increase_resolution(self):
        # Process to increase resolution
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
        
        # Value stream
        self.value_fc = tf.keras.layers.Dense(1, activation=None)
        
        # Advantage stream
        self.advantage_fc = tf.keras.layers.Dense(num_actions, activation=None)

    def call(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.flatten(x)
        x = self.lstm(tf.expand_dims(x, axis=0))  # LSTM layer
        x = self.fc1(x)
        value = self.value_fc(x)
        advantage = self.advantage_fc(x)
        q_values = value + (advantage - tf.reduce_mean(advantage, axis=1, keepdims=True))
        return q_values

# Definition of the agent (Supporting ARES-DQN)
class DQNAgent:
    def __init__(self, state_size, action_size):
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

    def _build_model(self):
        model = RecurrentDuelingDQN(self.action_size)
        model.compile(loss='mse', optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate))
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
    num_agents = 2  # Number of agents
    observation_range = 2  # Observation range of the agent
    resolution_levels = [0, 1, 2]  # Resolution levels

    # Initialize the environment and agents
    envs = [MultiResolutionPOMDPMazeEnvironment(grid_size=grid_size, observation_range=observation_range, 
                                                resolution_levels=resolution_levels) for _ in range(num_agents)]
    state_size = envs[0].state_size
    action_size = envs[0].action_size
    agents = [DQNAgent(state_size, action_size) for _ in range(num_agents)]
    
    episodes = 500
    scores = [[] for _ in range(num_agents)]
    cumulative_rewards = [[] for _ in range(num_agents)]
    
    for e in range(episodes):
        # Reset each agent's state
        states = [env.reset() for env in envs]
        states = [np.reshape(state, [1] + list(state.shape)) for state in states]
        total_rewards = [0] * num_agents

        for time in range(500):
            actions = [agent.act(state) for agent, state in zip(agents, states)]
            next_states, rewards, dones = [], [], []

            for i in range(num_agents):
                next_state, reward, done = envs[i].step(actions[i])
                next_states.append(np.reshape(next_state, [1] + list(next_state.shape)))
                rewards.append(reward)
                dones.append(done)
                total_rewards[i] += reward
                agents[i].remember(states[i], actions[i], reward, next_states[i], done)
                states[i] = next_states[i]

            if all(dones):
                break  # All agents are done

        # Change resolution for each agent
        for i in range(num_agents):
            if e % (episodes // len(resolution_levels)) == 0 and e != 0:
                envs[i].increase_resolution()
            scores[i].append(time)
            cumulative_rewards[i].append(total_rewards[i])
            agents[i].replay()

    # Plot the score (steps to goal) per episode for each agent
    for i in range(num_agents):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(scores[i])), scores[i])
        plt.xlabel('Episode')
        plt.ylabel('Steps to Goal')
        plt.title(f'Agent {i+1} Steps to Reach the Goal per Episode')
        plt.show()

    # Plot cumulative reward per episode for each agent
    for i in range(num_agents):
        plt.figure(figsize=(12, 6))
        plt.plot(range(len(cumulative_rewards[i])), cumulative_rewards[i])
        plt.xlabel('Episode')
        plt.ylabel('Cumulative Reward')
        plt.title(f'Agent {i+1} Cumulative Reward per Episode')
        plt.show()
