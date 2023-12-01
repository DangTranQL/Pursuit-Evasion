import numpy as np
import random
import math
import matplotlib.pyplot as plt

# Define the maze (grid)
class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = [(1, 2), (2, 2), (3, 2), (3, 3), (3, 4)]  # Define wall positions

    def is_wall(self, position):
        return position in self.walls

# Define the pursuer and evader
class Agent:
    def __init__(self, maze, position):
        self.maze = maze
        self.position = position

    def move(self, action):
        if action == 0:  # Up
            new_position = (self.position[0] - 1, self.position[1])
        elif action == 1:  # Down
            new_position = (self.position[0] + 1, self.position[1])
        elif action == 2:  # Left
            new_position = (self.position[0], self.position[1] - 1)
        elif action == 3:  # Right
            new_position = (self.position[0], self.position[1] + 1)
        else:
            raise ValueError("Invalid action")

        if (0 <= new_position[0] < self.maze.height and
                0 <= new_position[1] < self.maze.width and
                not self.maze.is_wall(new_position)):
            self.position = new_position

# Q-learning for the pursuer
class QLearning:
    def __init__(self, num_actions, learning_rate, discount_factor, epsilon):
        self.num_actions = num_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = {}

    def get_q_value(self, state, action):
        if state not in self.q_table:
            self.q_table[state] = [0] * self.num_actions
        return self.q_table[state][action]
    
    def choose_action(self, state):
        if state not in self.q_table:
            action = np.random.choice(range(self.num_actions))
        else:
            if np.random.uniform(0, 1) < self.epsilon:
                action = np.random.choice(range(self.num_actions))
            else:
                action = np.argmax(self.q_table[state])
                # action = np.argmax(self.q_table.get(state, [0] * self.num_actions))
        return action

    def update_q_value(self, state, action, reward, next_state):
        current_q = self.get_q_value(state, action)
        max_next_q = max(self.get_q_value(next_state, a) for a in range(self.num_actions))
        new_q = current_q + self.learning_rate * (reward + self.discount_factor * max_next_q - current_q)
        self.q_table[state][action] = new_q
    
def choose_evader_action(pursuer_position):
    evader_directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]
    
    # Predict pursuer's future position
    # future_pursuer_position = [pursuer_position[0] + pursuer_velocity[0], pursuer_position[1] + pursuer_velocity[1]]
    # Calculate distance from future pursuer position
    future_distances = [math.sqrt((evader.position[0] + direction[0] - pursuer_position[0])**2 + 
                                  (evader.position[1] + direction[1] - pursuer_position[1])**2)
                        for direction in evader_directions]
    # Choose the direction with the maximum future distance from the pursuer
    evader_action = np.argmax(future_distances)
    return evader_action

# Function to play the game using Q-learning with visualization
def play_game_with_qlearning_visualize(maze, pursuer, evader, q_learning, pursuer2, q_learning2, max_steps=300):
    visited = []
    repeating = []
    visited2 = []
    repeating2 = []
    plt.figure(figsize=(6, 6))
    for step in range(max_steps):
        plt.clf()
        plt.title(f"Step: {step + 1}")
        plt.xlim(-1, maze.width)
        plt.ylim(-1, maze.height)

        # Plot walls
        for wall in maze.walls:
            plt.fill_between([wall[1], wall[1] + 1], wall[0], wall[0] + 1, color='gray')

        # Plot pursuer and evader
        plt.scatter(pursuer.position[1], pursuer.position[0], color='blue', marker='o', s=100, label='Pursuer')
        plt.scatter(evader.position[1], evader.position[0], color='red', marker='x', s=100, label='Evader')
        plt.scatter(pursuer2.position[1], pursuer2.position[0], color='green', marker='o', s=100, label='Pursuer2')

        # Plot legend and display
        plt.legend()
        plt.draw()
        plt.pause(0.3)  # Pause to visualize each step

        pursuer_state = pursuer.position
        # evader_state = evader.position
        pursuer2_state = pursuer2.position

        # Choose the direction with the maximum distance from the pursuer
        evader_action = choose_evader_action(pursuer.position)
        evader.move(evader_action)

        # Pursuer moves based on Q-learning
        pursuer_action = q_learning.choose_action(pursuer_state)
        pursuer.move(pursuer_action)
        pursuer2_action = q_learning.choose_action(pursuer2_state)
        pursuer2.move(pursuer2_action)

        # Reward for the pursuer
        dist = math.sqrt((pursuer.position[0]-evader.position[0])**2 + (pursuer.position[1]-evader.position[1])**2)
        if pursuer.position == evader.position:
            reward = 10
        else:
            if maze.is_wall(pursuer.position):
                reward = -100
            elif pursuer.position == pursuer2.position:
                reward = -1
            else:
                if not (pursuer.position in visited):
                    visited.append(pursuer.position)
                    repeating.append(0)
                    reward = 1 - dist
                else:
                    idx = visited.index(pursuer.position)
                    repeating[idx] += 1
                    reward = 1 - dist - repeating[idx]

        # Update Q-values
        q_learning.update_q_value(pursuer_state, pursuer_action, reward, pursuer.position)
        
        # Reward for the pursuer
        dist2 = math.sqrt((pursuer2.position[0]-evader.position[0])**2 + (pursuer2.position[1]-evader.position[1])**2)
        if pursuer2.position == evader.position:
            reward2 = 10
        else:
            if maze.is_wall(pursuer2.position):
                reward2 = -100
            elif pursuer2.position == pursuer.position:
                reward2 = -1
            else:
                if not (pursuer2.position in visited2):
                    visited2.append(pursuer2.position)
                    repeating2.append(0)
                    reward2 = 1 - dist2
                else:
                    idx = visited2.index(pursuer2.position)
                    repeating2[idx] += 1
                    reward2 = 1 - dist2 - repeating2[idx]

        # Update Q-values
        q_learning2.update_q_value(pursuer2_state, pursuer2_action, reward2, pursuer2.position)

        if pursuer.position == evader.position or pursuer2.position == evader.position:
            plt.scatter(pursuer.position[1], pursuer.position[0], color='blue', marker='o', s=100, label='Pursuer')
            plt.scatter(pursuer2.position[1], pursuer2.position[0], color='green', marker='o', s=100, label='Pursuer2')
            plt.scatter(evader.position[1], evader.position[0], color='red', marker='x', s=100, label='Evader')

            # Plot legend and display
            plt.draw()
            plt.pause(0.3)  # Pause to visualize each step
            print("Caught the evader!")
            print("Step: ", step)
            break

# Initialize the maze and agents
maze = Maze(width=7, height=7)
pursuer = Agent(maze, position=(0, 0))
pursuer2 = Agent(maze, position=(3,6))
evader = Agent(maze, position=(4, 4))

# Initialize Q-learning for the pursuer
num_actions = 4  # Up, Down, Left, Right
q_learning = QLearning(num_actions, 0.005, 0.9, 0)
q_learning2 = QLearning(num_actions, 0.005, 0.9, 0)

# Play the game using Q-learning with visualization
play_game_with_qlearning_visualize(maze, pursuer, evader, q_learning, pursuer2, q_learning2)
plt.show()