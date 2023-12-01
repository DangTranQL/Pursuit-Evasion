import numpy as np
import math
import matplotlib.pyplot as plt
from maze import Maze
from agent import Agent
from qlearning import QLearning
from evader import choose_evader_action
from reward import Reward

# Function to play the game using Q-learning with visualization
def play_game_with_qlearning_visualize(maze, pursuer, evader, q_learning, pursuer2, q_learning2, max_steps=300):
    visited = []
    repeating = []
    visited2 = []
    repeating2 = []
    plt.figure(figsize=(6, 6))
    for step in range(max_steps):
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
        else:
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
            evader_action = choose_evader_action(pursuer.position, evader)
            evader.move(evader_action)

            # Pursuer moves based on Q-learning
            pursuer_action = q_learning.choose_action(pursuer_state)
            pursuer.move(pursuer_action)
            pursuer2_action = q_learning.choose_action(pursuer2_state)
            pursuer2.move(pursuer2_action)

            # Reward for the pursuer
            reward = Reward(pursuer, pursuer2, evader, maze, visited, repeating)

            # Update Q-values
            q_learning.update_q_value(pursuer_state, pursuer_action, reward, pursuer.position)
            
            # Reward for the pursuer
            reward2 = Reward(pursuer2, pursuer, evader, maze, visited2, repeating2)

            # Update Q-values
            q_learning2.update_q_value(pursuer2_state, pursuer2_action, reward2, pursuer2.position)

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