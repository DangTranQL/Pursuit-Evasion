import numpy as np
import math
import matplotlib.pyplot as plt
from maze import Maze
from agent import Agent
from qlearning import QLearning
from evader import choose_evader_action
from reward_pursuer import Reward
from reward_evader import Reward_Avoid

# Function to play the game using Q-learning with visualization
def train_agent(maze, pursuer, pursuer2, evader, q_learning, q_learning2, q_learning_avoid, max_steps=300):
    visited = []
    repeating = []
    visited2 = []
    repeating2 = []
    plt.figure(figsize=(6, 6))
    for step in range(max_steps):
        if pursuer.position == evader.position or pursuer2.position == evader.position:
            # if train_test == "test":
            #     plt.scatter(pursuer.position[1], pursuer.position[0], color='blue', marker='o', s=100, label='Pursuer')
            #     plt.scatter(pursuer2.position[1], pursuer2.position[0], color='green', marker='o', s=100, label='Pursuer2')
            #     plt.scatter(evader.position[1], evader.position[0], color='red', marker='x', s=100, label='Evader')

            #     # Plot legend and display
            #     plt.draw()
            #     plt.pause(0.3)  # Pause to visualize each step
            print("Caught the evader!")
            print("Step: ", step)
            break
        else:
            # if train_test == "test":
            #     plt.clf()
            #     plt.title(f"Step: {step + 1}")
            #     plt.xlim(-1, maze.width)
            #     plt.ylim(-1, maze.height)

            #     # Plot walls
            #     for wall in maze.walls:
            #         plt.fill_between([wall[1], wall[1] + 1], wall[0], wall[0] + 1, color='gray')

            #     # Plot pursuer and evader
            #     plt.scatter(pursuer.position[1], pursuer.position[0], color='blue', marker='o', s=100, label='Pursuer')
            #     plt.scatter(evader.position[1], evader.position[0], color='red', marker='x', s=100, label='Evader')
            #     plt.scatter(pursuer2.position[1], pursuer2.position[0], color='green', marker='o', s=100, label='Pursuer2')

            #     # Plot legend and display
            #     plt.legend()
            #     plt.draw()
            #     plt.pause(0.3)  # Pause to visualize each step

            pursuer_state = pursuer.position
            evader_state = evader.position
            pursuer2_state = pursuer2.position

            # Choose the direction with the maximum distance from the pursuer
            evader_action = q_learning_avoid.choose_action(pursuer_state)
            evader.move(evader_action)
            
            # Reward for the evader
            reward_evader = Reward_Avoid(pursuer, pursuer2, evader, maze)
            
            # Update Q-values
            q_learning_avoid.update_q_value(pursuer_state, evader_action, reward_evader, evader.position)
            q_learning_avoid.update_q_value(pursuer2_state, evader_action, reward_evader, evader.position)

            # Pursuer moves based on Q-learning
            pursuer_action = q_learning.choose_action(evader_state)
            pursuer.move(pursuer_action)
            pursuer2_action = q_learning2.choose_action(evader_state)
            pursuer2.move(pursuer2_action)

            # Reward for the pursuer
            reward = Reward(pursuer, pursuer2, evader, maze, visited, repeating)

            # Update Q-values
            q_learning.update_q_value(evader_state, pursuer_action, reward, evader.position)
            
            # Reward for the pursuer
            reward2 = Reward(pursuer2, pursuer, evader, maze, visited2, repeating2)

            # Update Q-values
            q_learning2.update_q_value(evader_state, pursuer2_action, reward2, evader.position)
            
def visualization(maze, pursuer, pursuer2, evader, q_learning, q_learning2, q_learning_avoid, max_steps=300):
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
            evader_state = evader.position
            pursuer2_state = pursuer2.position

            # Choose the direction with the maximum distance from the pursuer
            evader_action = q_learning_avoid.choose_action(pursuer_state)
            evader.move(evader_action)
            
            # # Reward for the evader
            # reward_evader = Reward_Avoid(pursuer, pursuer2, evader, maze)
            
            # # Update Q-values
            # q_learning_avoid.update_q_value(pursuer_state, evader_action, reward_evader, evader.position)
            # q_learning_avoid.update_q_value(pursuer2_state, evader_action, reward_evader, evader.position)

            # Pursuer moves based on Q-learning
            pursuer_action = q_learning.choose_action(evader_state)
            pursuer.move(pursuer_action)
            pursuer2_action = q_learning2.choose_action(evader_state)
            pursuer2.move(pursuer2_action)

            # # Reward for the pursuer
            # reward = Reward(pursuer, pursuer2, evader, maze, visited, repeating)

            # # Update Q-values
            # q_learning.update_q_value(evader_state, pursuer_action, reward, evader.position)
            
            # # Reward for the pursuer
            # reward2 = Reward(pursuer2, pursuer, evader, maze, visited2, repeating2)

            # # Update Q-values
            # q_learning2.update_q_value(evader_state, pursuer2_action, reward2, evader.position)