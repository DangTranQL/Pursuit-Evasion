from joblib import load
from visualize import visualization
from maze import Maze
from agent import Agent
from qlearning import QLearning

def test(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid):
    pursuer.reset()
    pursuer2.reset()
    evader.reset()
    visualization(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)

# Initialize the maze and agents
maze = Maze(width=8, height=8)
pursuer = Agent(maze, position=(0, 0))
pursuer2 = Agent(maze, position=(3,6))
evader = Agent(maze, position=(4, 4))

# Load Q-learning models
q_learning = load('q_learning_model.joblib')
q_learning2 = load('q_learning2_model.joblib')
q_learning_avoid = load('q_learning_avoid_model.joblib')

test(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)

