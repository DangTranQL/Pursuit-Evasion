from visualize import train_agent
from maze import Maze
from agent import Agent
from qlearning import QLearning
from joblib import dump

# Initialize the maze and agents
maze = Maze(width=7, height=7)
pursuer = Agent(maze, position=(0, 0))
pursuer2 = Agent(maze, position=(3,6))
evader = Agent(maze, position=(4, 4))

# Initialize Q-learning for the pursuer
num_actions = 4  # Up, Down, Left, Right
q_learning = QLearning(num_actions, 0.001, 0.9, 0)
q_learning2 = QLearning(num_actions, 0.001, 0.9, 0)
q_learning_avoid = QLearning(num_actions, 0.001, 0.9, 0)

def train(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid, epochs = 250):
    for epoch in range(epochs):
        pursuer.reset()
        pursuer2.reset()
        evader.reset()
        train_agent(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)
        print("Epoch: ", epoch)
    
train(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)

# Save Q-learning models
dump(q_learning, 'q_learning_model.joblib')
dump(q_learning2, 'q_learning2_model.joblib')
dump(q_learning_avoid, 'q_learning_avoid_model.joblib')


    