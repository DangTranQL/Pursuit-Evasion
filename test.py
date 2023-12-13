from maze import Maze
from agent import Agent
from visualize import testing
import matplotlib.pyplot as plt
from joblib import load

# Initialize the maze and agents
maze = Maze(width=8, height=8)
pursuer = Agent(maze, position=(0, 0))
pursuer2 = Agent(maze, position=(3,6))
evader = Agent(maze, position=(4, 4))

# Load Q-learning models
q_learning = load('q_learning_model.joblib')
q_learning2 = load('q_learning2_model.joblib')
q_learning_avoid = load('q_learning_avoid_model.joblib')

def test_agents(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid, cases=300):
    data_x = []
    data_y = []
    for case in range(cases):
        # pursuer.position = (0, 0)
        # pursuer2.position = (3,6)
        # evader.position = (4, 4)
        pursuer.reset()
        pursuer2.reset()
        evader.reset()
        step = testing(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)
        if step != None:
            data_x.append(case)
            data_y.append(step)
        print("Epoch: ", case, "Steps: ", step)

    # Plotting the data
    plt.figure(figsize=(6, 6))
    plt.plot(data_x, data_y, linestyle='-', color='b') 
    plt.title('Epochs vs Steps')
    plt.xlabel('Epochs')
    plt.ylabel('Steps')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
test_agents(maze, pursuer, pursuer2, evader, q_learning,q_learning2, q_learning_avoid)
    