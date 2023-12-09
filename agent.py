import random

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

        if (0 <= new_position[0] <= self.maze.height and
                0 <= new_position[1] <= self.maze.width and
                not self.maze.is_wall(new_position)):
            self.position = new_position
            
    def reset(self):
        while True:
            x = random.randint(0, self.maze.width)
            y = random.randint(0, self.maze.height)
            position = (x, y)
            if not self.maze.is_wall(position):
                return position