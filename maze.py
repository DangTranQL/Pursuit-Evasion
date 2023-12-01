# Define the maze (grid)
class Maze:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.walls = [(1, 2), (2, 2), (3, 2), (3, 3), (3, 4)]  # Define wall positions

    def is_wall(self, position):
        return position in self.walls