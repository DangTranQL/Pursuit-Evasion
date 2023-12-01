import math
import numpy as np

def choose_evader_action(pursuer_position, evader):
    evader_directions = [
            (-1, 0),  # Up
            (1, 0),   # Down
            (0, -1),  # Left
            (0, 1)    # Right
        ]
    
    # Calculate distance from future pursuer position
    future_distances = [math.sqrt((evader.position[0] + direction[0] - pursuer_position[0])**2 + 
                                  (evader.position[1] + direction[1] - pursuer_position[1])**2)
                        for direction in evader_directions]
    # Choose the direction with the maximum future distance from the pursuer
    evader_action = np.argmax(future_distances)
    return evader_action