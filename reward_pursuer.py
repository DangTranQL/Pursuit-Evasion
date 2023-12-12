import math

def Reward(pursuer, pursuer2, evader, maze, visited, repeating):
    dist = math.sqrt((pursuer.position[0]-evader.position[0])**2 + (pursuer.position[1]-evader.position[1])**2)
    sep = math.sqrt((pursuer.position[0]-pursuer2.position[0])**2 + (pursuer.position[1]-pursuer2.position[1])**2)
    if pursuer.position == evader.position:
        reward = 10
    else:
        if maze.is_wall(pursuer.position):
            reward = -100
        elif sep <= 3:
            reward = -50
        else:
            if not (pursuer.position in visited):
                visited.append(pursuer.position)
                repeating.append(0)
                reward = 1 - dist
            else:
                idx = visited.index(pursuer.position)
                repeating[idx] += 1
                reward = 1 - dist - repeating[idx]
    return reward