import math

def Reward_Avoid(pursuer, pursuer2, evader, maze):
    dist = math.sqrt((pursuer.position[0]-evader.position[0])**2 + (pursuer.position[1]-evader.position[1])**2)
    dist2 = math.sqrt((pursuer2.position[0]-evader.position[0])**2 + (pursuer2.position[1]-evader.position[1])**2)
    if pursuer.position == evader.position or  pursuer2.position == evader.position:
        reward = -100
    else:
        if maze.is_wall(evader.position):
            reward = -100
        else:
            if dist > 1 and dist2 > 1:
                reward = 1 + max(dist, dist2)
            else:
                reward = -10
    return reward