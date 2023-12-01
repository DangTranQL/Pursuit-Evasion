import pygame
import random
import math

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
PLAYER_SIZE = 10
ENEMY_SIZE = 10
PLAYER_SPEED = 5
ENEMY_SPEED = 3
WHITE = (255, 255, 255)
RED = (255, 0, 0)
BLUE = (0, 0, 255)

# Create the window
window = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Pursuit Evasion Game")

# Clock to control the frame rate
clock = pygame.time.Clock()

# Define the player and enemy positions
player_x, player_y = WIDTH // 2, HEIGHT // 2
enemy_x, enemy_y = random.randint(0, WIDTH), random.randint(0, HEIGHT)

# Game loop
running = True
while running:
    window.fill(WHITE)

    # Handle events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # Calculate the distance between player and enemy
    distance = math.sqrt((player_x - enemy_x) ** 2 + (player_y - enemy_y) ** 2)

    # Evasive movement for the player
    if distance < 150:  # If the enemy is within a certain distance
        dx = player_x - enemy_x
        dy = player_y - enemy_y
        angle = math.atan2(dy, dx)

        # Calculate the direction away from the pursuer
        evade_angle = angle  # Angle 180 degrees opposite from the pursuer

        # Calculate the new position
        new_player_x = player_x + PLAYER_SPEED * math.cos(evade_angle)
        new_player_y = player_y + PLAYER_SPEED * math.sin(evade_angle)

        # Boundary check for the player's new position
        if 0 <= new_player_x <= WIDTH - PLAYER_SIZE and 0 <= new_player_y <= HEIGHT - PLAYER_SIZE:
            player_x = new_player_x
            player_y = new_player_y
        else:
            # If hitting the border, change direction randomly
            new_evade_angle = random.uniform(0, math.pi)
            player_x += PLAYER_SPEED * math.cos(new_evade_angle)
            player_y += PLAYER_SPEED * math.sin(new_evade_angle)

    # Enemy movement (pursuit)
    if enemy_x < player_x:
        enemy_x += ENEMY_SPEED
    elif enemy_x > player_x:
        enemy_x -= ENEMY_SPEED

    if enemy_y < player_y:
        enemy_y += ENEMY_SPEED
    elif enemy_y > player_y:
        enemy_y -= ENEMY_SPEED

    # Boundary check for the enemy's position
    enemy_x = max(0, min(enemy_x, WIDTH - ENEMY_SIZE))
    enemy_y = max(0, min(enemy_y, HEIGHT - ENEMY_SIZE))

    # Draw player and enemy
    pygame.draw.rect(window, RED, (player_x, player_y, PLAYER_SIZE, PLAYER_SIZE))
    pygame.draw.rect(window, BLUE, (enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE))

    # Collision detection
    player_rect = pygame.Rect(player_x, player_y, PLAYER_SIZE, PLAYER_SIZE)
    enemy_rect = pygame.Rect(enemy_x, enemy_y, ENEMY_SIZE, ENEMY_SIZE)
    if player_rect.colliderect(enemy_rect):
        running = False
        print("Game Over! You were caught by the enemy.")

    # Update the display
    pygame.display.update()
    clock.tick(60)

# Quit Pygame
pygame.quit()
