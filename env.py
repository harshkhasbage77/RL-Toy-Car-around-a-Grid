import pygame
import math
import time

class Environment:
    def __init__(self, grid_size=30, cell_size=20):
        self.grid_size = grid_size
        self.cell_size = cell_size
        self.window_size = (grid_size * cell_size, grid_size * cell_size+50)
        self.car_x, self.car_y = grid_size // 2, 2 * grid_size // 3 
        self.car_direction = 'up'
        self.goal = (grid_size - 1, 0)
        self.screen = pygame.display.set_mode(self.window_size)
        pygame.display.set_caption('RL Car Environment Visualization')
        self.last_action = 'start'
        self.last_reward = 0
        self.last_done = False

    def _get_radar_signals(self):
        """Calculates radar signals (1 or 0) based on whether obstacles are in each direction."""
        radar_signals = [0, 0, 0, 0]  # Signals for [Up, Down, Left, Right]
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # (delta_y, delta_x)

        for i, direction in enumerate(directions):
            next_y = self.car_y + direction[0]
            next_x = self.car_x + direction[1]

            if next_x < 0 or next_x >= self.grid_size or next_y < 0 or next_y >= self.grid_size:
                radar_signals[i] = 0  # Obstacle (boundary) detected
            else:
                radar_signals[i] = 1  # No obstacle detected

        return radar_signals

    def draw_grid_with_circular_path(self):
        center_x, center_y = self.grid_size // 2, self.grid_size // 2  # Center of the grid
        radius = self.grid_size // 3  # Adjust the radius to control the size of the circular path
        
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                # Default grid color
                pygame.draw.rect(self.screen, (230, 230, 250), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)
                
                # Calculate the distance from the current cell to the center of the grid
                distance = math.sqrt((col - center_x) ** 2 + (row - center_y) ** 2)
                
                # If the cell is within the circular path (with some tolerance for grid fitting)
                if radius - 1 <= distance <= radius + 1:
                    pygame.draw.rect(self.screen, (139, 69, 19), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))

    def reset(self):
        """Resets the environment to the initial state and returns the state and radar signals."""
        # self.car_x, self.car_y = 0, self.grid_size
        self.car_x, self.car_y = self.grid_size // 2, 5 * self.grid_size // 6 + 1
        self.car_direction = 'up'
        # radar_signals = self._get_radar_signals()

    def step(self, action):
        """Updates the car position based on the action and returns the next state and radar signals."""
        self.last_action = action
        if action == 'forward':
            if self.car_direction == 'up':
                self.car_y = max(self.car_y - 1, 0)
            elif self.car_direction == 'down':
                self.car_y = min(self.car_y + 1, self.grid_size - 1)
            elif self.car_direction == 'left':
                self.car_x = max(self.car_x - 1, 0)
            elif self.car_direction == 'right':
                self.car_x = min(self.car_x + 1, self.grid_size - 1)

        elif action == 'right':
            if self.car_direction == 'up':
                self.car_direction = 'right'
            elif self.car_direction == 'down':
                self.car_direction = 'left'
            elif self.car_direction == 'left':
                self.car_direction = 'up'
            elif self.car_direction == 'right':
                self.car_direction = 'down'
        
        elif action == 'left':
            if self.car_direction == 'up':
                self.car_direction = 'left'
            elif self.car_direction == 'down':
                self.car_direction = 'right'
            elif self.car_direction == 'left':
                self.car_direction = 'down'
            elif self.car_direction == 'right':
                self.car_direction = 'up'

        radar_signals = self._get_radar_signals()  # Get radar signals after taking the action
        
        # isOnPath = (self.car_y == 0 and self.car_x == self.grid_size - 1) or (self.car_x == self.grid_size - 2) or (self.car_y == self.grid_size - 1 and self.car_x != self.grid_size - 1)
        
        # We need to define a circular isOnPath condition

        # Center of the grid
        center_x, center_y = self.grid_size // 2, self.grid_size // 2
        # Radius for the circular path
        radius = self.grid_size // 3

        # Calculate the distance from the car's current position to the center of the grid
        distance_from_center = math.sqrt((self.car_x - center_x) ** 2 + (self.car_y - center_y) ** 2)

        # Determine if the car is on the circular path
        isOnPath = (radius - 1 <= distance_from_center <= radius + 1)

        if (self.car_x, self.car_y) == self.goal:
            reward = 10
            done = True
        elif isOnPath:
            reward = 1
            done = False
        elif self.car_x < 0 or self.car_x >= self.grid_size or self.car_y < 0 or self.car_y >= self.grid_size:
            reward = -10
            done = True
        else:
            reward = -1
            done = False

        self.last_reward = reward
        self.last_done = done

        self._update_visualization()
        time.sleep(0.5)  # Delay for visualization

        return (self.car_x, self.car_y, self.car_direction), reward, done, {}

    def render(self):
        self._update_visualization()

    def _update_visualization(self):
        self.screen.fill((255, 255, 255))
        for row in range(self.grid_size):
            for col in range(self.grid_size):
                pygame.draw.rect(self.screen, (230, 230, 250), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size), 1)
                # if (row == 0 and col == self.grid_size - 1) or (col == self.grid_size - 2) or (row == self.grid_size - 1 and col != self.grid_size - 1):
                #     pygame.draw.rect(self.screen, (139, 69, 19), (col * self.cell_size, row * self.cell_size, self.cell_size, self.cell_size))
        env.draw_grid_with_circular_path()

        car_center = (self.car_x * self.cell_size + self.cell_size // 2, self.car_y * self.cell_size + self.cell_size // 2)
        car_side_length = self.cell_size // 2
        if self.car_direction == 'up':
            car_points = [(car_center[0], car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.car_direction == 'down':
            car_points = [(car_center[0], car_center[1] + car_side_length),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length)]
        elif self.car_direction == 'left':
            car_points = [(car_center[0] - car_side_length, car_center[1]),
                          (car_center[0] + car_side_length, car_center[1] - car_side_length),
                          (car_center[0] + car_side_length, car_center[1] + car_side_length)]
        elif self.car_direction == 'right':
            car_points = [(car_center[0] + car_side_length, car_center[1]),
                          (car_center[0] - car_side_length, car_center[1] - car_side_length),
                          (car_center[0] - car_side_length, car_center[1] + car_side_length)]

        pygame.draw.polygon(self.screen, (0, 0, 0), car_points)

        # Draw information box
        info_box = pygame.Surface((self.window_size[0], 50))
        info_box.fill((200, 200, 200))
        info_text = f"State: ({self.car_x}, {self.car_y}, {self.car_direction});  Action: {self.last_action};  Reward: {self.last_reward};  Done: {self.last_done}"
        font = pygame.font.Font(None, 24)
        text = font.render(info_text, True, (0, 0, 0))
        info_box.blit(text, (10, 10))
        self.screen.blit(info_box, (0, self.window_size[1] - 50))

        pygame.display.flip()

# Initialize Pygame
pygame.init()

# Create environment instance
env = Environment()

# Example of using the environment
state = env.reset()
env.render()
for _ in range(7):
    action = 'forward'
    next_state, reward, done, _ = env.step(action)
    env.render()

next_state, reward, done, _ = env.step('right')
env.render()

for i in range(10):
    action = 'forward'
    # print(i)
    next_state, reward, done, _ = env.step(action)
    env.render()

# Quit Pygame
pygame.quit()

