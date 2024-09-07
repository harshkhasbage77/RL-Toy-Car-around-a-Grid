import itertools
import numpy as np
from env import Environment
import random



DIRECTIONS = ['up', 'right', 'down', 'left']
ACTIONS = ['forward', 'left', 'right']

def generate_all_states(grid_size):
    states = []
    for x, y, direction in itertools.product(range(grid_size), range(grid_size), DIRECTIONS):
        states.append((x, y, direction))
    return states

def get_state_mappings(states):
    state_to_index = {state: idx for idx, state in enumerate(states)}
    index_to_state = {idx: state for idx, state in enumerate(states)}
    return state_to_index, index_to_state

# grid_size = 3
# all_states = generate_all_states(grid_size)
# state_to_index = {state: idx for idx, state in enumerate(all_states)}
# index_to_state = {idx: state for idx, state in enumerate(all_states)}

# states = generate_all_states(3)
# print(states)
# print(len(states))
# print(get_state_mappings(states))

def get_next_state(state, action, grid_size):
    x, y, direction = state
    new_direction = direction
    new_x, new_y = x, y

    if action == 'left':
        current_idx = DIRECTIONS.index(direction)
        new_direction = DIRECTIONS[(current_idx - 1) % 4]
    elif action == 'right':
        current_idx = DIRECTIONS.index(direction)
        new_direction = DIRECTIONS[(current_idx + 1) % 4]
    elif action == 'forward':
        if direction == 'up':
            new_y = max(y - 1, 0)
        elif direction == 'down':
            new_y = min(y + 1, grid_size - 1)
        elif direction == 'left':
            new_x = max(x - 1, 0)
        elif direction == 'right':
            new_x = min(x + 1, grid_size - 1)
    
    return (new_x, new_y, new_direction)

def get_reward(state, action, next_state, grid_size):
    car_x, car_y, _ = next_state
    goal = (grid_size - 1, 0)
    
    # Define path conditions based on the provided Environment class
    isOnPath = ((car_y == 0 and car_x == grid_size - 1) or 
               (car_x == grid_size - 2) or 
               (car_y == grid_size - 1 and car_x != grid_size - 1))
    
    if next_state[:2] == goal:
        return 10
    elif isOnPath:
        return 1
    elif (car_x < 0 or car_x >= grid_size or car_y < 0 or car_y >= grid_size):
        return -10
    else:
        return -1


def value_iteration(states, state_to_index, index_to_state, grid_size, gamma=0.9, theta=1e-4):
    num_states = len(states)
    num_actions = len(ACTIONS)
    
    V = np.zeros(num_states)
    policy = np.zeros(num_states, dtype=int)
    
    while True:
        delta = 0
        for state in states:
            s = state_to_index[state]
            v = V[s]
            action_values = []
            for a, action in enumerate(ACTIONS):
                next_state = get_next_state(state, action, grid_size)
                r = get_reward(state, action, next_state, grid_size)
                s_prime = state_to_index.get(next_state, None)
                if s_prime is not None:
                    action_values.append(r + gamma * V[s_prime])
                else:
                    # If next_state is invalid, assume zero value
                    action_values.append(r)
            V[s] = max(action_values)
            policy[s] = np.argmax(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    
    return V, policy

# i dont know what to do here
def q_learning(env, states, state_to_index, index_to_state, grid_size, Q, alpha, gamma, epsilon, epsilon_min, epsilon_decay, num_episodes):
    for episode in range(num_episodes):
        env.reset()
        state = (env.car_x, env.car_y, env.car_direction)
        done = False
        
        while not done:
            s = state_to_index[state]
            
            # Epsilon-greedy action selection
            if random.uniform(0, 1) < epsilon:
                a = random.randint(0, len(ACTIONS)-1)
            else:
                a = np.argmax(Q[s])
            
            action = ACTIONS[a]
            next_state_tuple, reward, done, _ = env.step(action)
            next_state = (next_state_tuple[0], next_state_tuple[1], env.car_direction)
            s_prime = state_to_index.get(next_state, None)
            
            if s_prime is not None:
                Q[s, a] = Q[s, a] + alpha * (reward + gamma * np.max(Q[s_prime]) - Q[s, a])
            else:
                # If next_state is invalid, assume zero value
                Q[s, a] = Q[s, a] + alpha * (reward - Q[s, a])
            
            state = next_state
        
        # Decay epsilon
        if epsilon > epsilon_min:
            epsilon *= epsilon_decay
        
        # Optionally, print progress
        if (episode+1) % 100 == 0:
            print(f"Episode {episode+1}/{num_episodes} completed.")
    
    return Q
