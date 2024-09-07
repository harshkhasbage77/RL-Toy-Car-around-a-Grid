import numpy as np
from env import Environment
from utils import generate_all_states, get_state_mappings, ACTIONS
from utils import q_learning

def main():
    grid_size = 8
    states = generate_all_states(grid_size)
    state_to_index, index_to_state = get_state_mappings(states)
    
    Q = np.zeros((len(states), len(ACTIONS)))
    
    # Define hyperparameters
    alpha = 0.1
    gamma = 0.9
    epsilon = 1.0
    epsilon_min = 0.1
    epsilon_decay = 0.995
    num_episodes = 1000
    
    env = Environment(grid_size=grid_size)
    
    Q = q_learning(env, states, state_to_index, index_to_state, grid_size, Q, alpha, gamma, epsilon, epsilon_min, epsilon_decay, num_episodes)
    
    # Save Q-table if needed
    np.save('q_table.npy', Q)
    
    print("Q-Learning completed.")
    # Optionally, visualize the policy
    # ...

if __name__ == "__main__":
    main()
