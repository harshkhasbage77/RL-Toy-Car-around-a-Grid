import numpy as np
from env import Environment
from utils import generate_all_states, get_state_mappings, ACTIONS
from utils import get_next_state, get_reward, value_iteration

def main():
    grid_size = 8
    states = generate_all_states(grid_size)
    state_to_index, index_to_state = get_state_mappings(states)
    
    V, policy = value_iteration(states, state_to_index, index_to_state, grid_size)
    
    # Save Value Function and Policy if needed
    np.save('value_function.npy', V)
    np.save('value_iteration_policy.npy', policy)
    
    print("Value Iteration completed.")
    # Optionally, visualize the policy
    # ...

if __name__ == "__main__":
    main()
