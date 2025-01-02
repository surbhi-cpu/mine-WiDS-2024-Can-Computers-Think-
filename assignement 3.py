import numpy as np
import matplotlib.pyplot as plt

# Constants
max_cars_at_location = 20  # Maximum number of cars at each location
rental_revenue = 10  # Revenue from renting out one car
move_cost = 2  # Cost of moving one car between locations
discount_factor = 0.9  # Discount factor (gamma)

# Define the state space
states = [(n1, n2) for n1 in range(max_cars_at_location + 1) for n2 in range(max_cars_at_location + 1)]

# Initialize the value function V(s) for each state
V = {state: 0 for state in states}  # V(s) = 0 initially

# Define the action space
def possible_actions(state):
    n1, n2 = state
    actions = []
    
    # Rent cars out from Location 1 (from 0 to n1 cars) or Location 2 (from 0 to n2 cars)
    for rent_from_location_1 in range(min(n1, 1) + 1):  # Renting a car from Location 1 (max 1 car)
        for rent_from_location_2 in range(min(n2, 1) + 1):  # Renting a car from Location 2 (max 1 car)
            # Moving cars between locations
            for move_from_location_1_to_2 in range(max_cars_at_location + 1):  # Move from Location 1 to Location 2
                for move_from_location_2_to_1 in range(max_cars_at_location + 1):  # Move from Location 2 to Location 1
                    actions.append((rent_from_location_1, rent_from_location_2, move_from_location_1_to_2, move_from_location_2_to_1))
    return actions

# Define the transition function and reward function
def transition(state, action):
    n1, n2 = state
    rent_from_location_1, rent_from_location_2, move_from_location_1_to_2, move_from_location_2_to_1 = action
    
    # Transition the state based on the action
    new_n1 = n1 - rent_from_location_1 - move_from_location_2_to_1 + move_from_location_1_to_2
    new_n2 = n2 - rent_from_location_2 - move_from_location_1_to_2 + move_from_location_2_to_1
    new_n1 = max(0, min(new_n1, max_cars_at_location))  # Ensure cars don't go below 0 or above the max
    new_n2 = max(0, min(new_n2, max_cars_at_location))  # Ensure cars don't go below 0 or above the max
    
    return (new_n1, new_n2)

def reward(state, action):
    rent_from_location_1, rent_from_location_2, move_from_location_1_to_2, move_from_location_2_to_1 = action
    # Reward is rental revenue for rented cars
    reward = rental_revenue * (rent_from_location_1 + rent_from_location_2)
    # Subtract cost of moving cars
    reward -= move_cost * (move_from_location_1_to_2 + move_from_location_2_to_1)
    return reward

# Value Iteration Algorithm
tolerance = 1e-6  # Convergence tolerance
converged = False
iterations = 0

while not converged:
    max_change = 0
    for state in states:
        # Update V(s) using the Bellman equation
        best_action_value = -float('inf')
        for action in possible_actions(state):
            next_state = transition(state, action)
            expected_value = reward(state, action) + discount_factor * V[next_state]
            best_action_value = max(best_action_value, expected_value)
        
        max_change = max(max_change, abs(V[state] - best_action_value))
        V[state] = best_action_value

    if max_change < tolerance:
        converged = True
    iterations += 1
    if iterations % 100 == 0:
        print(f"Iteration {iterations}, max_change = {max_change}")

print(f"Converged after {iterations} iterations")

# Create heatmap for the value function (V(s)) visualization
value_grid = np.zeros((max_cars_at_location + 1, max_cars_at_location + 1))

# Store the values in the grid
for state, value in V.items():
    n1, n2 = state
    value_grid[n1, n2] = value

# Plot the Value Function Heatmap
plt.figure(figsize=(8, 6))
plt.imshow(value_grid, cmap='inferno', interpolation='nearest')
plt.colorbar(label='Value Function V(s)')
plt.title("Value Function Heatmap (V(s))")
plt.xlabel('Location 2')
plt.ylabel('Location 1')
plt.show()
