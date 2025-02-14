#ass 3 more
import numpy as np
import matplotlib.pyplot as plt

def gambler_value_iteration(ph, goal=100, max_capital=99, epsilon=1e-6, max_iterations=1000):
    # Initialize the value function V(s) for states s = 0, 1, ..., goal
    V = np.zeros(goal + 1)
    V[goal] = 1  # The value at the goal state is 1 (we win if we reach $100)

    # Initialize a policy array to store the optimal stake for each state
    policy = np.zeros(goal + 1, dtype=int)

    # Value Iteration
    for iteration in range(max_iterations):
        delta = 0
        # We iterate over all states from 1 to goal-1 (no need to check state 0 and state goal)
        for s in range(1, goal):
            # Best action and value initialization
            best_value = -1
            best_action = 0

            # Try all possible stakes a, where a can range from 0 to min(s, goal - s)
            for a in range(0, min(s, goal - s) + 1):
                # Compute the expected value for this action
                expected_value = ph * V[s + a] + (1 - ph) * V[s - a]

                # Track the best action
                if expected_value > best_value:
                    best_value = expected_value
                    best_action = a

            # Update the value function for this state
            delta = max(delta, abs(V[s] - best_value))
            V[s] = best_value
            policy[s] = best_action

        # If the value function has converged, we can stop early
        if delta < epsilon:
            print(f"Value iteration converged after {iteration + 1} iterations.")
            break

    return V, policy

def plot_results(V, policy, goal=100):
    # Plot the value function
    plt.figure(figsize=(12, 6))

    plt.subplot(1, 2, 1)
    plt.plot(range(goal), V[1:])
    plt.xlabel('Capital')
    plt.ylabel('Value Function (Probability of reaching $100)')
    plt.title('Value Function')

    # Plot the optimal policy
    plt.subplot(1, 2, 2)
    plt.plot(range(goal), policy[1:])
    plt.xlabel('Capital')
    plt.ylabel('Optimal Stake')
    plt.title('Optimal Policy')

    plt.tight_layout()
    plt.show()

# Test the function with ph = 0.25 and ph = 0.55
ph_025 = 0.25
ph_055 = 0.55

# Run value iteration for ph = 0.25
V_025, policy_025 = gambler_value_iteration(ph_025)
# Run value iteration for ph = 0.55
V_055, policy_055 = gambler_value_iteration(ph_055)

# Plot the results for ph = 0.25
print("Results for ph = 0.25:")
plot_results(V_025, policy_025)

# Plot the results for ph = 0.55
print("Results for ph = 0.55:")
plot_results(V_055, policy_055)
