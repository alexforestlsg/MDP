import csv
from tqdm import tqdm

# Infinite horizon problem with discount factor
discount_factor = 0.99
max_iterations = 10000
convergence_threshold = 1e-6

# Initialize value function V(x) for all states x in [0, 10]
V = {x: 0 for x in range(0, 11)}
V_old = {x: 0 for x in range(0, 11)}

# Store optimal policy and value function history
optimal_policy = {}
value_history = []
iteration_history = []

def bellman_update(x, V):
    """
    Compute the Bellman optimality update for state x.
    Returns the minimum cost-to-go across all actions.
    """
    u_costs = []
    for u in range(0, 11):
        cost_over_w = 0
        for w in range(0, 11):
            # Immediate cost + discounted future cost
            next_state = max(0, min(10, x + u - w))  # Clamp to [0, 10]
            cost_over_w += u + 2 * abs(x + u - w) + discount_factor * V[next_state]
        avg_cost = cost_over_w / 11
        u_costs.append((u, avg_cost))
    
    # Return minimum cost and the action that achieves it
    min_idx = min(range(len(u_costs)), key=lambda i: u_costs[i][1])
    return u_costs[min_idx][1], u_costs[min_idx][0]

# Value iteration
print("Running value iteration for infinite horizon problem...")
print(f"Discount factor: {discount_factor}")
print(f"Convergence threshold: {convergence_threshold}\n")

for iteration in tqdm(range(max_iterations), desc="Value Iteration"):
    # Store old values
    V_old = V.copy()
    
    # Update value function for all states (0 to 10)
    new_policy = {}
    for x in range(0, 11):
        V[x], best_u = bellman_update(x, V_old)
        new_policy[x] = best_u
    
    # Check for convergence
    max_diff = max(abs(V[x] - V_old[x]) for x in V)
    
    # Record iteration history every 100 iterations
    if iteration % 100 == 0 or iteration == max_iterations - 1:
        iteration_history.append({
            'iteration': iteration,
            'max_value_diff': max_diff,
            'converged': max_diff < convergence_threshold
        })
    
    if max_diff < convergence_threshold:
        print(f"\nConverged after {iteration + 1} iterations")
        optimal_policy = new_policy
        break
    
    optimal_policy = new_policy

print(f"Final max value difference: {max_diff}")

# Save value function and optimal policy to CSV
u_cost_pairs = []
for x in range(0, 11):
    u_costs = []
    for u in range(0, 11):
        cost_over_w = 0
        for w in range(0, 11):
            next_state = max(0, min(10, x + u - w))  # Clamp to [0, 10]
            cost_over_w += u + 2 * abs(x + u - w) + discount_factor * V[next_state]
        avg_cost = cost_over_w / 11
        u_costs.append((u, avg_cost))
        u_cost_pairs.append((x, u, avg_cost))

# Save results to CSV
with open('results_infinite.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'u', 'cost'])
    for row in u_cost_pairs:
        writer.writerow(row)

# Save iteration history
with open('iteration_history.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['iteration', 'max_value_diff', 'converged'])
    writer.writeheader()
    writer.writerows(iteration_history)

# Print optimal policy
print("\nOptimal Policy (Infinite Horizon):")
for x in sorted(optimal_policy.keys()):
    print(f"  x={x}: u={optimal_policy[x]}")

print("\nValue Function (Infinite Horizon):")
for x in sorted(V.keys()):
    print(f"  V({x}) = {V[x]:.6f}")
