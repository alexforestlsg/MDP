import csv
from tqdm import tqdm

discount_factor = 0.99
max_iterations = 10000
convergence_threshold = 1e-6

V = {x: 0 for x in range(0, 11)}
V_old = {x: 0 for x in range(0, 11)}

optimal_policy = {}
value_history = []
iteration_history = []

def bellman_update(x, V):
    u_costs = []
    for u in range(0, 11):
        cost_over_w = 0
        for w in range(0, 11):
            next_state = max(0, min(10, x + u - w))
            cost_over_w += u + 2 * abs(x + u - w) + discount_factor * V[next_state]
        avg_cost = cost_over_w / 11
        u_costs.append((u, avg_cost))

    min_idx = min(range(len(u_costs)), key=lambda i: u_costs[i][1])
    return u_costs[min_idx][1], u_costs[min_idx][0]

print("Running value iteration for infinite horizon problem...")
print(f"Discount factor: {discount_factor}")
print(f"Convergence threshold: {convergence_threshold}\n")

for iteration in tqdm(range(max_iterations), desc="Value Iteration"):
    V_old = V.copy()

    new_policy = {}
    for x in range(0, 11):
        V[x], best_u = bellman_update(x, V_old)
        new_policy[x] = best_u

    max_diff = max(abs(V[x] - V_old[x]) for x in V)

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

u_cost_pairs = []
for x in range(0, 11):
    u_costs = []
    for u in range(0, 11):
        cost_over_w = 0
        for w in range(0, 11):
            next_state = max(0, min(10, x + u - w))
            cost_over_w += u + 2 * abs(x + u - w) + discount_factor * V[next_state]
        avg_cost = cost_over_w / 11
        u_costs.append((u, avg_cost))
        u_cost_pairs.append((x, u, avg_cost))

with open('results_infinite.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['x', 'u', 'cost'])
    for row in u_cost_pairs:
        writer.writerow(row)

with open('iteration_history.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['iteration', 'max_value_diff', 'converged'])
    writer.writeheader()
    writer.writerows(iteration_history)

print("\nOptimal Policy (Infinite Horizon):")
for x in sorted(optimal_policy.keys()):
    print(f"  x={x}: u={optimal_policy[x]}")

print("\nValue Function (Infinite Horizon):")
for x in sorted(V.keys()):
    print(f"  V({x}) = {V[x]:.6f}")
