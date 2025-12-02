import pandas as pd
import random
import csv
from tqdm import tqdm

df = pd.read_csv('results_infinite.csv')
df.columns = df.columns.str.strip()

# Extract optimal policy
optimal_policy = {}
for x in df['x'].unique():
    x_data = df[df['x'] == x]
    best_row = x_data.loc[x_data['cost'].idxmin()]
    optimal_policy[int(x)] = int(best_row['u'])

discount_factor = 0.99
num_simulations = 100
num_steps = 10000

print(f"Simulating {num_simulations} trajectories for {num_steps} steps")

optimal_total_costs = []
baseline_total_costs = {n: [] for n in range(1, 11)}

for sim in tqdm(range(num_simulations), desc="Running simulations"):
    # Optimal policy
    x = 0
    total_discounted_cost = 0
    discount_multiplier = 1.0

    for t in range(num_steps):
        u = optimal_policy[min(x, 10)]
        w = random.randint(0, 10)
        immediate_cost = u + 2 * abs(x + u - w)
        total_discounted_cost += discount_multiplier * immediate_cost
        x = max(0, min(10, x + u - w))
        discount_multiplier *= discount_factor

    optimal_total_costs.append(total_discounted_cost)

    # Baseline strategies
    for target_stock in tqdm(range(1, 11), leave=False, desc=f"Sim {sim}: Testing baselines"):
        x = 0
        total_discounted_cost = 0
        discount_multiplier = 1.0

        for t in range(num_steps):
            u = max(0, target_stock - x)
            w = random.randint(0, 10)
            immediate_cost = u + 2 * abs(x + u - w)
            total_discounted_cost += discount_multiplier * immediate_cost
            x = max(0, min(10, x + u - w))
            discount_multiplier *= discount_factor

        baseline_total_costs[target_stock].append(total_discounted_cost)

avg_optimal = sum(optimal_total_costs) / len(optimal_total_costs)

print(f"\nSimulated {num_simulations} trajectories for {num_steps} steps")
print(f"\nOptimal Policy:")
print(f"  Average total discounted cost: {avg_optimal:.4f}")
print(f"\nBaseline Strategies:")

baseline_comparison = []

for target_stock in range(1, 11):
    avg_baseline = sum(baseline_total_costs[target_stock]) / len(baseline_total_costs[target_stock])
    improvement = (avg_baseline - avg_optimal) / avg_baseline * 100
    
    print(f"  Stock up to {target_stock}: Avg cost = {avg_baseline:.4f}, Improvement = {improvement:.2f}%")
    
    baseline_comparison.append({
        'baseline_target_stock': target_stock,
        'average_discounted_cost': avg_baseline,
        'improvement_vs_optimal': improvement
    })

with open('baseline_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['baseline_target_stock', 'average_discounted_cost', 'improvement_vs_optimal'])
    writer.writeheader()
    writer.writerows(baseline_comparison)

print("\nOptimal Policy Details:")
for x in sorted(optimal_policy.keys()):
    print(f"  x={x}: u={optimal_policy[x]}")
