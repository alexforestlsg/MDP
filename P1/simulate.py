import pandas as pd
import random
import csv
from tqdm import tqdm

df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()

# Extract optimal policy
optimal_policy = {}
for k in df['k'].unique():
    optimal_policy[k] = {}
    k_data = df[df['k'] == k]
    for x in k_data['x'].unique():
        x_data = k_data[k_data['x'] == x]
        best_row = x_data.loc[x_data['cost'].idxmin()]
        optimal_policy[k][int(x)] = int(best_row['u'])

trajectories_optimal = []
trajectories_baselines = []
num_simulations = 100000

for sim in tqdm(range(num_simulations), desc="Running simulations"):
    # Optimal policy
    x = 0
    total_cost = 0

    for k in range(0, 5):
        u = optimal_policy[k][min(x, 10)]
        w = random.randint(0, 10)
        immediate_cost = u + 2 * abs(x + u - w)
        total_cost += immediate_cost
        x = max(0, x + u - w)

        trajectories_optimal.append({
            'strategy': 'optimal',
            'simulation': sim,
            'k': k,
            'x': x,
            'u': u,
            'w': w,
            'immediate_cost': immediate_cost,
            'cumulative_cost': total_cost
        })

    # Baseline strategies
    for target_stock in tqdm(range(1, 11), leave=False, desc=f"Sim {sim}: Testing baselines"):
        x = 0
        total_cost = 0

        for k in range(0, 5):
            u = max(0, target_stock - x)
            w = random.randint(0, 10)
            immediate_cost = u + 2 * abs(x + u - w)
            total_cost += immediate_cost
            x = max(0, x + u - w)

            trajectories_baselines.append({
                'strategy': f'baseline_stock_{target_stock}',
                'simulation': sim,
                'k': k,
                'x': x,
                'u': u,
                'w': w,
                'immediate_cost': immediate_cost,
                'cumulative_cost': total_cost
            })

all_trajectories = trajectories_optimal + trajectories_baselines
optimal_costs = [t['cumulative_cost'] for t in trajectories_optimal if t['k'] == 4]
avg_optimal = sum(optimal_costs) / len(optimal_costs)

print(f"Simulated {num_simulations} trajectories")
print(f"\nOptimal Policy:")
print(f"  Average final cumulative cost: {avg_optimal:.4f}")
print(f"\nBaseline Strategies:")

baseline_comparison = []

for target_stock in range(1, 11):
    baseline_costs = [t['cumulative_cost'] for t in trajectories_baselines if t['k'] == 4 and t['strategy'] == f'baseline_stock_{target_stock}']
    avg_baseline = sum(baseline_costs) / len(baseline_costs)
    improvement = (avg_baseline - avg_optimal) / avg_baseline * 100
    
    print(f"  Stock up to {target_stock}: Avg cost = {avg_baseline:.4f}, Improvement = {improvement:.2f}%")
    
    baseline_comparison.append({
        'baseline_target_stock': target_stock,
        'average_cost': avg_baseline,
        'improvement_vs_optimal': improvement
    })

with open('baseline_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['baseline_target_stock', 'average_cost', 'improvement_vs_optimal'])
    writer.writeheader()
    writer.writerows(baseline_comparison)

print("\nOptimal Policy Details:")
for k in sorted(optimal_policy.keys()):
    print(f"k={k}: {optimal_policy[k]}")
