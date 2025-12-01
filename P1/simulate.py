import pandas as pd
import random
import csv
from tqdm import tqdm

# Read the results to extract optimal policy
df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip()

# Extract optimal policy: for each (k, x), find u that minimizes cost
optimal_policy = {}
for k in df['k'].unique():
    optimal_policy[k] = {}
    k_data = df[df['k'] == k]
    for x in k_data['x'].unique():
        x_data = k_data[k_data['x'] == x]
        best_row = x_data.loc[x_data['cost'].idxmin()]
        optimal_policy[k][int(x)] = int(best_row['u'])

# Simulation: Start at x=0 and apply optimal policy forward from k=0 to k=4
trajectories_optimal = []
trajectories_baselines = []
num_simulations = 100000

for sim in tqdm(range(num_simulations), desc="Running simulations"):
    # OPTIMAL POLICY SIMULATION
    x = 0
    total_cost = 0
    
    for k in range(0, 5):
        # Get optimal control
        u = optimal_policy[k][min(x, 10)] 
        
        # w~unif(0,10)
        w = random.randint(0, 10)
        
        # g
        immediate_cost = u + 2 * abs(x + u - w)
        total_cost += immediate_cost
        
        # state transition  
        x = max(0, x + u - w)
        
        # Log trajectory
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
    
    # BASELINE STRATEGIES
    for target_stock in tqdm(range(1, 11), leave=False, desc=f"Sim {sim}: Testing baselines"):
        x = 0
        total_cost = 0
        
        for k in range(0, 5):
            # Baseline control
            u = max(0, target_stock - x)
            
            # w~unif(0,10)
            w = random.randint(0, 10)
            
            # g
            immediate_cost = u + 2 * abs(x + u - w)
            total_cost += immediate_cost
            
            # state transition
            x = max(0, x + u - w)
            
            # Logging
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

# Combine trajectories
all_trajectories = trajectories_optimal + trajectories_baselines

""" # Save trajectories
with open('simulation_trajectories.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['strategy', 'simulation', 'k', 'x', 'u', 'w', 'immediate_cost', 'cumulative_cost'])
    writer.writeheader()
    writer.writerows(all_trajectories) """

# Calculate statistics
optimal_costs = [t['cumulative_cost'] for t in trajectories_optimal if t['k'] == 4]
avg_optimal = sum(optimal_costs) / len(optimal_costs)

print(f"Simulated {num_simulations} trajectories")
print(f"\nOptimal Policy:")
print(f"  Average final cumulative cost: {avg_optimal:.4f}")
print(f"\nBaseline Strategies:")

# Save baseline comparison results
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

# Save baseline comparison to CSV
with open('baseline_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['baseline_target_stock', 'average_cost', 'improvement_vs_optimal'])
    writer.writeheader()
    writer.writerows(baseline_comparison)

# Print the optimal policy
print("\nOptimal Policy Details:")
for k in sorted(optimal_policy.keys()):
    print(f"k={k}: {optimal_policy[k]}")
