import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import csv

f_0 = lambda y: np.exp(-0.5 * y**2) / np.sqrt(2 * np.pi)
f_1 = lambda y: np.exp(-0.5 * (y - 1)**2) / np.sqrt(2 * np.pi)
l_0 = 25
l_1 = 15
C = 1
N = 5

def compute_cost_functions(N, p_values):
    n_p = len(p_values)
    n_samples = 1000
    y_vals = np.linspace(-5, 6, n_samples)

    f_0 = np.exp(-0.5 * y_vals**2) / np.sqrt(2 * np.pi)
    f_1 = np.exp(-0.5 * (y_vals - 1)**2) / np.sqrt(2 * np.pi)

    J = {}
    A = {}
    J[N] = np.minimum((1 - p_values) * l_0, p_values * l_1)
    for k in tqdm(range(N-1, -1, -1), desc="Computing stages backwards", total=N):
        J_next = J[k+1]

        J_k = np.zeros(n_p)
        A_k = np.zeros(n_p)

        for i, p in enumerate(p_values):
            expected_future = 0
            for j, y in enumerate(y_vals):
                num = p * f_1[j]
                denom = p * f_1[j] + (1 - p) * f_0[j]
                p_new = num / denom
                idx_new = np.argmin(np.abs(p_values - p_new))
                future_cost = J_next[idx_new]
                expected_future += future_cost * denom

            dy = y_vals[1] - y_vals[0]
            expected_future *= dy
            A_k[i] = expected_future

            stop_cost = min((1 - p) * l_0, p * l_1)
            continue_cost = C + expected_future

            J_k[i] = min(stop_cost, continue_cost)

        J[k] = J_k
        A[k] = A_k

    return J, A

print("Computing cost functions for the finite-horizon SHT problem...")
print(f"Parameters: lambda_0={l_0}, lambda_1={l_1}, C={C}, N={N}")
print()

p_values = np.linspace(0, 1, 401)
stages = range(N + 1)

results, A_values = compute_cost_functions(N, p_values)

with open('cost_functions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['p'] + [f'J_{k}' for k in stages])
    for i, p in enumerate(p_values):
        row = [p] + [results[k][i] for k in stages]
        writer.writerow(row)

print(f"Cost functions saved to 'cost_functions.csv'")

thresholds = {}
for k in range(N):
    stop_costs = np.minimum((1 - p_values) * l_0, p_values * l_1)
    continue_costs = C + A_values[k]
    diff = stop_costs - continue_costs
    sign_changes = np.where(np.diff(np.sign(diff)))[0]

    if len(sign_changes) >= 2:
        left_idx = sign_changes[0]
        right_idx = sign_changes[-1]

        if 'left' not in thresholds:
            thresholds['left'] = {}
        if 'right' not in thresholds:
            thresholds['right'] = {}

        thresholds['left'][k] = p_values[left_idx] + (p_values[left_idx+1] - p_values[left_idx]) * \
                                 abs(diff[left_idx]) / (abs(diff[left_idx]) + abs(diff[left_idx+1]))
        thresholds['right'][k] = p_values[right_idx] + (p_values[right_idx+1] - p_values[right_idx]) * \
                                  abs(diff[right_idx]) / (abs(diff[right_idx]) + abs(diff[right_idx+1]))

print("\nDecision thresholds:")
for k in range(N):
    left_val = thresholds.get('left', {}).get(k, None)
    right_val = thresholds.get('right', {}).get(k, None)
    if left_val is not None and right_val is not None:
        print(f"  Stage {k}: p in [{left_val:.4f}, {right_val:.4f}]")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for idx, k in enumerate(stages):
    if idx < len(axes):
        ax = axes[idx]
        ax.plot(p_values, results[k], linewidth=2, label=f'$J_{k}(p)$')
        ax.set_xlabel('Belief p (probability of H1)')
        ax.set_ylabel(f'Cost $J_{k}(p)$')
        ax.set_title(f'Cost Function at Stage {k}')
        ax.grid(True, alpha=0.3)
        ax.legend()

        if k in thresholds:
            ax.axvline(thresholds[k], color='r', linestyle='--', alpha=0.7, label=f'Threshold: {thresholds[k]:.3f}')
            ax.legend()

fig, ax = plt.subplots(figsize=(14, 8))
colors = plt.cm.viridis(np.linspace(0, 1, N + 1))
for k in stages:
    ax.plot(p_values, results[k], linewidth=2.5, label=f'$J_{k}(p)$', color=colors[k])
for k in range(N):
    left_val = thresholds.get('left', {}).get(k, None)
    right_val = thresholds.get('right', {}).get(k, None)
    
    if left_val is not None:
        ax.axvline(left_val, color=colors[k], linestyle=':', linewidth=2, alpha=0.7)
    if right_val is not None:
        ax.axvline(right_val, color=colors[k], linestyle=':', linewidth=2, alpha=0.7)

ax.set_xlabel('Belief on H1')
ax.set_ylabel('Cost')
ax.set_title('SHT Costs')
ax.grid(True)
ax.legend(loc='best')
plt.tight_layout()
plt.savefig('cost_functions_combined.png')
plt.close()

print("Combined plot saved to 'cost_functions_combined.png'")

