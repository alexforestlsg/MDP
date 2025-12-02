import numpy as np
import pandas as pd
import csv
from tqdm import tqdm

l_0 = 25
l_1 = 15
C = 1
N = 5

df = pd.read_csv('cost_functions.csv')
p_values = df['p'].values

results = {}
for k in range(N + 1):
    results[k] = df[f'J_{k}'].values

def get_optimal_action(p, k, A_values):
    if k == N:
        if p * l_1 < (1 - p) * l_0:
            return ('stop', 0)
        else:
            return ('stop', 1)

    idx = np.argmin(np.abs(p_values - p))
    stop_cost = min((1 - p) * l_0, p * l_1)
    continue_cost = C + A_values[k][idx]

    if continue_cost < stop_cost:
        return ('continue', None)
    else:
        if p * l_1 < (1 - p) * l_0:
            return ('stop', 0)
        else:
            return ('stop', 1)

def compute_A_values(p_values, N):
    n_p = len(p_values)
    n_samples = 1000
    y_vals = np.linspace(-5, 6, n_samples)

    f_0 = np.exp(-0.5 * y_vals**2) / np.sqrt(2 * np.pi)
    f_1 = np.exp(-0.5 * (y_vals - 1)**2) / np.sqrt(2 * np.pi)

    J = {}
    A = {}
    J[N] = np.minimum((1 - p_values) * l_0, p_values * l_1)

    for k in range(N-1, -1, -1):
        J_next = J[k+1]
        J_k = np.zeros(n_p)
        A_k = np.zeros(n_p)

        for i, p in enumerate(p_values):
            expected_future = 0
            for j, y in enumerate(y_vals):
                num = p * f_1[j]
                denom = p * f_1[j] + (1 - p) * f_0[j]
                prob_y = denom
                p_new = num / denom
                idx_new = np.argmin(np.abs(p_values - p_new))
                future_cost = J_next[idx_new]
                expected_future += future_cost * prob_y

            dy = y_vals[1] - y_vals[0]
            expected_future *= dy
            A_k[i] = expected_future

            stop_cost = min((1 - p) * l_0, p * l_1)
            continue_cost = C + expected_future
            J_k[i] = min(stop_cost, continue_cost)

        J[k] = J_k
        A[k] = A_k

    return J, A

print("Computing A values for optimal policy...")
_, A_values = compute_A_values(p_values, N)
print("Done!")

num_simulations = 10000
prior_p = 0.5

print(f"\nRunning {num_simulations} simulations...")

sprt_results = []
baseline_results = []

for sim in tqdm(range(num_simulations), desc="Simulating SPRT vs Baseline"):
    true_hypothesis = np.random.choice([0, 1])

    # SPRT (optimal policy)
    p = prior_p
    num_samples = 0

    for k in range(N):
        action, decision = get_optimal_action(p, k, A_values)

        if action == 'stop':
            type1_error = 1 if (true_hypothesis == 0 and decision == 1) else 0
            type2_error = 1 if (true_hypothesis == 1 and decision == 0) else 0
            total_cost = num_samples * C + type1_error * l_0 + type2_error * l_1

            sprt_results.append({
                'simulation': sim,
                'true_hypothesis': true_hypothesis,
                'decision': decision,
                'num_samples': num_samples,
                'type1_error': type1_error,
                'type2_error': type2_error,
                'total_cost': total_cost
            })
            break
        else:
            if true_hypothesis == 0:
                y = np.random.normal(0, 1)
            else:
                y = np.random.normal(1, 1)

            f_0_y = np.exp(-0.5 * y**2) / np.sqrt(2 * np.pi)
            f_1_y = np.exp(-0.5 * (y - 1)**2) / np.sqrt(2 * np.pi)

            num = p * f_1_y
            denom = p * f_1_y + (1 - p) * f_0_y
            p = num / denom

            num_samples += 1

    if num_samples == N:
        if p * l_1 < (1 - p) * l_0:
            decision = 0
        else:
            decision = 1

        type1_error = 1 if (true_hypothesis == 0 and decision == 1) else 0
        type2_error = 1 if (true_hypothesis == 1 and decision == 0) else 0
        total_cost = num_samples * C + type1_error * l_0 + type2_error * l_1

        sprt_results.append({
            'simulation': sim,
            'true_hypothesis': true_hypothesis,
            'decision': decision,
            'num_samples': num_samples,
            'type1_error': type1_error,
            'type2_error': type2_error,
            'total_cost': total_cost
        })

    # Baseline strategy
    p_baseline = prior_p

    if true_hypothesis == 0:
        y = np.random.normal(0, 1)
    else:
        y = np.random.normal(1, 1)

    f_0_y = np.exp(-0.5 * y**2) / np.sqrt(2 * np.pi)
    f_1_y = np.exp(-0.5 * (y - 1)**2) / np.sqrt(2 * np.pi)

    num = p_baseline * f_1_y
    denom = p_baseline * f_1_y + (1 - p_baseline) * f_0_y
    p_baseline = num / denom

    if p_baseline < 0.5:
        decision = 0
    else:
        decision = 1

    type1_error = 1 if (true_hypothesis == 0 and decision == 1) else 0
    type2_error = 1 if (true_hypothesis == 1 and decision == 0) else 0
    total_cost = 1 * C + type1_error * l_0 + type2_error * l_1

    baseline_results.append({
        'simulation': sim,
        'true_hypothesis': true_hypothesis,
        'decision': decision,
        'num_samples': 1,
        'type1_error': type1_error,
        'type2_error': type2_error,
        'total_cost': total_cost
    })

sprt_df = pd.DataFrame(sprt_results)
baseline_df = pd.DataFrame(baseline_results)

print("\n" + "="*60)
print("SIMULATION RESULTS")
print("="*60)

print("\nSPRT (Optimal Policy):")
print(f"  Average number of samples: {sprt_df['num_samples'].mean():.4f}")
print(f"  Type-I error probability: {sprt_df['type1_error'].mean():.4f}")
print(f"  Type-II error probability: {sprt_df['type2_error'].mean():.4f}")
print(f"  Average total cost: {sprt_df['total_cost'].mean():.4f}")

print("\nBaseline (1 sample, then p < 0.5 -> H0, else H1):")
print(f"  Average number of samples: {baseline_df['num_samples'].mean():.4f}")
print(f"  Type-I error probability: {baseline_df['type1_error'].mean():.4f}")
print(f"  Type-II error probability: {baseline_df['type2_error'].mean():.4f}")
print(f"  Average total cost: {baseline_df['total_cost'].mean():.4f}")

improvement = (baseline_df['total_cost'].mean() - sprt_df['total_cost'].mean()) / baseline_df['total_cost'].mean() * 100
print(f"\nSPRT improvement over baseline: {improvement:.2f}%")

comparison_results = [{
    'algorithm': 'SPRT',
    'avg_samples': sprt_df['num_samples'].mean(),
    'type1_error_prob': sprt_df['type1_error'].mean(),
    'type2_error_prob': sprt_df['type2_error'].mean(),
    'avg_total_cost': sprt_df['total_cost'].mean()
}, {
    'algorithm': 'Baseline',
    'avg_samples': baseline_df['num_samples'].mean(),
    'type1_error_prob': baseline_df['type1_error'].mean(),
    'type2_error_prob': baseline_df['type2_error'].mean(),
    'avg_total_cost': baseline_df['total_cost'].mean()
}]

with open('simulation_comparison.csv', 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['algorithm', 'avg_samples', 'type1_error_prob', 'type2_error_prob', 'avg_total_cost'])
    writer.writeheader()
    writer.writerows(comparison_results)

print("\nResults saved to 'simulation_comparison.csv'")
