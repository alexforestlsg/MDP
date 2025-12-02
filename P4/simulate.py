import numpy as np
import pandas as pd
import csv
from tqdm import tqdm
from scipy.stats import norm

f_0 = lambda y: norm.pdf(y, 0, 1)  
f_1 = lambda y: norm.pdf(y, 1, 1)  
lambda_param = 25  
rho = 0.01 

df = pd.read_csv('qcd_results.csv')
p_values = df['p'].values
J_star_values = df['J_star'].values
stop_costs = df['stop_cost'].values
continue_costs = df['continue_cost'].values

#get the optimal A*
diff = stop_costs - continue_costs
sign_changes = np.where(np.diff(np.sign(diff)))[0]

if len(sign_changes) > 0:
    thresh_idx = sign_changes[-1]
    A_star = p_values[thresh_idx] + (p_values[thresh_idx+1] - p_values[thresh_idx]) * \
             abs(diff[thresh_idx]) / (abs(diff[thresh_idx]) + abs(diff[thresh_idx+1]))
else:
    A_star = 0.5  # Default

print(f"Optimal threshold A* = {A_star:.6f}")
print()

def optimal_policy(p, A_star):
    return p >= A_star

def baseline_policy_threshold(p, threshold):
    return p >= threshold

def baseline_policy_count(observations, K, suspicious_threshold=1.5):
    return sum(1 for y in observations if y > suspicious_threshold) >= K

def simulate_qcd(policy_func, policy_name, num_simulations=10000, max_time=1000):
    detection_delays = []
    false_alarms = []

    for _ in tqdm(range(num_simulations), desc=f"Simulating {policy_name}"):
        true_change_time = np.random.randint(1, max_time)

        p = 0.0  
        t = 0
        detected = False

        while t < max_time and not detected:
            if policy_func(p):
                detected = True

                if t < true_change_time:
                    false_alarms.append(1)
                    detection_delays.append(np.nan)  
                else:
                    delay = t - true_change_time
                    detection_delays.append(delay)
                    false_alarms.append(0)
                break

            t += 1

            if t < true_change_time:
                y = np.random.normal(0, 1)
            else:
                y = np.random.normal(1, 1)

            p_tilde = (1 - rho) * p + rho

            num = p_tilde * f_1(y)
            denom = p_tilde * f_1(y) + (1 - p_tilde) * f_0(y)

            if denom > 1e-10:
                p = num / denom
                p = np.clip(p, 0, 1)

        if not detected:
            detection_delays.append(max_time - true_change_time)
            false_alarms.append(0)

    return detection_delays, false_alarms

def simulate_qcd_count(K, suspicious_threshold, policy_name, num_simulations=10000, max_time=1000):
    detection_delays = []
    false_alarms = []

    for _ in tqdm(range(num_simulations), desc=f"Simulating {policy_name}"):
        true_change_time = np.random.randint(1, max_time)
        observations = []
        t = 0
        detected = False

        while t < max_time and not detected:
            t += 1
            if t < true_change_time:
                y = np.random.normal(0, 1)
            else:
                y = np.random.normal(1, 1)

            observations.append(y)

            if baseline_policy_count(observations, K, suspicious_threshold):

                detected = True

                if t < true_change_time:
                    false_alarms.append(1)
                    detection_delays.append(np.nan) 
                else:
                    delay = t - true_change_time
                    detection_delays.append(delay)
                    false_alarms.append(0)
                break
        #failsafe
        if not detected:
            detection_delays.append(max_time - true_change_time)
            false_alarms.append(0)

    return detection_delays, false_alarms

num_simulations = 1000
max_time = 500

# Optimal policy
print(f"Simulating optimal policy (A* = {A_star:.6f})...")
opt_delays, opt_false_alarms = simulate_qcd(
    lambda p: optimal_policy(p, A_star),
    f"Optimal (A*={A_star:.4f})",
    num_simulations,
    max_time
)

opt_false_alarm_prob = np.mean(opt_false_alarms)
print(f"Optimal policy false alarm probability: {opt_false_alarm_prob:.6f}")
print()

# Grid search over k and suspicious threshold
best_K = 0
best_threshold = 0.5
best_fa_diff = float('inf')

print(f"Target false alarm rate: {opt_false_alarm_prob:.6f}")
print()

# Find a naive test that matches false alarm rate
for K_test in [6, 7, 8]:
    for threshold_test in np.arange(0.0, 2.5, 0.05):
        test_delays, test_false_alarms = simulate_qcd_count(
            K_test,
            threshold_test,
            f"Test K={K_test}, threshold={threshold_test:.2f}",
            num_simulations,
            max_time
        )
        test_fa_prob = np.mean(test_false_alarms)
        fa_diff = abs(test_fa_prob - opt_false_alarm_prob)

        print(f"  K={K_test}, threshold={threshold_test:.2f}: FA prob = {test_fa_prob:.6f} (diff = {fa_diff:.6f})")

        if fa_diff < best_fa_diff:
            best_fa_diff = fa_diff
            best_K = K_test
            best_threshold = threshold_test

print()
print(f"Best match: K={best_K}, suspicious_threshold={best_threshold:.2f}")
print(f"  False alarm difference: {best_fa_diff:.6f}")
print()
print(f"Running full simulation with K={best_K}, threshold={best_threshold:.2f}...")
print()

baseline_delays, baseline_false_alarms = simulate_qcd_count(
    best_K,
    best_threshold,
    f"Manager's rule (K={best_K}, threshold={best_threshold:.2f})",
    num_simulations,
    max_time
)

# Calculate comparison stats for optimal vs baseline
opt_valid_delays = [d for d in opt_delays if not np.isnan(d)]
baseline_valid_delays = [d for d in baseline_delays if not np.isnan(d)]

opt_false_alarm_prob = np.mean(opt_false_alarms)
baseline_false_alarm_prob = np.mean(baseline_false_alarms)

opt_avg_delay = np.mean(opt_valid_delays) if len(opt_valid_delays) > 0 else np.nan
baseline_avg_delay = np.mean(baseline_valid_delays) if len(baseline_valid_delays) > 0 else np.nan

# Compare
print("Comparison:")
print(f"  False alarm difference: {abs(opt_false_alarm_prob - baseline_false_alarm_prob):.6f}")

if not np.isnan(opt_avg_delay) and not np.isnan(baseline_avg_delay):
    delay_improvement = (baseline_avg_delay - opt_avg_delay) / baseline_avg_delay * 100
    print(f"  Detection delay improvement: {delay_improvement:.2f}%")
    print(f"  Optimal avg delay: {opt_avg_delay:.4f}")
    print(f"  Manager's rule avg delay: {baseline_avg_delay:.4f}")

    if delay_improvement > 0:
        print(f"\n  >> Optimal QCD achieves {delay_improvement:.2f}% faster detection than manager's rule!")
        print(f"     (With similar false alarm rates)")
    else:
        print(f"\n  >> Manager's rule has {-delay_improvement:.2f}% lower delay")
        print(f"     (But check false alarm rates - they may differ)")
