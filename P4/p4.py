import numpy as np
from scipy.stats import norm
import csv

f_0 = lambda y: norm.pdf(y, 0, 1)
f_1 = lambda y: norm.pdf(y, 1, 1)
lambda_param = 25
rho = 0.01

print("QCD Problem - Value Iteration")
print(f"Parameters: lambda={lambda_param}, rho={rho}")
print(f"f_0 = N(0,1), f_1 = N(1,1)")
print()

n_belief = 201
p_values = np.linspace(0, 1, n_belief)

max_iterations = 100
tolerance = 1e-2

J_star = np.zeros(n_belief)

n_samples = 200
y_vals = np.linspace(-5, 6, n_samples)
f_0_vals = f_0(y_vals)
f_1_vals = f_1(y_vals)
dy = y_vals[1] - y_vals[0]


for iteration in range(max_iterations):
    J_new = np.zeros(n_belief)

    for i, p in enumerate(p_values):
        p_tilde = (1 - rho) * p + rho

        expected_continue = 0
        for j, y in enumerate(y_vals):
            num = p_tilde * f_1_vals[j]
            denom = p_tilde * f_1_vals[j] + (1 - p_tilde) * f_0_vals[j]

            if denom > 1e-10:
                p_new = num / denom
                p_new = np.clip(p_new, 0, 1)
                idx_new = np.argmin(np.abs(p_values - p_new))
                future_cost = J_star[idx_new]
            else:
                future_cost = J_star[i]

            prob_y = denom
            expected_continue += future_cost * prob_y

        expected_continue *= dy
        continue_cost = p + expected_continue

        stop_cost = (1 - p) * lambda_param
        J_new[i] = min(stop_cost, continue_cost)

    max_change = np.max(np.abs(J_new - J_star))

    if iteration % 10 == 0:
        print(f"Iteration {iteration}: max change = {max_change:.6f}")

    J_star = J_new.copy()

    if max_change < tolerance:
        print(f"Converged after {iteration} iterations!")
        break

continue_costs = np.zeros(n_belief)
for i, p in enumerate(p_values):
    p_tilde = (1 - rho) * p + rho
    expected_continue = 0

    for j, y in enumerate(y_vals):
        num = p_tilde * f_1_vals[j]
        denom = p_tilde * f_1_vals[j] + (1 - p_tilde) * f_0_vals[j]

        if denom > 1e-10:
            p_new = num / denom
            p_new = np.clip(p_new, 0, 1)
            idx_new = np.argmin(np.abs(p_values - p_new))
            future_cost = J_star[idx_new]
        else:
            future_cost = J_star[i]

        prob_y = denom
        expected_continue += future_cost * prob_y

    expected_continue *= dy
    continue_costs[i] = p + expected_continue

stop_costs = (1 - p_values) * lambda_param

diff = stop_costs - continue_costs
sign_changes = np.where(np.diff(np.sign(diff)))[0]

if len(sign_changes) > 0:
    thresh_idx = sign_changes[-1]
    A_star = p_values[thresh_idx] + (p_values[thresh_idx+1] - p_values[thresh_idx]) * \
             abs(diff[thresh_idx]) / (abs(diff[thresh_idx]) + abs(diff[thresh_idx+1]))
    print(f"Optimal threshold A* = {A_star:.6f}")
else:
    A_star = None

with open('qcd_results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['p', 'J_star', 'stop_cost', 'continue_cost'])
    for i, p in enumerate(p_values):
        writer.writerow([p, J_star[i], stop_costs[i], continue_costs[i]])

print("Results saved to 'qcd_results.csv'")
