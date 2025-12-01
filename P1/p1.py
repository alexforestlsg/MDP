import random
import csv

# Manual cache mapping (x,k) -> cost-to-go
value_cache = {}


def ctg(x, k):
    # Check the manual cache before any computation
    if (x, k) in value_cache:
        return value_cache[(x, k)]
    # Base case
    if k == 5:
        value_cache[(x, k)] = 0
        return 0

    u_costs = []
    for u in range(0, 11):
        cost_over_w = 0
        for w in range(0, 11):
            cost_over_w += u + 2 * abs(x + u - w) + ctg(max(0, x + u - w), k + 1)
        avg_cost = cost_over_w / 11
        u_costs.append((u, avg_cost))

    min_cost = min(u_costs, key=lambda pair: pair[1])[1]
    # store in manual cache before returning
    value_cache[(x, k)] = min_cost
    return min_cost

u_cost_pairs = []
# Backward recursion from k=4 down to k=0
for k in range(4, -1, -1):
    for x in range(0, 11):
        per_u = []
        for u in range(0, 11):
            cost_over_w = 0
            for w in range(0, 11):
                cost_over_w += u + 2 * abs(x + u - w) + ctg(max(0, x + u - w), k + 1)
            avg_cost = cost_over_w / 11
            per_u.append((u, avg_cost))
            u_cost_pairs.append((k, x, u, avg_cost))
        min_cost = min(per_u, key=lambda pair: pair[1])[1]
        value_cache[(x, k)] = min_cost

# Save to CSV
with open('results.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['k', 'x', 'u', 'cost'])
    for row in u_cost_pairs:
        writer.writerow(row)