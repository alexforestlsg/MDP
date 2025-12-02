import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results_infinite.csv')
df.columns = df.columns.str.strip()  # Remove any trailing whitespace

# Plot cost vs action for each state
fig, ax = plt.subplots(figsize=(10, 6))

# Get all unique x values
x_values = sorted(df['x'].unique())

for x in x_values:
    x_data = df[df['x'] == x].sort_values('u')
    ax.plot(x_data['u'], x_data['cost'], marker='.', label=f'x={x}')
    
    # Optimal policy finder
    min_idx = x_data['cost'].idxmin()
    min_u = x_data.loc[min_idx, 'u']
    min_cost = x_data.loc[min_idx, 'cost']
    ax.plot(min_u, min_cost, '*', markersize=10)

ax.set_xlabel('u')
ax.set_ylabel('J')
ax.set_title('Cost vs Action for Infinite Horizon Problem')
ax.legend(loc='upper left')
ax.grid(True)
plt.tight_layout()
plt.savefig('cost_infinite_horizon.png')
plt.close()

print("Plot saved as 'cost_infinite_horizon.png'")
