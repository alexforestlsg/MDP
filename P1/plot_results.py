import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

df = pd.read_csv('results.csv')
df.columns = df.columns.str.strip() 

k_values = sorted(df['k'].unique())

for k in k_values:
    fig, ax = plt.subplots(figsize=(10, 6))
    k_data = df[df['k'] == k]
    x_values = sorted(k_data['x'].unique())
    
    for x in x_values:
        x_data = k_data[k_data['x'] == x].sort_values('u')
        ax.plot(x_data['u'], x_data['cost'], marker='.', label=f'x={x}')
        
        # Optimal policy finder
        min_idx = x_data['cost'].idxmin()
        min_u = x_data.loc[min_idx, 'u']
        min_cost = x_data.loc[min_idx, 'cost']
        ax.plot(min_u, min_cost, '*', markersize=10)
    
    ax.set_xlabel('u')
    ax.set_ylabel('J')
    ax.set_title(f'Cost vs Action for k={k}')
    ax.legend(loc='upper left')
    ax.grid(True)
    plt.savefig(f'cost_k{k}.png')
    plt.close()
