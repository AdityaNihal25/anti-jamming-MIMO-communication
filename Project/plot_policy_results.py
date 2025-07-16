# plot_policy_results.py
# Creates final visualization of PPO vs baseline policies for SINR, BER, and Reward

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

# Load CSV
df = pd.read_csv('results/policy_comparison.csv')

# Ensure save folder exists
import os
if not os.path.exists('results'):
    os.makedirs('results')

# Individual Metric Bar Charts
metrics = ['Avg SINR', 'Avg BER', 'Avg Reward']
colors = ['steelblue', 'orange', 'seagreen']

for i, metric in enumerate(metrics):
    plt.figure(figsize=(8,5))
    plt.bar(df['Policy'], df[metric], color=colors[i])
    plt.ylabel(metric)
    plt.title(f'{metric} by Policy')
    plt.grid(axis='y')
    plt.tight_layout()
    plt.savefig(f'results/{metric.lower().replace(" ", "_")}_bar.png', dpi=300)
    plt.close()

# Grouped Bar Chart for All Metrics
x = np.arange(len(df['Policy']))
width = 0.25

plt.figure(figsize=(10,6))
plt.bar(x - width, df['Avg SINR'], width, label='SINR (dB)', color='steelblue')
plt.bar(x, df['Avg BER'], width, label='BER', color='orange')
plt.bar(x + width, df['Avg Reward'], width, label='Reward', color='seagreen')

plt.xticks(x, df['Policy'])
plt.ylabel('Value')
plt.title('Comparison of PPO and Baselines (All Metrics)')
plt.legend()
plt.grid(True, axis='y')
plt.tight_layout()
plt.savefig('results/grouped_comparison.png', dpi=300)
plt.show()
