# compare_policies.py
# Compare PPO agent with static baseline strategies: fixed QPSK, 16QAM, random

import numpy as np
import pandas as pd
from ppo_env import MIMOAIAntiJammingEnv
from stable_baselines3 import PPO

# Load environment
env = MIMOAIAntiJammingEnv(
    data_csv_path='balanced_dataset_20000.csv',
    rf_model_path='rf_model.pkl',
    rf_scaler_path='rf_scaler.pkl'
)

# Load PPO model
ppo_model = PPO.load('ppo_mimo_anti_jamming')

strategies = {
    'Fixed QPSK': lambda obs: [0, 1, 0],           # QPSK, medium power, no null
    'Fixed 16-QAM': lambda obs: [1, 1, 0],         # 16-QAM, medium power
    'Random': lambda obs: env.action_space.sample(),
    'PPO Agent': lambda obs: ppo_model.predict(obs, deterministic=True)[0]
}

results = []
n_episodes = 100

for name, policy in strategies.items():
    sinrs = []
    bers = []
    rewards = []
    print(f"[INFO] Evaluating: {name}")
    for _ in range(n_episodes):
        obs, _ = env.reset()
        total_reward = 0
        total_sinr = 0
        total_ber = 0
        done = False
        step = 0
        while not done:
            action = policy(obs)
            obs, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward
            total_sinr += info['sinr']
            total_ber += info['ber']
            step += 1

        avg_sinr = total_sinr / step
        avg_ber = total_ber / step
        sinrs.append(avg_sinr)
        bers.append(avg_ber)
        rewards.append(total_reward)

    results.append({
        'Policy': name,
        'Avg SINR': np.mean(sinrs),
        'Avg BER': np.mean(bers),
        'Avg Reward': np.mean(rewards)
    })

# Save and print
df = pd.DataFrame(results)
df.to_csv('results/policy_comparison.csv', index=False)
print("\n[INFO] Policy comparison complete:\n")
print(df)

# Optional plot
import matplotlib.pyplot as plt

plt.figure(figsize=(8, 5))
plt.bar(df['Policy'], df['Avg SINR'], color='skyblue')
plt.ylabel('Avg SINR (dB)')
plt.title('Post-filter SINR by Policy')
plt.grid(axis='y')
plt.tight_layout()
plt.savefig('results/policy_comparison_sinr.png', dpi=300)
plt.show()
