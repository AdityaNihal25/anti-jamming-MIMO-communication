# evaluate_trained_ppo.py
# Evaluate trained PPO agent and log performance metrics

from stable_baselines3 import PPO
from ppo_env import MIMOAIAntiJammingEnv
import numpy as np
import pandas as pd

# Load environment and model
env = MIMOAIAntiJammingEnv(
    data_csv_path='balanced_dataset_20000.csv',
    rf_model_path='rf_model.pkl',
    rf_scaler_path='rf_scaler.pkl'
)
model = PPO.load("ppo_mimo_anti_jamming")

# Evaluation settings
num_episodes = 100
metrics = []

for ep in range(num_episodes):
    obs, _ = env.reset()
    total_reward = 0
    ep_sinr, ep_ber = [], []

    for step in range(env.max_steps):
        action, _states = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)

        total_reward += reward
        ep_sinr.append(info['sinr'])
        ep_ber.append(info['ber'])

        if terminated or truncated:
            break

    metrics.append({
        'episode': ep + 1,
        'avg_sinr': np.mean(ep_sinr),
        'avg_ber': np.mean(ep_ber),
        'total_reward': total_reward
    })

# Save to CSV
results_df = pd.DataFrame(metrics)
results_df.to_csv('results/ppo_evaluation_log.csv', index=False)
print("[INFO] Evaluation complete. Saved to results/ppo_evaluation_log.csv")

# Summary
print("\nAverage over", num_episodes, "episodes:")
print("Avg SINR:", results_df['avg_sinr'].mean())
print("Avg BER:", results_df['avg_ber'].mean())
print("Avg Reward:", results_df['total_reward'].mean())
