import pandas as pd
import matplotlib.pyplot as plt
import os

# Load your PPO evaluation log
df = pd.read_csv("results/ppo_evaluation_log.csv")

# Create the plot
plt.figure(figsize=(12, 6))

# Plot each metric
plt.plot(df["episode"], df["avg_sinr"], label="Avg SINR (dB)", color="blue", marker='o')
plt.plot(df["episode"], df["avg_ber"], label="Avg BER", color="red", marker='s')
plt.plot(df["episode"], df["total_reward"], label="Total Reward", color="green", marker='x')

# Customize plot
plt.title("PPO Evaluation Metrics Over Episodes")
plt.xlabel("Episode")
plt.grid(True)
plt.legend()
plt.tight_layout()

# Save plot
os.makedirs("results", exist_ok=True)
output_path = "results/ppo_training_reward_curve.png"
plt.savefig(output_path, dpi=300)
print(f"[INFO] Plot saved to {output_path}")

# Display plot
plt.show()
