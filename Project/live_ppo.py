import time
import matplotlib.pyplot as plt
from ppo_env import MIMOAIAntiJammingEnv
from stable_baselines3 import PPO
import numpy as np

def live_demo(episodes=1, max_steps=100):
    env = MIMOAIAntiJammingEnv(
        data_csv_path='balanced_dataset_20000.csv',
        rf_model_path='rf_model.pkl',
        rf_scaler_path='rf_scaler.pkl'
    )
    model = PPO.load('ppo_mimo_anti_jamming')

    plt.ion()
    fig, ax = plt.subplots()
    sinr_vals, ber_vals, step_vals = [], [], []

    for ep in range(episodes):
        obs, _ = env.reset()
        print(f"Starting Episode {ep+1}")

        for step in range(max_steps):
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)

            sinr_vals.append(info['sinr'])
            ber_vals.append(info['ber'])
            step_vals.append(step)

            # Console output
            print(f"Step {step+1}: Action={action}, SINR={info['sinr']:.2f} dB, BER={info['ber']:.4f}, Reward={reward:.2f}")

            # Plot live SINR and BER
            ax.clear()
            ax.plot(step_vals, sinr_vals, label='SINR (dB)', color='blue')
            ax.plot(step_vals, ber_vals, label='BER', color='red')
            ax.set_xlabel('Step')
            ax.set_title('Live SINR and BER During PPO Agent Episode')
            ax.legend()
            ax.grid(True)
            plt.pause(0.05)  # pause for plot update

            if terminated or truncated:
                print("Episode finished")
                break

        # Reset lists for next episode
        sinr_vals.clear()
        ber_vals.clear()
        step_vals.clear()

    plt.ioff()
    plt.show()

if __name__ == "__main__":
    live_demo(episodes=1, max_steps=100)