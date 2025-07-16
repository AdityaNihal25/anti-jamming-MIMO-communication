import gym
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
from ppo_env import MIMOAIAntiJammingEnv
import os

def main():
    # Paths to data and models
    data_csv_path = 'balanced_dataset_20000.csv'
    rf_model_path = 'rf_model.pkl'
    rf_scaler_path = 'rf_scaler.pkl'
    
    # Create environment
    env = MIMOAIAntiJammingEnv(data_csv_path, rf_model_path, rf_scaler_path)
    env = Monitor(env)  # For logging
    
    # Create PPO model
    model = PPO('MlpPolicy', env, verbose=1, tensorboard_log='./ppo_tensorboard/')
    
    # Eval callback (optional)
    eval_env = MIMOAIAntiJammingEnv(data_csv_path, rf_model_path, rf_scaler_path)
    eval_callback = EvalCallback(eval_env, best_model_save_path='./logs/',
                                 log_path='./logs/', eval_freq=1000,
                                 deterministic=True, render=False)
    
    # Train model
    model.learn(total_timesteps=100000, callback=eval_callback)
    
    # Save model
    model.save('ppo_mimo_anti_jamming')
    print("Model saved as ppo_mimo_anti_jamming.zip")

if __name__ == "__main__":
    main()
