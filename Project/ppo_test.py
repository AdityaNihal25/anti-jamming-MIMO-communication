from ppo_env import MIMOAIAntiJammingEnv

env = MIMOAIAntiJammingEnv(
    data_csv_path='balanced_dataset_20000.csv',
    rf_model_path='rf_model.pkl',
    rf_scaler_path='rf_scaler.pkl'
)

obs = env.reset()
print('Initial observation:', obs)

action = env.action_space.sample()
print('Sample action:', action)

obs, reward, done, info = env.step(action)
print('Step results:', reward, done, info)
