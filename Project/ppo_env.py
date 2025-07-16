import gymnasium as gym
from gymnasium import spaces
import numpy as np
import joblib
import pandas as pd

class MIMOAIAntiJammingEnv(gym.Env):
    """
    Gymnasium-compatible environment for MIMO anti-jamming adaptive control.
    Observation:
      - Wireless features (8 floats)
      - Jammer type (0=passive,1=active)
    Action (Discrete):
      - Modulation: 0=QPSK,1=16QAM,2=64QAM
      - Power level: 0=low,1=medium,2=high
      - Nulling: 0=none,1=partial,2=full
    Reward:
      - Positive reward proportional to simulated SINR
      - Penalty for power use and BER (simulated)
    """

    def __init__(self, data_csv_path, rf_model_path, rf_scaler_path):
        super().__init__()

        # Load dataset
        self.df = pd.read_csv(data_csv_path)
        self.features = self.df.drop(columns=['label']).values.astype(np.float32)
        self.labels = self.df['label'].values.astype(int)

        # Load RF classifier & scaler
        self.rf_model = joblib.load(rf_model_path)
        self.rf_scaler = joblib.load(rf_scaler_path)

        # Action space: Modulation(3) x Power(3) x Nulling(3)
        self.action_space = spaces.MultiDiscrete([3, 3, 3])

        # Observation space: 8 features + 1 jammer label (0 or 1)
        obs_low = np.array([-np.inf]*8 + [0], dtype=np.float32)
        obs_high = np.array([np.inf]*8 + [1], dtype=np.float32)
        self.observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.max_steps = 100
        self.current_step = 0
        self.current_idx = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.current_step = 0
        self.current_idx = np.random.randint(0, len(self.df))

        feat_raw = self.features[self.current_idx].reshape(1, -1)
        feat_scaled = self.rf_scaler.transform(feat_raw).flatten()
        jammer_label = self.labels[self.current_idx]

        obs = np.append(feat_scaled, jammer_label).astype(np.float32)
        info = {}
        return obs, info

    def step(self, action):
        self.current_step += 1

        modulation, power_level, nulling = action

        feat_raw = self.features[self.current_idx].reshape(1, -1)
        feat_scaled = self.rf_scaler.transform(feat_raw).flatten()
        jammer_label = self.labels[self.current_idx]

        sinr = self.simulate_sinr(modulation, power_level, nulling, feat_scaled, jammer_label)
        ber = self.simulate_ber(sinr, modulation)

        power_penalty = power_level * 0.1
        ber_penalty = ber * 10

        reward = sinr - power_penalty - ber_penalty

        obs = np.append(feat_scaled, jammer_label).astype(np.float32)

        terminated = self.current_step >= self.max_steps
        truncated = False

        info = {'sinr': sinr, 'ber': ber, 'reward': reward}

        return obs, reward, terminated, truncated, info

    def simulate_sinr(self, modulation, power_level, nulling, features, jammer_label):
        base_sinr = 20 if jammer_label == 0 else 5
        mod_factor = {0:1.0, 1:0.8, 2:0.5}[modulation]
        power_factor = 1 + 0.2*power_level
        nulling_factor = 1 + 0.3*nulling
        noise_factor = np.clip(np.sum(features)/1000, 0.5, 1.5)

        sinr = base_sinr * mod_factor * power_factor * nulling_factor * noise_factor
        return sinr

    def simulate_ber(self, sinr, modulation):
        mod_order = {0:2, 1:4, 2:6}[modulation]
        ber = 0.5 * np.exp(-sinr / (mod_order * 2))
        return ber

    def render(self, mode='human'):
        pass
