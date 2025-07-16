import matplotlib.pyplot as plt
from matplotlib.widgets import Button, RadioButtons
from ppo_env import MIMOAIAntiJammingEnv
from stable_baselines3 import PPO
import numpy as np

class LiveDemoGUI:
    def __init__(self):
        # Load environment and model
        self.env = MIMOAIAntiJammingEnv(
            data_csv_path='balanced_dataset_20000.csv',
            rf_model_path='rf_model.pkl',
            rf_scaler_path='rf_scaler.pkl'
        )
        self.model = PPO.load('ppo_mimo_anti_jamming')
        
        # Reset env (make sure env.reset() returns what you expect)
        self.obs, _ = self.env.reset()
        self.done = False
        self.auto_mode = True

        # State tracking lists for plotting
        self.sinr_vals = []
        self.ber_vals = []
        self.steps = []

        # Setup figure and axis
        self.fig, self.ax = plt.subplots()
        plt.subplots_adjust(bottom=0.35)
        self.line_sinr, = self.ax.plot([], [], label='SINR (dB)', color='blue')
        self.line_ber, = self.ax.plot([], [], label='BER', color='red')
        self.ax.set_xlim(0, 100)
        self.ax.set_ylim(0, 25)
        self.ax.set_xlabel('Step')
        self.ax.set_title('Live SINR and BER')
        self.ax.legend()
        self.ax.grid(True)

        # Reset Environment Button
        ax_reset = plt.axes([0.7, 0.15, 0.1, 0.075])
        self.btn_reset = Button(ax_reset, 'Reset Env')
        self.btn_reset.on_clicked(self.reset_env)
        self.fig.text(0.75, 0.12, 'Reset Environment', ha='center', fontsize=9)

        # Toggle Auto/Manual Button
        ax_mode = plt.axes([0.81, 0.15, 0.15, 0.075])
        self.btn_mode = Button(ax_mode, 'Toggle Auto')
        self.btn_mode.on_clicked(self.toggle_mode)
        self.fig.text(0.885, 0.12, 'Toggle Auto/Manual', ha='center', fontsize=9)

        # Radio Buttons for Manual Action Selection
        ax_mod = plt.axes([0.1, 0.25, 0.15, 0.1])
        self.radio_mod = RadioButtons(ax_mod, ('QPSK', '16-QAM', '64-QAM'))
        self.radio_mod.on_clicked(self.manual_action_change)
        self.fig.text(0.175, 0.22, 'Modulation', ha='center', fontsize=9)

        ax_pow = plt.axes([0.3, 0.25, 0.15, 0.1])
        self.radio_pow = RadioButtons(ax_pow, ('Low', 'Medium', 'High'))
        self.radio_pow.on_clicked(self.manual_action_change)
        self.fig.text(0.375, 0.22, 'Power Level', ha='center', fontsize=9)

        ax_null = plt.axes([0.5, 0.25, 0.15, 0.1])
        self.radio_null = RadioButtons(ax_null, ('None', 'Partial', 'Full'))
        self.radio_null.on_clicked(self.manual_action_change)
        self.fig.text(0.575, 0.22, 'Nulling Strength', ha='center', fontsize=9)

        # Default manual action: [modulation, power, nulling]
        self.manual_action = [0, 1, 0]

        # Start the animation timer
        self.fig.canvas.mpl_connect('close_event', self.on_close)
        self.timer = self.fig.canvas.new_timer(interval=100)
        self.timer.add_callback(self.update)
        self.timer.start()

    def reset_env(self, event):
        self.obs, _ = self.env.reset()
        self.done = False
        self.sinr_vals.clear()
        self.ber_vals.clear()
        self.steps.clear()
        print("Environment reset")

    def toggle_mode(self, event):
        self.auto_mode = not self.auto_mode
        print(f"Auto mode {'enabled' if self.auto_mode else 'disabled'}")

    def manual_action_change(self, label):
        # Map the currently selected labels from each radio widget to an action index.
        mod_map = {'QPSK': 0, '16-QAM': 1, '64-QAM': 2}
        pow_map = {'Low': 0, 'Medium': 1, 'High': 2}
        null_map = {'None': 0, 'Partial': 1, 'Full': 2}
        self.manual_action[0] = mod_map[self.radio_mod.value_selected]
        self.manual_action[1] = pow_map[self.radio_pow.value_selected]
        self.manual_action[2] = null_map[self.radio_null.value_selected]
        print(f"Manual action set to: {self.manual_action}")

    def update(self):
        if self.done:
            return

        if self.auto_mode:
            # Predict action using the trained model
            action, _ = self.model.predict(self.obs, deterministic=True)
        else:
            # Use manual action chosen by the user
            action = self.manual_action

        # Handle environments that use either the Gymnasium (5-tuple) or Gym (4-tuple) API.
        result = self.env.step(action)
        if len(result) == 5:
            self.obs, reward, terminated, truncated, info = result
            self.done = terminated or truncated
        elif len(result) == 4:
            self.obs, reward, done, info = result
            self.done = done
        else:
            raise ValueError("Unexpected number of return values from env.step")

        # Update logs for plotting
        step = len(self.steps) + 1
        self.steps.append(step)
        self.sinr_vals.append(info['sinr'])
        self.ber_vals.append(info['ber'])

        # Update plotted data
        self.line_sinr.set_data(self.steps, self.sinr_vals)
        self.line_ber.set_data(self.steps, self.ber_vals)
        self.ax.set_xlim(0, max(100, step + 10))
        # Combine both lists to dynamically adjust the y-limit
        current_max = max(self.sinr_vals + self.ber_vals)
        self.ax.set_ylim(0, max(25, current_max + 5))
        self.fig.canvas.draw_idle()

        print(f"Step {step}: Action={action}, SINR={info['sinr']:.2f}, BER={info['ber']:.4f}")

    def on_close(self, event):
        self.timer.stop()
        print("Demo closed.")

if __name__ == "__main__":
    demo = LiveDemoGUI()
    plt.show()