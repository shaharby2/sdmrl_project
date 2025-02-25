import numpy as np
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
from stable_baselines3.common.env_util import make_vec_env
from electricity_market_env import ElectricityMarketEnv
import pandas as pd
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.logger import configure
import os
from stable_baselines3.common.callbacks import BaseCallback


class RewardLoggingCallback(BaseCallback):
    """
    Callback for logging rewards and timesteps at regular intervals.
    This helps track the performance of the agent during training.
    """

    def __init__(self, log_interval=100, verbose=1):
        super().__init__(verbose)
        self.log_interval = log_interval
        self.episode_rewards = []
        self.total_steps = 0

    def _on_step(self) -> bool:
        """
        Called at every step during training to log rewards and timesteps.
        """
        self.total_steps += 1

        # Extract reward
        reward = np.mean(self.locals["rewards"]) if "rewards" in self.locals else 0
        self.episode_rewards.append(reward)

        # Log every `log_interval` steps
        if self.total_steps % self.log_interval == 0:
            avg_reward = np.mean(self.episode_rewards) if self.episode_rewards else 0

            # Log total timesteps explicitly
            self.logger.record("time/total_timesteps", self.total_steps)
            self.logger.record("rollout/ep_rew_mean", avg_reward)
            self.logger.dump(self.total_steps)

            if self.verbose:
                print(f"Step {self.total_steps}: Avg Reward (Last {self.log_interval} Steps): {avg_reward:.2f}")

            self.episode_rewards = []

        return True




def evaluate_policy(model, env, episodes=10):
    """
    Evaluates the trained model on the given environment for a fixed number of episodes.
    Returns average profit and demand fulfillment percentage.
    """
    total_profits = []
    demand_fulfilled = []

    for ep in range(episodes):
        reset_result = env.reset()
        obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

        total_profit = 0
        demand_met = 0
        total_demand = 0

        for _ in range(24):  # Simulate 24 time steps (there are 24 hours in a day)
            action, _ = model.predict(obs, deterministic=True)
            step_result = env.step(action)
            obs = step_result[0] if isinstance(step_result, tuple) else step_result
            reward = step_result[1]
            info = step_result[4]

            # Calculate actual profit separately
            grid_sale = info.get("energy_sold", 0)
            price = info.get("price", 0)
            actual_profit = grid_sale * price  # True monetary profit from selling energy

            total_profit += actual_profit  # Only add sales revenue to profit

            # Ensure demand fulfillment is non-negative
            demand = max(0, obs[1])
            battery_discharge = max(-action[0], 0)
            demand_met += min(battery_discharge, demand)
            total_demand += demand

        total_profits.append(total_profit)
        demand_fulfilled.append(demand_met / total_demand if total_demand > 0 else 0)

    avg_profit = np.mean(total_profits)
    avg_demand_fulfilled = np.mean(demand_fulfilled)

    return avg_profit, avg_demand_fulfilled


def visualize_policy_behavior(model, env):
    """
    Simulates one episode using the trained model and visualizes battery state of charge,
    energy transactions, and demand fulfillment over time.
    """
    reset_result = env.reset()
    obs = reset_result[0] if isinstance(reset_result, tuple) else reset_result

    soc_levels = []
    energy_sold = []
    demand_fulfilled = []

    for _ in range(24):  # Simulate 24 time steps (there are 24 hours in a day)
        action, _ = model.predict(obs, deterministic=True)

        step_result = env.step(action)
        obs = step_result[0] if isinstance(step_result, tuple) else step_result

        reward = step_result[1]
        done = step_result[2] if len(step_result) > 2 else False

        soc_levels.append(obs[0])  # Track State of Charge
        demand = obs[1]
        battery_discharge = max(-action[0], 0)
        demand_fulfilled.append(min(battery_discharge, demand))
        energy_sold.append(max(-action[0] - demand, 0))

        if done:
            break  # Stop simulation if episode ends early

    # Plot Battery SoC Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(soc_levels)), soc_levels, label="State of Charge", marker='o')
    plt.xlabel("Time (Hours)")
    plt.ylabel("Battery SoC (kWh)")
    plt.title("Battery SoC Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()

    # Plot Energy Transactions Over Time
    plt.figure(figsize=(10, 5))
    plt.plot(range(len(demand_fulfilled)), demand_fulfilled, label="Demand Met", linestyle='dashed', marker='x')
    plt.plot(range(len(energy_sold)), energy_sold, label="Energy Sold to Grid", marker='s')
    plt.xlabel("Time (Hours)")
    plt.ylabel("Energy (kWh)")
    plt.title("Energy Transactions Over Time")
    plt.legend()
    plt.grid(True)
    plt.show()


def plot_training_rewards(log_dirs):
    """
    Plots the training rewards for multiple RL algorithms from their respective log directories.
    Parameters:
        log_dirs (dict): A dictionary where keys are algorithm names and values are log directory paths.
    """
    plt.figure(figsize=(10, 5))

    for algo_name, log_dir in log_dirs.items():
        log_path = os.path.join(log_dir, "progress.csv")

        if not os.path.exists(log_path):
            print(f" Warning: Log file not found for {algo_name}: {log_path}")
            continue

        log_data = pd.read_csv(log_path)

        # Ensure reward column exists
        reward_column = "rollout/ep_rew_mean"
        if reward_column not in log_data.columns:
            print(f" Warning: Column {reward_column} not found in {log_path}. Skipping {algo_name}.")
            continue

        # Clean and sort data
        log_data = log_data[["time/total_timesteps", reward_column]].dropna()
        log_data = log_data.sort_values("time/total_timesteps")  # Ensure sorted order

        min_timestep = log_data["time/total_timesteps"].min()
        max_timestep = log_data["time/total_timesteps"].max()
        print(f"{algo_name} - Timestep range: {min_timestep} to {max_timestep}")

        plt.plot(
            log_data["time/total_timesteps"],
            log_data[reward_column],
            label=f"{algo_name} Reward"
        )

    # Plot Formatting
    plt.xlabel("Timesteps")
    plt.ylabel("Reward")
    plt.title("Training Reward Curve for Different RL Algorithms")
    plt.legend()
    plt.grid(True)
    plt.xlim(left=0)  # Ensure x-axis starts at 0

    plt.show()

#---------------------------------------
ppo_profits = []
ppo_demand_fulfilleds = []
sac_profits = []
sac_demand_fulfilleds = []
td3_profits = []
td3_demand_fulfilleds = []
a2c_profits = []
a2c_demand_fulfilleds = []
ddpg_profits = []
ddpg_demand_fulfilleds = []


for i in range(20):
    # Create the environment
    env = make_vec_env(lambda: ElectricityMarketEnv(), n_envs=1)

    #----------------------------------------
    # Train using PPO
    # Define log directory
    log_dir = "./logs/ppo/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with `Monitor`
    env = Monitor(ElectricityMarketEnv(), log_dir)

    # Configure logging for CSV only
    new_logger = configure(log_dir, ["stdout", "csv"])

    # Train PPO with logging every 100 steps
    ppo_model = PPO(
        "MlpPolicy",
        env,
        learning_rate=2.5e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        verbose=1
    )
    ppo_model.set_logger(new_logger)  # Set CSV logger
    ppo_model.learn(total_timesteps=10000, callback=RewardLoggingCallback(log_interval=100))
    ppo_model.save("ppo_electricity_market")

    # Evaluate PPO Model
    ppo_profit, ppo_demand_fulfilled = evaluate_policy(ppo_model, env)
    print(f"PPO - Avg Profit: {ppo_profit}, Avg Demand Fulfilled: {ppo_demand_fulfilled}")
    ppo_profits.append(ppo_profit)
    ppo_demand_fulfilleds.append(ppo_demand_fulfilled)

    #----------------------------------------
    # Train using SAC
    # Define log directory
    log_dir = "./logs/sac/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with `Monitor`
    env = Monitor(ElectricityMarketEnv(), log_dir)

    # Configure logging for CSV only (disable TensorBoard)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # Train SAC with logging every 100 steps
    sac_model = SAC("MlpPolicy", env, verbose=1)
    sac_model.set_logger(new_logger)  # Set CSV logger
    sac_model.learn(total_timesteps=10000, callback=RewardLoggingCallback(log_interval=100))
    sac_model.save("sac_electricity_market")
    # Evaluate SAC Model
    sac_profit, sac_demand_fulfilled = evaluate_policy(sac_model, env)
    print(f"SAC - Avg Profit: {sac_profit}, Avg Demand Fulfilled: {sac_demand_fulfilled}")
    sac_profits.append(sac_profit)
    sac_demand_fulfilleds.append(sac_demand_fulfilled)

    #-----------------------------
    # Train using TD3
    # Define log directory
    log_dir = "./logs/td3/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with `Monitor`
    env = Monitor(ElectricityMarketEnv(), log_dir)

    # Configure logging for CSV only (disable TensorBoard)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # Train TD3 with logging every 100 steps
    TD3_model = TD3("MlpPolicy", env, learning_rate=3e-4, batch_size=100, gamma=0.99, tau=0.005, policy_delay=2, verbose=1, tensorboard_log=log_dir)
    TD3_model.set_logger(new_logger)  # Set CSV logger
    TD3_model.learn(total_timesteps=10000, callback=RewardLoggingCallback(log_interval=100))
    TD3_model.save("td3_electricity_market")

    # Evaluate TD3 Model
    td_profit, td_demand_fulfilled = evaluate_policy(TD3_model, env)
    print(f"TD3 - Avg Profit: {td_profit}, Avg Demand Fulfilled: {td_demand_fulfilled}")
    td3_profits.append(td_profit)
    td3_demand_fulfilleds.append(td_demand_fulfilled)
    # #-----------------------------
    # Train using DDPG
    # Define log directory
    log_dir = "./logs/ddpg/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with `Monitor`
    env = Monitor(ElectricityMarketEnv(), log_dir)

    # Configure logging for CSV only (disable TensorBoard)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # Train DDPG_model with logging every 100 steps
    DDPG_model = DDPG("MlpPolicy", env, learning_rate=1e-3, batch_size=100, gamma=0.99, tau=0.005, verbose=1, tensorboard_log=log_dir)
    DDPG_model.set_logger(new_logger)  # Set CSV logger
    DDPG_model.learn(total_timesteps=10000, callback=RewardLoggingCallback(log_interval=100))
    DDPG_model.save("ddpg_electricity_market")

    # Evaluate DDPG Model
    ddpg_profit, ddpg_demand_fulfilled = evaluate_policy(DDPG_model, env)
    print(f"DDPG - Avg Profit: {ddpg_profit}, Avg Demand Fulfilled: {ddpg_demand_fulfilled}")
    ddpg_profits.append(ddpg_profit)
    ddpg_demand_fulfilleds.append(ddpg_demand_fulfilled)
    # #-----------------------------
    # Train using A2C
    # Define log directory
    log_dir = "./logs/a2c/"
    os.makedirs(log_dir, exist_ok=True)

    # Wrap environment with `Monitor`
    env = Monitor(ElectricityMarketEnv(), log_dir)

    # Configure logging for CSV only (disable TensorBoard)
    new_logger = configure(log_dir, ["stdout", "csv"])

    # Train A2C with logging every 100 steps
    A2C_model = A2C("MlpPolicy", env, learning_rate=7e-4, n_steps=5, gamma=0.99, gae_lambda=0.95, vf_coef=0.5, ent_coef=0.01, verbose=1, tensorboard_log=log_dir)
    A2C_model.set_logger(new_logger)  # Set CSV logger
    A2C_model.learn(total_timesteps=10000, callback=RewardLoggingCallback(log_interval=100))
    A2C_model.save("a2c_electricity_market")
    # Evaluate A2C Model
    a2c_profit, a2c_demand_fulfilled = evaluate_policy(A2C_model, env)
    print(f"A2C - Avg Profit: {a2c_profit}, Avg Demand Fulfilled: {a2c_demand_fulfilled}")
    a2c_profits.append(a2c_profit)
    a2c_demand_fulfilleds.append(a2c_demand_fulfilled)
    print("iteration {i} complited")

    #-----------------------------


print("ppo_profits = ",ppo_profits)
print("ppo_demand_fulfilleds = ",ppo_demand_fulfilleds)
print("sac_profits = ",sac_profits)
print("sac_demand_fulfilleds = ",sac_demand_fulfilleds)
print("td3_profits = ",td3_profits)
print("td3_demand_fulfilleds = ",td3_demand_fulfilleds)
print("a2c_profits = ",a2c_profits)
print("a2c_demand_fulfilleds = ",a2c_demand_fulfilleds)
print("ddpg_profits = ",ddpg_profits)
print("ddpg_demand_fulfilleds = ",ddpg_demand_fulfilleds)

# Visualize training curves
log_dirs = {
    "PPO": "./logs/ppo/",
    "SAC": "./logs/sac/",
    "TD3": "./logs/td3/",
    "DDPG": "./logs/ddpg/",
    "A2C": "./logs/a2c/"
}
plot_training_rewards(log_dirs)
