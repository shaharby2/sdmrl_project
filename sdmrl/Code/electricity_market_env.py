import gymnasium as gym
from gymnasium.spaces import Box
import numpy as np


class ElectricityMarketEnv(gym.Env):

    # A reinforcement learning environment simulating an electricity market with a battery storage system.
    # The agent can charge or discharge the battery based on electricity prices and household demand.
    # The goal is to optimize energy usage, maximize grid sales at high prices, and minimize demand penalties.

    def __init__(self, battery_capacity=100.0, max_demand=150.0, max_price=10.0):

        # Initializes the electricity market environment.
        # Args:
        # battery_capacity (float): Maximum battery storage capacity.
        # max_demand (float): Maximum energy demand.
        # max_price (float): Maximum electricity price.

        super(ElectricityMarketEnv, self).__init__()

        # Battery Parameters
        self.battery_capacity = battery_capacity
        self.soc = battery_capacity / 2  # Start at 50% charge
        self.seed()  # Ensure reproducibility

        # Market Parameters
        self.time_step = 0
        self.max_demand = max_demand
        self.max_price = max_price
        self.time_step = 0

        self.price_history = [] # Stores past prices for trend analysis

        # Action Space: Continuous [-Battery Capacity, Battery Capacity]
        # - Negative values mean discharging (selling energy)
        # - Positive values mean charging (buying energy)
        self.action_space = Box(
            low=np.array([-self.battery_capacity], dtype=np.float32),
            high=np.array([self.battery_capacity], dtype=np.float32),
            shape=(1,),
            dtype=np.float32
        )

        # Observation Space: [SoC, Demand, Price]
        self.observation_space = Box(
            low=np.array([0, 0, 0]), # Minimum values for SoC, demand, and price
            high=np.array([self.battery_capacity, np.inf, np.inf]), # Upper bounds
            dtype=np.float32,
        )

    def reset(self, seed=None, options=None):

        # Resets the environment for a new episode.
        # Args:
        # seed (int, optional): Random seed for reproducibility.
        # options (dict, optional): Additional options.
        # Returns:
        # np.array: Initial state [SoC, Demand, Price].
        # dict: Additional reset info (empty in this case).

        super().reset(seed=seed)
        if seed is not None:
            np.random.seed(seed)

        self.soc = self.battery_capacity / 2  # Reset battery charge
        self.time_step = 0
        return self._get_state(), {}

    def seed(self, seed=None):

        # Sets the random seed for reproducibility.
        # Args:
        # seed (int, optional): Seed value.

        np.random.seed(seed)

    def step(self, action):

        # Executes a time step in the environment given an action.
        # Args:
        # action (float or np.ndarray): The amount of energy to charge/discharge.
        # Returns:
        # np.array: New state [SoC, Demand, Price].
        # float: Reward received for the action.
        # bool: Whether the episode has ended (always False for now).
        # bool: Whether the episode was truncated (always False for now).
        # dict: Additional info about the step (e.g., demand fulfillment, price).

        self.time_step += 1

        # Ensure action is a scalar
        if isinstance(action, np.ndarray):
            action = action.item()

        # Clip action to battery limits
        action = np.clip(action, -self.soc, self.battery_capacity - self.soc)

        # Compute market demand and price
        demand = self._compute_demand(self.time_step)
        price = self._compute_price(self.time_step)

        self.price_history.append(price)
        if len(self.price_history) > 100:
            self.price_history.pop(0)  # Keep last 100 prices

        # Define battery discharge and energy transactions
        if action < 0:  # Discharging battery
            battery_discharge = min(-action, demand)  # Prioritize serving household
            grid_sale = max(-action - battery_discharge, 0)  # Sell only excess power
        else:  # Charging battery
            battery_discharge = 0
            grid_sale = 0

        # Update battery SoC
        self.soc = np.clip(self.soc + action, 0, self.battery_capacity)

        # reward function
        demand_fulfillment_reward = battery_discharge * 0.5  # Reward for meeting demand
        grid_sale_reward = grid_sale * price * 0.7  # incentive for selling
        unmet_demand_penalty = -max(0, demand - battery_discharge) * 0.3  # Penalize unmet demand

        reward = demand_fulfillment_reward + grid_sale_reward + unmet_demand_penalty

        return self._get_state(), reward, False, False, {
            "demand_fulfilled": battery_discharge,
            "energy_sold": grid_sale,
            "price": price
        }
    # def step(self, action):
    #     self.time_step += 1
    #
    #     # Ensure action is a scalar
    #     if isinstance(action, np.ndarray):
    #         action = action.item()
    #
    #     # Clip action to battery limits
    #     action = np.clip(action, -self.soc, self.battery_capacity - self.soc)
    #
    #     # Compute market demand and price
    #     demand = self._compute_demand(self.time_step)
    #     price = self._compute_price(self.time_step)
    #
    #     self.price_history.append(price)
    #     if len(self.price_history) > 100:
    #         self.price_history.pop(0)  # Keep last 100 prices
    #
    #     # Define battery discharge and energy transactions
    #     if action < 0:  # Discharging battery
    #         battery_discharge = min(-action, demand)  # Prioritize serving household
    #         grid_sale = max(-action - battery_discharge, 0)  # Sell only excess power
    #         charge = 0  # No charging
    #     else:  # Charging battery
    #         battery_discharge = 0
    #         grid_sale = 0
    #         charge = min(action, self.battery_capacity - self.soc)  # Store charging amount
    #
    #     # Update battery SoC
    #     self.soc = np.clip(self.soc + action, 0, self.battery_capacity)
    #
    #     # Improved reward function
    #     #demand_fulfillment_reward = battery_discharge * 0.5  # Reward for meeting demand
    #     grid_sale_reward = grid_sale * price   # Lower incentive for selling
    #     unmet_demand_penalty = -max(0, demand - battery_discharge)   # Penalize unmet demand
    #
    #     # New charging incentive: Reward cheap charging, penalize expensive charging
    #     if charge > 0:
    #         charging_reward = -charge * price * 0.2  # Penalize expensive charging
    #         if price < np.percentile(self.price_history, 30):  # Charge when price is low
    #             charging_reward += charge * 0.3  # Incentivize charging at low prices
    #     else:
    #         charging_reward = 0  # No effect if not charging
    #
    #     # Final reward function
    #     reward = grid_sale_reward* 0.6 + unmet_demand_penalty*0.2 + charging_reward*0.2
    #
    #     return self._get_state(), reward, False, False, {
    #         "demand_fulfilled": battery_discharge,
    #         "energy_sold": grid_sale,
    #         "energy_charged": charge,
    #         "price": price
    #     }


    def _get_state(self):

        # Retrieves the current state of the environment.
        # Returns:
        # np.array: State representation [SoC, Demand, Price].

        demand = self._compute_demand(self.time_step)
        price = self._compute_price(self.time_step)
        return np.array([self.soc, demand, price], dtype=np.float32)

    def _compute_demand(self, t):

        # Simulates electricity demand using a periodic function with noise.
        # Args:
        # t (int): Current time step.
        # Returns:
        # float: Simulated electricity demand.

        return 100 * np.exp(-((t % 24 - 8) ** 2) / (2 * (2 ** 2))) + \
               120 * np.exp(-((t % 24 - 18) ** 2) / (2 * (3 ** 2))) + np.random.normal(0, 10)

    def _compute_price(self, t):

        # Simulates electricity prices as a stochastic function.
        # Args:
        # t (int): Current time step.
        # Returns:
        # float: Simulated electricity price.

        return 5 + 3 * np.sin(2 * np.pi * (t % 24) / 24) + np.random.normal(0, 1)

    def render(self):
        print(f"Time: {self.time_step}, SoC: {self.soc:.2f}, Demand: {self._compute_demand(self.time_step):.2f}, Price: {self._compute_price(self.time_step):.2f}")
