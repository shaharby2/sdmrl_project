import numpy as np
import pandas as pd
import torch
from torch import nn
from stable_baselines3 import PPO, SAC, TD3, DDPG, A2C
from stable_baselines3.common.monitor import Monitor
from electricity_market_env import ElectricityMarketEnv
from training import evaluate_policy
from torch.utils.data import Dataset, DataLoader
import torch.optim as optim
from stable_baselines3.sac.policies import MlpPolicy
from gymnasium.spaces import Box

def generate_expert_data(episodes=200):

    #  Simulates an expert trading strategy in the electricity market and collects training data.
    #  Runs the environment for a specified number of episodes.
    #  Uses a rule-based strategy to decide when to buy/sell electricity based on:
    #  Historical price percentiles (low vs. high prices).
    #   Rolling price trend (using linear regression on recent prices).
    #  State of Charge (SoC) and demand constraints.
    #  Saves the expert state-action pairs in `smart_expert_data.csv` for imitation learning.
    #  Args:
    #  episodes (int): Number of episodes to simulate.
    #  Returns:
    #  None (saves data to CSV).

    env = ElectricityMarketEnv()
    data = []

    for ep in range(episodes):
        obs, _ = env.reset()
        for t in range(24):  # Simulate 24 hours
            soc, demand, price = obs  # Extract state variables

            # Ensure enough price history before calculating percentiles
            if len(env.price_history) < 30:
                action = 0  # No action if not enough history
            else:
                # Calculate price percentiles
                low_price_threshold = np.percentile(env.price_history, 25)  # Charge if price is in lowest 25%
                high_price_threshold = np.percentile(env.price_history, 75)  # Sell if price is in highest 25%

                # Rolling price trend (helps anticipate future prices)
                recent_prices = np.array(env.price_history[-10:])  # Look at last 10 prices
                price_trend = np.polyfit(range(len(recent_prices)), recent_prices, 1)[0]  # Linear regression slope

                # Smart trading strategy with demand consideration
                if price < low_price_threshold and soc < env.battery_capacity * 0.75:
                    action = env.battery_capacity * 0.5  # Charge when price is low
                elif price > high_price_threshold and soc > demand:
                    excess_energy = soc - demand  # Only sell what's left after demand
                    action = -min(excess_energy, soc * 0.5)  # Sell up to 50% of remaining energy
                elif price > high_price_threshold and price_trend > 0 and soc > demand * 1.5:
                    excess_energy = soc - demand  # Ensure demand is met before selling aggressively
                    action = -min(excess_energy, soc * 0.7)  # Sell more if price is rising
                elif soc > demand:
                    action = -demand  # Meet household demand first, but don't sell extra
                elif price < low_price_threshold and price_trend < 0 and soc < env.battery_capacity * 0.5:
                    action = env.battery_capacity * 0.25  # Charge slightly when price is low
                else:
                    action = 0  # No action

            next_obs, reward, _, _, _ = env.step(action)

            # Store state as a **properly formatted JSON string**
            data.append([list(map(float, obs)), action, reward, list(map(float, next_obs))])
            obs = next_obs

    df = pd.DataFrame(data, columns=["state", "action", "reward", "next_state"])

    # Save states as JSON-like strings
    df["state"] = df["state"].apply(lambda x: str(x))
    df["next_state"] = df["next_state"].apply(lambda x: str(x))

    df.to_csv("smart_expert_data.csv", index=False)
    print("Expert data saved to smart_expert_data.csv")


if __name__ == "__main__":
    #Step 1: Create an Expert Demonstration Dataset
    #generate_expert_data(episodes=200)

    #Step 2: Pretrain the Agent Using Imitation Learning
    df = pd.read_csv("smart_expert_data.csv")
    X = np.vstack(df["state"].apply(eval).values)  # Convert state strings to lists
    y = df["action"].values.astype(np.float32)  # Convert actions to float32

    # Define a PyTorch Dataset
    class ExpertDataset(Dataset):

        #  Custom dataset for storing expert state-action pairs for imitation learning.
        #  Args:
        #  X (np.array): Array of state features.
        #  y (np.array): Array of corresponding actions.
        #  Methods:
        #  __len__(): Returns the size of the dataset.
        #  __getitem__(idx): Returns a single state-action pair at the given index.

        def __init__(self, X, y):
            self.X = torch.tensor(X, dtype=torch.float32)
            self.y = torch.tensor(y, dtype=torch.float32).view(-1, 1)

        def __len__(self):
            return len(self.X)

        def __getitem__(self, idx):
            return self.X[idx], self.y[idx]


    dataset = ExpertDataset(X, y)
    torch.save(dataset, "expert_dataset.pth")
    dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

    # Define the observation and action space correctly using Gymnasium's Box
    observation_space = Box(low=-np.inf, high=np.inf, shape=(X.shape[1],), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Initialize SAC's policy network (MlpPolicy)
    policy_net = MlpPolicy(observation_space=observation_space, action_space=action_space,
                           lr_schedule=lambda _: 0.0003)

    optimizer = optim.AdamW(policy_net.parameters(), lr=0.0003, weight_decay=0.005)

    loss_fn = nn.SmoothL1Loss()

    # Training loop
    best_loss = float("inf")  # Track best loss for saving the best model
    for epoch in range(200):
        total_loss = 0
        for batch_X, batch_y in dataloader:
            optimizer.zero_grad()
            pred = policy_net(batch_X)
            loss = loss_fn(pred, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        avg_loss = total_loss / len(dataloader)

        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(policy_net.state_dict(), "pretrained_expert_policy.pth")

        if epoch % 10 == 0:
            print(f"Epoch {epoch}: Loss = {avg_loss}")

    print("Pretrained policy saved.")



    # Step 3: Fine-Tune With RL
    # Initialize environment
    env = Monitor(ElectricityMarketEnv())

    # Define the observation and action space
    observation_space = Box(low=-np.inf, high=np.inf, shape=(env.observation_space.shape[0],), dtype=np.float32)
    action_space = Box(low=-1.0, high=1.0, shape=(1,), dtype=np.float32)

    # Load pretrained expert policy weights
    pretrained_state_dict = torch.load("pretrained_expert_policy.pth")

    # Initialize models
    models = {
        "PPO": PPO("MlpPolicy", env, verbose=1),
        "A2C": A2C("MlpPolicy", env, verbose=1),
        "SAC": SAC("MlpPolicy", env, verbose=1),
        "TD3": TD3("MlpPolicy", env, verbose=1),
        "DDPG": DDPG("MlpPolicy", env, verbose=1)
    }


    def pretrain_policy(model, dataloader, epochs=20, lr=0.0001):

        # Pretrains a reinforcement learning policy using Behavior Cloning (BC).
        # Uses supervised learning to train an RL model to imitate expert decisions.
        # Optimizes the model using the Smooth L1 loss function.
        # Applies gradient clipping to stabilize training.
        # Args:
        # model: The RL model to pretrain.
        # dataloader: The DataLoader providing expert demonstration data.
        # epochs (int): Number of training epochs.
        # lr (float): Learning rate for Adam optimizer.
        # Returns:
        # None (updates model weights).

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        policy_net = model.policy.mlp_extractor.policy_net
        optimizer = torch.optim.Adam(policy_net.parameters(), lr=lr)
        criterion = nn.SmoothL1Loss()

        print(f"Pretraining {type(model).__name__} with Behavior Cloning...")

        for epoch in range(epochs):
            total_loss = 0
            for state, expert_action in dataloader:
                state = state.to(device)
                expert_action = expert_action.to(device)

                # Check for NaNs BEFORE training
                if torch.isnan(state).any() or torch.isnan(expert_action).any():
                    print("Found NaN in dataset! Skipping this batch.")
                    continue

                # Clip expert actions (avoid extreme values)
                expert_action = torch.clamp(expert_action, min=-1.0, max=1.0)

                # Avoid division by zero in normalization
                action_std = expert_action.std() + 1e-8
                expert_action = (expert_action - expert_action.mean()) / action_std

                # Predict actions
                predicted_action = policy_net(state)

                # Compute loss
                loss = criterion(predicted_action, expert_action)

                # Gradient Clipping
                torch.nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=1.0)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                total_loss += loss.item()

            avg_loss = total_loss / len(dataloader)
            print(f"Epoch {epoch + 1}: Loss = {avg_loss:.4f}")

            # Stop training if loss is NaN
            if torch.isnan(torch.tensor(avg_loss)):
                print("Stopping training due to NaN loss!")
                break

        print(f"Behavior Cloning Completed for {type(model).__name__}")


    # Prepare Expert Dataset for Behavior Cloning
    torch.serialization.add_safe_globals([ExpertDataset])
    # Load the dataset safely
    expert_dataset = torch.load("expert_dataset.pth", weights_only=False)
    print("Expert dataset loaded successfully!")


    # Train and evaluate each model
    hybrid_results = {
        "PPO": {"profits": [], "demand_fulfilleds": []},
        "SAC": {"profits": [], "demand_fulfilleds": []},
        "TD3": {"profits": [], "demand_fulfilleds": []},
        "A2C": {"profits": [], "demand_fulfilleds": []},
        "DDPG": {"profits": [], "demand_fulfilleds": []},
    }

    for i in range(5):
        print("Iteration ",i)
        for algo_name, model in models.items():
            print(f"Training Hybrid {algo_name} Model...")

            # Directly load weights for SAC, TD3, and DDPG
            if algo_name in ["SAC", "TD3", "DDPG"]:
                model.policy.actor.mu.load_state_dict(pretrained_state_dict, strict=False)
                print(f"Loaded Pretrained Weights into {algo_name}")

            # Pretrain PPO and A2C using Behavior Cloning
            elif algo_name in ["PPO", "A2C"]:
                pretrain_policy(model, DataLoader(expert_dataset, batch_size=128, shuffle=True))

            # Fine-tune the model using RL
            model.learn(total_timesteps=10000)
            model.save(f"fine_tuned_{algo_name.lower()}_model")

            # Evaluate performance
            avg_profit, avg_demand_fulfilled = evaluate_policy(model, env)
            print(f"Hybrid {algo_name} - Avg Profit: {avg_profit}, Avg Demand Fulfilled: {avg_demand_fulfilled}")
            hybrid_results[algo_name]["profits"].append(avg_profit)
            hybrid_results[algo_name]["demand_fulfilleds"].append(avg_demand_fulfilled)


    print(hybrid_results)