import gymnasium as gym
from stable_baselines3 import PPO, DQN

# Function to create the environment
def create_env():
    return gym.make("LunarLander-v3")

# Function to get the model class
def get_model_class(method_name):
    methods = {
        "PPO": PPO,
        "DQN": DQN,
    }
    if method_name not in methods:
        raise ValueError(f"Unsupported method: {method_name}. Supported methods are: {list(methods.keys())}")
    return methods[method_name]