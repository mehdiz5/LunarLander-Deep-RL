import gymnasium as gym
from stable_baselines3 import PPO, DQN
import argparse

# Function to create the environment
def create_env(render_mode= None):
    if render_mode is not None:
        return gym.make("LunarLander-v3", render_mode=render_mode)
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

# Function to create argument parser
def create_argument_parser():
    parser = argparse.ArgumentParser(description="Train and save reinforcement learning models.")

    # Required arguments
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="Name of the experiment (used for saving models)."
    )
    parser.add_argument(
        "--method",
        type=str,
        required=True,
        help="RL method to use for training (e.g., PPO, DQN)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="Number of timesteps for training."
    )

    # Optional arguments with default values
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0 for silent, 1 for info, 2 for debug."
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-4,
        help="Learning rate for the model."
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.99,
        help="Discount factor for future rewards."
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for training."
    )
    parser.add_argument(
        "--n_steps",
        type=int,
        default=2048,
        help="Number of steps to run for each environment update (only for PPO)."
    )
    parser.add_argument(
        "--mlp_architecture",
        type=str,
        default="64,64",
        help="Comma-separated list of integers representing the sizes of hidden layers in the MLP (e.g., '64,64' or '128,128,64')."
    )

    return parser