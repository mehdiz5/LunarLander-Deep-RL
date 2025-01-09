import argparse
import gymnasium as gym
from stable_baselines3 import PPO, DQN
import sys
import os

module_directory = "src/"
if module_directory not in sys.path:
    sys.path.append(module_directory)
    
from utils import create_env, get_model_class

# Main function
def main():
    # Parse arguments
    parser = argparse.ArgumentParser(description="Train and save reinforcement learning models.")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="default_experiment",
        help="Name of the experiment (used for saving models)."
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        default=100000,
        help="Number of timesteps for training."
    )
    parser.add_argument(
        "--verbose",
        type=int,
        default=1,
        choices=[0, 1, 2],
        help="Verbosity level: 0 for silent, 1 for info, 2 for debug."
    )
    parser.add_argument(
        "--method",
        type=str,
        default="PPO",
        help="RL method to use for training (e.g., PPO, DQN)."
    )
    args = parser.parse_args()

    # Define save path
    save_dir = os.path.join("models", args.experiment_name, args.method)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.timesteps}-model.zip")

    # Check if the model already exists
    if os.path.exists(save_path):
        print(f"Model already exists at: {save_path}. Skipping experiment.")
        return

    # Create environment
    env = create_env()

    # Create the model
    model_class = get_model_class(args.method)
    model = model_class("MlpPolicy", env, verbose=args.verbose)

    # Train the model
    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    # Save the model
    model.save(save_path)
    print(f"Model saved at: {save_path}")

# Run main
if __name__ == "__main__":
    main()
