import argparse
import gymnasium as gym
from stable_baselines3 import PPO, DQN
import sys
import os

module_directory = "src/"
if module_directory not in sys.path:
    sys.path.append(module_directory)

from utils import create_env, get_model_class, create_argument_parser

# Main function
def main():
    parser = create_argument_parser()
    args = parser.parse_args()

    # Define save path for the trained model
    save_dir = os.path.join("models", args.experiment_name, args.method)
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, f"{args.timesteps}-{args.mlp_architecture}-{args.learning_rate}-{args.gamma}-{args.batch_size}-{args.target_update_interval}-model.zip")

    # Check if the model already exists to avoid repeating the same experiment
    if os.path.exists(save_path):
        print(f"Model already exists at: {save_path}. Skipping experiment.")
        return

    env = create_env(fuel=args.limit_fuel)

    # Create the model
    # Parse MLP architecture
    mlp_architecture = tuple(map(int, args.mlp_architecture.split(",")))

    model_params = {
        "policy": "MlpPolicy",
        "env": env,
        "verbose": args.verbose,
        "learning_rate": args.learning_rate,
        "gamma": args.gamma,
        "n_steps": args.n_steps,
        "policy_kwargs": {"net_arch": mlp_architecture},
    }

    model = PPO(**model_params)

    print(f"Starting training for {args.timesteps} timesteps...")
    model.learn(total_timesteps=args.timesteps, progress_bar=True)

    model.save(save_path)
    print(f"Model saved at: {save_path}")

# Run main
if __name__ == "__main__":
    main()