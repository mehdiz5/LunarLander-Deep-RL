import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os

module_directory = "src/"
if module_directory not in sys.path:
    sys.path.append(module_directory)

from utils import get_model_class

def main(args):
    experiment_name = args.experiment_name
    method_name = args.method_name
    timesteps = args.timesteps
    env_name = args.env_name

    # Build paths
    models_dir = f"models/{experiment_name}/{method_name}/"
    model_path = f"{models_dir}{timesteps}-model.zip"

    # Check if the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    
    # Load the environment
    env = gym.make(env_name)

    model_class = get_model_class(method_name)
    
    # Load the model
    model = model_class.load(model_path)
    print("Model loaded successfully.")
    
    # Evaluate the model
    print("Evaluating the model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    # Render the agent
    print("Rendering the agent...")
    render_env = gym.make(env_name, render_mode="human")
    obs, _ = render_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = render_env.step(action)
    
    render_env.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate a trained RL model.")
    
    parser.add_argument(
        "--experiment_name",
        type=str,
        required=True,
        help="The name of the experiment.",
    )
    parser.add_argument(
        "--method_name",
        type=str,
        required=True,
        choices=["PPO", "DQN", "A2C"],  # Add any other methods you may use
        help="The RL algorithm used for training.",
    )
    parser.add_argument(
        "--timesteps",
        type=int,
        required=True,
        help="The number of timesteps the model was trained for.",
    )
    parser.add_argument(
        "--env_name",
        type=str,
        default="LunarLander-v3",
        help="The environment to evaluate the model on.",
    )

    args = parser.parse_args()
    main(args)
