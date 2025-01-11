import argparse
import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
import sys
import os

module_directory = "src/"
if module_directory not in sys.path:
    sys.path.append(module_directory)

from utils import get_model_class, create_argument_parser, create_env

def main(args):
    experiment_name = args.experiment_name
    method_name = args.method
    timesteps = args.timesteps

    # Build paths
    models_dir = f"models/{experiment_name}/{method_name}/"
    model_path = f"{models_dir}{timesteps}-{args.mlp_architecture}-{args.learning_rate}-{args.gamma}-{args.batch_size}-{args.target_update_interval}-model.zip"

    # Check if the model exists
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model not found at {model_path}. Please train the model first.")
    
    print(f"Loading model from {model_path}...")
    
    env = create_env()

    model_class = get_model_class(method_name)
    
    model = model_class.load(model_path)
    print("Model loaded successfully.")
    
    print("Evaluating the model...")
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward}, Std reward: {std_reward}")
    
    print("Rendering the agent...")
    render_env = create_env(render_mode="human")
    obs, _ = render_env.reset()
    done = False
    
    while not done:
        action, _ = model.predict(obs)
        obs, reward, done, _, _ = render_env.step(action)
    
    render_env.close()
    print("Evaluation completed.")

if __name__ == "__main__":
    parser = create_argument_parser()
    args = parser.parse_args()
    main(args)
