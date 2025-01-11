# LunarLander-Deep-RL

By : Mehdi Zakaria ADJAL

This project was done as part of the AI research internship recruitment project for InstaDeep. It focuses on training and evaluating reinforcement learning (RL) agents to solve the **Lunar Lander** environment using deep RL methods such as DQN PPO.

Thank you InstaDeep for the opportunity.

## 📋 Table of Contents
- [Setup Instructions](#setup-instructions)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Steps of the Solution](#steps-of-the-solution)
- [Future Extensions](#future-extensions)

---

## 🚀 Setup Instructions

### Step 1: Clone the Repository
Start by cloning the repository to your local machine using the following command:
```bash
git clone https://github.com/mehdiz5/LunarLander-Deep-RL.git
cd LunarLander-Deep-RL
```

### Step 2: Set Up a Virtual Environment
It is recommended to use a virtual environment to manage dependencies:
```bash
python -m venv venv
source venv/bin/activate   # For Linux/Mac
venv\\Scripts\\activate    # For Windows
```

### Step 3: Install Requirements
Install all the required Python libraries using:
```bash
pip install -r requirements.txt
```

### Step 4: Verify Installation
Test the setup by running the following command:
```bash
python src/test/eval.py --help
```
If the help message of the evaluation script is displayed, the setup is successful.

---

## 💻 Usage

### Training an RL Agent
Train a reinforcement learning model on the Lunar Lander environment using the following command:
```bash
python src/train/DQN.py --experiment_name lunar_exp --method DQN --timesteps 1000000 --mlp_architecture 512,256 --batch_size 64
```

- **`--experiment_name`**: Name of the experiment. Used to save the model and logs.  
- **`--method`**: The RL method, e.g., `DQN` or `PPO`.  
- **`--timesteps`**: Number of timesteps for training.  
- **`--mlp_architecture`**: Comma-separated sizes of hidden layers, e.g., `512,256`.

*Note : Additional arguments can be used for details use `python src/train/DQN.py --help`

### Evaluating a Trained Model
Evaluate a previously trained model using the following command:
```bash
python src/test/eval.py --experiment_name lunar_exp --method DQN --timesteps 1000000 --mlp_architecture 512,256 --batch_size 64
```
Make sure that a model with the same parameters exists in the appropriate saved directory.

*Note: Some pre-trained Agents are already saved in `/models`. You can evaluate them directly to save time, or delete them to test reproducibility*

---

## 📂 Project Structure
- **`src/`**: Contains the core source code for training and evaluation.
- **`src/train/`**: Script to train Agents.  
- **`src/test/eval.py`**: Script for evaluating trained RL agents.  
- **`requirements.txt`**: Lists the project dependencies.  

---

## 🛠 Steps of the Solution
1. Designed an argument parser to standardize and simplify the training and evaluation processes (just one parser due to lazyness).
2. Used DQN and PPO with custom hyperparameters to train on the Lunar Lander environment.
3. Implemented saving logic to avoid re-training models with identical parameters.
4. Developed evaluation scripts to verify performance and log results.

## ✨ Possible Extensions
- Add support for additional RL algorithms (e.g., PPO, A2C).
- Use tensorboard or wandb.
- Add flag files to check experiment status.
