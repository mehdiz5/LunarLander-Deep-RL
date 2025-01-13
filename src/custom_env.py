import gymnasium as gym

class FuelRewardWrapper(gym.Wrapper):
    def __init__(self, env, max_fuel=400):
        """
        Wrapper to introduce limited fuel and penalties for inefficient fuel usage.

        Args:
            env: The original Gym environment to wrap.
            max_fuel (int): Maximum fuel available.
        """
        super().__init__(env)
        self.max_fuel = max_fuel
        self.fuel = max_fuel

    def reset(self, **kwargs):
        """
        Resets the environment and initializes the fuel level.
        """
        self.fuel = self.max_fuel
        return self.env.reset(**kwargs)

    def step(self, action):
        """
        Steps through the environment, deducting fuel and stoping if fuel exhausted.

        Args:
            action: The action taken by the agent.
        
        Returns:
            Tuple containing:
                - observation
                - reward
                - terminated
                - truncated
                - info
        """
        # Check if fuel is exhausted; disallow engine actions if true
        if self.fuel <= 0:
            action = 0 

        
        fuel_cost = 0
        if action == 1:  # Left engine
            fuel_cost = 0.5
        elif action == 2:  # Main engine
            fuel_cost = 1.5
        elif action == 3:  # Right engine
            fuel_cost = 0.5

        # Decrease fuel and track remaining amount
        self.fuel = max(0, self.fuel - fuel_cost)

        # Step through the original environment
        obs, base_reward, terminated, truncated, info = self.env.step(action)

        # Compute total reward
        reward = base_reward

        # Update info with fuel status
        info["fuel"] = self.fuel
        return obs, reward, terminated, truncated, info
