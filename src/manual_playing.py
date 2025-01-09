import gymnasium as gym
from gymnasium.utils.play import play, PlayPlot

# Create the environment
env = gym.make("LunarLander-v3", render_mode="rgb_array")

# Plot the reward
def callback(obs_t, obs_tp1, action, rew, terminated, truncated, info):
    return [rew]

plotter = PlayPlot(callback, 150, ["reward"])

# Play manually
play(env, keys_to_action={
    'z': 2,
    'q': 1,
    's': 0,
    'd': 3
}, noop = 0, callback=plotter.callback)