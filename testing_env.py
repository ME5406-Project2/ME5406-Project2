import gym
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym.spaces import MultiDiscrete
from CustomNetwork import CustomMultiCategoricalLSTMPolicy

# Define the Gym environment
env = gym.make('Pendulum-v1')

# Define the new action space
n_actions = 100
action_space = MultiDiscrete([n_actions])

# Update the environment with the new action space
env.action_space = action_space

# Create a vectorized environment
env = make_vec_env(lambda: env, n_envs=1)
policy_kwargs = dict(net_arch=[dict(vf=[128], pi=[64])])
# Define the PPO model
model = PPO(CustomMultiCategoricalLSTMPolicy, env, verbose=1, policy_kwargs=policy_kwargs)
print(model.policy)

# Define the function to convert the multi-discrete action to a continuous action
def convert_action(action):
    continuous_action = np.array([(a - (n_actions-1)/2) / ((n_actions-1)/2) for a in action])
    return continuous_action

# Define the function to convert the continuous action to a multi-discrete action
def invert_action(continuous_action):
    action = np.array([np.around((a * (n_actions-1)/2) + (n_actions-1)/2) for a in continuous_action])
    return action

# Train the model and log the progress to TensorBoard
model.learn(total_timesteps=100000, progress_bar=True)

# Test the model
obs = env.reset()
for i in range(1000):
    # Get the model's continuous action
    continuous_action, _states = model.predict(obs)
    
    # Convert the continuous action to the multi-discrete action
    action = invert_action(continuous_action)
    
    # Take the action in the environment
    obs, rewards, dones, info = env.step(action)
    env.render()
