import gym
import numpy as np
from stable_baselines3 import PPO, SAC, DDPG, TD3
#from sb3_contrib import TRPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from gym.spaces import MultiDiscrete
from CustomNetwork import CustomMultiCategoricalLSTMPolicy, LSTMPolicy
from DiscreteWrapper import DiscreteActionWrapper
# Define the Gym environment
env = gym.make('CartPole-v1')

# Create a vectorized environment
#env = make_vec_env(lambda: env, n_envs=1)
env.action_space = MultiDiscrete([2])
env = DiscreteActionWrapper(env)
policy_kwargs = dict(net_arch=dict(qf=[128], pi=[64]))
# Define the PPO model
model = TD3('MlpPolicy', env, verbose=1, policy_kwargs=policy_kwargs)
print(model.policy)

# Train the model and log the progress to TensorBoard
model.learn(total_timesteps=10000, progress_bar=True)

# Test the model
obs = env.reset()
for i in range(1000):
    # Get the model's continuous action
    action, _states = model.predict(obs)
    
    # Take the action in the environment
    obs, rewards, done, info = env.step(action)
    if (done):
        env.reset()
    env.render()
