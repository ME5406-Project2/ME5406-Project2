import gym
from CustomNetwork import CustomMultiCategoricalLSTMPolicy
from gym.spaces import Box
import numpy as np

############# Hyper parameters #############
# LSTM params
lstm_features = 64
lstm_layers = 1
lstm_dropout = 0.2
############# ################ #############

# Define and train the LSTM-based PPO policy
# How to use
# env = gym.make('YourEnvName')
# vf = critic, [64] : 1 layer with 64 neurons
# pi = actor, [64] : 1 layer with 64 neurons
# For 1D observation space, a 2 layers fully connected net is used with:
# 64 units (per layer) for PPO/A2C/DQN
# 256 units for SAC
# [400, 300] units for TD3/DDPG (values are taken from the original TD3 paper)
# policy_kwargs = dict(net_arch=dict(vf=[64], pi=[64]))
# replace PPO with other algorithms
# model = PPO(CustomMultiCategoricalLSTMPolicy, env, policy_kwargs=policy_kwargs, verbose=1)
# model.learn(total_timesteps=int(1e5))

class DiscreteActionWrapper(gym.ActionWrapper):
    def __init__(self, env: gym.Env) -> None:
        super().__init__(env)
        
        # Get the original action space of environment
        original_action_space = env.action_space
        print(original_action_space)
        # Save the original action space of environment
        self.original_action_space = [original_action_space.nvec[i] for 
                                      i in range(len(original_action_space.nvec))]
        print("original action space: ",self.original_action_space)
        # Define the new action space
        self.action_space = Box(low=-1, high=1, shape=(len(self.original_action_space),),
                                 dtype=np.float32)
        
    # convert the continuous outputs from RL to discrete actions in env
    def action(self, action):
        
        def contToDisc(act, num_bins):
            actions = np.linspace(-1,1,num_bins)
            discrete_action = np.argmin(np.abs(actions - act))
            return discrete_action
        
        actions = np.zeros(len(self.original_action_space), dtype=np.int32)
        for i, num_bins in enumerate(self.original_action_space):
            actions[i] = contToDisc(action[i], num_bins)

        return actions.squeeze()
