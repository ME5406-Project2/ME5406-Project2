import gym
from gym.spaces import Box
import numpy as np

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
