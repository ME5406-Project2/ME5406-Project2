import torch as th
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.common.distributions import MultiCategoricalDistribution
from typing import Tuple
from stable_baselines3.common.distributions import (
    CategoricalDistribution,
    Distribution,
    MultiCategoricalDistribution,
    make_proba_distribution,
)

############# Hyper parameters #############
# LSTM params (might shift this to front page)
lstm_features = 64
lstm_layers = 1
lstm_dropout = 0.0
############# ################ #############

# create an Actor Critic Policy with LSTM as feature extractor
class CustomActorCriticPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, observation_space, features_dim=lstm_features):
        super().__init__()

        self.observation_space = observation_space
        self.features_dim = features_dim

        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=lstm_features, num_layers=lstm_layers,
                            batch_first=True, dropout=lstm_dropout)
        self.linear = nn.Linear(lstm_features, features_dim)

    def forward(self, obs):
        obs = obs.reshape((-1,) + self.observation_space.shape)

        # Extract features using LSTM
        h_lstm, (hn, cn) = self.lstm(obs)
        features = self.linear(h_lstm[0, :])

        return features

# create an Actor Critic Policy with LSTM and support multidiscrete actions 
class CustomMultiCategoricalLSTMPolicy(CustomActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # creates a list of number of actions in each action dimension
        action_space = self.action_space
        n_actions = [action_space.nvec[i] for i in range(len(action_space.nvec))]
        # Define a MultiCategorical distribution for the output layer
        self.action_dist = MultiCategoricalDistribution(n_actions)
        # Save the output size of actor network
        self.actor_output_size = sum(n_actions)
    
    def _get_action_dist_from_latent(self, latent_pi: th.Tensor) -> Distribution:
        """
        Retrieve action distribution given the latent codes.

        Here we directly assume that distribution is MultiCategoricalDistribution

        :param latent_pi: Latent code for the actor
        :return: Action distribution
        """

        mean_actions = self.action_net(latent_pi)
        
        # Need to reshape before they can be used
        mean_actions = th.reshape(mean_actions, (1, self.actor_output_size))
        
        # Here mean_actions are the flattened logits
        return self.action_dist.proba_distribution(action_logits=mean_actions)