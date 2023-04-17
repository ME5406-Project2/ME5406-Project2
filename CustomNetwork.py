import torch
import torch.nn as nn
from stable_baselines3.common.policies import ActorCriticPolicy
from stable_baselines3.td3.policies import TD3Policy
from stable_baselines3.sac.policies import SACPolicy


device = (
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)
print(f"Using {device} device")

# create a custom LSTM feature extractor
class LSTMFeatureExtractor(nn.Module):
    """
    LSTM features extractor class

    :param observation_space:
    :param features_dim: Number of features extracted.
    """
    def __init__(self, observation_space, features_dim:int=64, lstm_layers:int = 1, lstm_dropout:float = 0.0):
        super().__init__()

        self._observation_space = observation_space
        self.features_dim = features_dim
        self.lstm_layers = lstm_layers
        self.lstm_dropout = lstm_dropout

        self.lstm = nn.LSTM(input_size=observation_space.shape[0], hidden_size=self.features_dim, num_layers=self.lstm_layers,
                            batch_first=True, dropout=self.lstm_dropout)

    def forward(self, obs):
        obs = obs.reshape((-1,) + self._observation_space.shape)

        # Extract features using LSTM
        h_lstm, (hn, cn) = self.lstm(obs)
        return h_lstm
    
# create an TD3 policy with LSTM as feature extractor
class Custom_TD3_Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)

# create a PPO policy with LSTM as feature extractor
class Custom_PPO_Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)

# create a TRPO policy with LSTM as feature extractor
class Custom_TRPO_Policy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)

# create a SAC policy with LSTM as feature extractor
class Custom_SAC_Policy(SACPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)

# create a DDPG policy with LSTM as feature extractor
class Custom_DDPG_Policy(TD3Policy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs, features_extractor_class=LSTMFeatureExtractor)