import string
import gym
from DiscreteWrapper import DiscreteActionWrapper
from stable_baselines3 import PPO, SAC, DDPG, TD3
from sb3_contrib import TRPO
from CustomNetwork import (Custom_DDPG_Policy, Custom_PPO_Policy, 
                           Custom_SAC_Policy, Custom_TD3_Policy, 
                           Custom_TRPO_Policy)
from gym.spaces import MultiDiscrete
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
import os

# Trains the agent and logs the training process onto tensorboard
def Train(algorithm: string, env:gym.Env, num_vectorized_env: int = 1,
          num_timesteps: int=1e6, num_features: int = 64,
          show_net_arch: bool = False,
          use_LSTM: bool = True, verbose: int = 0, share_features_extractor: bool = True,
          lstm_layers: int = 1, lstm_dropout: float = 0.0,
          learning_rate: float = 0.0001, gamma: float = 0.99, batch_size: int = 256,
          training_name: string = "unnamed_training", save_freq: int = 10000, eval_freq:int = 5000):
    """
    Trains the agent and logs the training process onto tensorboard
    :param algorithm: algorithm name
    :param env: gym environment to be used
    :param num_vectorized_env: number of vectorized environment to be used
    
    :param num_timesteps: number of timesteps to train
    :param num_features: number of features to be extracted
    :param show_net_arch: display the network architecture
    :param use_LSTM: use LSTM as feature extractor else use FlattenExtractor
                    Note: TRPO does not support LSTM
    :param lstm_layers: number of layers of lstm
    :param lstm_dropout: dropout rate for lstm only applicable if more than 1 lstm layers are used
    :param verbose: 0 for no output, 1 for info messages, 2 for debug messages
    :param share_features_extractor: Whether or not to share the features extractor between the actor and the critic (this saves computation time)
    
    :param learning_rate: learning rate
    :param gamma: discount rate
    :param batch_size: batch size

    :param training_name: specify name of model trained
    :param save_freq: frequency to save model
    """

    ############# Hyper parameters #############
    # LSTM Params
    lstm_layers = 1
    lstm_dropout = 0.0
    ############################################

    # testing code (to be replaced with actual env)
    env = make_env()

    # to add in vectorized environment (not implemented yet)
    
    # discretize actions using wrapper
    # env = DiscreteActionWrapper(env)

    # path to save model
    save_path = "./trained_models/"+training_name+"/"
    best_model_path = save_path+"best_model"
    results_path = save_path+"results"
    # to open tensorboard 
    # 1. navigate to home directory in terminal
    # 2. call the following command
    # tensorboard --logdir ./<tensorboard_path>/
    tensorboard_path = save_path+"tensorboard_log"

    # make directory to save files
    os.makedirs(save_path, exist_ok=True)
    os.makedirs(best_model_path, exist_ok=True)
    os.makedirs(results_path, exist_ok=True)
    os.makedirs(tensorboard_path, exist_ok=True)

    if (algorithm=="DDPG"):
        # Defining policy to be used
        if use_LSTM:
            policy = Custom_DDPG_Policy
            features_extractor_kwargs = dict(features_dim=num_features, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        else:
            policy = "MlpPolicy"
            features_extractor_kwargs = dict(features_dim=num_features)
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[400, 300], pi=[400, 300]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor)
        # Defining the DDPG model
        model = DDPG(policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    elif (algorithm=="PPO"):
        # Defining policy to be used
        if use_LSTM:
            policy = Custom_PPO_Policy
            features_extractor_kwargs = dict(features_dim=num_features, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        else:
            policy = "MlpPolicy"
            features_extractor_kwargs = dict(features_dim=num_features)
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(vf=[64,64], pi=[64,64]), # network architecture for value function (vf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor)
        # Defining the PPO model
        model = PPO(policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    elif (algorithm=="SAC"):
        # Defining policy to be used
        if use_LSTM:
            policy = Custom_SAC_Policy
            features_extractor_kwargs = dict(features_dim=num_features, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        else:
            policy = "MlpPolicy"
            features_extractor_kwargs = dict(features_dim=num_features)
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[256], pi=[256]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor,
                             learning_rate=learning_rate)
        # Defining the SAC model
        model = SAC(policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    elif (algorithm=="TRPO"):
        policy = "MlpPolicy"
        features_extractor_kwargs = dict(features_dim=num_features)
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(vf=[64,64], pi=[64,64]), # network architecture for value function (vf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor,
                             learning_rate=learning_rate)
        # Defining the TRPO model
        model = TRPO(policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    elif (algorithm=="TD3"):
        # Defining policy to be used
        if use_LSTM:
            policy = Custom_TD3_Policy
            features_extractor_kwargs = dict(features_dim=num_features, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        else:
            policy = "MlpPolicy"
            features_extractor_kwargs = dict(features_dim=num_features)
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[400, 300], pi=[400, 300]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor,
                             learning_rate=learning_rate)
        # Defining the TD3 model
        model = TD3(policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    else:
        raise ValueError("Invalid algorithm name: {}".format(algorithm))
    
    # print network architecture
    if show_net_arch:
        print(model.policy)

    # callback for saving model at regular intervals
    checkpoint_cb = CheckpointCallback(save_freq=save_freq, 
                                       save_path=save_path,
                                       name_prefix=training_name)
    
    # callback to terminate training on no model imporovement
    stop_train_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=3,
                                                     min_evals=10)
    
    # create seperate evaluation environment for evaluation
    eval_env = make_env()
    eval_env = Monitor(eval_env)

    # callback for regular evaluation and save best model
    eval_cb = EvalCallback(eval_env=eval_env, 
                           best_model_save_path=best_model_path,
                           log_path=results_path, eval_freq=eval_freq,
                           n_eval_episodes=10, callback_after_eval=stop_train_cb)
    
    # callback list
    cb_list = CallbackList([checkpoint_cb, eval_cb])

    # Train model
    model.learn(total_timesteps=int(num_timesteps), progress_bar=True, callback=cb_list)


def make_env():
    env = gym.make('CartPole-v1')
    env.action_space = MultiDiscrete([2])
    # discretize actions using wrapper
    env = DiscreteActionWrapper(env)
    return env