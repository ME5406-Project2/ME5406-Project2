import shutil
import string
import gym
from DiscreteWrapper import DiscreteActionWrapper
from stable_baselines3 import PPO, SAC, DDPG, TD3
# from sb3_contrib import TRPO
from CustomNetwork import (Custom_DDPG_Policy, Custom_PPO_Policy, 
                           Custom_SAC_Policy, Custom_TD3_Policy, 
                           Custom_TRPO_Policy)
from gym.spaces import MultiDiscrete
from stable_baselines3.common.callbacks import CallbackList, EvalCallback, CheckpointCallback, StopTrainingOnNoModelImprovement
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import SubprocVecEnv, DummyVecEnv
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecMonitor

import os
from LeggedEnv import LeggedEnv

# to use dummy env or actual env
use_dummy = False

# Trains the agent and logs the training process onto tensorboard
def Train(algorithm: string, num_vectorized_env: int = 10,
          load_path: string = None,
          num_timesteps: int=1e6, num_features: int = 64,
          show_net_arch: bool = False,
          use_LSTM: bool = True, verbose: int = 0, share_features_extractor: bool = True,
          lstm_layers: int = 1, lstm_dropout: float = 0.0,
          learning_rate: float = 0.0001, gamma: float = 0.99, batch_size: int = 256,
          training_name: string = "unnamed_training", save_freq: int = 20000, eval_freq:int = 20000
          ):
    """
    Trains the agent and logs the training process onto tensorboard
    :param algorithm: algorithm name
    :param num_vectorized_env: number of vectorized environment to be used
    :param load_path: path to model to continue training (Default is None)
    
    :param num_timesteps: number of timesteps to train
    :param num_features: number of features to be extracted
    :param show_net_arch: display the network architecture
    :param use_LSTM: use LSTM as feature extractor else use FlattenExtractor
                    NOTE: TRPO does not support LSTM
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
    if (num_vectorized_env>1):
        env = make_vec_env(env_id=LeggedEnv,
                           n_envs=num_vectorized_env,
                           vec_env_cls=SubprocVecEnv,
                           wrapper_class=DiscreteActionWrapper)
    else:
        # to use dummy env or actual env
        if use_dummy:
            env = make_dummy_env()
        # create legged environment
        else:
            env = make_env()

    # to add in vectorized environment (not implemented yet)
    
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
            features_extractor_kwargs = dict()
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[400, 300], pi=[400, 300]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor)
        # Defining the DDPG model
        if load_path is not None:
            # load a pre trained model
            # model at specified path will be overwritten if save path is the same as load path
            model = DDPG.load(path=load_path, policy=policy, env=env, verbose=verbose,
                        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                        policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)
            print("Model loaded from {}".format(load_path))
        else:
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
            features_extractor_kwargs = dict()
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(vf=[64,64], pi=[64,64]), # network architecture for value function (vf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor)
        # Defining the PPO model
        if load_path is not None:
            # load a pre trained model
            # model at specified path will be overwritten if save path is the same as load path
            model = PPO.load(path=load_path, policy=policy, env=env, verbose=verbose,
                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)
            print("Model loaded from {}".format(load_path))
        else:
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
            features_extractor_kwargs = dict()
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[256], pi=[256]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor,
                            )
        # Defining the SAC model
        if load_path is not None:
            # load a pre trained model
            # model at specified path will be overwritten if save path is the same as load path
            model = SAC.load(path=load_path, policy=policy, env=env, verbose=verbose,
                        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                        tensorboard_log=tensorboard_path)
            print("Model loaded from {}".format(load_path))
        else:
            model = SAC(policy=policy, env=env, verbose=verbose,
                        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                        policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    # elif (algorithm=="TRPO"):
    #     policy = "MlpPolicy"
    #     features_extractor_kwargs = dict()
    #     # Defining hyperparameters
    #     policy_kwargs = dict(net_arch=dict(vf=[64,64], pi=[64,64]), # network architecture for value function (vf) and policy function (pi)
    #                          features_extractor_kwargs=features_extractor_kwargs,
    #                          share_features_extractor=share_features_extractor,
    #                         )
    #     # Defining the TRPO model
    #     if load_path is not None:
    #         # load a pre trained model
    #         # model at specified path will be overwritten if save path is the same as load path
    #         model = TRPO.load(path=load_path, policy=policy, env=env, verbose=verbose,
    #                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
    #                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)
    #         print("Model loaded from {}".format(load_path))
    #     else:
    #         model = TRPO(policy=policy, env=env, verbose=verbose,
    #                     learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
    #                     policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    elif (algorithm=="TD3"):
        # Defining policy to be used
        if use_LSTM:
            policy = Custom_TD3_Policy
            features_extractor_kwargs = dict(features_dim=num_features, lstm_layers=lstm_layers, lstm_dropout=lstm_dropout)
        else:
            policy = "MlpPolicy"
            features_extractor_kwargs = dict()
        # Defining hyperparameters
        policy_kwargs = dict(net_arch=dict(qf=[400, 300], pi=[400, 300]), # network architecture for q function (qf) and policy function (pi)
                             features_extractor_kwargs=features_extractor_kwargs,
                             share_features_extractor=share_features_extractor,
                            )
        # Defining the TD3 model
        if load_path is not None:
            # load a pre trained model
            # model at specified path will be overwritten if save path is the same as load path
            model = TD3.load(path=load_path, policy=policy, env=env, verbose=verbose,
                        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                        policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)
            print("Model loaded from {}".format(load_path))
        else:
            model = TD3(policy=policy, env=env, verbose=verbose,
                        learning_rate=learning_rate, batch_size=batch_size, gamma=gamma,
                        policy_kwargs=policy_kwargs, tensorboard_log=tensorboard_path)

    else:
        raise ValueError("Invalid algorithm name: {}".format(algorithm))
    
    # print network architecture
    if show_net_arch:
        print(model.policy)

    # callback for saving model at regular intervals
    checkpoint_cb = CheckpointCallback(save_freq=max(save_freq // num_vectorized_env, 1), 
                                       save_path=save_path,
                                       name_prefix=training_name)
    
    # callback to terminate training on no model imporovement
    stop_train_cb = StopTrainingOnNoModelImprovement(max_no_improvement_evals=10,
                                                     min_evals=100)
    
    # create seperate evaluation environment for evaluation
    if use_dummy:
        eval_env = make_dummy_env()
        eval_env = Monitor(eval_env)
    else:
        #eval_env = make_env()
        eval_env = make_vec_env(env_id=LeggedEnv,
                           n_envs=5,
                           vec_env_cls=SubprocVecEnv,
                           wrapper_class=DiscreteActionWrapper)
    

    # callback for regular evaluation and save best model
    eval_cb = EvalCallback(eval_env=eval_env, 
                           best_model_save_path=best_model_path,
                           log_path=results_path, eval_freq=max(eval_freq // num_vectorized_env, 1),
                           n_eval_episodes=5, callback_after_eval=stop_train_cb)
    
    # callback list
    cb_list = CallbackList([checkpoint_cb, eval_cb])

    # Continue logging to the original graph if is not training from scratch
    if load_path is not None:
        reset_num_timestep = False
        copy_log_file(load_path=load_path, dst=tensorboard_path, algorithm=algorithm)
    else:
        reset_num_timestep = True
    
    # Train model
    model.learn(total_timesteps=int(num_timesteps), progress_bar=True, callback=cb_list, reset_num_timesteps=reset_num_timestep)

def make_dummy_env():
    env = gym.make('CartPole-v1')
    env.action_space = MultiDiscrete([2])
    # discretize actions using wrapper
    env = DiscreteActionWrapper(env)
    return env

def make_env(use_gui=False):
    env = LeggedEnv(use_gui=use_gui)
    # discretize actions using wrapper
    env = DiscreteActionWrapper(env)
    return env

def copy_log_file(load_path, dst, algorithm):
    """
    Copys the log file in dst folder and copy to src folder
    :param load_path: path where pretrained model is located
    :param dst: destination folder (tensorboard log folder for new model)
    :param algorithm: name of algorithm
    """
    # set the src to the tensorboard_log folder of pretrained model
    if load_path[:2] != "./":
        load_path = "./"+load_path
    prefix = "./trained_models/"
    src = load_path[:load_path.find('/', load_path.find('/', len(prefix))) + 1] + "tensorboard_log" 
    # get the path of the latest updated file assumed to be tensorboard log in src folder
    file_path = max([os.path.join(src, f) for f in os.listdir(src)], key=os.path.getmtime).replace("\\", "/")
    # create a folder in dest dir
    dst += "/"+algorithm+"_0"
    # remove folder if exists
    if os.path.exists(dst):
        shutil.rmtree(dst, ignore_errors=True)
    os.makedirs(dst, exist_ok=True)
    # copy the file to dst folder
    # to plot a continuous graph
    shutil.copytree(src=file_path, dst=dst, dirs_exist_ok=True)    

# testing code
if __name__ == "__main__":
    # Train("PPO", num_timesteps=5e4)
    # Train("PPO", num_timesteps=2e4, training_name="unnamed_training2", load_path="./trained_models/unnamed_training/unnamed_training_50000_steps.zip")
    # Train("SAC", num_timesteps=5e5, training_name='SACtest')
    #Train("SAC", num_timesteps=1e6, training_name='timesteptest')
    # increase vel rwd mulitplier to 10 & max forces = 20
    # Train("SAC", num_timesteps=1e6, training_name='increase_vel_rwd', num_vectorized_env=15)
    # increase vel rwd mulitplier to 100 & max forces = inf
    # Train("SAC", num_timesteps=1e6, training_name='increase_1000_vel_rwd', num_vectorized_env=20)
    # same as above but no LSTM model
    Train("SAC", num_timesteps=1e6, training_name='increase_1000_vel_rwd_no_LSTM', num_vectorized_env=20, use_LSTM=False)
