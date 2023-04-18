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
          training_name: string = "unnamed_training", save_freq: int = 20000, eval_freq:int = 40000
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
        policy_kwargs = dict(net_arch=dict(qf=[64, 64], pi=[64, 64]), # network architecture for q function (qf) and policy function (pi)
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
        policy_kwargs = dict(net_arch=dict(qf=[64,64], pi=[64,64]), # network architecture for q function (qf) and policy function (pi)
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
        policy_kwargs = dict(net_arch=dict(qf=[64, 64], pi=[64, 64]), # network architecture for q function (qf) and policy function (pi)
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
        # eval_env = make_vec_env(env_id=LeggedEnv,
        #                    n_envs=5,
        #                    vec_env_cls=SubprocVecEnv,
        #                    wrapper_class=DiscreteActionWrapper)
        eval_env = env
    

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
    # change observations to match that of paper
    #Train("SAC", num_timesteps=1e6, training_name='test1', num_vectorized_env=20, use_LSTM=False)
    # added additional terminal conditions and dead penalty, increased multiplier for moving to 30
    #Train("SAC", num_timesteps=1e6, training_name='test2', num_vectorized_env=20, use_LSTM=False)
    # train test2 for another 1 mil steps
    # Train("SAC", num_timesteps=1e6, training_name='test3', num_vectorized_env=20, use_LSTM=False, load_path="./trained_models/test2/test2_1000000_steps.zip")
    # increase dead penalty to -500 and add LSTM, update prev_action in the correct
    #Train("SAC", num_timesteps=1e6, training_name='test4', num_vectorized_env=20, use_LSTM=True)
    # change back move multiplier to 1 removed actions in obs space 
    # Train("SAC", num_timesteps=1e6, training_name='test5', num_vectorized_env=20, use_LSTM=True)
    # removed timestep penalty added alive reward
    #Train("SAC", num_timesteps=1e6, training_name='test5', num_vectorized_env=20, use_LSTM=True)
    # increase vel reward multiplier to 10, increase pitch penalty multiplier to 100
    #Train("SAC", num_timesteps=1e6, training_name='test6', num_vectorized_env=20, use_LSTM=True)
    # increase pitch penalty multiplier to 200
    #Train("SAC", num_timesteps=1e6, training_name='test7', num_vectorized_env=25, use_LSTM=True)
    # reduce max steps to 2000, cap forward velocity reward at 0.2, scale down pitch penalty to 100, removed position reward
    # Train("SAC", num_timesteps=1e6, training_name='test8', num_vectorized_env=25, use_LSTM=True)
    # same as test 8 but remove terminal condition for being too close to ground and reduce pitch penalty multiplier to 10
    #Train("SAC", num_timesteps=1e6, training_name='test9', num_vectorized_env=25, use_LSTM=True)
    # add back remove terminal condition for being too close to ground and reduce pitch penalty multiplier to 5
    # Train("SAC", num_timesteps=1e6, training_name='test10', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # added position reward
    #Train("SAC", num_timesteps=1e6, training_name='test11', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # added contact forces as obs, fixed CoM bugs, updated position reward
    #Train("SAC", num_timesteps=1e6, training_name='test12', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # scaled down multiplier for move_reward to 0.01
    #Train("SAC", num_timesteps=1e6, training_name='test13', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # Added roll penalty
    #Train("SAC", num_timesteps=1e6, training_name='test14', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # Relaxed terminating condition (removed the height too low constraint)
    #Train("SAC", num_timesteps=1e6, training_name='test15', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # Relax roll and pitch penalty by having allowable range
    #Train("SAC", num_timesteps=1e6, training_name='test16', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # reduce allowable range to 5 deg
    #Train("SAC", num_timesteps=1e6, training_name='test17', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # change xyz dist to x dist
    #Train("SAC", num_timesteps=1e6, training_name='test18', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # added penalty for robot having all 4 legs on ground to encourage lifting of legs, reduce move reward to 0.1, remove tolerance for pitch 
    #Train("SAC", num_timesteps=1e6, training_name='test19', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # remove all terminal conditions except larg pitch / roll
    #Train("SAC", num_timesteps=1e6, training_name='test20', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # increase alive reward (to 0.1) add back terminal conditions
    #Train("SAC", num_timesteps=1e6, training_name='test21', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # Add joint angles as observations
    #Train("SAC", num_timesteps=1e6, training_name='test22', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # removed leg penalty
    #Train("SAC", num_timesteps=1e6, training_name='test23', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # shift goal further away to 3m
    #Train("SAC", num_timesteps=1e6, training_name='test24', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # unlimit velocity reward
    #Train("SAC", num_timesteps=1e6, training_name='test25', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # increase pitch penalty
    #Train("SAC", num_timesteps=1e6, training_name='test26', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # reduce learning rate to 0.001
    # Train("SAC", num_timesteps=1e6, training_name='test27', num_vectorized_env=25, use_LSTM=True, learning_rate=0.001, batch_size=512)
    # reduce learning rate to 0.0001
    #Train("SAC", num_timesteps=1e6, training_name='test28', num_vectorized_env=25, use_LSTM=True, learning_rate=0.0001, batch_size=512)
    # use new reward system
    #Train("SAC", num_timesteps=1e6, training_name='test29', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # added reward for diagonal legs, halve distance reward
    #Train("SAC", num_timesteps=1e6, training_name='test30', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # apply same leg rewards only to upper legs
    #Train("SAC", num_timesteps=1e6, training_name='test31', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # remove contact forces
    #Train("SAC", num_timesteps=1e6, training_name='test32', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # add back contact forces, increase alive reward to 0.25
    #Train("SAC", num_timesteps=1e6, training_name='test33', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # exclude 0 from leg reward
    #Train("SAC", num_timesteps=1e6, training_name='test34', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # increase alive reward to 0.45 from 0.25
    #Train("SAC", num_timesteps=1e6, training_name='test35', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # increase vel reward to 0.2, reduce alive reward back tp 0.25
    #Train("SAC", num_timesteps=1e6, training_name='test36', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # reduce vel reward to 0.1, introduce penalty for not moving / negative vel
    #Train("SAC", num_timesteps=1e6, training_name='test37', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # removed termination condition for too low / touch ground
    #Train("SAC", num_timesteps=1e6, training_name='test38', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    #Added reward to encourage gait like behaviour to replace leg reward
    #Train("SAC", num_timesteps=1e6, training_name='test39', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # improved gait rewards
    #Train("SAC", num_timesteps=1e6, training_name='test40', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # Add back termination conditions, added conditions for gait and lowered limits
    #Train("SAC", num_timesteps=1e6, training_name='test41', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # fixed bugs in direction
    #Train("SAC", num_timesteps=1e6, training_name='test42', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    # increase lower limit
    #Train("SAC", num_timesteps=1e6, training_name='test43', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)
    #increase pitch penalty
    Train("SAC", num_timesteps=1e6, training_name='test44', num_vectorized_env=25, use_LSTM=True, learning_rate=0.01, batch_size=512)