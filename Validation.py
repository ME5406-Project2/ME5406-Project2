from stable_baselines3.common.evaluation import evaluate_policy
import string
from stable_baselines3 import PPO, SAC, DDPG, TD3
from sb3_contrib import TRPO
from Train import make_dummy_env, make_env

use_dummy = False

def Validation(algorithm: string, save_path: string, eval_eps: int = 15):
    """
    Loads the pretrained model and validate its performance
    :param algorithm: algorithm name
    :param save_path: path to pretrained model e.g. leggedPPO.zip
    :param eval_eps: number of episodes to evaluate agent
    """
    # create environment
    if use_dummy:
        env = make_dummy_env()
    else:
        env = make_env(use_gui=True)

    if (algorithm=="DDPG"):
        # Defining the DDPG model
        model = DDPG.load(path=save_path, env=env)

    elif (algorithm=="PPO"):
        # Defining the PPO model
        model = PPO.load(path=save_path, env=env)

    elif (algorithm=="SAC"):
        # Defining the SAC model
        model = SAC.load(path=save_path, env=env)

    elif (algorithm=="TRPO"):
        policy = "MlpPolicy"
        # Defining the TRPO model
        model = TRPO.load(path=save_path, env=env)

    elif (algorithm=="TD3"):
        # Defining the TD3 model
        model = TD3.load(path=save_path, env=env)

    else:
        raise ValueError("Invalid algorithm name: {}".format(algorithm))
    
    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, model.get_env(), n_eval_episodes=eval_eps)
    print("Mean rewards =", mean_reward)
    print("Std rewards =", std_reward)
    
    # render the environment forever until ctrl-c is pressed
    obs = env.reset()
    while True:
        action, _states = model.predict(obs, deterministic=True)
        obs, _rewards, done, _ = env.step(action=action)
        if use_dummy:
            env.render()
        if done:
            obs = env.reset()

# testing code
if __name__ == "__main__":
    #Validation("SAC", "./trained_models/test1/test1_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test1/best_model/best_model.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test2/test2_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test2/best_model/best_model.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test3/best_model/best_model.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test3/test3_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test4/test4_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test4/best_model/best_model.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test5/test5_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test5/best_model/best_model.zip", eval_eps=1)
    Validation("SAC", "./trained_models/test6/test6_1000000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/test6/best_model/best_model.zip", eval_eps=1)