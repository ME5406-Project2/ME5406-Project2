from stable_baselines3.common.evaluation import evaluate_policy
import string
from stable_baselines3 import PPO, SAC, DDPG, TD3, A2C
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
        eval_env = env
    else:
        env = make_env(use_gui=True, randomize_env=False)
        eval_env = make_env(use_gui=False, randomize_env=False)

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
    
    elif (algorithm=="A2C"):
        # Defining the A2C model
        model = A2C.load(path=save_path, env=env)

    else:
        raise ValueError("Invalid algorithm name: {}".format(algorithm))
    
    # Evaluate policy
    mean_reward, std_reward = evaluate_policy(model, eval_env, n_eval_episodes=eval_eps)
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
    
    # Validation("A2C", "./trained_models/SACTEST/SACTEST_20000_steps.zip", eval_eps=1)
    Validation("SAC", "./trained_models/training73_SDe/training73_largeA_rand_SAC_SDE_550000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/SAC_test/best_model/best_model.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/SAC_test/SAC_test_260000_steps.zip", eval_eps=1)
    #Validation("SAC", "./trained_models/training31_box_0.8/best_model.zip", eval_eps=1)
    # Best model for SAC (best model out of all algorithms)
    # Validation("SAC", "./trained_models/training70_rand_CPG/best_model/best_model.zip", eval_eps=5)
    # Best model for PPO
    # Validation("PPO", "./trained_models/training70_rand_PPO/best_model/best_model.zip", eval_eps=5)
    # Best model for A2C
    # Validation("A2C", "./trained_models/training72_rand_A2C/best_model/best_model.zip", eval_eps=5)
    # Best model for TD3
    # Validation("TD3", "./trained_models/training72_rand_TD3/best_model/best_model.zip", eval_eps=5)