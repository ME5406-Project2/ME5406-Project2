from stable_baselines3.common.evaluation import evaluate_policy
import string
from stable_baselines3 import PPO, SAC, DDPG, TD3
from sb3_contrib import TRPO
from Train import make_dummy_env, make_env

use_dummy = True

def Validation(algorithm: string, save_path: string, eval_eps: int = 15):
    """
    Loads the pretrained model and validate its performance
    :param algorithm: algorithm name
    :param save_path: path to pretrained model e.g. leggedPPO.zip
    :param eval_eps: number of episodes to evaluate agent
    """
    # create actual environment
    if use_dummy:
        env = make_dummy_env()
    else:
        env = make_env()

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
        env.render()
        if done:
            obs = env.reset()

# testing code
if __name__ == "__main__":
    Validation("PPO", "./trained_models/unnamed_training/unnamed_training_50000_steps.zip")