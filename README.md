# ME5406 Project 2
Title: **CPG-Based Terrain-Aware Locomotion with Deep Reinforcement Learning**
<br>Authors: Elango Praveen, Lim Gang Le, Mong Juin Hwaye

<br>For this project, we aim to train our quadrupedal robot make it terrain aware whereby the robot is able to adapt its gait based on the terrain it is in. The code is written in Python 3.8.2 and the following Python libraries are required for our code to work.

    ale_py==0.7.4
    gym==0.21.0
    numpy==1.24.2
    pybullet==3.2.5
    sb3_contrib==1.7.0
    stable_baselines3==1.7.0
    torch==2.0.0
    

The code suppots the following algorithms
- SAC
- A2C
- TD3
- PPO
- DDPG

We will now walk through the files.<br>
# Folders
## Assembly
This folder contains all URDF files and associated STL files for the quadrupedal robot.
## trained_models
This folder contains the trained models using the various algorithms.
# Source Code
## Train.py
This file allows the user to set up the training session. The user can select the algorithm to be used and the hyperparameters for training.
## Validation.py
This file allows the user to validate and test a trained model. Visualization of trained model is also done using this file.
## LeggedEnv.py
This file contains the environment used for training. This uses pybullet and loads quadrupedal robot and the environemnt into the physics engine. 
## CustomNetwork.py
This file contains the custom  feature class (LSTM) for each algorithm.
## DiscreteWrapper.py
This file contains the DiscreteWrapper Class. It converts the box (continuous outputs in range [-1, 1]) into MultiDiscrete actions as required by the environment.

# How to train a model
To train the model from scratch, open up `Train.py` to edit hyperparameters at the bottom.<br>

    if __name__ == "__main__":
        Train(
            algorithm = "PPO", num_vectorized_env = 10,
            load_path = None,
            num_timesteps = 1e6, num_features = 64,
            show_net_arch = False,
            use_LSTM = True, verbose = 0, 
            share_features_extractor = True,
            lstm_layers = 1, lstm_dropout = 0.0,
            learning_rate = 0.0001, gamma = 0.99, batch_size = 256,
            training_name = "unnamed_training", 
            save_freq = 10000, eval_freq = 5000
            ):

To start training from an existing model,  enter the relative path to the trained model in the load path parameter.<br>

    if __name__ == "__main__":
        Train("PPO", num_timesteps=2e6, training_name="unnamed_training2", 
                load_path="./trained_models/unnamed_training/unnamed_training_50000_steps.zip"
                )

Once done, save the file and run the python code <br>

    > python Train.py

The trained models will be saved in `./trained_models/[name of training]` as a .zip file. The model that produces the best results during evaluation will be saved in `./trained_models/[name of training]/best_model/` as `best_model.zip`.

# How to run trained models
Open up `Validation.py` to edit the parameters at the bottom.<br>

    if __name__ == "__main__":
        Validation(
            algorithm = "SAC", 
            save_path = ./trained_models/SAC_test/SAC_test_260000_steps.zip, 
            eval_eps = 15
        )

Once done, save the file and run the python code <br>

    > python Validation.py

# Video Demostration

https://user-images.githubusercontent.com/60381094/233598965-0089bb38-8bfd-41e9-99c0-e4f59646cce4.mp4

