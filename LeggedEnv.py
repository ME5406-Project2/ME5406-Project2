import math
import gym
import pybullet as p
import pybullet_data
import time
import numpy as np
from Surface import Surface
from gym.spaces import MultiDiscrete
from gym import GoalEnv

class LeggedEnv(gym.Env):
    """
    LeggedEnv Robot Environment
    The goal of the four-legged agent in this environment is to locomote to a randomly generated
    rectangular goal-space in the workspace.
    
    """
    def __init__(self, use_gui=False):

        # Connect to PyBullet client
        if use_gui:
            self.physics_client = p.connect(p.GUI)
        else:
            self.client = p.connect(p.DIRECT)

        # Set visualisation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # Store observations in dict
        self.obs = {}

        # Termination condition parameter
        self.termination_pos_dist = 0.5
        self.max_steps = 3000
        self.env_step_count = 0
        self.prev_dist = 0
        self.move_reward = 0

        # Initialise rewards
        self.goal_reward = 0
        self.position_reward = 0
        self.time_reward = 0
        self.stability_reward = 0
        self.contact_reward = 0
        self.reward = 0
        self.prev_base_lin_vel = 0
        self.work_done_reward = 0
        self.prev_base_pos = 0

        # Initialize isdead
        self.is_dead = False

        self.prev_contact_pos = None
        self.stride_length = 0

        # Contact distance between front left leg and terrain
        self.contact_dist = 0

        # Define joint-to-action mapping
        # self.joint_to_action_map = {
        #     0: np.array([-10, -5, 0, 5, 10]),
        #     1: np.array([-10, -5, 0, 5, 10]),
        #     2: np.array([-10, -5, 0, 5, 10]),
        #     3: np.array([-10, -5, 0, 5, 10]),
        #     4: np.array([-10, -5, 0, 5, 10]),
        #     5: np.array([-10, -5, 0, 5, 10]),
        #     6: np.array([-10, -5, 0, 5, 10]),
        #     7: np.array([-10, -5, 0, 5, 10]),
        # }
        # self.joint_to_action_map = {
        #     0: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     1: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     2: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     3: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     4: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     5: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     6: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        #     7: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
        #                  5, 10, 15, 20, 25, 30, 35, 40, 45]),
        # }         
        # self.joint_to_action_map = {
        #     0: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     1: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     2: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     3: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     4: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     5: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     6: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        #     7: np.array([-0.25, -0.20, -0.15, -0.10, -0.05, 0, 0.05, 0.10, 0.15, 0.20, 0.25]),
        # }

        # self.joint_to_action_map = {
        #     0: np.array([0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]), # Left
        #     1: np.array([0, 0.025, 0.05, 0.075, 0.10, 0.125, 0.15, 0.175, 0.20, 0.225, 0.25, 0.275, 0.30]), # Right
        # }

        # Control parameters
        # self.joint_to_action_map = {
        #     0: np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]), # FL Frequency
        #     1: np.array([0.3, 0.4, 0.5, 0.6, 0.7]), # FL Amplitude
        #     2: np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]), # FR Frequency
        #     3: np.array([0.3, 0.4, 0.5, 0.6, 0.7]), # FR Amplitude
        #     4: np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]), # BL Frequency
        #     5: np.array([0.3, 0.4, 0.5, 0.6, 0.7]), # BL Amplitude
        #     6: np.array([2, 2.1, 2.2, 2.3, 2.4, 2.5, 2.6, 2.7, 2.8, 2.9, 3]), # BR Frequency
        #     7: np.array([0.3, 0.4, 0.5, 0.6, 0.7]), # BR Amplitude
        # }

        self.joint_to_action_map = {
            0: np.array([2, 2.5, 3]), # Frequency
            1: np.array([0.3, 0.4, 0.5]), # Amplitude
        }

        # self.joint_to_action_map = {
        #     0: np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), # Upper Left
        #     1: np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), # Upper Right
        #     2: np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), # Lower Left
        #     3: np.array([0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30]), # Lower Right
        # }

        # Load the initial parameters again
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeId = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        # Load surface that the robot will walk on
        self.surface = Surface(
            texture_path="wood.png",
            lateral_friction=1.0,
            spinning_friction=1.0,
            rolling_friction=0.0)
        
        # Spawn the robot and goal box in simulation
        self.spawn_robot()
        self.generate_goal()
        
        self.continuous_action_space = False

        if self.continuous_action_space:
            self.action_space = gym.spaces.Box(
            low=-50.0, high=50.0, shape=(8,), dtype=np.float64)
        else:
            # Define action space
            actions = [len(self.joint_to_action_map[key]) for key in range(len(self.joint_to_action_map))]
            self.action_space = MultiDiscrete(actions)
        # self.action_space = gym.spaces.Box(
        #     low=-10, high=10, shape=(8,), dtype=np.float64)

        # Define observation spaces
        # obs_shape = self.get_observation().shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(120,), dtype=np.float64)
        
        # Buffer for history stacking of observations
        self.buffer_size = 4
        self.obs_buffer = np.zeros((self.buffer_size, 30))

        # Robot weight
        self.weight = self.compute_weight()
        
        # CPG timestep
        self.t = 0
        self.cpg_first = True
        self.cpg_cnt = 0
        self.store_joint_vel = []

    def spawn_robot(self):
        """
        Instantiates the robot in the simulation.
        """
        # Set the start pose of the robot
        self.robot_start_pos = [0, 0, 0.5]
        self.robot_start_rpy = [0, 0, np.deg2rad(180)]
        self.robot_start_orn = p.getQuaternionFromEuler(self.robot_start_rpy)
        
        # Load the robot URDF into PyBullet
        self.robot = p.loadURDF(
            "assembly/Assem1_v5.SLDASM/urdf/Assem1_v5.SLDASM.urdf", 
            self.robot_start_pos, 
            self.robot_start_orn)

        # Get the total number of robot joints
        self.num_of_joints = p.getNumJoints(self.robot)

        # Get the actuator joints
        # NOTE: Revolute and Prismatic joints are motorised by default
        self.actuators  = [joint for joint in range(self.num_of_joints) 
                           if p.getJointInfo(self.robot, joint)[2] != p.JOINT_FIXED]
        
        # Get the information of the actuator joints
        self.actuators_info = [p.getJointInfo(self.robot, joint) for joint in self.actuators]

        # print(self.actuators_info)

        self.start_joint_pos = [
            0, #front left upper
            0, #front left lower
            0, #front right upper
            0, #front right lower
            0, #back left upper
            0, #back left lower
            0, #back right upper
            0  #back right lower
        ]

        # Set the joint positions
        for i in range(self.num_of_joints):
            p.resetJointState(self.robot, self.actuators[i], self.start_joint_pos[i])
        
        # Set all the motors to position or velocity control
        # p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.POSITION_CONTROL)
        p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.VELOCITY_CONTROL)

        # Upper and lower joint indeces
        self.upper_joint_indeces = [0, 2, 4, 6]
        self.lower_joint_indeces = [1, 3, 5, 7]

        # Enable force torque sensors on all actuator joints
        for joint in self.actuators:
            p.enableJointForceTorqueSensor(self.robot, joint, 1)

        self.prev_joint_states = p.getJointStates(self.robot, self.actuators)

    def reset(self):

        self.reward = 0
        self.env_step_count = 0
        # Reset simulation
        p.resetSimulation()
        # Load the initial parameters again
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeId = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        # Load surface that the robot will walk on
        self.surface = Surface(
            texture_path="wood.png",
            lateral_friction=1.0,
            spinning_friction=1.0,
            rolling_friction=0.0)
        
        # Spawn the robot and goal box in simulation
        self.spawn_robot()
        self.generate_goal()

        self.is_dead = False
        # wait until robot is stable for 100 consecutive counts
        # stable_count = 0
        # while (stable_count<100): #test
        #     p.stepSimulation()
        #     #stable_count += -1 if self.check_no_feet_on_ground() else 1
        #     #stable_count = max(0, stable_count)
        #     if self.check_no_feet_on_ground():
        #         stable_count = 0
        #     else:
        #         stable_count+=1

        # Step the simulation and return the initial observation
        # p.stepSimulation()
        # time.sleep(1/240)
        self.t = 0
        self.cpg_first = True

        return self.get_observation()
    
    def step(self, action):
        
        # CPG controller learning
        timestep = (self.env_step_count+1) * (1/240)
        control_params = []
        for control_param, index in enumerate(action):
            param_val  = self.joint_to_action_map[control_param][index]
            control_params.append(param_val)
        cmd_joint_pos = self.cpg_position_controller(timestep, control_params[0], control_params[1])
        print(control_params)
        # Crawl gait position control
        # joint_positions = []
        # # Find the actions based on pre-defined mappings
        # for joint, index in enumerate(action):
        #     joint_pos = self.joint_to_action_map[joint][index]
        #     joint_positions.append(joint_pos)
        # cmd_joint_pos = []
        # cmd_joint_pos.extend(joint_positions)
        # for joint_pos in joint_positions:
        #     negate_joint_pos = joint_pos * -1.0
        #     cmd_joint_pos.append(negate_joint_pos)
        # # print(cmd_joint_pos)
        p.setJointMotorControlArray(self.robot, self.upper_joint_indeces,
                                    p.POSITION_CONTROL, targetPositions=-np.array(cmd_joint_pos))
        p.setJointMotorControlArray(self.robot, self.lower_joint_indeces,
                                    p.POSITION_CONTROL, targetPositions=cmd_joint_pos)

        # p.setJointMotorControlArray(self.robot, self.upper_joint_indeces,
        #                             p.POSITION_CONTROL, targetPositions=-np.array(action))
        # p.setJointMotorControlArray(self.robot, self.lower_joint_indeces,
        #                             p.POSITION_CONTROL, targetPositions=action)
        # Send action velocities to robot joints
        # p.setJointMotorControlArray(self.robot, self.actuators, 
        #                             p.VELOCITY_CONTROL, targetVelocities=joint_velocities)
        
        # self.store_joint_vel.append(joint_velocities)
        
        # Step the simulation
        p.stepSimulation()
        # time.sleep(1/240)
        self.env_step_count += 1
        
        # Get the observation
        observation = self.get_observation()
        # # Update buffer
        # self.buffer[:-1] = self.buffer[1:]
        # self.buffer[-1] = observation

        # # Concatenate observations
        # stacked_observations = np.concatenate(self.buffer, axis=0)

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            print("Goal Reached!")
            done = True
        elif self.check_is_unrecoverable():
            done = True
            self.is_dead = True
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
        else:
            done = False

        reward = self.get_reward(control_params)
        # if isinstance(reward, np.ndarray):
        #     reward = reward[0]
        # self.reward += 0
        self.prev_dist = self.xyz_obj_dist_to_goal()

        self.prev_joint_velocities = self.joint_velocities
        self.prev_base_lin_vel = p.getBaseVelocity(self.robot)[0][0]
        self.prev_joint_states = p.getJointStates(self.robot, self.actuators)
        self.prev_base_pos = p.getBasePositionAndOrientation(self.robot)[0][2]
        
        
        return observation, reward, done, {}
        
    def xyz_obj_dist_to_goal(self):

        dist = np.linalg.norm(self.base_pos - self.goal_pos)
        # print("dist",dist)
        return dist
    
    def generate_goal(self):
        
        box_pos = [5.5, -0.25, 0]
        box_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.box_collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                                          halfExtents=[0.1, 0.1, 0.2])
        
        self.goal_id = p.createMultiBody(baseMass=1,
                                        baseCollisionShapeIndex=self.box_collision_shape,
                                        baseVisualShapeIndex=-1,
                                        basePosition=box_pos,
                                        baseOrientation=box_orn)
        
        self.start_joint_pos = [
            0, #front left upper
            0, #front left lower
            0, #front right upper
            0, #front right lower
            0, #back left upper
            0, #back left lower
            0, #back right upper
            0  #back right lower
        ]

        self.generate_terrain()
        
    def generate_terrain(self):
        # create a collision shape for the mud
        half_size = [2, 2, 0.15]
        block_shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=half_size)

        # create a multi-body object for the triangular block
        block_position = [4, 0, 0]
        block_orientation = p.getQuaternionFromEuler([0, 0, 0])
        self.mud = p.createMultiBody(
            baseMass=0,
            baseCollisionShapeIndex=block_shape,
            basePosition=block_position,
            baseOrientation=block_orientation,
        )

        p.changeVisualShape(self.mud, -1, rgbaColor=[101/255, 67/255, 33/255, 1])
        light_direction = [1, 1, 1]  # Direction of the light
        light_color = [1, 1, 1]  # Color of the light (white)
        light_id = p.addUserDebugParameter("light", -1, 1, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SHADOWS, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_SEGMENTATION_MARK_PREVIEW, 0)

        p.changeDynamics(
            self.mud, -1,
            contactStiffness=0.01,
            contactDamping=1.0,
            restitution=10.0,
            lateralFriction=1000.0,
            rollingFriction=1000.0,
            spinningFriction=1000.0,
            frictionAnchor=True,
            anisotropicFriction=[1.0, 0.05, 0.05])
        

    # Check if robot legs contacting the terrain
    def check_robot_legs_in_mud(self):
        # Link IDs of the end-effectors
        # Respectively: FL, FR, BL, BR
        foot_link_ids = [1, 3, 5, 7]
        foot_contacts = [False] * 4

        for i, foot_id in enumerate(foot_link_ids):
            contact_points = p.getContactPoints(bodyA=self.robot, 
                                                bodyB=self.mud, 
                                                linkIndexA=foot_id)
            if (len(contact_points)) != 0:
                if foot_id == 1:
                    self.contact_dist = contact_points[0][8]
            else:
                self.contact_dist = 0
    
    def cpg_position_controller_decoupled(self, t, control_params):
        
        fl_amplitude = control_params[0]
        fl_frequency = control_params[1]
        fr_amplitude = control_params[2]
        fr_frequency = control_params[3]
        bl_amplitude = control_params[4]
        bl_frequency = control_params[5]
        br_amplitude = control_params[6]
        br_frequency = control_params[7]
        self.phase_offset = 0.5

        # Calculate the CPG output for each leg
        front_left_leg_pos = fl_amplitude * np.sin(
            2 * np.pi * fl_frequency * t + self.phase_offset)
        front_right_leg_pos = fr_amplitude * np.sin(
            2 * np.pi * fr_frequency * t + np.pi / 2 + self.phase_offset) 
        back_left_leg_pos = bl_amplitude * np.sin(
            2 * np.pi * bl_frequency * t + np.pi + self.phase_offset) 
        back_right_leg_pos = br_amplitude * np.sin(
            2 * np.pi * br_frequency * t + 3 * np.pi / 2 + self.phase_offset)

        # Return the CPG output for all 4 legs
        return [front_left_leg_pos, front_right_leg_pos, 
                back_left_leg_pos, back_right_leg_pos]
    
    def cpg_position_controller(self, t, f, amp):
        
        # Set the CPG parameters
        self.frequency = f #3 #2 #1 
        self.amplitude = amp #0.3 #0.5 #0.7
        self.phase_offset = 0.5

        # Calculate the CPG output for each leg
        front_left_leg_pos = self.amplitude * np.sin(
            2 * np.pi * self.frequency * t + self.phase_offset)
        front_right_leg_pos = self.amplitude * np.sin(
            2 * np.pi * self.frequency * t + np.pi / 2 + self.phase_offset) 
        back_left_leg_pos = self.amplitude * np.sin(
            2 * np.pi * self.frequency * t + np.pi + self.phase_offset) 
        back_right_leg_pos = self.amplitude * np.sin(
            2 * np.pi * self.frequency * t + 3 * np.pi / 2 + self.phase_offset)

        # Return the CPG output for all 4 legs
        return [front_left_leg_pos, front_right_leg_pos, 
                back_left_leg_pos, back_right_leg_pos]
    
    def cpg_step(self, t):

        if self.check_no_feet_on_ground():
            self.cpg_cnt+=1

        freq = 2
        amp = 0.6
        leg_positions = self.cpg_position_controller(t, freq, amp)
        # leg_velocities = [pos / (1/240) for pos in leg_positions]
        # print(max(leg_velocities))
        observation = self.get_observation()
        cmd_joint_pos = [0] * 8

        for idx, joint in enumerate(self.upper_joint_indeces):
            cmd_joint_pos[joint] = -leg_positions[idx]

        for idx, joint in enumerate(self.lower_joint_indeces):
            cmd_joint_pos[joint] = leg_positions[idx]
        leg_vel = (cmd_joint_pos - self.joint_positions)/(1/240)

        p.setJointMotorControlArray(self.robot, self.upper_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=leg_positions)
        p.setJointMotorControlArray(self.robot, self.lower_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=-np.array(leg_positions))
        # p.setJointMotorControlArray(self.robot, range(len(self.actuators)), 
        #                            p.VELOCITY_CONTROL, targetVelocities=[10, 5, 10, 10, 10, 0, -5, -5])
        p.stepSimulation()
        time.sleep(1/240)
        self.env_step_count += 1

        # Get the observation
        observation = self.get_observation()

        # Update buffer
        # self.buffer[:-1] = self.buffer[1:]
        # self.buffer[-1] = observation

        # # Concatenate observations
        # stacked_observations = np.concatenate(self.buffer, axis=0)

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
            print("GOAL REACHED")
            print("Episode Reward",self.reward)
            print("Episode Length", self.env_step_count)
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
            print('EPISODE LENGTH EXCEEDED')
        else:
            done = False

        reward = self.get_reward([0,0])
        self.reward += reward
        self.prev_dist = self.xyz_obj_dist_to_goal()

        stridelen = self.calc_stride_length()

        self.prev_base_lin_vel = p.getBaseVelocity(self.robot)[0][0]
        self.prev_joint_states = p.getJointStates(self.robot, self.actuators)
        self.prev_base_pos = p.getBasePositionAndOrientation(self.robot)[0][2]
    
        return observation, reward, done
        
    def get_end_effector_pose(self):
        
        front_left_EE_pos = p.getLinkState(self.robot, 1)[0]
        front_left_EE_orn = p.getLinkState(self.robot, 1)[1]

        front_right_EE_pos = p.getLinkState(self.robot, 3)[0]
        front_right_EE_orn = p.getLinkState(self.robot, 3)[1]

        back_left_EE_pos = p.getLinkState(self.robot, 5)[0]
        back_left_EE_orn = p.getLinkState(self.robot, 5)[1]

        back_right_EE_pos = p.getLinkState(self.robot, 7)[0]
        back_right_EE_orn = p.getLinkState(self.robot, 7)[1]

        legs_pos = [front_left_EE_pos, front_right_EE_pos, 
                    back_left_EE_pos, back_right_EE_pos]
        
        legs_orn = np.array([front_left_EE_orn, front_right_EE_orn, 
                             back_left_EE_orn, back_right_EE_orn])


        return legs_pos, legs_orn
    
    def check_no_feet_on_ground(self):

        # Link IDs of the end-effectors
        # Respectively: FL, FR, BL, BR
        foot_link_ids = [1, 3, 5, 7]
        foot_contacts = [False] * 4

        for i, foot_id in enumerate(foot_link_ids):
            contact_points = p.getContactPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=foot_id)
            if len(contact_points) > 0:
                foot_contacts[i] = True

        not_on_ground = all(element == False for element in foot_contacts)
        
        if not_on_ground:
            return True
        else:
            return False

    def calc_stride_length(self):
        contact_pos_list = []
        contact_points = p.getContactPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=1)

        if len(contact_points) > 0:
            contact_pos = contact_points[0][5]
            contact_pos_list.append(contact_pos)

        if len(contact_points) == 0:
            return None

        curr_contact_pos = np.mean(contact_pos_list, axis=0)

        if self.prev_contact_pos is not None:
            self.stride_length = np.linalg.norm(curr_contact_pos - self.prev_contact_pos)

        self.prev_contact_pos = curr_contact_pos

        return self.stride_length

        
    def get_observation(self):

        self.check_robot_legs_in_mud()
        
        # Positions of all 8 joints
        self.joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
        # print(self.joint_positions)
        # relative joint angle in radians
        # range = [-pi/2, pi/2] -> divide by pi/2 to normalize to [-1,1]
        self.normalized_joint_angles = np.array(self.joint_positions / (np.pi / 2), dtype=np.float64)

        # Velocities of all 8 joints
        self.joint_velocities = np.array([p.getJointState(self.robot, self.actuators[i])[1] 
                                       for i in range(self.num_of_joints)])
        # v_max (set in urdf file) = p.getJointInfo()[11]
        # normalize joint (angular) velocities [-v_max,v_max] -> [-1,1] (divide by v_max)
        max_angular_velocity = np.array([self.actuators_info[i][11] for i in range(self.num_of_joints)])
        self.normalized_joint_velocities = np.array(np.divide(self.joint_velocities, max_angular_velocity), dtype=np.float64)

        # Reaction forces of all 8 joints
        self.joint_reaction_forces = np.array([p.getJointState(self.robot, self.actuators[i])[2] 
                                       for i in range(self.num_of_joints)])
        
        # NOTE f_max not defined in URDF (might be useful for slippage stuff?)
        # f_max = p.getJointInfo()[10]
        # normalize reaction forces [-f_max, f_max] -> [-1,1]
        # max_reaction_force = np.array([self.actuators_info[i][10] for i in range(self.num_of_joints)])
        # self.normalized_joint_reaction_forces = np.divide(self.joint_reaction_forces, max_reaction_force)

        # Linear and Angular velocity of the robot
        self.base_lin_vel = np.array(p.getBaseVelocity(self.robot)[0])
        self.base_ang_vel = np.array(p.getBaseVelocity(self.robot)[1])
        # print("baselinvel",self.base_lin_vel)

        # normalize linear and angular velocities [] -> [-1,1]
        # limits for linear and angular velocity cannot be set in URDF
        # need to find other methods to implement / constraint
        # proposed method to estimate linear velocity: model legs as a 2 wheel differential drive (overestimate)
        # length of leg = 0.575m, dist between legs = 0.565
        # lin_vel_max = pi*0.575 (assume all 4 legs in sync and max angle leg can cover in 1s pi rad)
        # ang_vel_max = 0.575*2pi / 0.565 ~= 2pi 
        base_lin_vel_limit, base_ang_vel_limit = 0.575*np.pi , 2*np.pi
        self.normalized_base_lin_vel = np.array(self.base_lin_vel / base_lin_vel_limit, dtype=np.float64)
        self.normalized_base_ang_vel = np.array(self.base_ang_vel / base_ang_vel_limit, dtype=np.float64)

        # Robot Position
        self.base_pos = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        # Robot Orientation (Quaternion)
        self.base_orn = np.array(p.getBasePositionAndOrientation(self.robot)[1])

        # normalize orientation to [-1,1] by dividing by norm
        base_orn_length = np.sqrt(np.dot(self.base_orn, self.base_orn))
        self.normalized_base_orn = np.array(self.base_orn / base_orn_length, dtype=np.float64)

        # Robot Orientation (Euler Angles)
        self.base_rpy = np.array(p.getEulerFromQuaternion(self.base_orn))

        # normalize orientation (Euler Angles) [-pi, pi] -> [-1,1]
        self.normalized_rpy = np.array(self.base_rpy / np.pi, dtype=np.float64)

        # Robot end-effector position
        self.robot_legs_EE_pos = np.array(self.get_end_effector_pose()[0])

        # Robot end-effector position
        self.robot_legs_EE_orn = np.array(self.get_end_effector_pose()[1])

        # Goal position
        self.goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])

        # Goal orientation
        self.goal_orn = np.array(p.getBasePositionAndOrientation(self.goal_id)[1])

        # Convert robot's orientation to rotation matrix
        rotation_matrix = np.array(p.getMatrixFromQuaternion(self.base_orn))
        rotation_matrix = rotation_matrix.reshape(3,3)

        # Create homogeneous rotation matrix
        T = np.eye(4)
        # transpose to convert from world frame to robot frame
        T[:3, :3] = rotation_matrix.T
        T[:3, 3] = -self.base_pos # Subtract the translation vector

        # Transform goal coordinates from global frame into robot frame
        relative_goal_pos = np.dot(T, np.append(self.goal_pos, 1))[:3]

        # Relative distance to goal (not normalized)
        self.relative_goal_dist = np.sqrt(np.dot(relative_goal_pos, relative_goal_pos))
        # Relative vector to goal (normalized)
        self.relative_goal_vect = relative_goal_pos / self.relative_goal_dist

        leg_pos_robot_frame_norm = []
        for leg_pos in self.robot_legs_EE_pos:
            p_world = leg_pos - self.base_pos
            p_robot = np.dot(rotation_matrix.T, p_world)
            p_robot_dist = np.sqrt(np.dot(p_robot, p_robot))
            p_robot_norm = p_robot / p_robot_dist
            leg_pos_robot_frame_norm.append(p_robot_norm)
        leg_pos_robot_frame_norm = np.array(leg_pos_robot_frame_norm)
        leg_pos_robot_frame_norm = leg_pos_robot_frame_norm.flatten()

        # Normal force for each foot upon contact
        # foot_link_ids = [1, 3, 5, 7]

        # for i, foot_id in enumerate(foot_link_ids):
        #     contact_points = p.getContactPoints(bodyA=self.robot, 
        #                                             bodyB=self.surface.plane_id, 
        #                                             linkIndexA=foot_id)
        #     if len(contact_points) == 1 and foot_id ==3:
        #         print("normal_force", foot_id, contact_points[0][9])
    
                
        # print(f"Position{self.base_pos}")
        # print(f"Ornrpy{self.base_rpy}")


        # Get contact forces and frictional forces
        contact_forces = []
        frictional_forces = []
        z_force_values = []
        # Link IDs of the end-effectors
        # Respectively: FL, FR, BL, BR
        foot_link_ids = [1, 3, 5, 7]
        
        # divide the weight of robot into 4 legs
        average_force = self.weight / 4
        for i, foot_id in enumerate(foot_link_ids):
            contact_points = p.getContactPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=foot_id)
            if (len(contact_points) == 0):
                # contact_forces.append(0)
                # frictional_forces.append(0)
                contact_forces.append(np.array([0, 0, 0]))
                z_force_values.append(0)
            else:
                # contact_forces.append((contact_points[9] - average_force)/(average_force))
                # frictional_forces.append(contact_points[10])
                normal_forces = contact_points[0][7]
                normal_forces /= np.linalg.norm(normal_forces)
                contact_forces.append(np.array(normal_forces))
                z_force_values.append(contact_points[0][9])
        # flatten contact forces
        contact_forces = np.concatenate(contact_forces)
        # print(z_force_values)


        # Normalised observations
        observation = np.hstack([
            *self.normalized_joint_angles,
            *self.normalized_joint_velocities,
            *self.normalized_base_lin_vel,
            *self.normalized_base_ang_vel,
            *self.normalized_base_orn,
            np.array(self.relative_goal_dist, dtype=np.float64),
            np.array(self.relative_goal_vect, dtype=np.float64)
        ])
        # Update buffer
        self.obs_buffer[:-1] = self.obs_buffer[1:]
        self.obs_buffer[-1] = observation

        # Concatenate observations
        stacked_observations = np.concatenate(self.obs_buffer, axis=0)
        # # Additions to be made
        # CoG in robot frame
        return stacked_observations
    
    def get_reward(self, control_params):
        # Goal reached
        # if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
        #     self.goal_reward = 500
        # else:
        #     self.goal_reward = 0

        # Robot is moving towards goal - Position
        # if self.prev_dist > self.xyz_obj_dist_to_goal():
        self.position_reward = 0.75 * self.xyz_obj_dist_to_goal()
        # Robot is moving 
        # self.move_reward = 0.75 * (self.base_lin_vel[0] - self.prev_base_lin_vel)
        # self.move_reward = 1.0 * self.normalized_base_lin_vel[0]
        # self.move_reward = min(self.move_reward, 0.2)
        # Penalise work done
        # Get the joint states for the current and next states
        # current_joint_states = p.getJointStates(self.robot, self.actuators)
        # work_done = 0.0
        # for i, joint_state in enumerate(self.prev_joint_states):
        #     joint_torque = joint_state[3]
        #     joint_displacement = current_joint_states[i][0] - joint_state[0]
        #     joint_work = joint_torque * joint_displacement
        #     work_done += joint_work

        # self.work_done_reward = 0.0001*work_done

        # Time penalty
        # self.time_reward = -0.05

        alive_reward = 0.005

        # Stability penalty
        # self.stability_reward = -0.1 * abs(self.base_pos[2] - self.prev_base_pos)
        
        # dead_penalty = 0
        # if self.is_dead:
        #     dead_penalty = -2

        # print("pos_reward", self.position_reward)
        # print("movreward", self.move_reward)

        # print("prev_dist", self.prev_dist)
        # print("xyz_dist_to_goal", self.xyz_obj_dist_to_goal())
            
        # Time-based penalty
        # if self.env_step_count >= self.max_steps:
        #     self.time_reward = -0.01

        # # Encourage stability
        # # Value of 1 means perfect stability, 0 means complete instability
        roll = self.base_rpy[0]
        pitch = self.base_rpy[1]
        # z_pos = self.base_pos[2]
        # if 1 - abs(roll) - abs(pitch) - abs(z_pos - 0.275):
        #     self.stability_reward = 0.1

        # penalize for too much tilting forward or backwards
        pitch_penalty = -5 * pitch**2
        roll_penalty = -5 * roll**2

        gait_reward = 0
        # Encourage conservative gait in rough terrain
        if self.contact_dist > 0.0:
            # Reward high amplitude
            if control_params[0] > 0.4:
                gait_reward += 0.005
            # Reward low frequency
            if control_params[1] < 2.5:
                gait_reward += 0.005
        else:
            if control_params[0] < 0.4:
                gait_reward += 0.005
            if control_params[1] > 2.5:
                gait_reward += 0.005
        # if self.check_no_feet_on_ground():
        #     self.contact_reward = -0.01
        # ADDITIONS TO BE MADE
        # Penalise staying in same place
        # Penalise sudden joint accelerations
        # Ensure that joint angles don't deviate too much

        # Sum of all rewards
        # reward = -(self.position_reward - self.move_reward + self.work_done_reward - self.stability_reward)
        reward = -self.position_reward  + pitch_penalty + roll_penalty + alive_reward + gait_reward
        return reward
    
    def process_and_cmd_vel(self):
        joint_pos_arr = []
        for joint_vel in self.store_joint_vel:
            joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
            for idx, vel in enumerate(joint_vel):
                change_in_joint_pos = vel 
                joint_pos = joint_positions[idx] + change_in_joint_pos
                joint_pos_arr.append(joint_pos)
            # print(joint_pos_arr)
            p.setJointMotorControlArray(self.robot, self.actuators, 
                                        p.POSITION_CONTROL, targetVelocities=joint_pos_arr)
            p.stepSimulation()
            joint_pos_arr.clear()


    def check_is_unrecoverable(self):
        closest_points = p.getClosestPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=-1,
                                                distance=0.10)
        contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.surface.plane_id, linkIndexA=-1)

        roll = self.base_rpy[0]
        pitch = self.base_rpy[1]
        
        is_unrecoverable = False
        # Height of torso is too low (less than 0.2 of original height)
        # if (self.normalized_base_height < -0.8 and not -0.01 <= self.normalized_base_height <= 0.01):
        #     is_unrecoverable = True
        #     print("height too low")
        # Torso of robot touches ground
        if (len(contact_points) > 0):
            is_unrecoverable = True
            # print("touch ground")
        # Torso of robot is very close to ground
        if (len(closest_points) > 0):
            is_unrecoverable = True
            # print("too close to ground")
        # Pitch and Roll is too large
        if (abs(pitch) > math.radians(40) or abs(roll) > math.radians(40)):
            is_unrecoverable = True
        return is_unrecoverable

    def compute_weight(self):
        total_mass = 0
        base_link_info = p.getDynamicsInfo(self.robot, -1)  # Get dynamics info of base link (-1)
        base_link_mass = base_link_info[0]
        total_mass += base_link_mass
        # Iterate through each link in the robot
        for link_idx in range(self.num_of_joints):
            # Get the link's URDF data
            link_urdf_data = p.getDynamicsInfo(self.robot, link_idx)

            # Get the link's mass
            link_mass = link_urdf_data[0]
            total_mass += link_mass
        return total_mass*9.81



if __name__ == "__main__":
    env = LeggedEnv(use_gui=True)
    env.reset()
    done = False
    t = 0
    while not done:
        # print("on_ground", env.check_no_feet_on_ground())
        obs, reward, done = env.cpg_step(t)
        t+=(1/240)
    # env.process_and_cmd_vel()
    

        
