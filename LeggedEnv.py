import math
import gym
import pybullet as p
import pybullet_data
import time
import numpy as np
from Surface import Surface
from gym.spaces import MultiDiscrete

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
        self.max_steps = 1000
        self.env_step_count = 0
        self.prev_dist = 0

        # Initialise rewards
        self.goal_reward = 0
        self.position_reward = 0
        self.time_reward = 0
        self.stability_reward = 0

        # Define joint-to-action mapping
        self.joint_to_action_map = {
            0: np.array([-10, -5, 0, 5, 10]),
            1: np.array([-10, -5, 0, 5, 10]),
            2: np.array([-10, -5, 0, 5, 10]),
            3: np.array([-10, -5, 0, 5, 10]),
            4: np.array([-10, -5, 0, 5, 10]),
            5: np.array([-10, -5, 0, 5, 10]),
            6: np.array([-10, -5, 0, 5, 10]),
            7: np.array([-10, -5, 0, 5, 10]),
        }
                
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
        
        # Define action space
        actions = [len(self.joint_to_action_map[key]) for key in range(len(self.joint_to_action_map))]
        self.action_space = MultiDiscrete(actions)

        # Define observation spaces
        obs_shape = self.get_observation().shape
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(33,), dtype=np.float64)

    def spawn_robot(self):
        """
        Instantiates the robot in the simulation.
        """
        # Set the start pose of the robot
        self.robot_start_pos = [0, 0, 0.5]
        self.robot_start_rpy = [math.radians(90), 0, 0]
        self.robot_start_orn = p.getQuaternionFromEuler(self.robot_start_rpy)
        
        # Load the robot URDF into PyBullet
        self.robot = p.loadURDF(
            "assembly/Assem1_v3.SLDASM/urdf/Assem1_v3.SLDASM.urdf", 
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

    def reset(self):
        # Reset reward and step count for episode
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

        # Step the simulation and return the initial observation
        p.stepSimulation()
        # time.sleep(1/240)

        return self.get_observation()
    
    def step(self, action):

        joint_velocities = []

        # Find the actions based on pre-defined mappings
        for joint, index in enumerate(action):
            joint_velocity = self.joint_to_action_map[joint][index]
            joint_velocities.append(joint_velocity)

        # Check if joint limits exceeded
        commanded_joint_positions = [0] * self.num_of_joints
        current_joint_positions = self.joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
        
        for index in range(self.num_of_joints):
            change_in_joint_pos = joint_velocities[index] * (1 / 240.0)
            commanded_joint_positions[index] = current_joint_positions[index] + change_in_joint_pos
        
        # Joint limits actually exceeded
        for index, joint_pos in enumerate(commanded_joint_positions):
            if joint_pos < -1.57 or joint_pos > 1.57:
                joint_velocities = self.prev_joint_velocities
                

        # Send action velocities to robot joints
        p.setJointMotorControlArray(self.robot, self.actuators, 
                                    p.VELOCITY_CONTROL, targetVelocities=joint_velocities)
        
        # Step the simulation
        p.stepSimulation()
        # time.sleep(1/240)
        self.env_step_count += 1
        
        # Get the observation
        observation = self.get_observation()

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
        # Episode timeout
        if self.env_step_count >= self.max_steps:
            done = True
        else:
            done = False

        self.prev_dist = self.xyz_obj_dist_to_goal()

        reward = self.get_reward()

        self.prev_joint_velocities = self.joint_velocities
    
        return observation, reward, done, {}
        
    def xyz_obj_dist_to_goal(self):

        dist = np.linalg.norm(self.base_pos - self.goal_pos)
        # print(dist)
        return dist
    
    def generate_goal(self):
        
        box_pos = [2, 0, 0]
        box_orn = p.getQuaternionFromEuler([0, 0, 0])

        self.box_collision_shape = p.createCollisionShape(p.GEOM_BOX,
                                                          halfExtents=[0.1, 0.1, 0.1])
        
        self.goal_id = p.createMultiBody(baseMass=1,
                                        baseCollisionShapeIndex=self.box_collision_shape,
                                        baseVisualShapeIndex=-1,
                                        basePosition=box_pos,
                                        baseOrientation=box_orn)
        
    def cpg_position_controller(self, t):
        
        # Set the CPG parameters
        self.frequency = 3
        self.amplitude = 0.3
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
        
        leg_positions = self.cpg_position_controller(t)
        p.setJointMotorControlArray(self.robot, self.upper_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=-np.array(leg_positions))
        p.setJointMotorControlArray(self.robot, self.lower_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=leg_positions)
        p.stepSimulation()
        # time.sleep(1/240)
        self.env_step_count += 1

        # Get the observation
        observation = self.get_observation()

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
            print("GOAL REACHED")
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
            print('EPISODE LENGTH EXCEEDED')
        else:
            done = False

        self.prev_dist = self.xyz_obj_dist_to_goal()

        reward = self.get_reward()
    
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

        legs_pos = np.array([front_left_EE_pos, front_right_EE_pos, 
                             back_left_EE_pos, back_right_EE_pos])
        
        legs_orn = np.array([front_left_EE_orn, front_right_EE_orn, 
                             back_left_EE_orn, back_right_EE_orn])

        return legs_pos, legs_orn
    
    def check_no_feet_on_ground(self):

        # Link IDs of the end-effectors
        # Respectively: FL, FR, BL, BR
        foot_link_ids = [1, 3, 5, 7]
        foot_contacts = [False] * 4

        for i, foot_id in enumerate(foot_link_ids):
            contact_points = p.getContactPoints(bodyA=self.robot, bodyB=self.surface.plane_id, linkIndexA=foot_id)
            if len(contact_points) > 0:
                foot_contacts[i] = True

        not_on_ground = all(element == False for element in foot_contacts)
        if not_on_ground:
            return True
        else:
            return False

    def get_observation(self):
        
        # Positions of all 8 joints
        self.joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
        
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

        # print(f"Position{self.base_pos}")
        # print(f"Ornrpy{self.base_rpy}")

        observation = np.hstack([
            *self.normalized_joint_angles,
            *self.normalized_joint_velocities,
            *self.normalized_base_lin_vel,
            *self.normalized_base_ang_vel,
            *self.normalized_base_orn,
            *self.normalized_rpy,
            np.array(self.relative_goal_dist, dtype=np.float64),
            np.array(self.relative_goal_vect, dtype=np.float64)
        ])

        # Additions to be made
        # CoG in robot frame

        return observation
    
    def get_reward(self):
        # Goal reached
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            self.goal_reward = 10
        # Robot is moving towards goal - Position
        elif self.prev_dist > self.xyz_obj_dist_to_goal():
            self.position_reward = 0.1 * self.xyz_obj_dist_to_goal()
            
        # Time-based penalty
        if self.env_step_count >= self.max_steps:
            self.time_reward = -0.05

        # Encourage stability
        # Value of 1 means perfect stability, 0 means complete instability
        roll = self.base_rpy[0]
        pitch = self.base_rpy[1]
        z_pos = self.base_pos[2]
        if 1 - abs(roll) - abs(pitch) - abs(z_pos - 0.275):
            self.stability_reward = 0.02

        # ADDITIONS TO BE MADE
        # Penalise staying in same place
        # Penalise sudden joint accelerations
        # Ensure that joint angles don't deviate too much

        # Sum of all rewards
        reward = (self.goal_reward + self.position_reward + 
                  self.time_reward + self.stability_reward)
        
        return reward

if __name__ == "__main__":
    env = LeggedEnv(use_gui=True)
    env.reset()
    done = False
    t = 0
    while not done:
        print("on_ground", env.check_no_feet_on_ground())
        obs, reward, done = env.cpg_step(t)
        t+=(1/240)

        
