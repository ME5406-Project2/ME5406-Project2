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
        self.termination_pos_dist = 0.5 #0.5
        self.max_steps = 2000
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

        # Initialize isdead
        self.is_dead = False

        # Initialize previous actions
        self.prev_actions = np.zeros(shape=(8,))

        # Define joint-to-action mapping
        self.joint_to_action_map = {
             0: np.array([-10, -5, 0, 5, 10]),
             1: np.array([-10, -5, 0, 5, 10]),
             2: np.array([-10, -5, 0, 5, 10]),
             3: np.array([-10, -5, 0, 5, 10])
        }
        """
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
        """
        """
        self.joint_to_action_map = {
            0: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            1: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            2: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            3: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            4: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            5: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            6: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
            7: np.array([-45, -40, -35, -30, -25, -20, -15, -10, -5, 0, 
                         5, 10, 15, 20, 25, 30, 35, 40, 45]),
        }
        """         
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

        # self.action_space = gym.spaces.Box(
        #     low=-10, high=10, shape=(8,), dtype=np.float64)

        # robot weight
        self.weight = self.compute_weight()
        
        # Define observation spaces
        obs_shape = self.get_observation().shape
        self.observation_space = gym.spaces.Box(
            low=-1, high=1, shape=(obs_shape), dtype=np.float64)
        
        # time interval between each timesteps
        self.timestep = 0.004#0.004
        p.setTimeStep(self.timestep)

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
        self.robot_start_pos = [0, 0, 0.50]
        self.robot_start_rpy = [0, 0, 0]
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
        
        # Get max torque of the actuator joints
        self.max_joint_torques = [p.getJointInfo(self.robot, joint)[10] for joint in self.actuators]

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
        
        # Set all the motors to position or velocity or torque control
        p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.POSITION_CONTROL)
        # p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.VELOCITY_CONTROL)
        # p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.TORQUE_CONTROL)

        # Upper and lower joint indeces
        self.upper_joint_indeces = [0, 2, 4, 6]
        self.lower_joint_indeces = [1, 3, 5, 7]

        # Enable force torque sensors on all actuator joints
        for joint in self.actuators:
            p.enableJointForceTorqueSensor(self.robot, joint, 1)
        
        # wait until robot is stable for 100 consecutive counts
        stable_count = 0
        while (stable_count<100):
            p.stepSimulation()
            #stable_count += -1 if self.check_no_feet_on_ground() else 1
            #stable_count = max(0, stable_count)
            if self.check_no_feet_on_ground():
                stable_count = 0
            else:
                stable_count+=1

        # Get initial height of robot's COM
        self.robot_init_height = np.array(p.getBasePositionAndOrientation(self.robot)[0])[2]

    def reset(self):
        # # open the file in write mode
        # with open("jointvel.txt", "w") as file:
        #     # iterate over the list of lists
        #     for inner_list in self.store_joint_vel:
        #         # convert the inner list to a string
        #         inner_list_string = ",".join([str(x) for x in inner_list])

        #         # write the inner list as a line in the file
        #         file.write(inner_list_string + "\n")        
        # file.close()
        # Reset reward and step count for episode

        # with open("jointvel.txt", "r") as file:

        #     # read each line of the file and split it into a list
        #     for line in file:
        #         inner_list = [int(x) for x in line.strip().split(",")]
        #         self.store_joint_vel.append(inner_list)
        # print(self.reward)
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

        self.t = 0
        self.cpg_first = True

        # Reset previous actions
        self.prev_actions = np.zeros(shape=(8,))

        return self.get_observation()
    
    def step(self, action):
        self.cpg_first = False
        joint_velocities = []

        """
        0, #front left upper
        1, #front left lower
        2, #front right upper
        3, #front right lower
        4, #back left upper
        5, #back left lower
        6, #back right upper
        7  #back right lower
        """
        # Find the actions based on pre-defined mappings
        for joint, index in enumerate(action):
            joint_velocity = self.joint_to_action_map[joint][index]
            joint_velocities.append(joint_velocity)
        
        # Append the same action to the diagonal back legs
        joint_velocities.append(joint_velocities[2]) # apply front right upper actions to back left upper
        joint_velocities.append(joint_velocities[3]) # apply front right lower actions to back left lower
        joint_velocities.append(joint_velocities[0]) # apply front left upper actions to back right upper
        joint_velocities.append(joint_velocities[1]) # apply front left lower actions to back right lower

        # Check if joint limits exceeded
        commanded_joint_positions = [0] * self.num_of_joints
        current_joint_positions = self.joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                    for i in range(self.num_of_joints)])
        current_joint_velocities = np.array([p.getJointState(self.robot, self.actuators[i])[1] 
                                    for i in range(self.num_of_joints)])
        for index in range(self.num_of_joints):
            # find change in angle using s = ut + 0.5 a t^2
            joint_acceleration = (joint_velocities[index] - current_joint_velocities[index]) / self.timestep
            change_in_joint_pos = joint_velocities[index] * self.timestep + 0.5 * joint_acceleration * self.timestep ** 2
            commanded_joint_positions[index] = current_joint_positions[index] + change_in_joint_pos
        
        # Joint limits actually exceeded
        for index, joint_pos in enumerate(commanded_joint_positions):
            if joint_pos < -math.radians(45):
                commanded_joint_positions[index] = -math.radians(45)
            elif joint_pos > math.radians(90):
                commanded_joint_positions[index] = math.radians(90)
        
        #print("commanded_joint_positions", commanded_joint_positions)
        # # Joint limits actually exceeded
        # for index, joint_pos in enumerate(commanded_joint_positions):
        #     if joint_pos < -1.57 or joint_pos > 1.57:
        #         joint_velocities = self.prev_joint_velocities
        p.setJointMotorControlArray(self.robot, self.actuators,
                                    p.POSITION_CONTROL, targetPositions=commanded_joint_positions,
                                    forces=self.max_joint_torques)

        # Send action velocities to robot joints
        # p.setJointMotorControlArray(self.robot, self.actuators, 
        #                             p.VELOCITY_CONTROL, targetVelocities=joint_velocities)
        
        self.store_joint_vel.append(joint_velocities)
        
        # Step the simulation
        p.stepSimulation()
        # time.sleep(1/240)
        self.env_step_count += 1
        
        # Update previous actions
        # Convert actions back to range [-1,1]
        self.prev_actions = self.revert_actions(action)

        # Get the observation
        observation = self.get_observation()
        
        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
            print("GOAL REACHED")
        elif(self.check_is_unrecoverable()):
            done = True
            self.is_dead=True
            # print("Unrecoverable")
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
        else:
            done = False

        reward = self.get_reward()
        self.reward += 0
        self.prev_dist = self.xyz_obj_dist_to_goal()

        self.prev_joint_velocities = self.joint_velocities

        # else:

        #     leg_positions = self.cpg_position_controller(self.t)
        #     p.setJointMotorControlArray(self.robot, self.upper_joint_indeces, 
        #                                 p.POSITION_CONTROL, targetPositions=-np.array(leg_positions))
        #     p.setJointMotorControlArray(self.robot, self.lower_joint_indeces, 
        #                                 p.POSITION_CONTROL, targetPositions=leg_positions)
        #     p.stepSimulation()
        #     # time.sleep(1/240)
        #     self.env_step_count += 1
        #     # Get the observation
        #     observation = self.get_observation()

        #     reward = 0

        #     self.prev_joint_velocities = self.joint_velocities
        #     self.t+=1/240

        #     done = False

    
        return observation, reward, done, {}
        
    def xyz_obj_dist_to_goal(self):

        dist = np.linalg.norm(self.base_pos[0] - self.goal_pos[0])
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

        if self.check_no_feet_on_ground():
            self.cpg_cnt+=1

        leg_positions = self.cpg_position_controller(t)
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
                                    p.POSITION_CONTROL, targetPositions=-np.array(leg_positions))
        p.setJointMotorControlArray(self.robot, self.lower_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=leg_positions)
        # p.setJointMotorControlArray(self.robot, range(len(self.actuators)), 
        #                            p.VELOCITY_CONTROL, targetVelocities=[10, 5, 10, 10, 10, 0, -5, -5])
        p.stepSimulation()
        self.env_step_count += 1

        # Get the observation
        observation = self.get_observation()

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
            print("GOAL REACHED")
            print(self.reward)
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
            print('EPISODE LENGTH EXCEEDED')
        else:
            done = False

        reward = self.get_reward()
        self.reward += reward
        self.prev_dist = self.xyz_obj_dist_to_goal()
    
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
    
    def check_all_feet_on_ground(self):

        # Link IDs of the end-effectors
        # Respectively: FL, FR, BL, BR
        foot_link_ids = [1, 3, 5, 7]
        foot_contacts = [False] * 4

        for i, foot_id in enumerate(foot_link_ids):
            closest_points = p.getClosestPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=foot_id,
                                                distance=0.05)
            if len(closest_points) > 0:
                foot_contacts[i] = True
        # print(foot_contacts)
        # all_on_ground = all(element == True for element in foot_contacts)
        all_on_ground = foot_contacts.count(True)>2
        return all_on_ground


    def get_observation(self):
        
        # Positions of all 8 joints
        self.joint_positions = np.array([p.getJointState(self.robot, self.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
        # print(self.joint_positions)
        # relative joint angle in radians
        # range = [-pi/2, pi/2] -> divide by pi/2 to normalize to [-1,1]
        normalized_joint_angles = np.array(self.joint_positions / (np.pi / 2), dtype=np.float64)

        # Velocities of all 8 joints
        self.joint_velocities = np.array([p.getJointState(self.robot, self.actuators[i])[1] 
                                       for i in range(self.num_of_joints)])
        # v_max (set in urdf file) = p.getJointInfo()[11]
        # normalize joint (angular) velocities [-v_max,v_max] -> [-1,1] (divide by v_max)
        max_angular_velocity = np.array([self.actuators_info[i][11] for i in range(self.num_of_joints)])
        normalized_joint_velocities = np.array(np.divide(self.joint_velocities, max_angular_velocity), dtype=np.float64)

        # Reaction forces of all 8 joints
        joint_reaction_forces = np.array([p.getJointState(self.robot, self.actuators[i])[2] 
                                    for i in range(self.num_of_joints)])
        
        # NOTE f_max not defined in URDF (might be useful for slippage stuff?)
        # f_max = p.getJointInfo()[10]
        # normalize reaction forces [-f_max, f_max] -> [-1,1]
        # max_reaction_force = np.array([self.actuators_info[i][10] for i in range(self.num_of_joints)])
        # self.normalized_joint_reaction_forces = np.divide(self.joint_reaction_forces, max_reaction_force)

        # Linear and Angular velocity of the robot
        base_lin_vel = np.array(p.getBaseVelocity(self.robot)[0])
        base_ang_vel = np.array(p.getBaseVelocity(self.robot)[1])
        # print("baselinvel",self.base_lin_vel)

        # normalize linear and angular velocities [] -> [-1,1]
        # limits for linear and angular velocity cannot be set in URDF
        # need to find other methods to implement / constraint
        # proposed method to estimate linear velocity: model legs as a 2 wheel differential drive (overestimate)
        # length of leg = 0.575m, dist between legs = 0.565
        # lin_vel_max = pi*0.575 (assume all 4 legs in sync and max angle leg can cover in 1s pi rad)
        # ang_vel_max = 0.575*2pi / 0.565 ~= 2pi 
        base_lin_vel_limit, base_ang_vel_limit = 0.575*np.pi , 2*np.pi
        self.normalized_base_lin_vel = np.array(base_lin_vel / base_lin_vel_limit, dtype=np.float64)
        normalized_base_ang_vel = np.array(base_ang_vel / base_ang_vel_limit, dtype=np.float64)

        # Robot Position
        self.base_pos = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        # Height of robot's COM (normalized to -> [-1,1])
        self.normalized_base_height = (self.base_pos[2] - self.robot_init_height) / self.robot_init_height

        # Robot Orientation (Quaternion)
        self.base_orn = np.array(p.getBasePositionAndOrientation(self.robot)[1])

        # normalize orientation to [-1,1] by dividing by norm
        base_orn_length = np.sqrt(np.dot(self.base_orn, self.base_orn))
        normalized_base_orn = np.array(self.base_orn / base_orn_length, dtype=np.float64)

        # Robot Orientation (Euler Angles)
        self.base_rpy = np.array(p.getEulerFromQuaternion(self.base_orn))

        # normalize orientation (Euler Angles) [-pi, pi] -> [-1,1]
        normalized_rpy = np.array(self.base_rpy / np.pi, dtype=np.float64)

        # Robot end-effector position
        self.robot_legs_EE_pos = np.array(self.get_end_effector_pose()[0])

        # Robot end-effector position
        self.robot_legs_EE_orn = np.array(self.get_end_effector_pose()[1])

        # Goal position
        self.goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])

        # Goal orientation
        goal_orn = np.array(p.getBasePositionAndOrientation(self.goal_id)[1])

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
        relative_goal_dist = np.sqrt(np.dot(relative_goal_pos, relative_goal_pos))
        # Normalized distance to goal (initial distance to goal)
        self.normalized_goal_dist = relative_goal_dist / np.sqrt(np.dot(self.goal_pos, self.goal_pos))
        # Relative vector to goal (normalized)
        relative_goal_vect = relative_goal_pos / relative_goal_dist

        # Compute Center of Mass of Robot
        normalized_center_of_mass = self.compute_center_of_mass()

        # print(f"Position{self.base_pos}")
        # print(f"Ornrpy{self.base_rpy}")

        # Get contact forces and frictional forces
        contact_forces = []
        frictional_forces = []
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
            else:
                # contact_forces.append((contact_points[9] - average_force)/(average_force))
                # frictional_forces.append(contact_points[10])
                normal_forces = contact_points[0][7]
                normal_forces /= np.linalg.norm(normal_forces)
                contact_forces.append(np.array(normal_forces))
        # flatten contact forces
        contact_forces = np.concatenate(contact_forces)
        
        observation = np.hstack([
            #*self.prev_actions,
            *normalized_center_of_mass, #in paper
            *normalized_base_orn, #in paper
            *normalized_joint_velocities, #in paper
            *normalized_joint_angles,
            *self.normalized_base_lin_vel, #in paper
            *normalized_base_ang_vel, #in paper
            # np.array(self.normalized_base_height),
            contact_forces,
            *normalized_rpy,
            np.array(self.normalized_goal_dist, dtype=np.float64),
            np.array(relative_goal_vect, dtype=np.float64)
        ])

        return observation
    
    def get_reward(self):
        # Goal reached
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            self.goal_reward = 500
        else:
            self.goal_reward = 0
        # Robot is moving towards goal - Position
        self.position_reward = 100.0 * np.round(self.xyz_obj_dist_to_goal() - self.prev_dist, 3) #10
        # Reward increases as robot approaches goal
        # self.position_reward = 0.01 / 2**self.normalized_goal_dist
        # Robot is moving
        self.move_reward = 1.0*self.normalized_base_lin_vel[0] # 10.0 * self.base_lin_vel[0]
        self.move_reward = min(self.move_reward, 0.1)
        # self.move_reward = max(1, self.move_reward)
        # time step penalty
        # time_step_penalty = -0.005

        # alive reward
        alive_reward = 0.05

        dead_penalty = 0
        # robot is deemed to be in an unrecoverable / undesired position
        if self.is_dead:
            dead_penalty = -10 #-500
        # print("pos_reward", self.position_reward)
        # print("movreward", self.move_reward)

        # print("prev_dist", self.prev_dist)
        # print("xyz_dist_to_goal", self.xyz_obj_dist_to_goal())
            
        # Time-based penalty
        # if self.env_step_count >= self.max_steps:
        #     self.time_reward = -0.01

        # Encourage stability
        # Value of 1 means perfect stability, 0 means complete instability
        roll = self.base_rpy[0]
        pitch = self.base_rpy[1]
        z_pos = self.base_pos[2]
        # if 1 - abs(roll) - abs(pitch) - abs(z_pos - 0.275):
        #     self.stability_reward = 1 #0.1

        # penalize for too much tilting forward or backwards or sideways
        pitch_penalty = 0
        roll_penalty = 0
        #if abs(pitch) > math.radians(0):
        pitch_penalty = -5 * pitch**2
        if abs(roll) > math.radians(5):
            roll_penalty = -5 * roll**2
        
        # reward for diagonal legs moving in the same direction
        """
        0, #front left upper
        1, #front left lower
        2, #front right upper
        3, #front right lower
        4, #back left upper
        5, #back left lower
        6, #back right upper
        7  #back right lower
        """
        """
        def is_same_direction(a, b):
            if ((a <= 0 and b <= 0) or (a >= 0 and b >= 0)):
                return True
            return False
        same_leg_reward = 0
        # front left upper and back right upper
        if is_same_direction(self.joint_velocities[0], self.joint_velocities[6]):
            same_leg_reward += 1
        # front left lower and back right lower
        if is_same_direction(self.joint_velocities[1], self.joint_velocities[7]):
            same_leg_reward += 1
        # front right upper and back left upper
        if is_same_direction(self.joint_velocities[2], self.joint_velocities[4]):
            same_leg_reward += 1
        # front right lower and back right lower
        if is_same_direction(self.joint_velocities[3], self.joint_velocities[5]):
            same_leg_reward += 1
        same_leg_reward *= 0.05
        """
        # check if at least 3 feet are on the ground to encourage lifting of legs to walk
        # leg_penalty = 0
        # if self.check_all_feet_on_ground():
        #     leg_penalty = -0.05

        # if self.check_no_feet_on_ground():
        #     self.contact_reward = -0.01
        # ADDITIONS TO BE MADE
        # Penalise staying in same place
        # Penalise sudden joint accelerations
        # Ensure that joint angles don't deviate too much

        print("roll penalty:", roll_penalty)
        print("pitch penalty:", pitch_penalty)
        print("alive_reward:", alive_reward)
        print("move_reward:", self.move_reward)
        print("position_reward:", self.position_reward)
        print("self.goal_reward:", self.goal_reward)
        
        # Sum of all rewards
        reward = (self.goal_reward + alive_reward + pitch_penalty + roll_penalty + self.position_reward)
        print("total reward:", reward)
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
    
    def revert_actions(self, actions):
        original_actions = []
        # convert discrete actions back into values in range [-1,1]
        for index in range(len(self.joint_to_action_map)):
            num_actions = len(self.joint_to_action_map[index])
            # only able to recover original actions up to nearest 2/num_actions
            act = actions[index]*(2/num_actions)-1
            original_actions.append(act)
        return np.array(original_actions)

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
        #     # print("height too low")
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
             
    def compute_center_of_mass(self):
        # Get the robot's world transform
        robot_world_transform = p.getBasePositionAndOrientation(self.robot)

        # Initialize variables for center of mass position and total mass
        center_of_mass_position = [0, 0, 0]
        total_mass = 0

        # Add the contribution of the base link's mass
        base_link_info = p.getDynamicsInfo(self.robot, -1)  # Get dynamics info of base link (-1)
        base_link_mass = base_link_info[0]
        total_mass += base_link_mass

        # Iterate through each link in the robot
        for link_idx in range(self.num_of_joints):
            # Get the link's info
            link_info = p.getLinkState(self.robot, link_idx, computeLinkVelocity=0)

            # Get the link's URDF data
            link_urdf_data = p.getDynamicsInfo(self.robot, link_idx)

            # Get the link's mass
            link_mass = link_urdf_data[0]

            # Get the link's local position of the center of mass
            link_local_com_position = link_info[2]

            # Compute the link's world transform by multiplying the link's local transform with the robot's world transform
            link_world_transform = p.multiplyTransforms(robot_world_transform[0], robot_world_transform[1], link_local_com_position, [0, 0, 0, 1])

            # Compute the link's center of mass position in the world frame by multiplying the link's mass with the link's world transform
            link_com_position = [link_mass * link_world_transform[0][0], link_mass * link_world_transform[0][1], link_mass * link_world_transform[0][2]]

            # Update the total mass
            total_mass += link_mass

            # Update the center of mass position
            center_of_mass_position[0] += link_com_position[0]
            center_of_mass_position[1] += link_com_position[1]
            center_of_mass_position[2] += link_com_position[2]

        # Divide the center of mass position by the total mass to get the weighted average
        center_of_mass_position[0] /= total_mass
        center_of_mass_position[1] /= total_mass
        center_of_mass_position[2] /= total_mass

        # Transform the center of mass position from the world frame to the robot frame using the inverse of the robot's world transform
        robot_world_transform_inv = p.invertTransform(robot_world_transform[0], robot_world_transform[1])
        center_of_mass_position_robot_frame = p.multiplyTransforms(robot_world_transform_inv[0], robot_world_transform_inv[1], center_of_mass_position, [0, 0, 0, 1])

        # print("Center of Mass Position in Robot Frame: ", center_of_mass_position_robot_frame[0])
        normalized_center_of_mass = np.array(center_of_mass_position_robot_frame[0]) / np.linalg.norm(center_of_mass_position_robot_frame[0])
        # print(normalized_center_of_mass)
        return normalized_center_of_mass

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

    def test(self):
        # Robot Position
        self.base_pos = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        # Height of robot's COM (normalized to -> [-1,1])
        self.normalized_base_height = (self.base_pos[2] - self.robot_init_height) / self.robot_init_height

        #print("robot init height",self.robot_init_height)
        #print("normalized base height", self.normalized_base_height)
        #print(np.array(p.getBasePositionAndOrientation(self.robot)[0]))
        if (self.normalized_base_height < 0 and not -0.1 <= self.normalized_base_height <= 0.1):
            print("negative", self.normalized_base_height)
            print(self.base_pos)
        closest_points = p.getClosestPoints(bodyA=self.robot, 
                                                bodyB=self.surface.plane_id, 
                                                linkIndexA=-1,
                                                distance=0.3)
        # print(len(closest_points))
        # print("total reward",self.get_reward())
        # print("dist rwd", self.position_reward)
        # self.prev_dist = self.xyz_obj_dist_to_goal()
        
        # Robot Orientation (Quaternion)
        #self.base_orn = np.array(p.getBasePositionAndOrientation(self.robot)[1])

        # normalize orientation to [-1,1] by dividing by norm
        #base_orn_length = np.sqrt(np.dot(self.base_orn, self.base_orn))
        #self.normalized_base_orn = np.array(self.base_orn / base_orn_length, dtype=np.float64)

        #print(self.normalized_base_orn)
        # Step the simulation
        p.stepSimulation()
        # print(env.get_observation())
        self.get_reward()

if __name__ == "__main__":
    env = LeggedEnv(use_gui=True)
    env.reset()
    done = False
    t = 0
    
    # env.test_step(joint_num=6)
    #env.compute_center_of_mass()
    #env.get_reward()
    while (True):
        env.test()
        
    """
    while not done:
        # print("on_ground", env.check_no_feet_on_ground())
        obs, reward, done = env.cpg_step(t)
        t+=(1/240)
    # env.process_and_cmd_vel()
    
    """
        
