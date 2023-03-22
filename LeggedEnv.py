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
    def __init__(self):
        # Connect to PyBullet client
        self.physics_client = p.connect(p.GUI)

        # Set visualisation
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 0)
        p.configureDebugVisualizer(p.COV_ENABLE_GUI, 0)
        
        # Store observations in dict
        self.obs = {}

        # Define action and observation spaces
        self.observation_space = gym.spaces.Box(
            low=-np.inf, high=np.inf, shape=(44,), dtype=np.float32)
        self.action_space = gym.spaces.Box(
            low=-1, high=1, shape=(8,5), dtype=np.float32
        )

        # Termination condition parameter
        self.termination_pos_dist = 0.2
        self.max_steps = 10000
        self.env_step_count = 0

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

    def spawn_robot(self):
        """
        Instantiates the robot in the simulation.
        """
        # Set the start pose of the robot
        self.robot_start_pos = [0, 0, 0]
        self.robot_start_rpy = [90, 0, 0]
        self.robot_start_orn = p.getQuaternionFromEuler(self.robot_start_rpy)
        
        # Load the robot URDF into PyBullet
        self.robot = p.loadURDF(
            "assembly/Assem1.SLDASM/urdf/Assem1.SLDASM.urdf", 
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
        p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.POSITION_CONTROL)
        # p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.VELOCITY_CONTROL)

        # Upper and lower joint indeces
        self.upper_joint_indeces = [0, 2, 4, 6]
        self.lower_joint_indeces = [1, 3, 5, 7]

        # Enable force torque sensors on all actuator joints
        for joint in self.actuators:
            p.enableJointForceTorqueSensor(self.robot, joint, 1)

    def reset(self):
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
        time.sleep(1/240)

        return self.get_observation()
    
    def step(self, action):
        joint_velocities = []

        # Find the actions based on pre-defined mappings
        for joint, index in enumerate(action):
            joint_velocity = self.joint_to_action_map[joint][index]
            joint_velocities.append(joint_velocity)

        # Send action velocities to robot joints
        p.setJointMotorControlArray(env.robot, self.actuators, 
                                    p.VELOCITY_CONTROL, targetVelocities=joint_velocities)
        
        # Step the simulation
        p.stepSimulation()
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

        return observation, done
        
    def xyz_obj_dist_to_goal(self):

        dist = np.linalg.norm(self.base_pos - self.goal_pos)
        print(dist)
        return dist
    
    def generate_goal(self):
        
        box_pos = [1, 0, 0]
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
        
        leg_positions = env.cpg_position_controller(t)
        p.setJointMotorControlArray(env.robot, env.upper_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=-np.array(leg_positions))
        p.setJointMotorControlArray(env.robot, env.lower_joint_indeces, 
                                    p.POSITION_CONTROL, targetPositions=leg_positions)
        p.stepSimulation()
        time.sleep(1/240)
        self.env_step_count += 1

        # Get the observation
        observation = self.get_observation()

        # Terminating conditions
        # Reached goal
        if self.xyz_obj_dist_to_goal() < self.termination_pos_dist:
            done = True
            print("GOALREACHED")
        # Episode timeout
        elif self.env_step_count >= self.max_steps:
            done = True
            print('EPISODETERM')
        else:
            done = False
    

        return observation, done
    
    # def get_end_effector_force(self):

    #     # Append the joint torque forces for each leg
    #     joint_T_front_left = []
    #     joint_T_front_right = []
    #     joint_T_back_left = []
    #     joint_T_back_right = []

    #     # Robot end-effector position
    #     robot_legs_EE_pos = np.array(self.get_end_effector_pose()[0])

    #     # Robot end-effector position
    #     robot_legs_EE_orn = np.array(self.get_end_effector_pose()[1])
        
    #     joint_T_front_left.append(p.getJointState(self.robot, self.actuators[0])[3])
    #     joint_T_front_left.append(p.getJointState(self.robot, self.actuators[1])[3])

    #     p.calculateJacobian(self.robot, 1, )

    #     joint_T_front_right.append(p.getJointState(self.robot, self.actuators[2])[3])
    #     joint_T_front_right.append(p.getJointState(self.robot, self.actuators[3])[3])

    #     joint_T_back_left.append(p.getJointState(self.robot, self.actuators[4])[3])
    #     joint_T_back_left.append(p.getJointState(self.robot, self.actuators[5])[3])

    #     joint_T_back_right.append(p.getJointState(self.robot, self.actuators[6])[3])
    #     joint_T_back_right.append(p.getJointState(self.robot, self.actuators[7])[3])
        

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

    def get_observation(self):
        
        # Positions of all 8 joints
        self.joint_positions = np.array([p.getJointState(self.robot, env.actuators[i])[0] 
                                       for i in range(self.num_of_joints)])
        
        # Velocities of all 8 joints
        self.joint_velocities = np.array([p.getJointState(self.robot, env.actuators[i])[1] 
                                       for i in range(self.num_of_joints)])
        
        # Reaction forces of all 8 joints
        self.joint_reaction_forces = np.array([p.getJointState(env.robot, env.actuators[i])[2] 
                                       for i in range(self.num_of_joints)])
        
        # Linear and Angular velocity of the robot
        self.base_lin_vel = np.array(p.getBaseVelocity(self.robot)[0])
        self.base_ang_vel = np.array(p.getBaseVelocity(self.robot)[1])

        # Robot Position
        self.base_pos = np.array(p.getBasePositionAndOrientation(self.robot)[0])

        # Robot Orientation (Quaternion)
        self.base_orn = np.array(p.getBasePositionAndOrientation(self.robot)[1])

        # Robot Orientation (Euler Angles)
        self.base_rpy = np.array(p.getEulerFromQuaternion(self.base_orn))

        # Robot end-effector position
        self.robot_legs_EE_pos = np.array(self.get_end_effector_pose()[0])

        # Robot end-effector position
        self.robot_legs_EE_orn = np.array(self.get_end_effector_pose()[1])

        # Goal position
        self.goal_pos = np.array(p.getBasePositionAndOrientation(self.goal_id)[0])

        # Goal orientation
        self.goal_orn = np.array(p.getBasePositionAndOrientation(self.goal_id)[1])

        observation = np.hstack([
            *self.base_lin_vel,
            *self.base_ang_vel,
            *self.base_pos,
            *self.base_orn,
            *self.robot_legs_EE_pos,
            *self.robot_legs_EE_orn
        ])

        return observation


if __name__ == "__main__":
    env = LeggedEnv()
    env.reset()
    done = False
    t = 0
    while not done:
        obs, done = env.cpg_step(t)
        t+=(1/240)


        
