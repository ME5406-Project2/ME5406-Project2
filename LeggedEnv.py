import gym
import pybullet as p
import pybullet_data
import time
import numpy as np
from Surface import Surface

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

        # Load plane to the world and set gravity
        p.setAdditionalSearchPath(pybullet_data.getDataPath()) 
        planeId = p.loadURDF("plane.urdf")
        p.setGravity(0, 0, -9.81)

        self.wood = Surface(
            texture_path="wood.png",
            lateral_friction=1.0,
            spinning_friction=1.0,
            rolling_friction=0.0)

        self.spawn_world()

    def spawn_robot(self):
        """
        Instantiates the robot in the simulation.
        """
        # Set the start pose of the robot
        self.robot_start_pos = [0, 0, 0]
        self.robot_start_rpy = [0, 0, 0]
        self.robot_start_orn = p.getQuaternionFromEuler(self.robot_start_rpy)
        
        # Load the robot URDF into PyBullet
        self.robot = p.loadURDF(
            "urdf/robot.urdf", 
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

        # Set all the motors to velocity control
        p.setJointMotorControlArray(self.robot, self.actuators, controlMode=p.VELOCITY_CONTROL)

    def spawn_world(self):
        while True:
            p.stepSimulation()

if __name__ == "__main__":
    env = LeggedEnv()