import numpy as np

# Define the joint limits in the world frame
lower_limit_world = -np.pi/2
upper_limit_world = np.pi/2

# Define the transformation matrix that defines the orientation of the joint frame in the world frame
euler_angles = [0, 0, 0]  # Assumes the joint axis is aligned with the z-axis
R_joint_parent = np.eye(3)
R_joint_parent[:3, :3] = np.array([
    [np.cos(euler_angles[2]), -np.sin(euler_angles[2]), 0],
    [np.sin(euler_angles[2]), np.cos(euler_angles[2]), 0],
    [0, 0, 1]
])

# Define the inverse transformation matrix that defines the orientation of the parent frame in the joint frame
R_parent_joint = np.linalg.inv(R_joint_parent)

R_joint_world = R_parent_joint.dot(R_joint_parent)
R_world_joint = R_joint_world.T

# Transform the joint limits from the world frame to the joint frame
joint_limits_world = [lower_limit_world, 0, 0], [upper_limit_world, 0, 0]
joint_limits_joint = []
for joint_limit_world in joint_limits_world:
    joint_limit_joint = np.dot(R_world_joint, joint_limit_world)
    joint_limits_joint.append(joint_limit_joint[2])

lower_limit_joint, upper_limit_joint = joint_limits_joint
print(f"Joint limits in the joint frame: [{lower_limit_joint}, {upper_limit_joint}]")