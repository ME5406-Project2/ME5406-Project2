import pybullet as p
import pybullet_data
import time
import numpy as np

class Surface(object):
    """
    Instantiates the surface to be traversed on in the simulation.
    """
    def __init__(self, texture_path, lateral_friction, spinning_friction, rolling_friction):

        # Initialise variables to set the surface
        self.plane_visual = -1
        self.plane_pos = [0, 0, 0]
        self.plane_rpy = [0, 0, 0]
        self.plane_orn = p.getQuaternionFromEuler(self.plane_rpy)

        # Create a plane collision shape for the surface
        self.plane_collision = p.createCollisionShape(p.GEOM_PLANE)

        # Create a body for the surface using the collision shape
        self.plane_id = p.createMultiBody(
                baseCollisionShapeIndex=self.plane_collision,
                baseVisualShapeIndex=self.plane_visual,
                basePosition=self.plane_pos,
                baseOrientation=self.plane_orn)
        self.texture_id = p.loadTexture(texture_path)
        
        # Load texture image to reflect on surface
        p.changeVisualShape(self.plane_id, -1, textureUniqueId=self.texture_id)
        p.configureDebugVisualizer(p.COV_ENABLE_RENDERING, 1)
        
        # Set the friction of the surface
        p.changeDynamics(
            self.plane_id, -1, 
            lateralFriction=lateral_friction, 
            spinningFriction=spinning_friction, 
            rollingFriction=rolling_friction)