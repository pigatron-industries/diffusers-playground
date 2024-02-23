from diffuserslib.batch.BatchRunner import evaluateArguments
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody, MovingBodiesFuncType
import numpy as np


class BouncingPoints2DNode(FunctionalNode):
    def __init__(self, 
                 init_bodies:MovingBodiesFuncType,
                 name:str = "bouncing_points_2d"):
        super().__init__(name)
        self.addParam("init_bodies", init_bodies, List[MovingBody])
        self.bodies:List[MovingBody] = []
        self.dt = 0.1
        self.init_frames()


    def init_frames(self):
        params = self.evaluateParams()
        self.bodies = params["init_bodies"]


    def process(self, init_bodies:List[MovingBody]) -> List[Vector]:
        positions:List[Vector] = []
        for body in self.bodies:
            body.update_position(self.dt)
            positions.append(body.position)
        return positions
    