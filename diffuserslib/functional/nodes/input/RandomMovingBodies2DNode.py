from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody
import numpy as np
import random
import math
from typing import List


class RandomMovingBodies2DNode(FunctionalNode):
    def __init__(self, 
                 name:str = "random_moving_bodies",
                 num_bodies:IntFuncType = 20,
                 speed:FloatFuncType = 0.1):
        super().__init__(name)
        self.addParam("num_bodies", num_bodies, int)
        self.addParam("speed", speed, float)


    def process(self, num_bodies:int, speed:float) -> List[MovingBody]:
        moving_bodies = []
        for _ in range(num_bodies):
            position = Vector(random.random(), random.random())
            angle = random.uniform(0, 2*math.pi)
            velocity_x = speed * math.cos(angle)
            velocity_y = speed * math.sin(angle)
            velocity = Vector(velocity_x, velocity_y)
            moving_bodies.append(MovingBody(position, velocity))
        return moving_bodies
    