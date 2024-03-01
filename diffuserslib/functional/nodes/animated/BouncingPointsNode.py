from diffuserslib.batch.BatchRunner import evaluateArguments
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody, MovingBodiesFuncType


class BouncingPoints2DNode(FunctionalNode):
    def __init__(self, 
                 init_bodies:MovingBodiesFuncType,
                 dt:FloatFuncType = 0.01,
                 name:str = "bouncing_points_2d"):
        super().__init__(name)
        self.addInitParam("init_bodies", init_bodies, List[MovingBody])
        self.addParam("dt", dt, float)
        self.bodies:List[MovingBody] = []
        self.reset()


    def init(self, init_bodies:List[MovingBody]):
        self.bodies = init_bodies


    def process(self, dt:float) -> List[Vector]:
        positions:List[Vector] = []
        for body in self.bodies:
            body.update_position(dt)
            positions.append(body.position)
        return positions
    