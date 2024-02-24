from diffuserslib.batch.BatchRunner import evaluateArguments
from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
from diffuserslib.functional.types import Vector, MovingBody, MovingBodiesFuncType


class BouncingPoints2DNode(FunctionalNode):
    def __init__(self, 
                 init_bodies:MovingBodiesFuncType,
                 dt:FloatFuncType = 0.1,
                 name:str = "bouncing_points_2d"):
        super().__init__(name)
        self.addParam("init_bodies", init_bodies, List[MovingBody])
        self.addParam("dt", dt, float)
        self.bodies:List[MovingBody] = []
        self.reset()


    def reset(self):
        super().reset()
        params = self.evaluateParams()
        self.bodies = params["init_bodies"]


    def process(self, init_bodies:List[MovingBody], dt:float) -> List[Vector]:
        positions:List[Vector] = []
        for body in self.bodies:
            body.update_position(dt)
            positions.append(body.position)
        return positions
    