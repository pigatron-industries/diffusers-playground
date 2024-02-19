from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import numpy as np


class BoundingPoints2DNode(FunctionalNode):
    def __init__(self, 
                 points:Points2DFuncType,
                 name:str = "bounce_points_2d"):
        super().__init__(name)
        self.addParam("points", points, Points2DType)


    def process(self, points:Points2DType) -> Points2DType:
        # TODO
        return points 
    