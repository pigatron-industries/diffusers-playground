from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import numpy as np


class RandomPoints2DNode(FunctionalNode):
    def __init__(self, 
                 name:str = "random_points_2d",
                 num_points:IntFuncType = 20):
        super().__init__(name)
        self.addParam("num_points", num_points, int)


    def process(self, num_points:int) -> Points2DType:
        return list(np.random.uniform(low=0, high=1, size=(num_points, 2)))
    