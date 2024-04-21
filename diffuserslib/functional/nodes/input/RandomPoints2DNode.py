from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector
from diffuserslib.functional.nodes.user import *
import itertools
import numpy as np


class RandomPoints2DNode(UserInputNode):
    """ This is a placeholder to allow and actual node to be selected by the user """
    def __init__(self, 
                 name:str = "random_points_2d"):
        super().__init__(name)

    def process(self) -> List[Vector]:
        return []


class RandomPoints2DUniformNode(FunctionalNode):
    def __init__(self, 
                 name:str = "random_points_2d_uniform",
                 num_points:IntFuncType = 20):
        super().__init__(name)
        self.addParam("num_points", num_points, int)


    def process(self, num_points:int) -> List[Vector]:
        vectors:List[Vector] = []
        points = np.random.uniform(low=0, high=1, size=(num_points, 2))
        for point in points:
            vectors.append(Vector(*point))
        return vectors
    

class RandomPoints2DEdgePowerNode(FunctionalNode):
    def __init__(self, 
                 name:str = "random_points_2d_edge_power",
                 num_points:IntFuncType = 20,
                 power:FloatFuncType = 1.0):
        super().__init__(name)
        self.addParam("num_points", num_points, int)
        self.addParam("power", power, float)


    def process(self, num_points:int, power:float) -> List[Vector]:
        vectors:List[Vector] = []

        points = np.random.power(a=power, size=(num_points, 2))
        for i in itertools.product(*[range(s) for s in (num_points, 2)]):
            if (np.random.random() < 0.5):
                points[i] = 1 - points[i]

        for point in points:
            vectors.append(Vector(*point))
        return vectors
