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

    def getValue(self):
        return []
    
    def setValue(self, value):
        pass

    def processValue(self) -> List[Vector]:
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
                 power_x:FloatFuncType = 1.0,
                 power_y:FloatFuncType = 1.0,
                 mirror_x:FloatFuncType = 0,
                 mirror_y:FloatFuncType = 0):
        super().__init__(name)
        self.addParam("num_points", num_points, int)
        self.addParam("power_x", power_x, float)
        self.addParam("power_y", power_y, float)
        self.addParam("mirror_x", mirror_x, float)
        self.addParam("mirror_y", mirror_y, float)


    def process(self, num_points:int, power_x:float, power_y:float, mirror_x:float, mirror_y:float) -> List[Vector]:
        vectors:List[Vector] = []
        xs = np.random.power(a=power_x, size=num_points)
        ys = np.random.power(a=power_y, size=num_points)
        for i in range(num_points):
            x = self.randomizeSide(xs[i], mirror_x) if mirror_x > 0 else xs[i]
            y = self.randomizeSide(ys[i], mirror_y) if mirror_y > 0 else ys[i]
            vectors.append(Vector(x, y))
        return vectors


    def randomizeSide(self, num, ratio):
        if (np.random.random() < 0.5):
            return ratio + num*(1-ratio)
        else:
            return ratio - num*ratio
