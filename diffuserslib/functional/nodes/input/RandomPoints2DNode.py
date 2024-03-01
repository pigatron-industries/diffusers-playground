from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
from diffuserslib.functional.types import Vector
import itertools
import numpy as np


class RandomPoints2DNode(FunctionalNode):
    def __init__(self, 
                 name:str = "random_points_2d",
                 num_points:IntFuncType = 20):
        super().__init__(name)
        self.addParam("num_points", num_points, int)


    def process(self, num_points:int) -> List[Vector]:
        vectors:List[Vector] = []
        points = np.random.uniform(low=0, high=1, size=(num_points, 2))
        for point in points:
            vectors.append(Vector(*point))
        return vectors
    

# TODO allow different distribution functions to be selected
    
def edge_power_distribution(a, size):
    power_distribution = np.random.power(a=a, size=size)
    for idx in itertools.product(*[range(s) for s in size]):
        if (np.random.random() < 0.5):
            power_distribution[idx] = 1 - power_distribution[idx]
    return power_distribution.tolist()

def uniform_distribution(size):
    return np.random.uniform(low=0, high=1, size=size).tolist()