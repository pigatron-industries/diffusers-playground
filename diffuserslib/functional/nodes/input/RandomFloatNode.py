from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
import random


class RandomFloatNode(FunctionalNode):
    """ Select a random number between min and max """
    def __init__(self, min_max:MinMaxFloatFuncType=(0, 1), name:str="random_int"):
        super().__init__(name)
        self.addParam("min_max", min_max, Tuple[float, float])
        
        
    def process(self, min_max:Tuple[float, float]) -> float:
        return random.uniform(min_max[0], min_max[1])
