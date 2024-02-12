from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import random


class RandomIntNode(FunctionalNode):
    """ Select a random number between min and max """
    def __init__(self, min_max:Tuple[int, int]=(0, 100), name:str="random_int"):
        super().__init__(name)
        self.addParam("min_max", min_max, TypeInfo(ParamType.INT, size=2, labels=["min", "max"]))
        
        
    def process(self, min_max:Tuple[int, int]) -> int:
        return random.randint(min_max[0], min_max[1])
