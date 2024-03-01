from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.types.FunctionalTyping import *
import random


class RandomIntNode(FunctionalNode):
    """ Select a random number between min and max """
    def __init__(self, min_max:MinMaxIntFuncType=(0, 100), name:str="random_int"):
        super().__init__(name)
        self.addParam("min_max", min_max, Tuple[int, int])
        
        
    def process(self, min_max:Tuple[int, int]) -> int:
        return random.randint(min_max[0], min_max[1])
