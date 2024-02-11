from diffuserslib.functional.FunctionalNode import *
from diffuserslib.functional.FunctionalTyping import *
import random


class RandomFloatNode(FunctionalNode):
    """ Select a random number between min and max """
    def __init__(self, min:FloatFuncType=0, max:FloatFuncType=1, name:str="random_int"):
        super().__init__(name)
        self.addParam("min", min, TypeInfo(ParamType.FLOAT))
        self.addParam("max", max, TypeInfo(ParamType.FLOAT))
        
        
    def process(self, min:float, max:float) -> float:
        return random.uniform(min, max)
